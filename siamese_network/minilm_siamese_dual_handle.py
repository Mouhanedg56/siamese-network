import copy
import functools

import numpy as np
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, MSELoss
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput

from siamese_network.base_siamese_model import BaseSiameseModel, mean_pooling

HANDLE_MAP = {0: 0, 1: 1, 2: 2}


@functools.lru_cache(maxsize=10)
def onehot_handle(handle_ix):
    vec = [0] * len(HANDLE_MAP)
    ix = HANDLE_MAP.get(handle_ix)
    if ix is not None:
        vec[ix] = 1
    return vec


ORGS_MAP = {
    262: 0,
    390: 1,
    142: 2,
    144: 3,
    529: 4,
    533: 5,
    28: 6,
    36: 7,
    553: 8,
    554: 9,
    300: 10,
    304: 11,
    436: 12,
    564: 13,
    58: 14,
    316: 15,
    577: 16,
    70: 17,
    71: 18,
    200: 19,
    75: 20,
    77: 21,
    461: 22,
    589: 23,
    470: 24,
    88: 25,
    473: 26,
    98: 27,
    358: 28,
    619: 29,
    375: 30,
    378: 31,
    251: 32,
    254: 33,
}


@functools.lru_cache(maxsize=100)
def onehot_org(org_ix):
    vec = [0] * len(ORGS_MAP)
    ix = ORGS_MAP.get(org_ix)
    if ix is not None:
        vec[ix] = 1
    return vec


class ResponseGenSiameseModelMiniDualHandle(BaseSiameseModel):
    BASE_MODELNAME = "sentence-transformers/all-MiniLM-L6-v2"

    @classmethod
    def ensure_loaded(cls, model_path=None):
        model_path = model_path if model_path else cls.BASE_MODELNAME
        if cls.default_config is None:
            cls.default_config = AutoConfig.from_pretrained(model_path)
        if cls.tokenizer is None:
            cls.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def __init__(self, config=None, model_path=None, num_mid_nodes=100, keyword_loss_weight=0.05):
        self.ensure_loaded(model_path)
        config = config or self.default_config
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = AutoModel.from_config(config=config, add_pooling_layer=False)
        self.extra_inputs = len(ORGS_MAP) + len(HANDLE_MAP)
        num_mid_nodes = num_mid_nodes or config.hidden_size
        num_inputs = self.extra_inputs + 3 * config.hidden_size
        self.pre_classifier = nn.Linear(num_inputs, num_mid_nodes)
        self.classifier = nn.Linear(num_mid_nodes, 1)
        self.dropout = nn.Dropout(0.1)  # not in bert config, so hardcoding

        self.keyword_loss_weight = keyword_loss_weight
        self.keyword_classifier = nn.Linear(config.hidden_size, 1)

        # Initialize weights and apply final processing
        if hasattr(self, "post_init"):  # version differences
            self.post_init()
        else:
            self.init_weights()

    def pooled_output(self, input_ids, attention_mask, token_type_ids, output_attentions=False, **kwargs):
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            **kwargs,
        )
        return bert_output, mean_pooling(bert_output, attention_mask)

    def format_extra_inputs(self, org_id, handle_type, n):
        """For inference. could be refactored w format_data"""
        extra_inputs = torch.tensor([onehot_org(org_id) + onehot_handle(min(handle_type, 2))])
        extra_inputs_expanded = extra_inputs.expand(n, -1)
        return extra_inputs_expanded

    def logits_score(self, pooled_output_query, pooled_output_reply, extra_inputs):
        # suggested to add abs diff by sbert paper
        tocat = [
            pooled_output_query,
            pooled_output_reply,
            torch.abs(pooled_output_query - pooled_output_reply),
            extra_inputs,
        ]
        pooled_output = torch.cat(tocat, dim=-1)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, num_labels)
        return logits

    def forward(
        self,
        input_ids_query=None,
        attention_mask_query=None,
        token_type_ids_query=None,
        input_ids_reply=None,
        attention_mask_reply=None,
        token_type_ids_reply=None,
        keyword_targets=None,
        labels=None,
        extra_inputs=None,
        return_dict=None,
    ):
        bert_output_query, pooled_output_query = self.pooled_output(
            input_ids_query, attention_mask_query, token_type_ids_query
        )
        _, pooled_output_reply = self.pooled_output(input_ids_reply, attention_mask_reply, token_type_ids_reply)

        logits = self.logits_score(pooled_output_query, pooled_output_reply, extra_inputs)

        loss_fct = BCEWithLogitsLoss(reduction="sum")
        loss = loss_fct(logits.squeeze(), labels.squeeze().float())

        if self.keyword_loss_weight:
            mse = MSELoss(reduction="sum")
            pred_keywords_score = torch.sigmoid(self.keyword_classifier(bert_output_query[0]).squeeze(dim=2))
            loss += self.keyword_loss_weight * mse(pred_keywords_score, keyword_targets)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )

    def format_data(self, examples):
        """tokenizes text, padding to max length and truncating if needed"""
        tokenize_helper = lambda texts, suffix, is_reply: {
            k + suffix: v for k, v in self.normalize_tokenize(texts, is_reply=is_reply)[1].items()
        }
        query_tok = tokenize_helper(examples["incoming_text"], "_query", False)
        reply_tok = tokenize_helper(examples["outgoing_text"], "_reply", True)
        extra_inputs = [onehot_org(o) + onehot_handle(h) for o, h in zip(examples["org_id"], examples["handle_type"])]
        keyword_targets = np.zeros((len(examples["keywords"]), len(query_tok["input_ids_query"][0])))
        for i, keywords in enumerate(examples["keywords"]):
            if keywords:
                keyword_targets[i] = self.generate_keyword_target(
                    query_tok["input_ids_query"][i], keywords, examples["keywords_score"][i], self.tokenizer
                )

        return {
            "labels": examples["score"],
            "keyword_targets": keyword_targets,
            "extra_inputs": extra_inputs,
            **query_tok,
            **reply_tok,
        }


def data_formatter_augment(samples):
    """augments teams data, used in training"""
    for i, db in enumerate(copy.copy(samples["db"])):
        if db == "teams":
            for augment in [
                {"incoming_text"},
                {"outgoing_text"},
                {"org_id"},
                {"handle_type"},
                {"org_id", "handle_type"},
            ]:
                for k, vec in samples.items():
                    value = samples[k][i]
                    if k in augment:
                        if isinstance(value, str):
                            value = value.lower()
                        else:
                            value = -1  # zero out handles
                    vec.append(value)
    return samples
