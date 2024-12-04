import functools
from collections import defaultdict
from typing import Callable, Dict, List, Tuple

import torch
from transformers import BertPreTrainedModel

from siamese_network.text_utils import normalize_for_model


class BaseSiameseModel(BertPreTrainedModel):
    MAX_LENGTH = 256  # minilm model is 512, but 256 used in sentencetransformers
    tokenizer: Callable = None
    default_config = None

    def transformer_tokenize(self, normalized_texts: List[str], padding: str = "max_length") -> Dict:
        tokenized_texts = self.tokenizer(
            normalized_texts,
            padding=padding,
            truncation=True,
            max_length=self.MAX_LENGTH,
        )
        return tokenized_texts

    def normalize_tokenize(
        self, texts: List[str], padding: str = "max_length"
    ) -> Tuple[List[str], Dict]:
        normalized_texts = [normalize_for_model(t) for t in texts]
        tokenized_texts = self.transformer_tokenize(
            normalized_texts,
            padding=padding,
        )
        return normalized_texts, tokenized_texts

    @staticmethod
    def generate_keyword_target(input_ids, keywords, scores, tokenizer):
        kw_to_score = {k: s for k, s in zip(keywords, scores)}
        token_score = {
            tid: 1 - kw_to_score[k] for k in keywords for tid in tokenize_word(k, tokenizer)
        }  # higher score for lower freq
        return [token_score.get(tid, 0.0) for tid in input_ids]

    @staticmethod
    def get_detected_keywords(
        token_ids, attention_mask, keyword_score, tokenizer, token_score_threshold=0.1
    ) -> Dict[str, float]:
        """Get detected keywords over threshold from token scores"""
        query_keywords = {}
        special_tokens = set(tokenizer.all_special_tokens)
        i = 0
        while i < len(token_ids):
            decoded_token = tokenizer.decode(token_ids[i])
            if attention_mask[i] and not decoded_token.startswith("##") and decoded_token not in special_tokens:
                # a word might consist of multiple tokens, so we continue to include all after one over threshold
                start_i = i
                i += 1
                while i < len(token_ids) and attention_mask[i] and tokenizer.decode(token_ids[i]).startswith("##"):
                    i += 1  # include any suffix tokens
                # store max score
                decoded_combined_tokens = tokenizer.decode(token_ids[start_i:i])
                query_keywords[decoded_combined_tokens] = float(
                    max(max(keyword_score[start_i:i]), query_keywords.get(decoded_combined_tokens, 0))
                )
            else:
                i += 1
        # filter by threshold at the end
        return {tok: s for tok, s in query_keywords.items() if s >= token_score_threshold}

    def get_input_attention(self, bert_output, incoming_tokenized, threshold=0.01) -> Dict[str, float]:
        """tries to get the 'total attention fraction' of each input token, in a way that is undoubtedly wrong"""
        attention_sum = 0
        for a in bert_output.attentions:
            attention_sum += a.sum(dim=1).squeeze(dim=0).sum(dim=0).detach().numpy()
        self_attention = list(attention_sum)

        word_attention = defaultdict(float)
        for i, (a, t, m) in enumerate(
            zip(self_attention, incoming_tokenized["input_ids"][0], incoming_tokenized["attention_mask"][0])
        ):
            if m:
                ts = self.tokenizer.decode(t)
                if ts.startswith("##"):
                    continue
                for j in range(i + 1, len(incoming_tokenized["input_ids"][0])):
                    tj = self.tokenizer.decode(incoming_tokenized["input_ids"][0][j])
                    if not tj.startswith("##"):  # concatenate sub-tokens
                        break
                    ts += tj[2:]
                if len(ts) >= 3:  # ignore short words even if they get attention
                    word_attention[ts] = max(word_attention[ts], a) if not ts.startswith("[") else 0
        wa_sum = sum(word_attention.values())
        top_words = sorted([(wa / wa_sum, str) for str, wa in word_attention.items()], reverse=True)  # sort for logs
        return {str: score for score, str in top_words if score > threshold}


@functools.lru_cache(maxsize=10000)
def tokenize_word(word, tokenizer):
    return tokenizer(word, add_special_tokens=False)["input_ids"]


# from huggingface
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
