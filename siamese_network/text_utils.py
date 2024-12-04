import re
import warnings
import unidecode
from bs4 import MarkupResemblesLocatorWarning
from emoji import demojize

import lxml.html
import regex
from bs4 import BeautifulSoup
from itertools import groupby


warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)


def build_regex_or(entries, regexes=False):
    if regexes:  # regexes, add brackets
        strings = ["(?:" + s + ")" for s in entries]
    else:  # strings
        strings = [re.escape(s) for s in entries]
    return "(?:" + "|".join(strings) + ")"


NAME_REGEX = r"(?:[A-Z]\w*(?:\s+\w+){0,3})"
HANDLE_REGEX = r"(?<!\w)@\w[\w._-]+\b"

_http_url_regex = re.compile(
    r"(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})"
)

def replace_url(text: str, replacement: str = "URL"):
    return _http_url_regex.sub(replacement, text)

def encode_html(message):
    """
    Encodes html characters -- mostly emojis, greater-than, less-than and ampersands.
    """

    # Non-breaking spaces ("&nbsp;") do not get converted, so they are manually removed
    message = message.replace("&nbsp;", " ").replace("\ufeff", "").strip()

    try:
        if regex.search(r"&\w{2,10};", message, flags=regex.I):
            message_w_emojis = lxml.html.fromstring(message).text

            if message_w_emojis is not None:
                return message_w_emojis

    except lxml.etree.ParserError:
        return message

    return message


def shorten_message(message, max_len=5000, short_len=2000):
    if len(message) > max_len:
        message = message[:short_len]
    return message


def standardize_message(message):
    """
    Standardize message for text_cleaning removal
    """

    # encodes unicode, including emojis
    message = encode_unicode(message)

    soup = BeautifulSoup(message, "lxml")
    message = soup.text

    message = shorten_message(message)

    message = message.replace("“", '"').replace("’", "'").replace("`", "'")
    message = message.replace("↵", "")
    message = message.replace("…", "...")
    message = regex.sub(r"[ ﻿‌​\s]+", " ", message)

    message = encode_html(message)

    # Reduce sequences of the same character to a maximum of 3, except a short list of characters
    # This includes emojis
    def join_groups(s):
        s = list(s)
        character = s[0]
        # TODO: Consider if we should reduce this list of characters that are allowed to repeat
        # Most of these characters are used as separators or replacements during anonymization
        if (not character.isdigit()) and character not in [".", "_", "X", "\t", "-"]:
            return "".join(s)[:3]
        else:
            return "".join(s)

    message = "".join(join_groups(s) for _, s in groupby(message))

    message = regex.sub(r"\s+", " ", message)
    return message


def encode_unicode(message):
    """
    Encodes unicode (including emojis) and handles unicode surrogates.

    The first encode/decode basically converts escape characters. Incoming text is inconsistant, so we need to be able to interpret "this\\n\\n" and "\t\tthis\n".

    The second encode/decode handles unicode surrogates. Unicode surrogates are combinations of utf-16 unicode characters that reference code points beyond the utf-16 scheme. For us, this mostly means emojis.
    ie: "\ud83c\udf10" = 🌐

    Surrogate characters will freeze the code (without raising an error!).

    The loop handles two (compounding) problems:
    1) Truncated unicode characters raise an error, and these can happen when we truncate a long message.
    ie: "hey \\u2834" becomes "hey ⠴", but "hey \\u283" will crash

    The loop removes the truncated unicode at the end of the message.

    2) If a message ends in a surrogate character, the first iteration of the loop will only remove the second character, with no guarantee the first character can be interpreted.

    The loop cleans up the remaining character in that case.

    It runs 3 times because you can't be too careful!
    """
    for _ in range(3):
        try:
            message = (
                message.encode("latin1", "backslashreplace")
                .decode("unicode-escape")
                .encode("utf-16", "surrogatepass")
                .decode("utf-16")
            )
            break  # It worked! No need for more encode attempts
        except UnicodeDecodeError:
            message = regex.sub(r"\\u\w+$", "", message)

    return message


def normalize_for_model(text, author=None):
    """Normalize the message for training and inference."""
    text = standardize_message(text)
    text = replace_url(text, replacement="URL")
    text = demojize(text, delimiters=(":", ": "))

    return unidecode.unidecode(text)


# for fast deduping
ignore_words = {"nbsp", "br"}


def basic_tokenize(text):
    return [w for w in re.findall(r"\w{2,}", text.lower()) if w not in ignore_words]
