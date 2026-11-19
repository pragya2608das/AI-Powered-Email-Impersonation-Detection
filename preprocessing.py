"""
preprocessing.py
-----------------
Reusable text preprocessing utilities for the AI-Powered-Email-Impersonation-Detection.

The functions in this module clean raw email text
(messages, DMs, comments) before it is passed to the TF-IDF
vectorizer. Each cleaning step is implemented as its own function so
that the pipeline stays modular, testable, and easy to extend.
"""

import re
import string
from typing import List

try:
    import nltk
    from nltk.corpus import stopwords

    # Ensure the stopwords corpus is available. If it is not, download
    # it silently on first run so the project works out-of-the-box.
    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True)

    _STOPWORDS = set(stopwords.words("english"))
    _NLTK_AVAILABLE = True
except Exception:
    # If NLTK / stopwords corpus cannot be loaded for any reason, fall
    # back to a small built-in stopword list so the project never
    # crashes because of a missing NLTK resource.
    _NLTK_AVAILABLE = False
    _STOPWORDS = {
        "a", "an", "the", "is", "are", "am", "was", "were", "be", "been",
        "being", "in", "on", "at", "to", "for", "of", "and", "or", "but",
        "if", "with", "as", "by", "this", "that", "it", "you", "your",
        "i", "we", "they", "he", "she", "his", "her", "them", "my", "me",
    }

URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+)")
NUMBER_PATTERN = re.compile(r"\d+")
WHITESPACE_PATTERN = re.compile(r"\s+")
PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)


def to_lowercase(text: str) -> str:
    """Convert all characters in the text to lowercase."""
    return text.lower()


def remove_urls(text: str) -> str:
    """Remove URLs (http/https/www links) from the text."""
    return URL_PATTERN.sub(" ", text)


def remove_punctuation(text: str) -> str:
    """Remove all punctuation characters from the text."""
    return text.translate(PUNCTUATION_TABLE)


def remove_numbers(text: str) -> str:
    """Remove all standalone digits/numbers from the text."""
    return NUMBER_PATTERN.sub(" ", text)


def remove_extra_whitespace(text: str) -> str:
    """Collapse multiple whitespace characters into a single space
    and strip leading/trailing whitespace."""
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def remove_stopwords(text: str) -> str:
    """Remove common English stopwords from the text.

    This step is optional in the pipeline (see `clean_text`) because
    stopwords can sometimes carry weak signal for short, informal
    social-media messages. It is provided here as a reusable helper.
    """
    tokens: List[str] = text.split()
    filtered = [word for word in tokens if word not in _STOPWORDS]
    return " ".join(filtered)


def clean_text(text: str, remove_stopwords_flag: bool = False) -> str:
    """Run the full preprocessing pipeline on a single message.

    Steps (in order):
        1. Lowercase conversion
        2. URL removal
        3. Punctuation removal
        4. Number removal
        5. Extra whitespace removal
        6. Optional stopword removal

    Args:
        text: Raw input text (e.g. a social media message).
        remove_stopwords_flag: If True, also strips English stopwords.

    Returns:
        The cleaned, normalized text ready for TF-IDF vectorization.
    """
    if not isinstance(text, str):
        text = str(text)

    text = to_lowercase(text)
    text = remove_urls(text)
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = remove_extra_whitespace(text)

    if remove_stopwords_flag:
        text = remove_stopwords(text)

    return text


def clean_corpus(texts: List[str], remove_stopwords_flag: bool = False) -> List[str]:
    """Apply `clean_text` to a list/iterable of messages.

    Args:
        texts: An iterable of raw text messages.
        remove_stopwords_flag: If True, also strips English stopwords.

    Returns:
        A list of cleaned text strings, in the same order as input.
    """
    return [clean_text(t, remove_stopwords_flag=remove_stopwords_flag) for t in texts]


if __name__ == "__main__":
    # Quick manual sanity check when running this file directly.
    sample_messages = [
        "Verify your Instagram account immediately!! Click http://bit.ly/fake-verify NOW!!",
        "Are you coming to college tomorrow? Let's meet at 10am.",
    ]
    for msg in sample_messages:
        print(f"RAW    : {msg}")
        print(f"CLEANED: {clean_text(msg)}")
        print("-" * 60)
