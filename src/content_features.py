import re
import string

from nltk.corpus import stopwords


SCAM_KEYWORDS = {
    "act now",
    "today only",
    "important notice",
    "reactivate",
    "reset password",
    "otp",
    "cashback",
    "winner",
    "reward",
    "bonus",
    "prize",
    "congratulations",
    "lucky",
    "claim your",
    "click here to claim",
    "ssn",
    "social security",
    "verify identity",
    "we noticed unusual",
    "suspicious activity",
    "unusual activity",
    "your account has been",
    "has been compromised",
    "account locked",
    "security alert",
    "account suspended",
    "crypto giveaway",
    "wallet phrase",
    "seed phrase",
    "send bitcoin",
}

MAX_TEXT_LENGTH_FEATURE = 250
MAX_TOKEN_COUNT_FEATURE = 30
MAX_EXCLAMATION_FEATURE = 4


class ContentFeatureExtractor:
    def __init__(self, title_weight: int = 3):
        self.title_weight = max(title_weight, 1)
        try:
            self._stop_words = set(stopwords.words("english"))
        except LookupError:
            import nltk

            nltk.download("stopwords", quiet=True)
            self._stop_words = set(stopwords.words("english"))

        self._single_keywords = {kw for kw in SCAM_KEYWORDS if " " not in kw}
        self._multi_keywords = {kw for kw in SCAM_KEYWORDS if " " in kw}
        self._punctuation_table = str.maketrans({char: " " for char in string.punctuation})

    def _normalize_text(self, text: str) -> str:
        text = text.lower().translate(self._punctuation_table)
        return re.sub(r"\s+", " ", text).strip()

    def _clean_tokens(self, text: str) -> list[str]:
        normalized = self._normalize_text(text)
        return [token for token in normalized.split() if token and token not in self._stop_words]

    def _count_keyword_hits(self, tokens: list[str], normalized_text: str) -> int:
        token_counts: dict[str, int] = {}
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1

        hits = sum(token_counts.get(keyword, 0) for keyword in self._single_keywords)

        for phrase in self._multi_keywords:
            pattern = rf"(?<!\w){re.escape(phrase)}(?!\w)"
            hits += len(re.findall(pattern, normalized_text))

        return hits

    @staticmethod
    def _letter_caps_ratio(text: str) -> float:
        letters = [char for char in text if char.isalpha()]
        if not letters:
            return 0.0
        upper_count = sum(1 for char in letters if char.isupper())
        return upper_count / len(letters)

    def extract(self, title: str | None, text: str | None, html: str | None = None) -> dict:
        title = (title or "").strip()
        text = (text or "").strip()
        html = html or ""

        weighted_combined = ((title + " ") * self.title_weight + text).strip()
        normalized_text = self._normalize_text(weighted_combined)
        tokens = self._clean_tokens(weighted_combined)

        raw_token_count = len(tokens)
        raw_text_length = len(text)
        avg_word_length = (
            sum(len(token) for token in tokens) / raw_token_count if raw_token_count else 0.0
        )

        scam_count = self._count_keyword_hits(tokens, normalized_text)
        density = scam_count / raw_token_count if raw_token_count else 0.0

        has_form = int(re.search(r"<form\b", html, re.IGNORECASE) is not None)
        has_iframe = int(re.search(r"<iframe\b", html, re.IGNORECASE) is not None)
        raw_exclamation_count = text.count("!")
        caps_ratio = self._letter_caps_ratio(text)

        return {
            "text_length": min(raw_text_length, MAX_TEXT_LENGTH_FEATURE),
            "token_count": min(raw_token_count, MAX_TOKEN_COUNT_FEATURE),
            "scam_keyword_count": scam_count,
            "scam_keyword_density": round(density, 6),
            "has_form": has_form,
            "has_iframe": has_iframe,
            "exclamation_count": min(raw_exclamation_count, MAX_EXCLAMATION_FEATURE),
            "caps_ratio": round(caps_ratio, 4),
            "avg_word_length": round(avg_word_length, 3),
        }
