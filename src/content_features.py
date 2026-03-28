import re
from nltk.corpus import stopwords


# Expanded & categorised scam keyword lexicon
SCAM_KEYWORDS = {
    # Urgency / pressure
    "urgent", "immediately", "expire", "expires", "limited", "act now",
    "today only", "hurry", "deadline",
    # Credential harvesting
    "verify", "verification", "login", "password", "username", "account",
    "confirm", "authenticate", "security", "update", "reactivate",
    # Financial lures
    "bank", "credit card", "wire transfer", "payment", "billing",
    "invoice", "refund", "cashback",
    # Prize / reward scams
    "free", "win", "winner", "reward", "bonus", "prize", "gift",
    "congratulations", "lucky", "claim",
    # Click-bait
    "click here", "click", "link", "download", "install",
    # Personal info harvesting
    "ssn", "social security", "date of birth", "passport", "address",
    # Generic phishing phrases
    "we noticed", "suspicious activity", "unusual activity",
    "your account has been", "has been compromised", "locked",
}


class ContentFeatureExtractor:

    def __init__(self):
        try:
            self._stop_words = set(stopwords.words("english"))
        except LookupError:
            import nltk
            nltk.download("stopwords", quiet=True)
            self._stop_words = set(stopwords.words("english"))

    # helpers
    def _clean_tokens(self, text: str) -> list[str]:
        text = text.lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        return [
            w for w in text.split()
            if w and w not in self._stop_words
        ]

    @staticmethod
    def _count_keyword_hits(tokens: list[str], text_lower: str) -> int:
        """
        Count both single-word and multi-word keyword matches.
        Single-word: token-level (fast).
        Multi-word: substring search on the lowercased full text.
        """
        hits = 0
        single = {kw for kw in SCAM_KEYWORDS if " " not in kw}
        multi  = {kw for kw in SCAM_KEYWORDS if " " in kw}

        token_set = set(tokens)
        hits += len(token_set & single)

        for phrase in multi:
            if phrase in text_lower:
                hits += 1

        return hits

    # public API

    def extract(self, title: str | None, text: str | None) -> dict:
        title = title or ""
        text  = text  or ""

        # Title is weighted 3× — it's the most visible part of a page
        weighted_combined = (title + " ") * 3 + text
        text_lower = weighted_combined.lower()

        tokens      = self._clean_tokens(weighted_combined)
        token_count = len(tokens)
        text_length = len(text)          # length of body only (un-weighted)

        scam_count = self._count_keyword_hits(tokens, text_lower)

        density = scam_count / token_count if token_count > 0 else 0.0

        # Extra surface signals
        has_form   = 1 if re.search(r"<form", text, re.IGNORECASE) else 0
        has_iframe = 1 if re.search(r"<iframe", text, re.IGNORECASE) else 0
        exclamation_count = text.count("!")
        caps_ratio = (
            sum(1 for c in text if c.isupper()) / len(text)
            if text else 0.0
        )

        return {
            "text_length":           text_length,
            "token_count":           token_count,
            "scam_keyword_count":    scam_count,
            "scam_keyword_density":  round(density, 6),
            "has_form":              has_form,
            "has_iframe":            has_iframe,
            "exclamation_count":     exclamation_count,
            "caps_ratio":            round(caps_ratio, 4),
        }