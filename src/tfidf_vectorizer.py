import re
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.exceptions import NotFittedError

# Module-level constants — avoids class-scope reference issues in __init__
_TOKEN_PAT     = re.compile(r"\b[a-z0-9][a-z0-9]+\b", re.ASCII)
_TOKEN_PAT_STR = r"\b[a-z0-9][a-z0-9]+\b"


class TFIDFFeatureExtractor:
    """
    TF-IDF feature extractor with robust cleaning, safe fit/transform,
    and save/load support.
    """

    def __init__(
        self,
        max_features: int = 1200,
        max_df: float = 0.95,
        min_df: int = 1,
        sublinear_tf: bool = True,
    ):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            max_df=max_df,
            min_df=min_df,
            sublinear_tf=sublinear_tf,
            strip_accents=None,        # avoid Unicode normalizer mangling clean ASCII text
            analyzer="word",
            ngram_range=(1, 2),
            stop_words=None,
            lowercase=False,           # _clean() already lowercases; skip sklearn's unicode path
            norm="l2",
            dtype=np.float32,
            token_pattern=_TOKEN_PAT_STR,
        )
        self._is_fitted = False
        self._feature_names = []

    # ----------------------------------------
    # CORE METHODS
    # ----------------------------------------

    def fit_transform(self, texts, as_dataframe: bool = False):
        """Fit the vectorizer on texts and return the feature matrix."""
        cleaned = self._clean(texts)
        cleaned = [t[:5000] for t in cleaned]

        non_empty = [t for t in cleaned if t.strip() and t != "empty_doc"]
        if not non_empty:
            raise ValueError(
                "All input texts are empty after cleaning. "
                "Check your text_content column — it may be all URLs/HTML/numbers."
            )

        # Diagnostic (re.ASCII flag is Python 3.12 safe)
        sample_tokens = _TOKEN_PAT.findall(cleaned[0])
        total_tokens  = sum(len(_TOKEN_PAT.findall(t)) for t in cleaned)
        print(f"[TF-IDF] Sample tokens (doc 0): {sample_tokens[:10]}")
        print(f"[TF-IDF] Total tokens across train set: {total_tokens}")

        if total_tokens == 0:
            raise ValueError(
                "Token pattern matched 0 tokens. "
                f"First doc bytes: {cleaned[0].encode()[:100]}"
            )

        matrix = self.vectorizer.fit_transform(cleaned)
        self._is_fitted = True
        self._feature_names = self.vectorizer.get_feature_names_out()
        print(f"[TF-IDF] Vocabulary size: {len(self._feature_names)}")

        if as_dataframe:
            return pd.DataFrame(
                matrix.toarray(),
                columns=[f"tfidf_{f}" for f in self._feature_names],
            )
        return matrix

    def transform(self, texts, as_dataframe: bool = False):
        """Transform texts using the already-fitted vectorizer."""
        self._check_fitted()

        cleaned = self._clean(texts)
        cleaned = [t[:5000] for t in cleaned]

        matrix = self.vectorizer.transform(cleaned)

        if as_dataframe:
            return pd.DataFrame(
                matrix.toarray(),
                columns=[f"tfidf_{f}" for f in self._feature_names],
            )
        return matrix

    # ----------------------------------------
    # CLEANING
    # ----------------------------------------

    @staticmethod
    def _clean(texts) -> list[str]:
        """
        Clean raw texts:
          - Lowercase
          - Strip URLs, HTML tags
          - Keep ASCII letters + digits only
          - Encode/decode ASCII to strip hidden Unicode/BOM characters
          - Collapse whitespace
          - Fallback to 'empty_doc' so the vectorizer never sees a blank row
        """
        result = []
        for t in texts:
            t = str(t).lower().strip() if t is not None else ""
            t = re.sub(r"http\S+",     " ", t)   # remove URLs
            t = re.sub(r"<[^>]+>",     " ", t)   # remove HTML tags
            t = re.sub(r"[^a-z0-9\s]", " ", t)   # keep ASCII letters + digits
            t = re.sub(r"\s+",         " ", t).strip()
            t = t.encode("ascii", errors="ignore").decode("ascii")  # strip hidden Unicode/BOM
            result.append(t if t else "empty_doc")
        return result

    # ----------------------------------------
    # VALIDATION
    # ----------------------------------------

    def _check_fitted(self):
        if not self._is_fitted:
            raise NotFittedError(
                "TF-IDF vectorizer is not fitted yet. "
                "Call fit_transform() before transform()."
            )

    @property
    def vocabulary_size(self) -> int:
        self._check_fitted()
        return len(self._feature_names)

    @property
    def feature_names(self) -> list[str]:
        self._check_fitted()
        return list(self._feature_names)

    # ----------------------------------------
    # SAVE / LOAD
    # ----------------------------------------

    def save(self, path: str):
        """Save the fitted vectorizer to disk."""
        self._check_fitted()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, path)
        print(f"[TF-IDF] Saved vectorizer → {path}")

    def load(self, path: str):
        """Load a previously saved vectorizer from disk."""
        if not Path(path).exists():
            raise FileNotFoundError(f"No vectorizer found at: {path}")
        self.vectorizer = joblib.load(path)
        self._feature_names = self.vectorizer.get_feature_names_out()
        self._is_fitted = True
        print(f"[TF-IDF] Loaded vectorizer from {path} "
              f"(vocab size: {len(self._feature_names)})")