import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.exceptions import NotFittedError


class TFIDFFeatureExtractor:
    """
    Wrapper around sklearn's TfidfVectorizer with:
      - Safe transform (checks fitted state)
      - sublinear_tf for better term weighting
      - save / load helpers
      - direct DataFrame output option
    """

    def __init__(
        self,
        max_features: int = 100,
        ngram_range: tuple = (1, 2),
        max_df: float = 0.85,
        min_df: int = 1,          # 1 instead of 2 — safe for small datasets
        sublinear_tf: bool = True, # log-scale TF to reduce dominance of frequent words
    ):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            max_df=max_df,
            min_df=min_df,
            ngram_range=ngram_range,
            sublinear_tf=sublinear_tf,
            strip_accents="unicode",
            analyzer="word",
        )
        self._is_fitted = False

    # core methods

    def fit_transform(
        self, texts: list[str], as_dataframe: bool = False
    ) -> np.ndarray | pd.DataFrame:
        """Fit on texts and return transformed matrix."""
        cleaned = self._clean(texts)
        matrix = self.vectorizer.fit_transform(cleaned)
        self._is_fitted = True

        if as_dataframe:
            return pd.DataFrame(
                matrix.toarray(),
                columns=self.vectorizer.get_feature_names_out(),
            )
        return matrix

    def transform(
        self, texts: list[str], as_dataframe: bool = False
    ) -> np.ndarray | pd.DataFrame:
        """Transform new texts using the already-fitted vectorizer."""
        self._check_fitted()
        cleaned = self._clean(texts)
        matrix = self.vectorizer.transform(cleaned)

        if as_dataframe:
            return pd.DataFrame(
                matrix.toarray(),
                columns=self.vectorizer.get_feature_names_out(),
            )
        return matrix

    def get_feature_names(self) -> list[str]:
        """Return list of feature (token) names."""
        self._check_fitted()
        return self.vectorizer.get_feature_names_out().tolist()

    def vocab_size(self) -> int:
        """Number of features learned."""
        self._check_fitted()
        return len(self.vectorizer.vocabulary_)

    # persistence

    def save(self, path: str = "./pkl_models/tfidf_vectorizer.pkl") -> None:
        """Save the fitted vectorizer to disk."""
        self._check_fitted()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, path)
        print(f"[OK] TF-IDF vectorizer saved to {path}")

    def load(self, path: str = "./pkl_models/tfidf_vectorizer.pkl") -> None:
        """Load a previously saved vectorizer from disk."""
        if not Path(path).exists():
            raise FileNotFoundError(f"No vectorizer found at: {path}")
        self.vectorizer = joblib.load(path)
        self._is_fitted = True
        print(f"[OK] TF-IDF vectorizer loaded from {path}  "
              f"(vocab size: {len(self.vectorizer.vocabulary_)})")

    # helpers

    @staticmethod
    def _clean(texts: list[str]) -> list[str]:
        """Replace None / non-string entries with empty string."""
        return [t if isinstance(t, str) else "" for t in texts]

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise NotFittedError(
                "TFIDFFeatureExtractor is not fitted yet. "
                "Call fit_transform() or load() first."
            )