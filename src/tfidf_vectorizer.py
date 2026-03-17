from sklearn.feature_extraction.text import TfidfVectorizer


class TFIDFFeatureExtractor:

    def __init__(self):

        self.vectorizer = TfidfVectorizer(
            max_features=130,
            stop_words="english",
            max_df=0.85,
            min_df=2,
            ngram_range=(1,2)
        )

    def fit_transform(self, texts):

        matrix = self.vectorizer.fit_transform(texts)

        return matrix

    def transform(self, texts):

        return self.vectorizer.transform(texts)

    def get_feature_names(self):

        return self.vectorizer.get_feature_names_out()