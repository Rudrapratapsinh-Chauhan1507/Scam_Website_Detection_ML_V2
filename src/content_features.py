import re
from nltk.corpus import stopwords


class ContentFeatureExtractor:

    def __init__(self):

        self.stop_words = set(stopwords.words("english"))

        # phishing keywords
        self.scam_keywords = [
            "verify",
            "account",
            "password",
            "bank",
            "login",
            "update",
            "security",
            "confirm",
            "urgent",
            "click",
            "reward",
            "free",
            "win",
            "bonus",
            "limited",
            "offer"
        ]

    def clean_text(self, text):

        text = text.lower()

        text = re.sub(r'[^a-z\s]', '', text)

        tokens = text.split()

        tokens = [word for word in tokens if word not in self.stop_words]

        return tokens

    def extract(self, title, text):

        if text is None:
            text = ""

        combined = (title or "") + " " + text

        tokens = self.clean_text(combined)

        token_count = len(tokens)

        text_length = len(combined)

        scam_count = 0

        for word in tokens:

            if word in self.scam_keywords:

                scam_count += 1

        density = 0

        if token_count > 0:

            density = scam_count / token_count

        return {
            "text_length": text_length,
            "token_count": token_count,
            "scam_keyword_count": scam_count,
            "scam_keyword_density": density
        }