import joblib
import pandas as pd

from src.scraper import WebsiteScraper
from src.url_features import URLFeatureExtractor
from src.domain_features import DomainFeatureExtractor
from src.content_features import ContentFeatureExtractor

model = joblib.load("scam_detector_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")
scaler = joblib.load("scaler.pkl")

url = input("Enter URL: ")

scraper = WebsiteScraper()
data = scraper.scrape(url)

if data["status"] == "failed":
    print("Scraping failed:", data["error"])
    exit()

url_extract = URLFeatureExtractor()
url_features = url_extract.extract(url)

domain_extract = DomainFeatureExtractor()
domain_features = domain_extract.extract(url)

content_extract = ContentFeatureExtractor()
content_features = content_extract.extract(
    data["title"],
    data["text"]
)

features = {
    **url_features,
    **domain_features,
    **content_features
}

features_df = pd.DataFrame([features])

# tf-idf
tfidf_matrix = tfidf.transform([data["text"]])

tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=tfidf.get_feature_names_out()
)

features_df = pd.concat([features_df, tfidf_df], axis=1)

features_df = features_df.reindex(columns=feature_columns, fill_value=0)

features_df = pd.DataFrame(
    scaler.transform(features_df),
    columns=feature_columns
)

prob = model.predict_proba(features_df)[0][1]

if prob > 0.5:
    print("\nPrediction: Scam Website")
else:
    print("\nPrediction: Legitimate Website")

print("Confidence:", round(prob, 2))