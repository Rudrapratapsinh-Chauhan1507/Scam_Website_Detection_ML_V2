import sys
import joblib
import numpy as np
import pandas as pd
from urllib.parse import urlparse

from src.scraper import WebsiteScraper
from src.url_features import URLFeatureExtractor
from src.domain_features import DomainFeatureExtractor
from src.content_features import ContentFeatureExtractor

# LOAD MODELS
MODEL_PATH = "./pkl_models"
model = None
tfidf = None
scaler = None
all_columns = None


def load_artifacts():
    global model, tfidf, scaler, all_columns

    try:
        model = joblib.load(f"{MODEL_PATH}/random_forest.pkl")
        print("[INFO] Loaded Random Forest model")
    except:
        model = joblib.load(f"{MODEL_PATH}/logistic_regression.pkl")
        print("[INFO] Loaded Logistic Regression model")

    loaded = joblib.load(f"{MODEL_PATH}/tfidf_vectorizer.pkl")
    tfidf = loaded["vectorizer"] if isinstance(loaded, dict) else loaded

    all_columns = joblib.load(f"{MODEL_PATH}/feature_columns.pkl")

    try:
        scaler = joblib.load(f"{MODEL_PATH}/scaler.pkl")
        print("[INFO] Scaler loaded")
    except:
        scaler = None

# FEATURE EXTRACTION
def extract_features(url: str):
    scraper = WebsiteScraper()
    data = scraper.scrape(url)

    url_features = URLFeatureExtractor().extract(url)
    domain_features = DomainFeatureExtractor().extract(url)
    content_features = ContentFeatureExtractor().extract(
        data.get("title"), data.get("text")
    )

    features = {}
    features.update(url_features)
    features.update(domain_features)
    features.update(content_features)

    return features, data.get("text", "")

# BUILD FEATURE VECTOR
def build_feature_vector(struct_features: dict, text: str):
    df_struct = pd.DataFrame([struct_features])
    df_struct = df_struct.apply(pd.to_numeric, errors="coerce").fillna(0)

    text = str(text).strip() if text else "empty_doc"
    X_text = tfidf.transform([text]).toarray()

    tfidf_cols = [f"tfidf_{f}" for f in tfidf.get_feature_names_out()]
    df_text = pd.DataFrame(X_text, columns=tfidf_cols)

    df_final = pd.concat([df_struct, df_text], axis=1)

    for col in all_columns:
        if col not in df_final.columns:
            df_final[col] = 0

    df_final = df_final[all_columns]
    return df_final.values

# PREDICT
def predict(url: str):
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    print("\n" + "=" * 60)
    print(f"[INFO] Processing URL: {url}")
    print("=" * 60)

    domain = urlparse(url).netloc.lower()

    # TRUST SIGNALS (soft influence)
    trusted_keywords = [".edu", ".gov", ".ac.in", "bank", "university", "college"]

    struct_features, text = extract_features(url)

    # BUILD FEATURE VECTOR
    X = build_feature_vector(struct_features, text)

    if scaler is not None:
        X = scaler.transform(X)

    # MODEL PREDICTION
    proba = model.predict_proba(X)[0][1]

    # HYBRID ADJUSTMENTS (NO HARD OVERRIDE)
    adjustments = []

    # Domain age adjustment
    age = struct_features.get("domain_age_days", 0)
    if age and age > 1000:
        proba -= 0.15
        adjustments.append("Old domain (trusted signal)")

    if any(k in domain for k in trusted_keywords):
        proba -= 0.20
        adjustments.append("Trusted domain pattern")

    # Risk signals
    if struct_features.get("is_shortened") == 1:
        proba += 0.15
        adjustments.append("Shortened URL risk")

    if struct_features.get("suspicious_tld") == 1:
        proba += 0.20
        adjustments.append("Suspicious TLD")

    # Clamp
    proba = max(0.0, min(1.0, proba))
    # FINAL DECISION
    if proba >= 0.85:
        label = "🚨 SCAM (HIGH RISK)"
    elif proba >= 0.65:
        label = "⚠️ SUSPICIOUS"
    else:
        label = "✅ LEGIT"

    # OUTPUT
    print("\n===== RESULT =====")
    print(f"Prediction : {label}")
    print(f"Confidence : {proba:.4f}")

    # Explanation
    print("\n===== ANALYSIS =====")
    print(f"• Domain       : {domain}")
    print(f"• Domain Age   : {age} days")
    print(f"• HTTPS        : {struct_features.get('https')}")
    print(f"• URL Length   : {struct_features.get('url_length')}")
    print(f"• Suspicious TLD : {struct_features.get('suspicious_tld')}")

    if adjustments:
        print("\n[Adjustments Applied]")
        for adj in adjustments:
            print(f"  - {adj}")

    print("=" * 60)

# MAIN
if __name__ == "__main__":
    load_artifacts()

    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = input("Enter URL: ").strip()

    if not url:
        print("[ERROR] No URL provided.")
        sys.exit(1)

    predict(url)