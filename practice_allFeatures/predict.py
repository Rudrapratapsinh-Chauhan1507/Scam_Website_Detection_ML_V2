import sys
import os
import re
import joblib
import numpy as np
import pandas as pd
from urllib.parse import urlparse

from src.scraper import WebsiteScraper
from src.url_features import URLFeatureExtractor
from src.domain_features import DomainFeatureExtractor
from src.content_features import ContentFeatureExtractor

# ================= CONFIG =================
MODEL_PATH = "./pkl_models"

model = None
tfidf = None
all_columns = None
scaler = None  # will remain None for RF


# ================= CLEAN TEXT =================
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else "empty_doc"


# ================= LOAD ARTIFACTS =================
def load_artifacts():
    global model, tfidf, all_columns, scaler

    # Load best model
    if os.path.exists(f"{MODEL_PATH}/random_forest.pkl"):
        model = joblib.load(f"{MODEL_PATH}/random_forest.pkl")
        print("[INFO] Loaded Random Forest model")
        scaler = None  # RF doesn't need scaling

    elif os.path.exists(f"{MODEL_PATH}/logistic_regression.pkl"):
        model = joblib.load(f"{MODEL_PATH}/logistic_regression.pkl")
        print("[INFO] Loaded Logistic Regression model")

        # Load scaler ONLY for LR
        scaler_path = f"{MODEL_PATH}/scaler.pkl"
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            print("[INFO] Scaler loaded")
        else:
            scaler = None
    else:
        raise Exception("❌ No trained model found!")

    # Load TF-IDF
    loaded = joblib.load(f"{MODEL_PATH}/tfidf_vectorizer.pkl")
    tfidf = loaded["vectorizer"] if isinstance(loaded, dict) else loaded

    # Load feature columns
    all_columns = joblib.load(f"{MODEL_PATH}/feature_columns.pkl")


# ================= FEATURE EXTRACTION =================
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


# ================= BUILD FEATURE VECTOR =================
def build_feature_vector(struct_features: dict, text: str):
    df_struct = pd.DataFrame([struct_features])
    df_struct = df_struct.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Clean text (IMPORTANT)
    text = clean_text(text)

    X_text = tfidf.transform([text]).toarray()
    tfidf_cols = [f"tfidf_{f}" for f in tfidf.get_feature_names_out()]

    df_text = pd.DataFrame(X_text, columns=tfidf_cols)

    df_final = pd.concat([df_struct, df_text], axis=1)

    # Align columns
    for col in all_columns:
        if col not in df_final.columns:
            df_final[col] = 0

    df_final = df_final[all_columns]

    return df_final.values, df_struct.iloc[0].to_dict()


# ================= PREDICT =================
def predict(url: str):
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    print("\n" + "=" * 60)
    print(f"[INFO] Processing URL: {url}")
    print("=" * 60)

    domain = urlparse(url).netloc.lower()

    # Extract features
    struct_features, text = extract_features(url)

    # Build vector
    X, struct_features = build_feature_vector(struct_features, text)

    # Scale if needed
    if scaler is not None:
        X = scaler.transform(X)

    # Model prediction
    proba = model.predict_proba(X)[0][1]

    # ================= HYBRID RISK (V2) =================
    risk_score = 0

    if struct_features.get("is_shortened") == 1:
        risk_score += 0.15

    if struct_features.get("suspicious_tld") == 1:
        risk_score += 0.20

    if struct_features.get("num_subdomains", 0) > 3:
        risk_score += 0.10

    # Combine ML + risk
    proba = 0.8 * proba + 0.2 * risk_score
    proba = max(0.0, min(1.0, proba))

    # ================= FINAL LABEL =================
    if proba >= 0.80:
        label = "🚨 SCAM (HIGH RISK)"
    elif proba >= 0.60:
        label = "⚠️ SUSPICIOUS"
    else:
        label = "✅ LEGIT"

    # ================= OUTPUT =================
    print("\n===== RESULT =====")
    print(f"Prediction : {label}")
    print(f"Confidence : {proba:.4f}")

    print("\n===== ANALYSIS =====")
    print(f"Domain            : {domain}")
    print(f"HTTPS             : {struct_features.get('https')}")
    print(f"URL Length        : {struct_features.get('url_length')}")
    print(f"Suspicious TLD    : {struct_features.get('suspicious_tld')}")
    print(f"Shortened URL     : {struct_features.get('is_shortened')}")

    print("=" * 60)


# ================= MAIN =================
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