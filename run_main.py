import argparse
import sys
import joblib
import pandas as pd

from src.scraper import WebsiteScraper
from src.url_features import URLFeatureExtractor
from src.domain_features import DomainFeatureExtractor
from src.content_features import ContentFeatureExtractor

MODEL_DIR = "./pkl_models"

def load_artifacts(model_name: str = "best") -> tuple:
    """
    Load model artefacts from pkl_models/.
    model_name: 'best' | 'logistic_regression' | 'decision_tree' | 'random_forest'
    """
    try:
        if model_name == "best":
            model_path = f"{MODEL_DIR}/scam_detector_model.pkl"
        else:
            model_path = f"{MODEL_DIR}/model_{model_name}.pkl"

        model           = joblib.load(model_path)
        feature_columns = joblib.load(f"{MODEL_DIR}/feature_columns.pkl")
        tfidf           = joblib.load(f"{MODEL_DIR}/tfidf_vectorizer.pkl")
        scaler          = joblib.load(f"{MODEL_DIR}/scaler.pkl")
        imputer         = joblib.load(f"{MODEL_DIR}/imputer.pkl")

    except FileNotFoundError as e:
        print(
            f"[ERROR] Model file missing: {e}\n"
            "Run train_model.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    return model, feature_columns, tfidf, scaler, imputer


def predict(url: str, threshold: float = 0.5, model_name: str = "best") -> dict:
    # Normalise URL
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    model, feature_columns, tfidf, scaler, imputer = load_artifacts(model_name)

    # Scrape
    print(f"[INFO] Scraping: {url}")
    scraper = WebsiteScraper()
    data    = scraper.scrape(url)

    if data["status"] == "failed":
        print("[WARN] Scraping failed, using limited features...")
    
    url = data.get("final_url", url)
    print(f"[DEBUG] Final URL used: {url}")

    # Feature extraction
    url_features     = URLFeatureExtractor().extract(url)
    domain_features  = DomainFeatureExtractor().extract(url)
    content_features = ContentFeatureExtractor().extract(
        data.get("title"), data.get("text")
    )

    combined     = {**url_features, **domain_features, **content_features}
    features_df  = pd.DataFrame([combined])

    # TF-IDF
    text = data.get("text") or ""
    tfidf_matrix = tfidf.transform([text])
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=tfidf.get_feature_names_out(),
    )

    features_df = pd.concat([features_df, tfidf_df], axis=1)

    # Drop duplicate columns (ssl_valid == has_ssl)
    features_df = features_df.loc[:, ~features_df.columns.duplicated()]

    # Align to training columns (fill missing with NaN, drop extras)
    features_df = features_df.reindex(columns=feature_columns)

    # Impute NaNs (same imputer used during training) 
    features_df = pd.DataFrame(
        imputer.transform(features_df),
        columns=feature_columns,
    )

    # Scale
    features_df = pd.DataFrame(
        scaler.transform(features_df),
        columns=feature_columns,
    )

    # Predict
    proba = model.predict_proba(features_df)[0]

    scam_prob  = float(proba[1])
    legit_prob = float(proba[0])

    # NEW: probability difference logic
    diff = abs(scam_prob - legit_prob)

    if diff < 0.15:
        label = "Suspicious ⚠️"
        confidence = max(scam_prob, legit_prob)
    elif scam_prob > legit_prob:
        label = "Scam 🚨"
        confidence = scam_prob
    else:
        label = "Legitimate ✅"
        confidence = legit_prob

    if "Scam" in label and scam_prob < 0.65:
        label = "Suspicious ⚠️"
        confidence = max(scam_prob, legit_prob)
    
    # Short URL detection (FINAL FIX)
    shorteners = ["bit.ly", "tinyurl.com", "t.co", "goo.gl", "rb.gy"]

    if any(s in url for s in shorteners):
        label = "Suspicious ⚠️"

        scam_prob = max(scam_prob, 0.6)
        legit_prob = 1 - scam_prob
        confidence = max(scam_prob, legit_prob)

    return {
        "url": url,
        "prediction": label,
        "confidence": round(confidence, 4),
        "scam_prob": scam_prob,
        "legit_prob": legit_prob,
        "model_used": model_name,
        "error": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Scam website predictor")
    parser.add_argument("--url",       help="URL to classify (prompted if omitted)")
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Probability threshold for 'Scam' prediction (default: 0.5)",
    )
    parser.add_argument(
        "--model",
        choices=["best", "logistic_regression", "decision_tree", "random_forest"],
        default="best",
        help="Which saved model to use (default: best CV model)",
    )
    args = parser.parse_args()

    url = args.url or input("Enter URL: ").strip()
    if not url:
        print("[ERROR] No URL provided.", file=sys.stderr)
        sys.exit(1)

    result = predict(url, threshold=args.threshold, model_name=args.model)

    print("\n" + "=" * 45)
    print(f"URL        : {result['url']}")
    print(f"Model      : {result.get('model_used', 'best')}")
    if result["error"]:
        print(f"[FAIL]  {result['error']}")
    else:
        # Icon selection
        if "Scam" in result["prediction"]:
            icon = "🚨"
        elif "Suspicious" in result["prediction"]:
            icon = "⚠️"
        else:
            icon = "✅"

        print(f"Prediction : {icon}  {result['prediction']}")
        print(f"Confidence : {result['confidence']:.2%}")

        # Show both probabilities (correct way)
        scam_prob = result.get("scam_prob")
        legit_prob = result.get("legit_prob")

        if scam_prob is not None and legit_prob is not None:
            print(f"Legit Prob : {legit_prob:.2%}")
            print(f"Scam Prob  : {scam_prob:.2%}")

print("=" * 45)
if __name__ == "__main__":
    main()