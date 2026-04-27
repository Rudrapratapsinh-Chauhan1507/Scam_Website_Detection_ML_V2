import argparse
import csv
from pathlib import Path

import joblib
import pandas as pd

from src.content_features import ContentFeatureExtractor
from src.scraper import WebsiteScraper
from src.url_features import URLFeatureExtractor


MODEL_PATH = Path("pkl_models") / "best_scam_detector.pkl"
FEATURES_PATH = Path("pkl_models") / "model_features.pkl"


def normalize_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url


def load_urls(args: argparse.Namespace) -> list[str]:
    urls = []

    if args.url:
        urls.append(args.url)

    if args.input:
        input_path = Path(args.input)
        if input_path.suffix.lower() == ".csv":
            frame = pd.read_csv(input_path)
            if "url" not in frame.columns:
                raise ValueError("Input CSV must contain a 'url' column.")
            urls.extend(frame["url"].dropna().astype(str).tolist())
        else:
            urls.extend(
                line.strip()
                for line in input_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )

    urls = [normalize_url(url) for url in urls]
    return [url for url in urls if url]


def build_feature_frame(feature_columns: list[str], features: dict) -> pd.DataFrame:
    frame = pd.DataFrame([features])
    for column in feature_columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[feature_columns]


def predict_url(
    url: str,
    model,
    feature_columns: list[str],
    scraper: WebsiteScraper,
    url_extractor: URLFeatureExtractor,
    content_extractor: ContentFeatureExtractor,
) -> dict:
    scraped = scraper.scrape(url)
    final_url = scraped.get("final_url") or url

    features = {}
    features.update(url_extractor.extract(final_url))
    features.update(
        content_extractor.extract(
            title=scraped.get("title"),
            text=scraped.get("text"),
            html=scraped.get("html"),
        )
    )

    feature_frame = build_feature_frame(feature_columns, features)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(feature_frame)[0]
        scam_probability = float(probabilities[1])
    else:
        scam_probability = float(model.predict(feature_frame)[0])

    prediction = int(scam_probability >= 0.5)
    prediction_name = "SCAM" if prediction == 1 else "LEGIT"
    confidence = scam_probability if prediction == 1 else 1.0 - scam_probability

    return {
        "url": url,
        "final_url": final_url,
        "status": scraped.get("status"),
        "error": scraped.get("error"),
        "prediction": prediction_name,
        "prediction_label": prediction,
        "confidence": round(confidence, 4),
        "scam_probability": round(scam_probability, 4),
        "text_length": features.get("text_length", 0),
        "url_length": features.get("url_length", 0),
        "suspicious_tld": features.get("suspicious_tld", 0),
        "suspicious_word_count": features.get("suspicious_word_count", 0),
        "scam_keyword_count": features.get("scam_keyword_count", 0),
    }


def print_result(result: dict) -> None:
    print("=" * 72)
    print(f"URL: {result['url']}")
    print(f"Final URL: {result['final_url']}")
    print(f"Scrape status: {result['status']}")
    if result.get("error"):
        print(f"Scrape note: {result['error']}")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Scam probability: {result['scam_probability']:.2%}")


def save_results(path: Path, results: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict whether URLs are scam or legitimate.")
    parser.add_argument("--url", help="Single URL to predict.")
    parser.add_argument(
        "--input",
        help="Text file with one URL per line, or CSV file with a 'url' column.",
    )
    parser.add_argument("--output", help="Optional CSV path for prediction results.")
    parser.add_argument("--timeout", type=int, default=8, help="Scraping timeout in seconds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    urls = load_urls(args)

    if not urls:
        raise SystemExit("Provide --url or --input.")

    if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
        raise SystemExit("Model files not found. Run training_model.ipynb first.")

    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURES_PATH)
    scraper = WebsiteScraper(timeout=args.timeout)
    url_extractor = URLFeatureExtractor()
    content_extractor = ContentFeatureExtractor()

    results = []
    for url in urls:
        try:
            result = predict_url(
                url=url,
                model=model,
                feature_columns=feature_columns,
                scraper=scraper,
                url_extractor=url_extractor,
                content_extractor=content_extractor,
            )
        except Exception as exc:
            result = {
                "url": url,
                "final_url": url,
                "status": "error",
                "error": str(exc),
                "prediction": "ERROR",
                "prediction_label": -1,
                "confidence": 0.0,
                "scam_probability": 0.0,
                "text_length": 0,
                "url_length": 0,
                "suspicious_tld": 0,
                "suspicious_word_count": 0,
                "scam_keyword_count": 0,
            }

        results.append(result)
        print_result(result)

    if args.output:
        save_results(Path(args.output), results)
        print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
