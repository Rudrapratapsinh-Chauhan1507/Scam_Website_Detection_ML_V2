import argparse
import csv
import io
import zipfile
from pathlib import Path

import joblib
import pandas as pd
import requests

from src.content_features import ContentFeatureExtractor
from src.scraper import WebsiteScraper
from src.url_features import URLFeatureExtractor


MODEL_PATH = Path("pkl_models") / "best_scam_detector.pkl"
FEATURES_PATH = Path("pkl_models") / "model_features.pkl"
DEFAULT_OUTPUT = Path("dataset") / "real_site_prediction_results.csv"
DEFAULT_MARKDOWN = Path("real_site_prediction_results.md")

OPENPHISH_FEED = "https://openphish.com/feed.txt"
URLHAUS_RECENT_FEED = "https://urlhaus.abuse.ch/downloads/csv_recent/"

LEGIT_SITES = [
    "https://www.google.com",
    "https://www.youtube.com",
    "https://www.wikipedia.org",
    "https://www.github.com",
    "https://www.python.org",
    "https://www.microsoft.com",
    "https://www.apple.com",
    "https://www.amazon.com",
    "https://www.cloudflare.com",
    "https://www.mozilla.org",
    "https://stackoverflow.com",
    "https://www.linkedin.com",
    "https://www.bbc.com",
    "https://www.reuters.com",
    "https://www.nasa.gov",
    "https://www.harvard.edu",
    "https://www.stanford.edu",
    "https://www.mit.edu",
    "https://www.openai.com",
    "https://www.kaggle.com",
    "https://www.gutenberg.org/",
    "https://jsonlint.com/",
    "https://www.futureme.org/",
]


def normalize_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url


def dedupe_urls(urls: list[str]) -> list[str]:
    seen = set()
    unique = []
    for url in urls:
        normalized = normalize_url(url)
        key = normalized.rstrip("/").lower()
        if normalized and key not in seen:
            seen.add(key)
            unique.append(normalized)
    return unique


def fetch_openphish(limit: int, timeout: int) -> list[str]:
    try:
        response = requests.get(
            OPENPHISH_FEED,
            timeout=timeout,
            headers={"User-Agent": "ScamShieldEvaluator/1.0"},
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"[WARN] OpenPhish unavailable: {exc}")
        return []

    urls = [line.strip() for line in response.text.splitlines() if line.strip()]
    return dedupe_urls(urls)[:limit]


def fetch_urlhaus(limit: int, timeout: int) -> list[str]:
    try:
        response = requests.get(
            URLHAUS_RECENT_FEED,
            timeout=timeout,
            headers={"User-Agent": "ScamShieldEvaluator/1.0"},
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"[WARN] URLhaus unavailable: {exc}")
        return []

    content = response.content
    if zipfile.is_zipfile(io.BytesIO(content)):
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            csv_name = next((name for name in archive.namelist() if name.endswith(".csv")), None)
            if csv_name is None:
                return []
            text = archive.read(csv_name).decode("utf-8", errors="replace")
    else:
        text = content.decode("utf-8", errors="replace")

    urls = []
    rows = csv.reader(line for line in text.splitlines() if line and not line.startswith("#"))
    for row in rows:
        if len(row) >= 3 and row[2].startswith(("http://", "https://")):
            urls.append(row[2])
    return dedupe_urls(urls)[:limit]


def get_scam_sites(limit: int, timeout: int) -> list[str]:
    urls = fetch_openphish(limit, timeout)
    if len(urls) < limit:
        urls = dedupe_urls(urls + fetch_urlhaus(limit - len(urls), timeout))
    return urls[:limit]


def zero_content_features() -> dict:
    return {
        "text_length": 0,
        "token_count": 0,
        "scam_keyword_count": 0,
        "scam_keyword_density": 0.0,
        "has_form": 0,
        "has_iframe": 0,
        "exclamation_count": 0,
        "caps_ratio": 0.0,
        "avg_word_length": 0.0,
    }


def extract_features(url: str, scraper: WebsiteScraper, scrape_content: bool) -> tuple[dict, dict]:
    url = normalize_url(url)
    final_url = url
    scrape_status = "not_scraped"
    scrape_error = ""
    content_features = zero_content_features()

    if scrape_content:
        scraped = scraper.scrape(url)
        final_url = scraped.get("final_url") or url
        scrape_status = scraped.get("status") or "failed"
        scrape_error = scraped.get("error") or ""
        content_features = ContentFeatureExtractor().extract(
            title=scraped.get("title"),
            text=scraped.get("text"),
            html=scraped.get("html"),
        )

    features = URLFeatureExtractor().extract(final_url)
    features.update(content_features)

    meta = {
        "url": url,
        "final_url": final_url,
        "scrape_status": scrape_status,
        "scrape_error": scrape_error,
    }
    return features, meta


def predict_one(
    model,
    feature_columns: list[str],
    features: dict,
) -> tuple[int, str, float, float]:
    frame = pd.DataFrame([features])
    for column in feature_columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame = frame[feature_columns]

    if hasattr(model, "predict_proba"):
        scam_probability = float(model.predict_proba(frame)[0][1])
    else:
        scam_probability = float(model.predict(frame)[0])

    prediction = int(scam_probability >= 0.5)
    prediction_name = "SCAM" if prediction == 1 else "LEGIT"
    confidence = scam_probability if prediction == 1 else 1.0 - scam_probability

    return prediction, prediction_name, confidence, scam_probability


def write_markdown(path: Path, frame: pd.DataFrame) -> None:
    correct = int(frame["correct"].sum())
    total = len(frame)
    legit = frame[frame["expected_label"] == 0]
    scam = frame[frame["expected_label"] == 1]

    lines = [
        "# Real Site Prediction Results",
        "",
        f"- Total: {correct}/{total} correct",
        f"- Legit: {int(legit['correct'].sum())}/{len(legit)} correct",
        f"- Scam: {int(scam['correct'].sum())}/{len(scam)} correct",
        "",
        "| Expected | Predicted | Correct | Confidence | URL |",
        "|---|---|---:|---:|---|",
    ]

    for row in frame.to_dict(orient="records"):
        lines.append(
            "| {expected} | {predicted} | {correct} | {confidence:.2%} | {url} |".format(
                expected=row["expected_name"],
                predicted=row["predicted_name"],
                correct="yes" if row["correct"] else "no",
                confidence=row["confidence"],
                url=row["url"],
            )
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(args: argparse.Namespace) -> None:
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURES_PATH)

    scam_sites = get_scam_sites(args.scam_count, args.timeout)
    if len(scam_sites) < args.scam_count:
        print(f"[WARN] Only found {len(scam_sites)} scam URLs.")

    test_rows = (
        [{"url": url, "expected_label": 0, "expected_name": "LEGIT"} for url in LEGIT_SITES[: args.legit_count]]
        + [{"url": url, "expected_label": 1, "expected_name": "SCAM"} for url in scam_sites]
    )

    scraper = WebsiteScraper(timeout=args.timeout)
    results = []
    for index, item in enumerate(test_rows, start=1):
        scrape_content = item["expected_label"] == 0 or args.scrape_scam_content
        features, meta = extract_features(item["url"], scraper, scrape_content=scrape_content)
        prediction, prediction_name, confidence, scam_probability = predict_one(
            model,
            feature_columns,
            features,
        )

        result = {
            **item,
            **meta,
            "predicted_label": prediction,
            "predicted_name": prediction_name,
            "confidence": confidence,
            "scam_probability": scam_probability,
            "correct": int(prediction == item["expected_label"]),
        }
        results.append(result)
        print(
            f"[{index}/{len(test_rows)}] expected={result['expected_name']} "
            f"predicted={result['predicted_name']} correct={bool(result['correct'])} "
            f"confidence={confidence:.2%} {item['url']}"
        )

    frame = pd.DataFrame(results)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, encoding="utf-8")
    write_markdown(Path(args.markdown), frame)

    correct = int(frame["correct"].sum())
    print(f"[DONE] Correct: {correct}/{len(frame)}")
    print(f"[DONE] Results CSV: {output_path}")
    print(f"[DONE] Report: {args.markdown}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the saved model on real legit and scam URLs.")
    parser.add_argument("--legit-count", type=int, default=20)
    parser.add_argument("--scam-count", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=5)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--markdown", default=str(DEFAULT_MARKDOWN))
    parser.add_argument(
        "--scrape-scam-content",
        action="store_true",
        help="Also fetch scam pages for content features. Off by default for safer evaluation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
