import argparse
import random
from pathlib import Path

import pandas as pd

from src.content_features import ContentFeatureExtractor
from src.domain_features import DomainFeatureExtractor
from src.scraper import WebsiteScraper
from src.url_features import URLFeatureExtractor


REFERENCE_ROOT = Path(r"C:\Users\lenovo\Desktop\Scam Prac")
DEFAULT_OUTPUT = Path("dataset") / "reference_training_dataset_600.csv"
DEFAULT_ROWS = 600

REFERENCE_FILES = [
    REFERENCE_ROOT / "MainData" / "dataset.csv",
    REFERENCE_ROOT / "MainData" / "df_0.csv",
    REFERENCE_ROOT / "MainData" / "df_1.csv",
    REFERENCE_ROOT / "Dataset.csv",
    REFERENCE_ROOT / "df_0.csv",
    REFERENCE_ROOT / "df_1.csv",
]


def normalize_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url


def load_reference_rows(reference_root: Path) -> pd.DataFrame:
    frames = []
    files = [
        reference_root / file.relative_to(REFERENCE_ROOT)
        for file in REFERENCE_FILES
        if (reference_root / file.relative_to(REFERENCE_ROOT)).exists()
    ]

    for file in files:
        frame = pd.read_csv(file, low_memory=False)
        if "url" not in frame.columns or "label" not in frame.columns:
            continue

        keep_columns = ["url", "label"]
        if "text_content" in frame.columns:
            keep_columns.append("text_content")

        slim = frame[keep_columns].copy()
        slim["url"] = slim["url"].astype(str).map(normalize_url)
        slim["label"] = pd.to_numeric(slim["label"], errors="coerce")
        if "text_content" in slim.columns:
            slim["text_content"] = slim["text_content"].fillna("").astype(str)
        else:
            slim["text_content"] = ""
        slim["source_dataset"] = str(file)
        slim["has_reference_text"] = slim["text_content"].str.strip().ne("")
        slim = slim[slim["url"].ne("") & slim["label"].isin([0, 1])]
        frames.append(slim)

    if not frames:
        raise FileNotFoundError(f"No usable reference CSV files found under {reference_root}")

    combined = pd.concat(frames, ignore_index=True)
    combined["label"] = combined["label"].astype(int)

    combined = combined.sort_values(
        by=["has_reference_text", "source_dataset"],
        ascending=[False, True],
    )
    combined = combined.drop_duplicates(subset=["url"], keep="first")
    return combined


def balanced_sample(frame: pd.DataFrame, rows: int, seed: int, prefer_text: bool) -> pd.DataFrame:
    per_label = rows // 2
    remainder = rows % 2
    samples = []

    for label, needed in [(0, per_label + remainder), (1, per_label)]:
        group = frame[frame["label"] == label]
        if group.empty:
            raise ValueError(f"No rows found for label {label}")

        if prefer_text:
            text_group = group[group["has_reference_text"]]
            if len(text_group) >= needed:
                group = text_group
            elif not text_group.empty:
                remaining = needed - len(text_group)
                fallback = group[~group["has_reference_text"]]
                fallback_replace = len(fallback) < remaining
                samples.append(text_group.sample(frac=1, random_state=seed + label))
                samples.append(
                    fallback.sample(
                        n=remaining,
                        replace=fallback_replace,
                        random_state=seed + 100 + label,
                    )
                )
                continue

        replace = len(group) < needed
        samples.append(group.sample(n=needed, replace=replace, random_state=seed + label))

    sampled = pd.concat(samples, ignore_index=True)
    return sampled.sample(frac=1, random_state=seed).reset_index(drop=True)


def empty_domain_features() -> dict:
    return {
        "domain_age_days": -1,
        "domain_expiry_days": -1,
        "registrar": "",
        "missing_whois": 1,
        "has_ssl": 0,
        "ssl_valid": 0,
        "ssl_days_remaining": -1,
        "short_expiry_domain": 0,
        "is_new_domain": 0,
    }


def extract_row(
    record: dict,
    url_extractor: URLFeatureExtractor,
    content_extractor: ContentFeatureExtractor,
    domain_extractor: DomainFeatureExtractor | None,
    scraper: WebsiteScraper | None,
) -> dict:
    input_url = normalize_url(record["url"])
    title = ""
    text = (record.get("text_content") or "").strip()
    html = ""
    final_url = input_url
    scrape_status = "not_scraped"
    scrape_error = ""
    redirect_count = 0

    if scraper is not None and not text:
        scraped = scraper.scrape(input_url)
        final_url = scraped.get("final_url") or input_url
        title = scraped.get("title") or ""
        text = scraped.get("text") or ""
        html = scraped.get("html") or ""
        scrape_status = scraped.get("status") or "failed"
        scrape_error = scraped.get("error") or ""
        redirect_count = scraped.get("redirect_count") or 0

    features = {}
    features.update(url_extractor.extract(final_url))

    if domain_extractor is None:
        features.update(empty_domain_features())
    else:
        features.update(domain_extractor.extract(final_url))

    features.update(content_extractor.extract(title=title, text=text, html=html))

    row = {
        "url": input_url,
        "final_url": final_url,
        "label": int(record["label"]),
        "source_dataset": record.get("source_dataset", ""),
        "has_reference_text": int(bool((record.get("text_content") or "").strip())),
        "scrape_status": scrape_status,
        "scrape_error": scrape_error,
        "redirect_count": redirect_count,
    }
    row.update(features)
    return row


def build_dataset(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    reference_root = Path(args.reference_root)
    output_path = Path(args.output)

    references = load_reference_rows(reference_root)
    sampled = balanced_sample(
        references,
        rows=args.rows,
        seed=args.seed,
        prefer_text=not args.no_prefer_text,
    )

    url_extractor = URLFeatureExtractor()
    content_extractor = ContentFeatureExtractor()
    domain_extractor = DomainFeatureExtractor(timeout=args.timeout) if args.enrich_domain else None
    scraper = WebsiteScraper(timeout=args.timeout) if args.live_content else None

    rows = []
    for index, record in enumerate(sampled.to_dict(orient="records"), start=1):
        try:
            rows.append(
                extract_row(
                    record=record,
                    url_extractor=url_extractor,
                    content_extractor=content_extractor,
                    domain_extractor=domain_extractor,
                    scraper=scraper,
                )
            )
            if index % 50 == 0 or index == len(sampled):
                print(f"[INFO] Prepared {index}/{len(sampled)} rows")
        except Exception as exc:
            print(f"[WARN] Skipped {record.get('url')}: {exc}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_path, index=False, encoding="utf-8")

    print(f"[OK] Wrote {len(frame)} rows to {output_path}")
    print(f"[INFO] Label counts: {frame['label'].value_counts().sort_index().to_dict()}")
    print(f"[INFO] Columns: {len(frame.columns)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a balanced training dataset from the Scam Prac reference CSVs."
    )
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS, help="Total rows to output.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output CSV path.")
    parser.add_argument(
        "--reference-root",
        default=str(REFERENCE_ROOT),
        help="Folder containing Dataset.csv, df_0.csv, df_1.csv and MainData.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument("--timeout", type=int, default=6, help="Timeout for optional live enrichment.")
    parser.add_argument(
        "--no-prefer-text",
        action="store_true",
        help="Do not prefer reference rows that include text_content.",
    )
    parser.add_argument(
        "--enrich-domain",
        action="store_true",
        help="Run live WHOIS/SSL domain extraction. Slower, but fills domain features.",
    )
    parser.add_argument(
        "--live-content",
        action="store_true",
        help="Scrape pages when the reference row does not contain text_content.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    build_dataset(parse_args())
