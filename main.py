import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.exceptions import NotFittedError

from src.content_features import ContentFeatureExtractor
from src.database_mysql import DatabaseManager
from src.domain_features import DomainFeatureExtractor
from src.scraper import WebsiteScraper
from src.url_features import URLFeatureExtractor


DEFAULT_TFIDF_PATH = Path("pkl_models") / "tfidf_vectorizer.pkl"
logging.getLogger("mysql.connector").setLevel(logging.WARNING)


class ScamDataCollector:
    def __init__(self, tfidf_path: str | Path = DEFAULT_TFIDF_PATH):
        self.db = DatabaseManager()
        self.scraper = WebsiteScraper()
        self.url_extractor = URLFeatureExtractor()
        self.domain_extractor = DomainFeatureExtractor()
        self.content_extractor = ContentFeatureExtractor()
        self.tfidf_path = Path(tfidf_path)

    @staticmethod
    def _normalize_url(url: str) -> str:
        url = (url or "").strip()
        if not url:
            return ""
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        return url

    @staticmethod
    def _combined_text(data: dict) -> str:
        title = (data.get("title") or "").strip()
        text = (data.get("text") or "").strip()
        return " ".join(part for part in [title, text] if part).strip()

    def _prepare_tfidf_features(
        self,
        page_text: str,
        refit_tfidf: bool,
    ) -> tuple[pd.DataFrame | None, str]:
        if not page_text:
            return None, "No page text available for TF-IDF."

        corpus = self.db.get_text_corpus()
        corpus_size = len(corpus)

        if refit_tfidf:
            if corpus_size < 2:
                return None, "TF-IDF skipped: need at least 2 stored pages with text."

            self.tfidf_extractor.fit_transform(corpus)
            self.tfidf_extractor.save(str(self.tfidf_path))
            tfidf_status = (
                f"TF-IDF model refitted on {corpus_size} pages and saved to {self.tfidf_path}."
            )
        elif self.tfidf_path.exists():
            self.tfidf_extractor.load(str(self.tfidf_path))
            tfidf_status = f"TF-IDF model loaded from {self.tfidf_path}."
        elif corpus_size >= 2:
            self.tfidf_extractor.fit_transform(corpus)
            self.tfidf_extractor.save(str(self.tfidf_path))
            tfidf_status = (
                f"TF-IDF model auto-created from {corpus_size} stored pages and saved to {self.tfidf_path}."
            )
        else:
            return None, (
                f"TF-IDF waiting for more data: found {corpus_size} stored page(s), need at least 2."
            )

        try:
            tfidf_frame = self.tfidf_extractor.transform([page_text], as_dataframe=True)
            return tfidf_frame, tfidf_status
        except NotFittedError:
            return None, "TF-IDF model is not fitted yet."

    @staticmethod
    def _top_tfidf_terms(tfidf_frame: pd.DataFrame, top_n: int = 10) -> list[tuple[str, float]]:
        if tfidf_frame.empty:
            return []

        row = tfidf_frame.iloc[0]
        non_zero = row[row > 0].sort_values(ascending=False).head(top_n)
        return [
            (column.replace("tfidf_", "", 1), float(value))
            for column, value in non_zero.items()
        ]

    def collect(
        self,
        url: str,
        label: int | None = None,
        refit_tfidf: bool = False,
        enable_tfidf: bool = True,
        show_features: bool = False,
        force: bool = False,
    ) -> dict:
        url = self._normalize_url(url)
        if not url:
            raise ValueError("No URL provided.")

        if not force and self.db.url_exists(url):
            return {
                "status": "skipped",
                "url": url,
                "message": f"URL already exists in database: {url}",
            }

        print(f"[INFO] Processing: {url}")
        scraped = self.scraper.scrape(url)

        if scraped["status"] == "failed":
            print(f"[WARN] Scraping failed: {scraped['error']}")

        feature_url = scraped.get("final_url") or url
        url_features = self.url_extractor.extract(feature_url)
        domain_features = self.domain_extractor.extract(feature_url)
        content_features = self.content_extractor.extract(
            scraped.get("title"),
            scraped.get("text"),
            scraped.get("html"),
        )

        all_features = {}
        all_features.update(url_features)
        all_features.update(domain_features)
        all_features.update(content_features)

        self.db.insert_website(url, scraped)
        self.db.update_url_features(url, url_features)
        self.db.update_domain_features(url, domain_features)
        self.db.update_content_features(url, content_features)

        if label is not None:
            self.db.update_label(url, label)
            print(f"[INFO] Label saved: {label}")

        if enable_tfidf:
            page_text = self._combined_text(scraped)
            tfidf_frame, tfidf_status = self._prepare_tfidf_features(
                page_text=page_text,
                refit_tfidf=refit_tfidf,
            )

            tfidf_summary = {
                "status": tfidf_status,
                "vocabulary_size": self.tfidf_extractor.vocabulary_size
                if tfidf_frame is not None
                else 0,
                "non_zero_features": int((tfidf_frame.iloc[0] > 0).sum()) if tfidf_frame is not None else 0,
                "top_terms": self._top_tfidf_terms(tfidf_frame) if tfidf_frame is not None else [],
            }
        else:
            tfidf_summary = {
                "status": "TF-IDF skipped for this run.",
                "vocabulary_size": 0,
                "non_zero_features": 0,
                "top_terms": [],
            }

        if scraped["status"] == "success":
            print("[OK] Website scraped and stored successfully.")
        else:
            print("[OK] Partial/failure record stored in database.")

        print(f"[INFO] {tfidf_summary['status']}")

        if show_features:
            print("\n===== STRUCTURED FEATURES =====")
            for key in sorted(all_features):
                print(f"{key}: {all_features[key]}")

            if tfidf_summary["top_terms"]:
                print("\n===== TOP TF-IDF TERMS =====")
                for term, score in tfidf_summary["top_terms"]:
                    print(f"{term}: {score:.4f}")

        return {
            "status": scraped["status"],
            "url": url,
            "final_url": feature_url,
            "scrape": scraped,
            "features": all_features,
            "tfidf": tfidf_summary,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect website data and extract scam-detection features."
    )
    parser.add_argument("--url", help="URL to process (prompted if omitted)")
    parser.add_argument(
        "--label",
        type=int,
        choices=[0, 1],
        help="Label (0 = legit, 1 = scam)",
    )
    parser.add_argument(
        "--fit-tfidf",
        action="store_true",
        help="Refit the TF-IDF vectorizer using stored website text from the database.",
    )
    parser.add_argument(
        "--tfidf-path",
        default=str(DEFAULT_TFIDF_PATH),
        help="Path to save/load the TF-IDF vectorizer.",
    )
    parser.add_argument(
        "--show-features",
        action="store_true",
        help="Print extracted numerical features and top TF-IDF terms.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Process the URL even if it already exists in the database.",
    )
    parser.add_argument(
        "--skip-tfidf",
        action="store_true",
        help="Skip TF-IDF work for this run. Useful for large batch collection.",
    )
    return parser.parse_args()


def prompt_for_url() -> str:
    return input("Enter website URL: ").strip()


def prompt_for_label(existing_label: int | None) -> int | None:
    if existing_label is not None:
        return existing_label

    label_input = input("Enter label (0 = legit, 1 = scam, leave blank to skip): ").strip()
    if label_input == "":
        return None
    if label_input not in {"0", "1"}:
        print("[ERROR] Invalid label. Use 0 or 1.", file=sys.stderr)
        sys.exit(1)
    return int(label_input)


def main() -> None:
    args = parse_args()

    url = args.url or prompt_for_url()
    if not url:
        print("[ERROR] No URL provided.", file=sys.stderr)
        sys.exit(1)

    label = prompt_for_label(args.label)
    collector = ScamDataCollector(tfidf_path=args.tfidf_path)

    try:
        result = collector.collect(
            url=url,
            label=label,
            refit_tfidf=args.fit_tfidf,
            enable_tfidf=not args.skip_tfidf,
            show_features=args.show_features,
            force=args.force,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    if result["status"] == "skipped":
        print(f"[INFO] {result['message']}")


if __name__ == "__main__":
    main()
