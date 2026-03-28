import argparse
import sys

from src.scraper import WebsiteScraper
from src.database_mysql import DatabaseManager
from src.url_features import URLFeatureExtractor
from src.domain_features import DomainFeatureExtractor
from src.content_features import ContentFeatureExtractor


def collect(url: str) -> None:
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    db = DatabaseManager()

    if db.url_exists(url):
        print(f"[INFO] URL already in database, skipping insert: {url}")
        return

    print(f"[INFO] Processing: {url}")

    scraper = WebsiteScraper()
    data    = scraper.scrape(url)

    if data["status"] == "failed":
        print(f"[WARN] Scraping failed: {data['error']}")

    url_features     = URLFeatureExtractor().extract(url)
    domain_features  = DomainFeatureExtractor().extract(url)
    content_features = ContentFeatureExtractor().extract(
        data.get("title"), data.get("text")
    )

    db.insert_website(url, data, url_features)
    db.update_url_features(url, url_features)        # saves ALL url feature columns
    db.update_domain_features(url, domain_features)
    db.update_content_features(url, content_features)

    if data["status"] == "success":
        print("[OK] Website scraped and stored successfully.")
    else:
        print("[OK] Failure recorded in database.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scam-detection scraper")
    parser.add_argument("--url", help="URL to process (prompted if omitted)")
    args = parser.parse_args()

    url = args.url or input("Enter website URL: ").strip()

    if not url:
        print("[ERROR] No URL provided.", file=sys.stderr)
        sys.exit(1)

    collect(url)


if __name__ == "__main__":
    main()