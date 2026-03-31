import argparse
import sys

from src.scraper import WebsiteScraper
from src.database_mysql import DatabaseManager
from src.url_features import URLFeatureExtractor
from src.domain_features import DomainFeatureExtractor
from src.content_features import ContentFeatureExtractor


def collect(url: str, label: int | None = None) -> None:
    # Ensure proper URL format
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    db = DatabaseManager()

    if db.url_exists(url):
        print(f"[INFO] URL already in database, skipping insert: {url}")
        return

    print(f"[INFO] Processing: {url}")

    # Scrape website
    scraper = WebsiteScraper()
    data = scraper.scrape(url)

    if data["status"] == "failed":
        print(f"[WARN] Scraping failed: {data['error']}")

    # Extract features
    url_features = URLFeatureExtractor().extract(url)
    domain_features = DomainFeatureExtractor().extract(url)
    content_features = ContentFeatureExtractor().extract(
        data.get("title"),
        data.get("text"),
        data.get("html") 
    )

    # Store in DB
    db.insert_website(url, data)
    db.update_url_features(url, url_features)
    db.update_domain_features(url, domain_features)
    db.update_content_features(url, content_features)

    # Save label if provided
    if label is not None:
        db.update_label(url, label)
        print(f"[INFO] Label saved: {label}")

    # Final status
    if data["status"] == "success":
        print("[OK] Website scraped and stored successfully.")
    else:
        print("[OK] Failure recorded in database.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scam-detection scraper")
    parser.add_argument("--url", help="URL to process (prompted if omitted)")
    parser.add_argument("--label", type=int, choices=[0, 1],
                        help="Label (0 = legit, 1 = scam)")
    args = parser.parse_args()

    # URL input
    url = args.url or input("Enter website URL: ").strip()
    if not url:
        print("[ERROR] No URL provided.", file=sys.stderr)
        sys.exit(1)

    # Label input
    label = args.label
    if label is None:
        label_input = input("Enter label (0 = legit, 1 = scam, leave blank to skip): ").strip()
        if label_input != "":
            if label_input not in ("0", "1"):
                print("[ERROR] Invalid label. Use 0 or 1.", file=sys.stderr)
                sys.exit(1)
            label = int(label_input)

    # Run pipeline
    collect(url, label)


if __name__ == "__main__":
    main()