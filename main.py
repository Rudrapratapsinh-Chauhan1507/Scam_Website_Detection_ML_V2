from src.scraper import WebsiteScraper
from src.database_mysql import DatabaseManager
from src.url_features import URLFeatureExtractor
from src.domain_features import DomainFeatureExtractor
from src.content_features import ContentFeatureExtractor


scraper = WebsiteScraper()
feature_extractor = URLFeatureExtractor()
domain_extractor = DomainFeatureExtractor()
content_extractor = ContentFeatureExtractor()
db = DatabaseManager()


url = input("Enter website URL: ")

data = scraper.scrape(url)
features = feature_extractor.extract(url)
domain_features = domain_extractor.extract(url)
content_features = content_extractor.extract(data["title"], data["text"])
db.insert_website(url,data, features)

db.update_domain_features(url, domain_features)
db.update_content_features(url, content_features)

if data["status"] == "success":
    print("Website scraped and stored successfully")

else:
    print("Scraping failed but stored in database")
