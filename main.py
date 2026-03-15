from src.scraper import WebsiteScraper
from src.database_mysql import DatabaseManager
from src.url_features import URLFeatureExtractor

scraper = WebsiteScraper()
db = DatabaseManager()
feature_extractor = URLFeatureExtractor()

url = input("Enter website URL: ")

data = scraper.scrape(url)
features = feature_extractor.extract(url)
db.insert_website(url,data, features)

if data["status"] == "success":
    print("Website scraped and stored successfully")

else:
    print("Scraping failed but stored in database")
