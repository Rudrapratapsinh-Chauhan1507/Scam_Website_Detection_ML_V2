from src.database_mysql import DatabaseManager
from src.url_features import URLFeatureExtractor


db = DatabaseManager()
extractor = URLFeatureExtractor()

urls = db.get_all_urls()

for url in urls:

    features = extractor.extract(url)

    db.update_url_features(url, features)

    print("Updated:", url)