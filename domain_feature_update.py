import time
from src.database_mysql import DatabaseManager
from src.domain_features import DomainFeatureExtractor

db = DatabaseManager()
extractor = DomainFeatureExtractor()

urls = db.get_all_urls()

for url in urls:

    features = extractor.extract(url)

    db.update_domain_features(url, features)

    print("Updated:", url)

    time.sleep(2)