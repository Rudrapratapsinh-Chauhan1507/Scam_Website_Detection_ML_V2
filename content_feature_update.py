import time
from src.database_mysql import DatabaseManager
from src.content_features import ContentFeatureExtractor


db = DatabaseManager()
extractor = ContentFeatureExtractor()

rows = db.get_all_rows()

for row in rows:

    url = row["url"]
    title = row["title"]
    text = row["text_content"]

    features = extractor.extract(title, text)

    db.update_content_features(url, features)

    print("Content features updated:", url)

    time.sleep(1)