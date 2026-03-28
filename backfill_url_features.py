import sys
sys.path.insert(0, ".")

from src.database_mysql import DatabaseManager
from src.url_features   import URLFeatureExtractor

db      = DatabaseManager()
url_ext = URLFeatureExtractor()

# Fetch only rows where the new columns are still NULL
rows = db._execute(
    """
    SELECT url FROM websites
    WHERE num_underscores IS NULL
       OR num_percent     IS NULL
       OR num_digits      IS NULL
    """,
    fetch="all"
)

total = len(rows)
print(f"Found {total} rows with NULL url features. Backfilling …\n")

fixed = 0
for i, (url,) in enumerate(rows, 1):
    try:
        features = url_ext.extract(url)
        db.update_url_features(url, features)
        fixed += 1
        print(f"[{i}/{total}] OK  {url[:80]}")
    except Exception as e:
        print(f"[{i}/{total}] ERR {url[:80]}  → {e}")

print(f"\nDone. Fixed {fixed}/{total} rows.")
print("All new url feature columns should now be populated.")
