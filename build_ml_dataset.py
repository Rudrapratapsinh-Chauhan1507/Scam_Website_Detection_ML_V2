import pandas as pd
from src.database_mysql import DatabaseManager

db = DatabaseManager()

query = """
SELECT
url,
text_content,
url_length,
num_dots,
num_hyphen,
num_slashes,
https,
subdomains,
has_at_symbol,
has_ip,
has_ssl,
ssl_valid,
text_length,
token_count,
scam_keyword_count,
scam_keyword_density,
label
FROM websites
"""

db.cursor.execute(query)

rows = db.cursor.fetchall()

data = []

for row in rows:

    data.append({
        "url": row[0],
        "text_content" : row[1] if row[1] else "",
        "url_length": row[2],
        "num_dots": row[3],
        "num_hyphen": row[4],
        "num_slashes": row[5],
        "https": row[6],
        "subdomains": row[7],
        "has_at_symbol": row[8],
        "has_ip": row[9],
        "has_ssl": row[10],
        "ssl_valid": row[11],
        "text_length": row[12],
        "token_count": row[13],
        "scam_keyword_count": row[14],
        "scam_keyword_density": row[15],
        "label" : row[16]
    })


# convert to dataframe
df = pd.DataFrame(data)

# save dataset
df.to_csv("./dataset/ml_project_dataset.csv", index=False)

print("Dataset created successfully")