import os
import pandas as pd
from src.database_mysql import DatabaseManager

QUERY = """
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
    num_underscores,
    num_percent,
    num_digits,
    num_query_params,
    has_query,
    path_depth,
    suspicious_tld,
    brand_in_url,
    is_shortened,
    has_ssl,
    ssl_valid,
    ssl_days_remaining,
    is_new_domain,
    domain_age_days,
    domain_expiry_days,
    text_length,
    token_count,
    scam_keyword_count,
    scam_keyword_density,
    has_form,
    has_iframe,
    exclamation_count,
    caps_ratio,
    label
FROM websites
WHERE label IS NOT NULL
"""

OUTPUT_PATH = "./dataset/ml_project_dataset.csv"

os.makedirs("dataset", exist_ok=True)

db = DatabaseManager()
db._execute.__func__  # warm up pool

rows = db._execute(QUERY, fetch="all")

columns = [
    "url", "text_content",
    "url_length", "num_dots", "num_hyphen", "num_slashes",
    "https", "subdomains", "has_at_symbol", "has_ip",
    "num_underscores", "num_percent", "num_digits",
    "num_query_params", "has_query", "path_depth",
    "suspicious_tld", "brand_in_url", "is_shortened",
    "has_ssl", "ssl_valid", "ssl_days_remaining", "is_new_domain",
    "domain_age_days", "domain_expiry_days",
    "text_length", "token_count",
    "scam_keyword_count", "scam_keyword_density",
    "has_form", "has_iframe", "exclamation_count", "caps_ratio",
    "label",
]

df = pd.DataFrame(rows, columns=columns)
df["text_content"] = df["text_content"].fillna("")

print(f"Rows exported : {len(df)}")
print(f"Label counts  :\n{df['label'].value_counts()}\n")

df.to_csv(OUTPUT_PATH, index=False)
print(f"[OK] Dataset saved to {OUTPUT_PATH}")