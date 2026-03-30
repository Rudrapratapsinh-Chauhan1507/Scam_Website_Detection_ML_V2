import os
import pandas as pd
from src.database_mysql import DatabaseManager

QUERY = """
SELECT
    url,
    text_content,

    -- URL features
    url_length, num_dots, num_hyphen, num_slashes,
    num_underscores, num_percent, num_digits,
    https, subdomains, has_at_symbol, has_double_slash, has_ip,
    num_query_params, has_query, path_depth,
    suspicious_tld, brand_in_url, is_shortened,
    suspicious_word_count,

    -- Domain features
    has_ssl, ssl_valid, ssl_days_remaining,
    is_new_domain, domain_age_days, domain_expiry_days,
    short_expiry_domain,

    -- Content features
    text_length, token_count,
    scam_keyword_count, scam_keyword_density,
    has_form, has_iframe,
    exclamation_count, caps_ratio, avg_word_length,

    -- Scraper feature
    redirect_count,

    -- Label
    label

FROM websites
WHERE label IS NOT NULL
AND status = 'success'
"""

OUTPUT_PATH = "./dataset/ml_project_dataset.csv"

# SETUP
os.makedirs("dataset", exist_ok=True)

db = DatabaseManager()
rows = db._execute(QUERY, fetch="all")

columns = [
    "url",
    "text_content",

    # URL
    "url_length", "num_dots", "num_hyphen", "num_slashes",
    "num_underscores", "num_percent", "num_digits",
    "https", "subdomains", "has_at_symbol", "has_double_slash", "has_ip",
    "num_query_params", "has_query", "path_depth",
    "suspicious_tld", "brand_in_url", "is_shortened",
    "suspicious_word_count",

    # Domain
    "has_ssl", "ssl_valid", "ssl_days_remaining",
    "is_new_domain", "domain_age_days", "domain_expiry_days",
    "short_expiry_domain",

    # Content
    "text_length", "token_count",
    "scam_keyword_count", "scam_keyword_density",
    "has_form", "has_iframe",
    "exclamation_count", "caps_ratio", "avg_word_length",

    # Scraper
    "redirect_count",

    # Label
    "label",
]

df = pd.DataFrame(rows, columns=columns)

# DATA CLEANING

# Fill missing text
df["text_content"] = df["text_content"].fillna("")

# Fill numeric nulls
numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
df[numeric_cols] = df[numeric_cols].fillna(0)

# Convert all numeric columns to float
df[numeric_cols] = df[numeric_cols].astype(float)

# DEBUG INFO
print(f"Rows exported : {len(df)}")
print(f"Label counts  :\n{df['label'].value_counts()}\n")

# SAVE
df.to_csv(OUTPUT_PATH, index=False)
print(f"[OK] Dataset saved to {OUTPUT_PATH}")