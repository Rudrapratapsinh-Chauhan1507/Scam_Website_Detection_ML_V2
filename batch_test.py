import os
import joblib
import pandas as pd
from src.scraper import WebsiteScraper
from src.url_features import URLFeatureExtractor
from src.domain_features import DomainFeatureExtractor
from src.content_features import ContentFeatureExtractor
from urllib.parse import urlparse

WHITELIST = [
    "spectricssolutions.com",
    "www.spectricssolutions.com",
    "google.com",
    "www.google.com",
    "gtu.ac.in",
    "www.gtu.ac.in"
]

def impute_missing_features(df: pd.DataFrame) -> pd.DataFrame:
    mapped = df.copy()
    for col in ["domain_age_days", "domain_expiry_days", "ssl_days_remaining"]:
        if col in mapped.columns: mapped[col] = pd.to_numeric(mapped[col], errors="coerce").fillna(-1)
    if "domain_age_days" in mapped.columns: mapped["missing_whois"] = (mapped["domain_age_days"] == -1).astype(int)
    for col in ["has_ssl", "ssl_valid", "is_new_domain", "short_expiry_domain", "has_form", "has_iframe"]:
        if col in mapped.columns: mapped[col] = pd.to_numeric(mapped[col], errors="coerce").fillna(-1)
    for col in ["text_length", "token_count", "scam_keyword_count", "exclamation_count"]:
        if col in mapped.columns: mapped[col] = pd.to_numeric(mapped[col], errors="coerce").fillna(0)
    for col in ["scam_keyword_density", "caps_ratio", "avg_word_length"]:
        if col in mapped.columns: mapped[col] = pd.to_numeric(mapped[col], errors="coerce").fillna(0.0)
    return mapped

def analyze_batch():
    model_path = os.path.join("pkl_models", "best_scam_detector.pkl")
    features_path = os.path.join("pkl_models", "model_features.pkl")
    
    if not os.path.exists(model_path):
        print("[!] Error: Model not found. Run retrain_model.py first.")
        return

    model = joblib.load(model_path)
    feature_columns = joblib.load(features_path)
    
    urls_data = [
        # Legitimate
        {"url": "https://www.gtu.ac.in", "expected": "Legit"},
        {"url": "https://www.spectricssolutions.com/", "expected": "Legit"},
        {"url": "https://www.harvard.edu", "expected": "Legit"},
        {"url": "https://www.tcs.com", "expected": "Legit"},
        {"url": "https://docs.python.org", "expected": "Legit"},

        # Scams (Using some dead ones to test robust fallback, and some live ones if possible)
        {"url": "b-t-i-n-t-e-r-n-e-t-105959.weeblysite.com", "expected": "Scam"},
        {"url": "https://paypal-login-help.com", "expected": "Scam"},
        {"url": "http://account-check.ga", "expected": "Scam"},
        {"url": "https://att-102546.weeblysite.com/", "expected": "Scam"},
        {"url": "https://urlz.fr/uMHD", "expected": "Scam"}
    ]
    
    results = []
    
    print(f"Starting batch analysis of {len(urls_data)} URLs...")
    for i, item in enumerate(urls_data, 1):
        url = item["url"]
        expected = item["expected"]
        print(f"[{i}/{len(urls_data)}] Analyzing: {url}")
        
        domain = urlparse(url).netloc.lower() or url.lower()
        if domain in WHITELIST:
            results.append({
                "URL": url,
                "Expected": expected,
                "Predicted": "LEGITIMATE",
                "Confidence": "100.00%",
                "Note": "Whitelisted"
            })
            continue
            
        # Reduced timeout for batch testing to avoid hanging on dead scam domains
        scrape_result = WebsiteScraper(timeout=3).scrape(url)
        final_url = scrape_result.get("final_url") if scrape_result and scrape_result.get("final_url") else url
        
        url_feats = URLFeatureExtractor().extract(final_url)
        domain_feats = DomainFeatureExtractor(timeout=3).extract(final_url)
        content_feats = {}
        if scrape_result and scrape_result.get("status") != "failed":
            content_feats = ContentFeatureExtractor().extract(
                title=scrape_result.get("title"),
                text=scrape_result.get("text"),
                html=scrape_result.get("html")
            )
            
        raw_features = {**url_feats, **domain_feats, **content_feats}
        df = pd.DataFrame([raw_features])
        for col in feature_columns:
            if col not in df.columns: df[col] = pd.NA
        df = df[feature_columns]
        df = impute_missing_features(df)
        
        prediction = model.predict(df)[0]
        confidence = model.predict_proba(df)[0].max() * 100 if hasattr(model, "predict_proba") else 100.0
        
        pred_label = "SCAM" if prediction == 1 else "LEGITIMATE"
        
        results.append({
            "URL": url,
            "Expected": expected,
            "Predicted": pred_label,
            "Confidence": f"{confidence:.2f}%",
            "Note": "ML Prediction"
        })
        
    print("Writing markdown report...")
    with open("batch_results.md", "w", encoding="utf-8") as f:
        f.write("| URL | Expected | Result | Predicted | Confidence | Method |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in results:
            # Using ASCII characters for pass/fail to avoid Windows CP1252 encoding issues
            is_correct = (r["Expected"].upper() in r["Predicted"] or (r["Expected"] == "Legit" and r["Predicted"] == "LEGITIMATE"))
            status_icon = "[PASS]" if is_correct else "[FAIL]"
            f.write(f"| {r['URL']} | {r['Expected']} | {status_icon} | **{r['Predicted']}** | {r['Confidence']} | {r['Note']} |\n")
    
    print(f"[OK] Batch analysis complete. Results written to batch_results.md")

if __name__ == "__main__":
    analyze_batch()
