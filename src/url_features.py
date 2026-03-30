import re
from urllib.parse import urlparse, parse_qs


# TLDs commonly abused in phishing / scam campaigns
SUSPICIOUS_TLDS = {
    ".tk", ".ml", ".ga", ".cf", ".gq",  
    ".xyz", ".top", ".club", ".online",
    ".site", ".info", ".biz", ".pw",
}


class URLFeatureExtractor:

    def extract(self, url: str) -> dict:
        parsed = urlparse(url)
        netloc = parsed.netloc.lower()
        path = parsed.path.lower()
        full_url = url.lower()

        features: dict = {}

        # Basic length / character counts
        features["url_length"]   = len(url)
        features["num_dots"]     = url.count(".")
        features["num_hyphen"]   = url.count("-")
        features["num_slashes"]  = url.count("/")
        features["num_underscores"] = url.count("_")
        features["num_percent"]  = url.count("%")        # URL-encoding
        features["num_digits"]   = sum(c.isdigit() for c in url)

        # Scheme 
        features["https"] = 1 if parsed.scheme == "https" else 0

        # Host / subdomain analysis 
        domain_parts = netloc.split(".")
        # subdomains = parts minus the SLD + TLD
        features["subdomains"] = max(len(domain_parts) - 2, 0)

        # Suspicious symbols
        features["has_at_symbol"]    = 1 if "@" in url else 0
        features["has_double_slash"] = 1 if "//" in parsed.path else 0

        # IP address used instead of domain
        ip_pattern = r"(?<!\d)(\d{1,3}\.){3}\d{1,3}(?!\d)"
        features["has_ip"] = 1 if re.search(ip_pattern, netloc) else 0

        # Query string 
        query_params = parse_qs(parsed.query)
        features["num_query_params"] = len(query_params)
        features["has_query"]        = 1 if parsed.query else 0

        # Path depth
        path_parts = [p for p in path.split("/") if p]
        features["path_depth"] = len(path_parts)

        # Suspicious TLD
        tld = "." + netloc.rsplit(".", 1)[-1] if "." in netloc else ""
        features["suspicious_tld"] = 1 if tld in SUSPICIOUS_TLDS else 0

        # Brand impersonation signals
        brand_keywords = [
            "paypal", "amazon", "google", "facebook", "apple",
            "netflix", "microsoft", "ebay", "instagram", "whatsapp",
        ]
        combined = netloc + path
        features["brand_in_url"] = 1 if any(b in combined for b in brand_keywords) else 0

        # Shortening services
        shorteners = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "rb.gy"}
        features["is_shortened"] = 1 if netloc in shorteners else 0

        suspicious_words = ["login", "verify", "secure", "account", "update"]

        features["suspicious_word_count"] = sum(
            1 for word in suspicious_words if word in full_url
        )
        
        return features