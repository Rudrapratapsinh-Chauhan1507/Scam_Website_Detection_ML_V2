import re
from urllib.parse import urlparse


class URLFeatureExtractor:

    def extract(self, url):

        features = {}

        parsed = urlparse(url)

        # URL Length
        features["url_length"] = len(url)

        # number of dots
        features["num_dots"] = url.count(".")

        # number of hyphen
        features["num_hyphen"] = url.count("-")

        # number of slashes
        features["num_slashes"] = url.count("/")

        # HTTPS usage
        features["https"] = 1 if parsed.scheme == "https" else 0

        # subdomain count
        domain_parts = parsed.netloc.split(".")
        features["subdomains"] = max(len(domain_parts) - 2, 0)

        # @ symbol
        features["has_at_symbol"] = 1 if "@" in url else 0

        # detect IP address
        ip_pattern = r"\d+\.\d+\.\d+\.\d+"
        features["has_ip"] = 1 if re.search(ip_pattern, url) else 0

        return features