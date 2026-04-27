import ipaddress
import math
import re
from collections import Counter
from urllib.parse import parse_qs, urlparse

import tldextract

# TLDs commonly abused in phishing / scam campaigns
SUSPICIOUS_TLDS = {
    ".tk",
    ".ml",
    ".ga",
    ".cf",
    ".gq",
    ".xyz",
    ".top",
    ".club",
    ".online",
    ".site",
    ".info",
    ".biz",
    ".pw",
    ".win",
    ".loan",
    ".rest",
    ".run",
    ".work",
    ".click",
    ".link",
    ".buzz",
    ".icu",
}

BRAND_KEYWORDS = {
    "paypal",
    "amazon",
    "google",
    "facebook",
    "apple",
    "netflix",
    "microsoft",
    "ebay",
    "instagram",
    "whatsapp",
    "youtube",
    "twitter",
    "linkedin",
    "dropbox",
    "chase",
    "wellsfargo",
    "bankofamerica",
    "hdfc",
    "icici",
    "sbi",
    "adobe",
    "binance",
    "coinbase",
    "docusign",
    "gemini",
    "kucoin",
    "metamask",
    "roblox",
    "shopee",
    "trezor",
}

OFFICIAL_BRAND_DOMAINS = {
    "adobe": {"adobe.com"},
    "amazon": {"amazon.com"},
    "apple": {"apple.com"},
    "binance": {"binance.com"},
    "coinbase": {"coinbase.com"},
    "docusign": {"docusign.com"},
    "facebook": {"facebook.com"},
    "gemini": {"gemini.com"},
    "google": {"google.com"},
    "kucoin": {"kucoin.com"},
    "metamask": {"metamask.io"},
    "microsoft": {"microsoft.com"},
    "netflix": {"netflix.com"},
    "paypal": {"paypal.com"},
    "roblox": {"roblox.com"},
    "shopee": {"shopee.com"},
    "trezor": {"trezor.io"},
}

SHORTENING_SERVICES = {
    "bit.ly",
    "tinyurl.com",
    "t.co",
    "goo.gl",
    "ow.ly",
    "rb.gy",
    "is.gd",
    "buff.ly",
    "cutt.ly",
    "tiny.cc",
    "rebrand.ly",
    "shorturl.at",
    "urlz.fr",
    "short.ly",
    "u.to",
    "v.gd",
    "clck.ru",
    "snip.ly",
    "bl.ink",
    "trib.al",
}

FREE_HOSTING_DOMAINS = {
    "blogspot.com",
    "framer.app",
    "github.io",
    "ipfs.dweb.link",
    "netlify.app",
    "pages.dev",
    "typedream.app",
    "vercel.app",
    "webflow.io",
    "weebly.com",
    "weeblysite.com",
}

SUSPICIOUS_WORDS = {
    "login",
    "verify",
    "secure",
    "account",
    "update",
    "signin",
    "confirm",
    "banking",
    "password",
    "wallet",
    "support",
    "ebayisapi",
    "webscr",
    "cmd",
    "dispatch",
    "redirect",
    "checkout",
    "gift",
    "prize",
    "winner",
    "free",
    "alert",
    "airdrop",
    "bridge",
    "claim",
    "giveaway",
    "kyc",
    "restore",
    "unlock",
    "validation",
}

# Known TLD extensions — used to detect TLDs embedded in paths
_PATH_TLD_RE = re.compile(
    r"\.(com|net|org|info|biz|edu|gov|io|co|me|tk|ml|ga|cf|gq|xyz|top|club|online|site)(/|$)",
    re.IGNORECASE,
)

# Longest run of consecutive digits
_CONSEC_DIGITS_RE = re.compile(r"\d+")
_TLD_EXTRACTOR = tldextract.TLDExtract(cache_dir=None, suffix_list_urls=None)


def _shannon_entropy(text: str) -> float:
    """Compute Shannon entropy of a string (bits per character)."""
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


class URLFeatureExtractor:
    @staticmethod
    def _normalize_url(url: str) -> str:
        url = (url or "").strip()
        if not url:
            return ""
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        return url

    @staticmethod
    def _hostname(parsed) -> str:
        return (parsed.hostname or "").lower()

    @staticmethod
    def _has_ip_address(hostname: str) -> int:
        if not hostname:
            return 0
        try:
            ipaddress.ip_address(hostname)
            return 1
        except ValueError:
            return 0

    @staticmethod
    def _subdomain_count(hostname: str) -> int:
        if not hostname or URLFeatureExtractor._has_ip_address(hostname):
            return 0

        parts = [part for part in hostname.split(".") if part]
        if len(parts) <= 2:
            return 0

        # Approximation that works well for most cases while keeping the feature simple.
        common_second_level_suffixes = {"co", "com", "org", "net", "gov", "edu", "ac"}
        if len(parts) >= 3 and len(parts[-1]) == 2 and parts[-2] in common_second_level_suffixes:
            return max(len(parts) - 3, 0)

        return max(len(parts) - 2, 0)

    @staticmethod
    def _extract_tld(hostname: str) -> str:
        if not hostname or "." not in hostname:
            return ""
        return "." + hostname.rsplit(".", 1)[-1]

    @staticmethod
    def _digit_letter_ratio(hostname: str) -> float:
        """Ratio of digit characters to letter characters in hostname."""
        if not hostname:
            return 0.0
        digits = sum(c.isdigit() for c in hostname)
        letters = sum(c.isalpha() for c in hostname)
        if letters == 0:
            return 1.0 if digits > 0 else 0.0
        return round(digits / letters, 4)

    @staticmethod
    def _num_special_chars(url: str) -> int:
        """Count obfuscation-style special characters in URL."""
        return sum(url.count(c) for c in "~={}|^`")

    @staticmethod
    def _tld_in_path(path: str) -> int:
        """Detect if a TLD appears inside the URL path (common in phishing redirects)."""
        return int(bool(_PATH_TLD_RE.search(path)))

    @staticmethod
    def _longest_digit_run(url: str) -> int:
        """Longest consecutive run of digits (flags IP fragments or random domains)."""
        runs = _CONSEC_DIGITS_RE.findall(url)
        return max((len(r) for r in runs), default=0)

    @staticmethod
    def _uses_free_hosting(hostname: str) -> int:
        hostname = hostname.lower().strip(".")
        return int(
            any(hostname == domain or hostname.endswith(f".{domain}") for domain in FREE_HOSTING_DOMAINS)
        )

    @staticmethod
    def _registered_domain(hostname: str) -> str:
        extracted = _TLD_EXTRACTOR(hostname)
        if extracted.domain and extracted.suffix:
            return f"{extracted.domain}.{extracted.suffix}".lower()
        return hostname.lower()

    @staticmethod
    def _official_brand_domain(registered_domain: str, brand_hits: set[str]) -> int:
        return int(any(registered_domain in OFFICIAL_BRAND_DOMAINS.get(brand, set()) for brand in brand_hits))

    def extract(self, url: str) -> dict:
        normalized_url = self._normalize_url(url)
        parsed = urlparse(normalized_url)

        hostname = self._hostname(parsed)
        netloc = (parsed.netloc or "").lower()
        path = (parsed.path or "").lower()
        query = parsed.query or ""
        full_url = normalized_url.lower()

        query_params = parse_qs(query, keep_blank_values=True)
        path_parts = [part for part in path.split("/") if part]

        brand_hits = {brand for brand in BRAND_KEYWORDS if brand in f"{hostname}{path}"}
        registered_domain = self._registered_domain(hostname)
        official_brand_domain = self._official_brand_domain(registered_domain, brand_hits)
        brand_in_url = int(bool(brand_hits))
        uses_free_hosting = self._uses_free_hosting(hostname)
        suspicious_word_count = sum(1 for word in SUSPICIOUS_WORDS if word in full_url)

        features: dict = {
            # ── Original features 
            "url_length": len(normalized_url),
            "num_dots": normalized_url.count("."),
            "num_hyphen": normalized_url.count("-"),
            "num_slashes": normalized_url.count("/"),
            "https": int(parsed.scheme == "https"),
            "subdomains": self._subdomain_count(hostname),
            "has_at_symbol": int("@" in normalized_url),
            "has_double_slash": int("//" in path or re.search(r"/{2,}", path) is not None),
            "has_ip": self._has_ip_address(hostname),
            "num_underscores": normalized_url.count("_"),
            "num_percent": normalized_url.count("%"),
            "num_digits": sum(char.isdigit() for char in normalized_url),
            "num_query_params": len(query_params),
            "has_query": int(bool(query)),
            "path_depth": len(path_parts),
            "suspicious_tld": int(self._extract_tld(hostname) in SUSPICIOUS_TLDS),
            "brand_in_url": brand_in_url,
            "is_shortened": int(hostname in SHORTENING_SERVICES),
            "suspicious_word_count": suspicious_word_count,
            # ── New features 
            "url_entropy": round(_shannon_entropy(normalized_url), 4),
            "hostname_length": len(hostname),
            "digit_letter_ratio": self._digit_letter_ratio(hostname),
            "num_special_chars": self._num_special_chars(normalized_url),
            "tld_in_path": self._tld_in_path(path),
            "longest_digit_run": self._longest_digit_run(hostname),
            "uses_free_hosting": uses_free_hosting,
            "brand_on_free_hosting": int(brand_in_url and uses_free_hosting),
            "brand_domain_mismatch": int(brand_in_url and not official_brand_domain),
        }

        # Ensure credentials or ports in netloc still contribute only to existing features.
        if hostname and hostname != netloc:
            features["has_at_symbol"] = max(features["has_at_symbol"], int("@" in netloc))

        return features
