import re
import socket
import ssl
import subprocess
from datetime import datetime, timezone
from urllib.parse import urlparse

import tldextract

try:
    import whois as python_whois
except ImportError:
    python_whois = None


_WHOIS_PATTERNS = {
    "creation": [
        re.compile(r"(?:Creation Date|Created On|Domain Registration Date|Registered On):\s*(.+)", re.IGNORECASE),
        re.compile(r"(?:Created|Registered):\s*(.+)", re.IGNORECASE),
    ],
    "expiry": [
        re.compile(
            r"(?:Expir(?:y|ation) Date|Registrar Registration Expiration Date|Registry Expiry Date|Paid-till):\s*(.+)",
            re.IGNORECASE,
        ),
        re.compile(r"(?:Expires On|Expiry):\s*(.+)", re.IGNORECASE),
    ],
    "registrar": [
        re.compile(r"Registrar:\s*(.+)", re.IGNORECASE),
        re.compile(r"Registrar Name:\s*(.+)", re.IGNORECASE),
        re.compile(r"Sponsoring Registrar:\s*(.+)", re.IGNORECASE),
    ],
}


def _parse_date(raw: str) -> datetime | None:
    if not raw:
        return None

    raw = raw.strip().strip('"').strip("'")
    raw = raw.replace("(UTC)", "").replace(" UTC", "").strip()
    raw = re.sub(r"\.\d+$", "", raw)
    raw = re.sub(r"\s+[A-Z]{2,5}$", "", raw)
    raw = re.sub(r"([+-]\d{2}):?(\d{2})$", "", raw).strip()
    raw = raw.rstrip("Z").strip()

    formats = [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d",
        "%Y-%m-%d",
        "%d-%b-%Y",
        "%d-%b-%Y %H:%M:%S",
        "%d.%m.%Y",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%b %d %Y",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


class DomainFeatureExtractor:
    def __init__(self, timeout: int = 8):
        self.timeout = timeout
        self._tld_extractor = tldextract.TLDExtract(suffix_list_urls=None)

    def _extract_registered_domain(self, url: str) -> str:
        candidate = url if "://" in (url or "") else f"https://{url}"
        hostname = urlparse(candidate).hostname or candidate
        hostname = (hostname or "").lower()

        try:
            extracted = self._tld_extractor(hostname)
            if extracted.domain and extracted.suffix:
                return f"{extracted.domain}.{extracted.suffix}".lower()
        except Exception:
            pass

        parts = [part for part in hostname.split(".") if part]
        if len(parts) >= 3 and len(parts[-1]) == 2 and parts[-2] in {"co", "com", "org", "net", "gov", "edu", "ac"}:
            return ".".join(parts[-3:])
        if len(parts) >= 2:
            return ".".join(parts[-2:])
        return hostname

    @staticmethod
    def _match_first(raw_text: str, patterns: list[re.Pattern]) -> str | None:
        for pattern in patterns:
            match = pattern.search(raw_text)
            if match:
                value = match.group(1).strip()
                if value:
                    return value
        return None

    def _run_whois(self, domain: str) -> dict:
        result = {
            "domain_age_days": -1,
            "domain_expiry_days": -1,
            "registrar": "",
            "missing_whois": 1,
        }

        if not domain:
            return result

        python_result = self._run_python_whois(domain)
        if python_result.get("missing_whois") == 0 or python_result.get("registrar"):
            return python_result

        return self._run_system_whois(domain)

    @staticmethod
    def _first_value(value):
        if isinstance(value, (list, tuple)):
            return value[0] if value else None
        return value

    @staticmethod
    def _coerce_datetime(value) -> datetime | None:
        value = DomainFeatureExtractor._first_value(value)
        if value is None:
            return None
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc)
        if isinstance(value, str):
            return _parse_date(value)
        return None

    def _run_python_whois(self, domain: str) -> dict:
        result = {
            "domain_age_days": -1,
            "domain_expiry_days": -1,
            "registrar": "",
            "missing_whois": 1,
        }

        if python_whois is None:
            return result

        try:
            info = python_whois.whois(domain)
        except Exception:
            return result

        now = datetime.now(tz=timezone.utc)

        creation_dt = self._coerce_datetime(getattr(info, "creation_date", None))
        if creation_dt:
            result["domain_age_days"] = max((now - creation_dt).days, 0)
            result["missing_whois"] = 0

        expiry_dt = self._coerce_datetime(getattr(info, "expiration_date", None))
        if expiry_dt:
            result["domain_expiry_days"] = (expiry_dt - now).days
            result["missing_whois"] = 0

        registrar = self._first_value(getattr(info, "registrar", None))
        if registrar:
            result["registrar"] = str(registrar).strip()[:255]
            result["missing_whois"] = 0

        return result

    def _run_system_whois(self, domain: str) -> dict:
        result = {
            "domain_age_days": -1,
            "domain_expiry_days": -1,
            "registrar": "",
            "missing_whois": 1,
        }

        try:
            raw = subprocess.check_output(
                ["whois", domain],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=max(self.timeout, 20),
            )
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            return result

        if not raw.strip() or ("Connecting to" in raw and "Registrar:" not in raw):
            return result

        now = datetime.now(tz=timezone.utc)

        creation_raw = self._match_first(raw, _WHOIS_PATTERNS["creation"])
        creation_dt = _parse_date(creation_raw) if creation_raw else None
        if creation_dt:
            age_days = (now - creation_dt).days
            result["domain_age_days"] = max(age_days, 0)
            result["missing_whois"] = 0

        expiry_raw = self._match_first(raw, _WHOIS_PATTERNS["expiry"])
        expiry_dt = _parse_date(expiry_raw) if expiry_raw else None
        if expiry_dt:
            result["domain_expiry_days"] = (expiry_dt - now).days
            result["missing_whois"] = 0

        registrar = self._match_first(raw, _WHOIS_PATTERNS["registrar"])
        if registrar:
            result["registrar"] = registrar[:255]
            result["missing_whois"] = 0

        return result

    def _check_ssl(self, domain: str) -> dict:
        result = {"has_ssl": 0, "ssl_valid": 0, "ssl_days_remaining": -1}

        if not domain:
            return result

        try:
            with socket.create_connection((domain, 443), timeout=self.timeout) as sock:
                result["has_ssl"] = 1

                context = ssl.create_default_context()
                with context.wrap_socket(sock, server_hostname=domain) as secure_sock:
                    cert = secure_sock.getpeercert()
                    if cert:
                        result["ssl_valid"] = 1
                        not_after = cert.get("notAfter")
                        if not_after:
                            expiry_dt = datetime.strptime(
                                not_after, "%b %d %H:%M:%S %Y %Z"
                            ).replace(tzinfo=timezone.utc)
                            result["ssl_days_remaining"] = (
                                expiry_dt - datetime.now(tz=timezone.utc)
                            ).days
        except ssl.SSLCertVerificationError:
            result["has_ssl"] = 1
        except (ssl.SSLError, socket.timeout, socket.gaierror, ConnectionError, OSError):
            pass

        return result

    def extract(self, url: str) -> dict:
        domain = self._extract_registered_domain(url)

        features: dict = {}
        features.update(self._run_whois(domain))
        features.update(self._check_ssl(domain))

        expiry_days = features.get("domain_expiry_days")
        age_days = features.get("domain_age_days")

        features["short_expiry_domain"] = int(
            expiry_days is not None and expiry_days != -1 and expiry_days < 180
        )
        features["is_new_domain"] = int(
            age_days is not None and age_days != -1 and age_days < 180
        )

        return features
