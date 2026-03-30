import ssl
import socket
import tldextract
import subprocess
import re
from datetime import datetime, timezone


# Regex patterns for WHOIS field extraction
_WHOIS_PATTERNS = {
    "creation": re.compile(
        r"(?:Creation Date|Created On|Domain Registration Date):\s*(.+)", re.IGNORECASE
    ),
    "expiry": re.compile(
        r"(?:Expir(?:y|ation) Date|Registrar Registration Expiration Date):\s*(.+)",
        re.IGNORECASE,
    ),
    "registrar": re.compile(r"Registrar:\s*(.+)", re.IGNORECASE),
}


def _parse_date(raw: str) -> datetime | None:
    """Try common ISO / WHOIS date formats. Returns UTC-aware datetime or None."""
    raw = raw.strip().rstrip("Z")
    # Remove trailing timezone info like '+0000' or ' UTC'
    raw = re.sub(r"[\s+]\d{4}$", "", raw).replace(" UTC", "").strip()
    formats = [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%d-%b-%Y",  # 01-Jan-2020
        "%d/%m/%Y",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


class DomainFeatureExtractor:

    def _run_whois(self, domain: str) -> dict:
        """
        Run system whois and parse creation / expiry / registrar fields.
        Returns partial feature dict (values may be None on failure).
        """
        result = {
            "domain_age_days": None,
            "domain_expiry_days": None,
            "registrar": None,
        }
        try:
            raw = subprocess.check_output(
                ["whois", domain],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=15,
            )
        except (subprocess.SubprocessError, FileNotFoundError):
            return result

        now = datetime.now(tz=timezone.utc)

        creation_match = _WHOIS_PATTERNS["creation"].search(raw)
        if creation_match:
            dt = _parse_date(creation_match.group(1))
            if dt:
                result["domain_age_days"] = (now - dt).days

        expiry_match = _WHOIS_PATTERNS["expiry"].search(raw)
        if expiry_match:
            dt = _parse_date(expiry_match.group(1))
            if dt:
                result["domain_expiry_days"] = (dt - now).days

        registrar_match = _WHOIS_PATTERNS["registrar"].search(raw)
        if registrar_match:
            result["registrar"] = registrar_match.group(1).strip()

        return result

    def _check_ssl(self, domain: str) -> dict:
        """
        Attempt TLS handshake and inspect certificate.
        Returns has_ssl / ssl_valid / ssl_days_remaining.
        """
        result = {"has_ssl": 0, "ssl_valid": 0, "ssl_days_remaining": None}
        try:
            ctx = ssl.create_default_context()
            with socket.create_connection((domain, 443), timeout=5) as sock:
                with ctx.wrap_socket(sock, server_hostname=domain) as ssock:
                    cert = ssock.getpeercert()
                    if cert:
                        result["has_ssl"] = 1
                        result["ssl_valid"] = 1
                        # Calculate days until cert expiry
                        not_after = cert.get("notAfter")
                        if not_after:
                            expiry_dt = datetime.strptime(
                                not_after, "%b %d %H:%M:%S %Y %Z"
                            ).replace(tzinfo=timezone.utc)
                            result["ssl_days_remaining"] = (
                                expiry_dt - datetime.now(tz=timezone.utc)
                            ).days
        except (ssl.SSLError, socket.error, OSError):
            pass  # leave defaults (0 / None)
        return result

    def extract(self, url: str) -> dict:
        extracted = tldextract.extract(url)
        domain = f"{extracted.domain}.{extracted.suffix}"

        features: dict = {}
        features.update(self._run_whois(domain))
        features.update(self._check_ssl(domain))

        expiry_days = features.get("domain_expiry_days")

        features["short_expiry_domain"] = (
            1 if (expiry_days is not None and expiry_days < 180) else 0
        )
        # Derived flag: very new domains (< 180 days) are higher risk
        age = features.get("domain_age_days")
        features["is_new_domain"] = (
            1 if (age is not None and age < 180) else 0
        )

        return features