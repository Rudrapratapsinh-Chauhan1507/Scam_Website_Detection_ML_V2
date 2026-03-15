import subprocess
import ssl
import socket
import tldextract
import re
from datetime import datetime


class DomainFeatureExtractor:

    def extract(self, url):

        extracted = tldextract.extract(url)
        domain = extracted.domain + "." + extracted.suffix

        features = {
            "domain_age_days": None,
            "domain_expiry_days": None,
            "registrar": None,
            "has_ssl": 0,
            "ssl_valid": 0
        }

        # -------- WHOIS --------
        try:

            result = subprocess.check_output(["whois", domain], text=True)

            # Extract creation date
            creation_match = re.search(r"Creation Date:\s*(.+)", result)

            # Extract expiry date
            expiry_match = re.search(r"Expiration Date:\s*(.+)", result)
            if not expiry_match:
                expiry_match = re.search(r"Registrar Registration Expiration Date:\s*(.+)", result)

            # Extract registrar
            registrar_match = re.search(r"Registrar:\s*(.+)", result)

            if creation_match:
                creation_date = datetime.fromisoformat(creation_match.group(1).strip().replace("Z",""))
                features["domain_age_days"] = (datetime.now() - creation_date).days

            if expiry_match:
                expiry_date = datetime.fromisoformat(expiry_match.group(1).strip().replace("Z",""))
                features["domain_expiry_days"] = (expiry_date - datetime.now()).days

            if registrar_match:
                features["registrar"] = registrar_match.group(1).strip()

        except Exception as e:

            print("WHOIS parsing failed:", domain)

        # -------- SSL --------
        try:

            context = ssl.create_default_context()

            with socket.create_connection((domain, 443), timeout=5) as sock:

                with context.wrap_socket(sock, server_hostname=domain) as ssock:

                    cert = ssock.getpeercert()

                    if cert:
                        features["has_ssl"] = 1
                        features["ssl_valid"] = 1

        except:
            features["has_ssl"] = 0
            features["ssl_valid"] = 0

        return features