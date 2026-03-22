import requests
import certifi
import socket
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib.parse import urlparse


class WebsiteScraper:

    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
        }

        # Retry session
        self.session = requests.Session()

        retry = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504]
        )

        adapter = HTTPAdapter(max_retries=retry)

        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def scrape(self, url):

        try:
            parsed = urlparse(url)
            domain = parsed.netloc

            # ✅ DNS Check
            try:
                socket.gethostbyname(domain)
            except:
                return {
                    "status": "failed",
                    "title": None,
                    "meta_description": None,
                    "text": None,
                    "error": "DNS resolution failed"
                }

            # ✅ Try multiple URL formats
            urls_to_try = [
                url,
                url.replace("https://", "http://"),
                url.replace("www.", "")
            ]

            response = None

            for u in urls_to_try:
                try:
                    response = self.session.get(
                        u,
                        headers=self.headers,
                        timeout=10,
                        verify=certifi.where()
                    )

                    if response.status_code == 200:
                        break

                except Exception:
                    # fallback (ignore SSL)
                    try:
                        response = self.session.get(
                            u,
                            headers=self.headers,
                            timeout=10,
                            verify=False
                        )
                        if response.status_code == 200:
                            break
                    except:
                        continue

            if response is None or response.status_code != 200:
                return {
                    "status": "failed",
                    "title": None,
                    "meta_description": None,
                    "text": None,
                    "error": "Unable to fetch website"
                }

            # ✅ Parse HTML
            soup = BeautifulSoup(response.text, "lxml")

            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()

            title = soup.title.string.strip() if soup.title else ""

            meta_desc = ""
            meta = soup.find("meta", attrs={"name": "description"})
            if meta:
                meta_desc = meta.get("content", "")

            text = soup.get_text(separator=" ")
            clean_text = " ".join(text.split())

            return {
                "status": "success",
                "title": title,
                "meta_description": meta_desc,
                "text": clean_text,
                "error": None
            }

        except Exception as e:
            return {
                "status": "failed",
                "title": None,
                "meta_description": None,
                "text": None,
                "error": str(e)
            }