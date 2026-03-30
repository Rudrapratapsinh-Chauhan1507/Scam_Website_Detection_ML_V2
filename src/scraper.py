import requests
import certifi
import socket
import urllib3
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib.parse import urlparse

# Suppress SSL warnings when verify=False is used as fallback
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class WebsiteScraper:

    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml",
            "Connection": "keep-alive",
        }
        self.session = self._build_session()

    def _build_session(self) -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET"],        # modern urllib3 parameter
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def _resolve_dns(self, domain: str) -> bool:
        """Return True if domain resolves, False otherwise."""
        try:
            socket.gethostbyname(domain)
            return True
        except socket.gaierror:
            return False

    def _expand_url(self, url: str) -> str:
        """Force expand shortened URLs like bit.ly"""
        try:
            resp = requests.get(
                url,
                headers=self.headers,
                timeout=10,
                allow_redirects=False  
            )

            # If redirect exists → extract it
            if "Location" in resp.headers:
                return resp.headers["Location"]

            return url
        except:
            return url

    def _fetch(self, url: str) -> requests.Response | None:
        """
        Try to fetch the URL. Attempts verified SSL first,
        then falls back to unverified SSL for self-signed certs.
        Returns the first successful Response or None.
        """
        candidates = [url]

        # If the original URL uses https, also try http fallback
        if url.startswith("https://"):
            candidates.append(url.replace("https://", "http://", 1))

        # Also try dropping/adding www
        base = candidates[0]
        if "www." in base:
            candidates.append(base.replace("www.", "", 1))
        else:
            parsed = urlparse(base)
            candidates.append(
                f"{parsed.scheme}://www.{parsed.netloc}{parsed.path}"
            )

        for candidate in candidates:
            # First attempt: verified SSL
            for verify in (certifi.where(), False):
                try:
                    resp = self.session.get(
                        candidate,
                        headers=self.headers,
                        timeout=self.timeout,
                        verify=verify,
                        allow_redirects=True
                    )
                    if resp.status_code == 200:
                        # Force UTF-8 detection from content rather than header
                        resp.encoding = resp.apparent_encoding or "utf-8"
                        resp.final_url = resp.url
                        resp.redirect_count = len(resp.history)
                        return resp
                except requests.RequestException:
                    continue  # try next verify / next candidate

        return None

    def _parse(self, html: str) -> dict:
        """Extract title, meta description, and clean body text from HTML."""
        soup = BeautifulSoup(html, "lxml")

        for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
            tag.decompose()

        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()

        meta_desc = ""
        meta = soup.find("meta", attrs={"name": "description"})
        if meta:
            meta_desc = meta.get("content", "").strip()

        raw_text = soup.get_text(separator=" ")
        clean_text = " ".join(raw_text.split())

        return {
            "title": title,
            "meta_description": meta_desc,
            "text": clean_text,
            "html": html,
        }

    def scrape(self, url: str) -> dict:
        """
        Main entry point.  Returns a dict with keys:
            status, title, meta_description, text, error
        """
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        # FORCE EXPANSION
        expanded_url = self._expand_url(url)

        # Debug
        print(f"[DEBUG] Expanded URL: {expanded_url}")

        # Use expanded URL for further processing
        url = expanded_url

        parsed = urlparse(url)
        domain = parsed.netloc

        # DNS check
        if not self._resolve_dns(domain):
            return self._failure("DNS resolution failed")

        response = self._fetch(url)

        if response is None:
            return {
                "status": "partial",
                "title": "",
                "meta_description": "",
                "text": "",
                "final_url": url,
                "redirect_count": 0, 
                "error": "Scraping blocked or failed"
            }

        parsed_content = self._parse(response.text)

        return {
            "status": "success",
            "error": None,
            "final_url": response.url,
            "redirect_count": getattr(response, "redirect_count", 0),
            **parsed_content,
        }

    @staticmethod
    def _failure(reason: str) -> dict:
        return {
            "status": "failed",
            "title": None,
            "meta_description": None,
            "text": None,
            "error": reason,
        }