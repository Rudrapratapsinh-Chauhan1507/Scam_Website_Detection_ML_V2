import ipaddress
import random
import socket
import urllib3
from urllib.parse import urljoin, urlparse

import certifi
import requests
from bs4 import BeautifulSoup, Comment
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (X11; Linux x86_64; rv:122.0) Gecko/20100101 Firefox/122.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0"
]

# Suppress SSL warnings when verify=False is used as a last-resort fallback.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class WebsiteScraper:
    def __init__(self, timeout: int = 10, min_text_length: int = 100):
        self.timeout = timeout
        self.min_text_length = min_text_length
        self.session = self._build_session()

    def _get_headers(self) -> dict:
        return {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1"
        }

    def _build_session(self) -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=3,
            connect=3,
            read=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "HEAD"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.trust_env = False
        return session

    @staticmethod
    def _normalize_url(url: str) -> str:
        url = (url or "").strip()
        if not url:
            return ""
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        parsed = urlparse(url)
        if not parsed.netloc:
            return ""

        normalized_netloc = parsed.netloc.strip().lower()
        normalized_path = parsed.path or "/"
        return parsed._replace(netloc=normalized_netloc, path=normalized_path).geturl()

    @staticmethod
    def _extract_domain(url: str) -> str:
        return urlparse(url).hostname or ""

    @staticmethod
    def _homepage_url(url: str) -> str:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return url
        return parsed._replace(path="/", params="", query="", fragment="").geturl()

    @staticmethod
    def _is_public_domain(domain: str) -> bool:
        if not domain:
            return False
        try:
            ip = ipaddress.ip_address(domain)
            return not (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast)
        except ValueError:
            lowered = domain.lower()
            if lowered in {"localhost", "localhost.localdomain"}:
                return False
            return True

    def _resolve_dns(self, domain: str) -> bool:
        if not self._is_public_domain(domain):
            return False
        try:
            socket.gethostbyname(domain)
            return True
        except (socket.gaierror, OSError):
            return False

    def _expand_url(self, url: str) -> str:
        try:
            response = self.session.head(
                url,
                headers=self._get_headers(),
                timeout=self.timeout,
                allow_redirects=True,
                verify=certifi.where(),
            )
            if response.url:
                return response.url
        except requests.RequestException:
            pass

        try:
            response = self.session.get(
                url,
                headers=self._get_headers(),
                timeout=self.timeout,
                allow_redirects=True,
                stream=True,
                verify=certifi.where(),
            )
            final_url = response.url or url
            response.close()
            return final_url
        except requests.RequestException:
            return url

    @staticmethod
    def _candidate_urls(url: str) -> list[str]:
        parsed = urlparse(url)
        if not parsed.netloc:
            return [url]

        candidates: list[str] = []
        netloc = parsed.netloc
        path = parsed.path or "/"
        base_url = parsed._replace(path=path).geturl()
        candidates.append(base_url)

        if parsed.scheme == "https":
            candidates.append(parsed._replace(scheme="http", path=path).geturl())

        if netloc.startswith("www."):
            candidates.append(parsed._replace(netloc=netloc[4:], path=path).geturl())
        else:
            candidates.append(parsed._replace(netloc=f"www.{netloc}", path=path).geturl())

        seen: list[str] = []
        for candidate in candidates:
            if candidate not in seen:
                seen.append(candidate)
        return seen

    def _fetch(self, url: str) -> requests.Response | None:
        for candidate in self._candidate_urls(url):
            for verify in (certifi.where(), False):
                try:
                    response = self.session.get(
                        candidate,
                        headers=self._get_headers(),
                        timeout=self.timeout,
                        verify=verify,
                        allow_redirects=True,
                    )
                    content_type = response.headers.get("Content-Type", "").lower()
                    body_start = response.text[:1000].lower() if response.text else ""
                    
                    is_probably_html = (
                        "text/html" in content_type
                        or "application/xhtml+xml" in content_type
                        or ("<html" in body_start or "<body" in body_start)
                        or ("text/plain" in content_type and ("<" in body_start and ">" in body_start))
                        or not content_type
                    )
                    
                    if "image/" in content_type or "video/" in content_type or "application/pdf" in content_type:
                        is_probably_html = False

                    if response.ok and is_probably_html:
                        response.encoding = response.apparent_encoding or response.encoding or "utf-8"
                        response.redirect_count = len(response.history)
                        return response
                except requests.RequestException:
                    continue
        return None

    @staticmethod
    def _clean_text(text: str) -> str:
        return " ".join(text.split())

    def _parse(self, html: str, base_url: str = "") -> dict:
        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(
            [
                "script",
                "style",
                "noscript",
                "svg",
                "canvas",
                "template",
                "iframe",
                "header",
                "footer",
                "nav",
                "aside",
            ]
        ):
            tag.decompose()

        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()

        title = ""
        if soup.title and soup.title.string:
            title = self._clean_text(soup.title.string)

        meta_description = ""
        meta_tag = (
            soup.find("meta", attrs={"name": lambda value: value and value.lower() == "description"})
            or soup.find("meta", attrs={"property": "og:description"})
        )
        if meta_tag:
            meta_description = self._clean_text(meta_tag.get("content", ""))

        body = soup.body or soup
        visible_text = self._clean_text(body.get_text(separator=" ", strip=True))

        links = []
        for anchor in soup.find_all("a", href=True):
            absolute_url = urljoin(base_url, anchor["href"]) if base_url else anchor["href"]
            links.append(absolute_url)

        return {
            "title": title,
            "meta_description": meta_description,
            "text": visible_text,
            "html": html,
            "text_length": len(visible_text),
            "low_content": int(len(visible_text) < self.min_text_length),
            "link_count": len(links),
        }

    def scrape(self, url: str) -> dict:
        normalized_url = self._normalize_url(url)
        if not normalized_url:
            return self._failure("Invalid URL")

        expanded_url = self._expand_url(normalized_url)
        domain = self._extract_domain(expanded_url)

        if not domain:
            return self._failure("Invalid URL")

        response = self._fetch(expanded_url)
        used_homepage_fallback = False

        if response is None:
            homepage_url = self._homepage_url(expanded_url)
            if homepage_url != expanded_url:
                response = self._fetch(homepage_url)
                used_homepage_fallback = response is not None

        if response is None:
            return {
                **self._base_result(),
                "status": "partial",
                "final_url": expanded_url,
                "error": "Scraping blocked, non-HTML content, or request failed",
            }

        parsed_content = self._parse(response.text, base_url=response.url)
        if parsed_content["low_content"] and self._homepage_url(response.url) != response.url:
            homepage_url = self._homepage_url(response.url)
            homepage_response = self._fetch(homepage_url)
            if homepage_response is not None:
                homepage_content = self._parse(homepage_response.text, base_url=homepage_response.url)
                if homepage_content["text_length"] > parsed_content["text_length"]:
                    response = homepage_response
                    parsed_content = homepage_content
                    used_homepage_fallback = True

        error_message = None
        if used_homepage_fallback:
            error_message = "Homepage fallback used"

        return {
            **self._base_result(),
            "status": "success",
            "final_url": response.url,
            "redirect_count": getattr(response, "redirect_count", len(response.history)),
            "error": error_message,
            **parsed_content,
        }

    @staticmethod
    def _base_result() -> dict:
        return {
            "status": "failed",
            "title": "",
            "meta_description": "",
            "text": "",
            "html": "",
            "text_length": 0,
            "low_content": 0,
            "link_count": 0,
            "final_url": "",
            "redirect_count": 0,
            "error": None,
        }

    def _failure(self, reason: str) -> dict:
        return {
            **self._base_result(),
            "error": reason,
        }
