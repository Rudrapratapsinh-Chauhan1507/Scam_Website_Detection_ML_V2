import requests
import certifi
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class WebsiteScraper:

    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0"
        }

        # retry mechanism
        self.session = requests.Session()

        retry = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[500,502,503,504]
        )

        adapter = HTTPAdapter(max_retries=retry)

        self.session.mount("http://",adapter)
        self.session.mount("https://",adapter)

    def scrape(self, url):

        try:

            response = requests.get(url, headers=self.headers, timeout=10, verify=certifi.where())

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
                "status":"failed",
                "title":None,
                "meta_description":None,
                "text":None,
                "error":str(e)
            }