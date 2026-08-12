from __future__ import annotations
import html
import re
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
from .types import Evidence

USER_AGENT = "ButterflyAI educational local research agent"


def source_trust(url: str):
    host = urlparse(url).hostname or ""
    host = host.lower()
    if host.endswith(".gov") or ".gov." in host:
        return 0.95
    if host.endswith(".edu") or ".edu." in host:
        return 0.90
    if "wikipedia.org" in host:
        return 0.72
    return 0.50


class WebResearch:
    def __init__(self, timeout=8):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def fetch_text(self, url: str, max_chars=8000):
        r = self.session.get(url, timeout=self.timeout)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for bad in soup(["script", "style", "noscript"]):
            bad.decompose()
        text = " ".join(soup.stripped_strings)
        return text[:max_chars]

    def wikipedia_search(self, query: str, limit=3):
        endpoint = "https://es.wikipedia.org/w/api.php"
        params = {"action": "query", "list": "search", "srsearch": query, "format": "json", "utf8": 1}
        r = self.session.get(endpoint, params=params, timeout=self.timeout)
        r.raise_for_status()
        results = r.json().get("query", {}).get("search", [])[:limit]
        evidence = []
        for item in results:
            title = item.get("title", "")
            snippet = re.sub("<.*?>", "", html.unescape(item.get("snippet", "")))
            url = "https://es.wikipedia.org/wiki/" + title.replace(" ", "_")
            evidence.append(Evidence(url, snippet, source_trust(url), None))
        return evidence
