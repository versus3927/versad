"""Устойчивый HTTP-сканер без Playwright."""
import logging
import xml.etree.ElementTree as ET
from typing import Dict, List
from urllib.parse import parse_qs, quote, unquote, urlparse

import requests
import trafilatura
from bs4 import BeautifulSoup

from . import config

logger = logging.getLogger("scraper")
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36",
    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.7",
}


def _unique(urls: List[str]) -> List[str]:
    result, seen = [], set()
    for url in urls:
        url = (url or "").strip()
        if url.startswith("http") and url not in seen:
            seen.add(url)
            result.append(url)
    return result


def _rss_urls(response: requests.Response, limit: int) -> List[str]:
    root = ET.fromstring(response.content)
    return _unique([(item.findtext("link") or "") for item in root.findall(".//item")])[:limit]


def _search_bing_rss(topic: str, limit: int) -> List[str]:
    response = requests.get(
        "https://www.bing.com/search",
        params={"q": topic, "format": "rss", "count": limit},
        headers=HEADERS,
        timeout=15,
    )
    response.raise_for_status()
    return _rss_urls(response, limit)


def _search_google_news(topic: str, limit: int) -> List[str]:
    response = requests.get(
        "https://news.google.com/rss/search",
        params={"q": topic, "hl": "ru", "gl": "RU", "ceid": "RU:ru"},
        headers=HEADERS,
        timeout=15,
    )
    response.raise_for_status()
    return _rss_urls(response, limit)


def _resolve_ddg(href: str) -> str:
    if href.startswith("//"):
        href = "https:" + href
    if "duckduckgo.com/l/" in href:
        target = parse_qs(urlparse(href).query).get("uddg", [""])[0]
        return unquote(target) if target else href
    return href


def _search_ddg(topic: str, limit: int) -> List[str]:
    response = requests.get(
        "https://html.duckduckgo.com/html/",
        params={"q": topic},
        headers=HEADERS,
        timeout=15,
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    return _unique([_resolve_ddg(a.get("href", "")) for a in soup.select("a.result__a[href]")])[:limit]


def _search_bing_html(topic: str, limit: int) -> List[str]:
    response = requests.get(
        "https://www.bing.com/search",
        params={"q": topic, "count": limit, "setlang": "ru"},
        headers=HEADERS,
        timeout=15,
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    return _unique([a.get("href", "") for a in soup.select("li.b_algo h2 a[href]")])[:limit]


SEARCHERS = (_search_bing_rss, _search_google_news, _search_bing_html, _search_ddg)


def search_urls(topic: str, max_results: int) -> List[str]:
    urls: List[str] = []
    for searcher in SEARCHERS:
        remaining = max_results - len(_unique(urls))
        if remaining <= 0:
            break
        try:
            found = searcher(topic, remaining)
            urls.extend(found)
            logger.info("%s: '%s' -> %s ссылок", searcher.__name__, topic, len(found))
        except Exception as exc:
            logger.warning("%s failed for '%s': %s", searcher.__name__, topic, exc)
    if len(_unique(urls)) < max_results:
        title = topic.strip().replace(" ", "_")
        if title:
            urls.append("https://ru.wikipedia.org/wiki/" + quote(title))
    return _unique(urls)[:max_results]


def fetch_text(url: str) -> str | None:
    try:
        response = requests.get(url, headers=HEADERS, timeout=20, allow_redirects=True)
        response.raise_for_status()
        text = trafilatura.extract(
            response.text,
            include_comments=False,
            include_tables=False,
            url=response.url,
        )
        if text and len(text) >= config.MIN_CHARS_PER_PAGE:
            return text
        logger.info("слишком мало текста: %s", url)
    except Exception as exc:
        logger.warning("fetch failed for %s: %s", url, exc)
    return None


def collect_documents(topics: list[str] | None = None) -> List[Dict]:
    docs: List[Dict] = []
    for topic in topics or config.SEARCH_TOPICS:
        urls = search_urls(topic, config.PAGES_PER_TOPIC)
        valid = 0
        for url in urls:
            text = fetch_text(url)
            if text:
                docs.append({"topic": topic, "url": url, "text": text})
                valid += 1
        logger.info("topic '%s': собрано %s ссылок, годных текстов %s", topic, len(urls), valid)
    return docs
