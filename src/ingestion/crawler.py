from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import NamedTuple
from urllib.parse import urljoin, urlparse, urldefrag

import requests
from requests.exceptions import RequestException

log = logging.getLogger(__name__)

COPPERMIND_BASE = "https://coppermind.net"
COPPERMIND_WIKI = f"{COPPERMIND_BASE}/wiki"
DATA_DIR = Path(__file__).parent.parent.parent / "data"
CRAWL_DATA_DIR = DATA_DIR / "crawl"
CRAWL_DATA_DIR.mkdir(parents=True, exist_ok=True)

PAGES_FILE = CRAWL_DATA_DIR / "pages.jsonl"
GRAPH_FILE = CRAWL_DATA_DIR / "graph_edges.jsonl"
LINKS_FILE = CRAWL_DATA_DIR / "discovered_links.json"
ROBOTS_FILE = CRAWL_DATA_DIR / "robots.txt"

DISALLOWED_PATTERNS = [
    r"/w/",
    r"/mw/",
    r"/wiki/Special:",
    r"/wiki/special%3A",
]

SLEEP_INTERVAL = 1.0


class CrawledPage(NamedTuple):
    url: str
    title: str
    text: str
    links: list[str]


def _load_robots() -> set[str]:
    if ROBOTS_FILE.exists():
        content = ROBOTS_FILE.read_text()
    else:
        try:
            resp = requests.get(f"{COPPERMIND_BASE}/robots.txt", timeout=10)
            resp.raise_for_status()
            content = resp.text
            ROBOTS_FILE.write_text(content)
        except RequestException as e:
            log.warning("Failed to fetch robots.txt: %s", e)
            return set()

    disallowed: set[str] = set()
    for line in content.splitlines():
        if line.startswith("Disallow:"):
            path = line.split("Disallow:", 1)[1].strip()
            if path:
                disallowed.add(path)
    return disallowed


def _is_allowed(path: str) -> bool:
    for pattern in DISALLOWED_PATTERNS:
        if re.search(pattern, path):
            return False
    return True


def _normalize_url(url: str) -> tuple[str, str | None]:
    cleaned, frag = urldefrag(url)
    parsed = urlparse(cleaned)
    if parsed.scheme not in ("http", "https"):
        return "", None
    if not parsed.netloc.endswith("coppermind.net"):
        return "", None
    path = parsed.path
    if not path.startswith("/wiki/"):
        return "", None
    return cleaned, frag


def _fetch_page(url: str) -> tuple[str, str, list[str]]:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    html = resp.text

    title_match = re.search(r"<title>([^<]+)</title>", html)
    title = title_match.group(1) if title_match else ""

    content_match = re.search(
        r'<div[^>]*id=["\']mw-content-text["\'][^>]*>(.*?)</div>\s*</div>',
        html,
        re.DOTALL,
    )
    content = content_match.group(1) if content_match else ""

    text = re.sub(r"<[^>]+>", "", content)
    text = re.sub(r"\s+", " ", text).strip()

    link_pattern = re.compile(r'href="(/wiki/[^"#]+)"')
    links: list[str] = []
    for match in link_pattern.finditer(html):
        link = match.group(1)
        if _is_allowed(link):
            links.append(link)

    return title, text, links


def _load_discovered() -> set[str]:
    if LINKS_FILE.exists():
        return set(json.loads(LINKS_FILE.read_text()))
    return set()


def _save_discovered(links: set[str]) -> None:
    LINKS_FILE.write_text(json.dumps(list(links)))


def crawl(start_path: str = "/wiki/Main_Page", limit: int | None = None) -> None:
    robots_disallowed = _load_robots()
    discovered = _load_discovered()
    discovered.add(start_path)
    visited: set[str] = set()
    count = 0

    log.info("Starting crawl from %s", start_path)

    while discovered:
        path = discovered.pop()
        if path in visited:
            continue
        parsed = urlparse(path)
        if parsed.path in robots_disallowed or not _is_allowed(path):
            visited.add(path)
            continue

        url = urljoin(COPPERMIND_BASE, path)
        log.info("Crawling %s", url)

        try:
            title, text, links = _fetch_page(url)
            page_links: list[str] = []
            for link in links:
                normalized, frag = _normalize_url(urljoin(COPPERMIND_BASE, link))
                if normalized:
                    if normalized not in visited:
                        discovered.add(normalized)
                    if frag:
                        page_links.append(f"{normalized}#{frag}")
                    else:
                        page_links.append(normalized)

            page: CrawledPage = CrawledPage(
                url=url,
                title=title,
                text=text,
                links=page_links,
            )
            with PAGES_FILE.open("a", encoding="utf-8") as f:
                f.write(json.dumps(page._asdict(), ensure_ascii=False) + "\n")

            if page_links:
                with GRAPH_FILE.open("a", encoding="utf-8") as f:
                    for link in page_links:
                        f.write(json.dumps({"from": url, "to": link}) + "\n")

            visited.add(path)
            count += 1
            log.info("Crawled %d pages", count)

            if limit and count >= limit:
                log.info("Reached limit %d", limit)
                break

            _save_discovered(discovered)
            time.sleep(SLEEP_INTERVAL)

        except RequestException as e:
            log.error("Failed to fetch %s: %s", url, e)
            visited.add(path)
            continue

    log.info("Crawl complete. Crawled %d pages", count)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    crawl()
