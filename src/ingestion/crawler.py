from __future__ import annotations

import json
import logging
import random
import re
import time
from pathlib import Path
from typing import NamedTuple
from urllib.parse import unquote, urljoin, urlparse, urldefrag

import requests
from requests.exceptions import RequestException

from src.db.connection import get_crawl_conn

log = logging.getLogger(__name__)

HEADERS = {"User-Agent": "FanOfBrandon"}

COPPERMIND_BASE = "https://coppermind.net"
COPPERMIND_WIKI = f"{COPPERMIND_BASE}/wiki"
DATA_DIR = Path(__file__).parent.parent.parent / "data"
CRAWL_DATA_DIR = DATA_DIR / "crawl"
CRAWL_DATA_DIR.mkdir(parents=True, exist_ok=True)

MD_DIR = CRAWL_DATA_DIR / "markdown"
MD_DIR.mkdir(exist_ok=True)

PAGES_FILE = CRAWL_DATA_DIR / "pages.jsonl"
GRAPH_FILE = CRAWL_DATA_DIR / "graph_edges.jsonl"
ROBOTS_FILE = CRAWL_DATA_DIR / "robots.txt"

DISALLOWED_PATTERNS = [
    r"/w/",
    r"/mw/",
    r"\?diff",
    r"\?oldid=",
]

IGNORED_NAMESPACE_PREFIXES = (
    "Special:",
    "Talk:",
    "User:",
    "User talk:",
)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico", ".bmp")

MIN_SLEEP = 0.1
MAX_SLEEP = 0.5


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
    path_no_query = path.split("?", 1)[0]
    if path_no_query.lower().endswith(IMAGE_EXTENSIONS):
        return False
    normalized_path = (
        unquote(path_no_query.removeprefix("/wiki/")).replace(" ", "_").casefold()
    )
    for prefix in IGNORED_NAMESPACE_PREFIXES:
        if normalized_path.startswith(prefix.replace(" ", "_").casefold()):
            return False
    for pattern in DISALLOWED_PATTERNS:
        if re.search(pattern, path):
            return False
    return True


def _is_disallowed_query(query: str) -> bool:
    return (
        query.startswith("action")
        or query.startswith("oldver")
        or query.startswith("oldid")
        or query.startswith("diff")
    )


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
    if _is_disallowed_query(parsed.query):
        return "", None
    if not _is_allowed(parsed.path):
        return "", None
    return cleaned, frag


def _fetch_page(url: str) -> tuple[str, str, list[str]]:
    resp = requests.get(url, headers=HEADERS, timeout=30)
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


def _write_markdown(url: str, title: str, text: str, links: list[str]) -> None:
    slug = urlparse(url).path.split("/")[-1] or "index"
    slug = slug.replace("%", "-").replace(":", "-").replace("/", "_")

    md_content = f"""# {title}

{text}

---
Source: {url}

Links: {len(links)} outgoing
"""
    (MD_DIR / f"{slug}.md").write_text(md_content)


def _get_discovered_urls(limit: int = 100) -> list[str]:
    with get_crawl_conn() as conn:
        cur = conn.execute(
            "SELECT url FROM crawl_state WHERE status = 'DISCOVERED' LIMIT ?",
            (limit,),
        )
        return [row[0] for row in cur.fetchall()]


def _mark_visited(url: str) -> None:
    with get_crawl_conn() as conn:
        conn.execute(
            "UPDATE crawl_state SET status = 'VISITED', visited_at = CURRENT_TIMESTAMP WHERE url = ?",
            (url,),
        )
        conn.commit()


def _add_discovered_urls(urls: list[str]) -> None:
    with get_crawl_conn() as conn:
        for url in urls:
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO crawl_state (url, status) VALUES (?, 'DISCOVERED')",
                    (url,),
                )
            except Exception:
                pass
        conn.commit()


def _count_visited() -> int:
    with get_crawl_conn() as conn:
        cur = conn.execute("SELECT COUNT(*) FROM crawl_state WHERE status = 'VISITED'")
        return cur.fetchone()[0]


def crawl(start_path: str = "/wiki/Main_Page", limit: int | None = None) -> None:
    robots_disallowed = _load_robots()
    if _is_allowed(start_path):
        _add_discovered_urls([start_path])
    count = 0

    log.info("Starting crawl from %s", start_path)

    while True:
        should_sleep = True
        url = ""
        try:
            with get_crawl_conn() as conn:
                cur = conn.execute(
                    "SELECT url FROM crawl_state WHERE status = 'DISCOVERED' LIMIT 1"
                )
                row = cur.fetchone()
                if not row:
                    should_sleep = False
                    break
                url = row[0]
                path = url.replace(COPPERMIND_BASE, "")

            parsed = urlparse(path)
            if (
                parsed.path in robots_disallowed
                or _is_disallowed_query(parsed.query)
                or not _is_allowed(parsed.path)
            ):
                _mark_visited(url)
                continue

            log.info("Crawling %s", url)

            title, text, links = _fetch_page(url)
            page_links: list[str] = []
            new_urls: list[str] = []
            for link in links:
                normalized, frag = _normalize_url(urljoin(COPPERMIND_BASE, link))
                if normalized:
                    new_urls.append(normalized)
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

            _write_markdown(url, title, text, page_links)

            if page_links:
                with GRAPH_FILE.open("a", encoding="utf-8") as f:
                    for link in page_links:
                        f.write(json.dumps({"from": url, "to": link}) + "\n")

            _add_discovered_urls(new_urls)
            _mark_visited(url)
            count += 1
            visited_count = _count_visited()
            log.info("Crawled %d pages (visited: %d)", count, visited_count)

            if limit and count >= limit:
                log.info("Reached limit %d", limit)
                should_sleep = False
                break

        except RequestException as e:
            log.error("Failed to fetch %s: %s", url, e)
            _mark_visited(url)
            continue
        finally:
            if should_sleep:
                time.sleep(random.uniform(MIN_SLEEP, MAX_SLEEP))

    log.info("Crawl complete. Crawled %d pages", count)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    crawl()
