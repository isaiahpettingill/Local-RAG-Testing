from __future__ import annotations


def test_normalize_url_skips_oldver_and_diff_queries():
    from src.ingestion.crawler import _normalize_url

    assert _normalize_url("https://coppermind.net/wiki/Test?action=edit")[0] == ""
    assert _normalize_url("https://coppermind.net/wiki/Test?oldver=1")[0] == ""
    assert _normalize_url("https://coppermind.net/wiki/Test?oldid=123")[0] == ""
    assert _normalize_url("https://coppermind.net/wiki/Test?diff=123")[0] == ""
    assert _normalize_url("https://coppermind.net/wiki/Test?oldversion=1")[0] == ""


def test_is_disallowed_query_matches_prefixes():
    from src.ingestion.crawler import _is_disallowed_query

    assert _is_disallowed_query("action=edit")
    assert _is_disallowed_query("oldver=1")
    assert _is_disallowed_query("oldid=123")
    assert _is_disallowed_query("diff=123")
    assert _is_disallowed_query("oldversion=1")


def test_is_allowed_keeps_categories_but_skips_non_content_namespaces():
    from src.ingestion.crawler import _is_allowed

    assert _is_allowed("/wiki/Category:Books")
    assert not _is_allowed("/wiki/User:Example")
    assert not _is_allowed("/wiki/Special:RecentChanges")
    assert not _is_allowed("/wiki/Talk:Main_Page")


def test_crawl_sleeps_after_skipped_urls(monkeypatch):
    import src.ingestion.crawler as crawler

    class DummyCursor:
        def __init__(self, row):
            self._row = row

        def fetchone(self):
            return self._row

    class DummyConn:
        def __init__(self):
            self._queries = []

        def execute(self, query, params=None):
            self._queries.append((query, params))
            if "SELECT url FROM crawl_state" in query:
                if len(self._queries) == 1:
                    return DummyCursor(("https://coppermind.net/wiki/Test?oldid=123",))
                return DummyCursor(None)
            return DummyCursor(None)

        def commit(self):
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    conn = DummyConn()
    sleep_calls: list[float] = []

    monkeypatch.setattr(crawler, "get_crawl_conn", lambda: conn)
    monkeypatch.setattr(crawler, "_load_robots", lambda: set())
    monkeypatch.setattr(crawler, "_add_discovered_urls", lambda urls: None)
    monkeypatch.setattr(crawler, "_mark_visited", lambda url: None)
    monkeypatch.setattr(
        crawler.time, "sleep", lambda duration: sleep_calls.append(duration)
    )
    monkeypatch.setattr(crawler.random, "uniform", lambda a, b: a)

    crawler.crawl(limit=1)

    assert sleep_calls == [crawler.MIN_SLEEP]


def test_migrate_crawl_state_copies_rows(monkeypatch):
    import src.db.schema as schema
    import src.db.connection as connection

    class DummyCursor:
        def __init__(self, rows=None):
            self._rows = rows or []

        def fetchone(self):
            return self._rows[0] if self._rows else None

        def fetchall(self):
            return self._rows

    class DummyConn:
        def __init__(self, rows=None, has_table=True):
            self.rows = rows or []
            self.has_table = has_table
            self.executed = []

        def execute(self, query, params=None):
            self.executed.append((query, params))
            if "sqlite_master" in query:
                return DummyCursor([(1,)] if self.has_table else [])
            if (
                "SELECT url, status, discovered_at, visited_at FROM crawl_state"
                in query
            ):
                return DummyCursor(self.rows)
            return DummyCursor([])

        def commit(self):
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    staging = DummyConn(
        rows=[
            {
                "url": "https://coppermind.net/wiki/Test",
                "status": "VISITED",
                "discovered_at": "2026-04-09 00:00:00",
                "visited_at": "2026-04-09 00:01:00",
            }
        ]
    )
    crawl = DummyConn()

    monkeypatch.setattr(connection, "get_staging_conn", lambda: staging)
    monkeypatch.setattr(schema, "get_crawl_conn", lambda: crawl)

    schema.migrate_crawl_state()

    assert any(
        "INSERT OR IGNORE INTO crawl_state" in query for query, _ in crawl.executed
    )
