"""Crawl job descriptions from Chinese recruitment websites.

Owner: ztt

Anti-scraping note
------------------
Major platforms (Boss直聘, 拉勾) use dynamic rendering and require login
cookies that change frequently.  This module provides:

  - A Selenium-based implementation that renders JavaScript and can reuse a
    logged-in browser profile, which is the most reliable approach.
  - A CSV import fallback for datasets downloaded manually (e.g. from Kaggle).

Recommended workflow
--------------------
1. Log in to the target site in Chrome with the profile at
   ``--chrome-profile``.  This preserves session cookies between runs.
2. Run with ``--source lagou --max 50`` and adjust ``--request-interval``
   if you hit rate limits.
3. If scraping is blocked, export a CSV from the website's search results
   page (or use a Kaggle dataset) and import with ``--csv-path``.

Output
------
Each record is saved as ``data/raw/jd/<source>/<doc_id>.json``.

Usage
-----
    python -m optimize.data_collection.crawl_jd --source lagou --max 30
    python -m optimize.data_collection.crawl_jd --csv-path /path/to/jobs.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterator, Optional

from optimize.config import cfg
from optimize.data_collection.catalog import CatalogEntry, catalog
from optimize.utils.file_utils import ensure_dir, save_raw_doc
from optimize.utils.hash_utils import dict_sha256
from optimize.utils.logging_utils import get_pipeline_logger

logger = get_pipeline_logger("data_collection.crawl_jd")

_SNAPSHOT_DATE = date.today().isoformat()

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
]


@dataclass
class RawJD:
    """Platform-agnostic representation of a single job posting."""

    job_title:        str
    company_name:     str
    location:         str
    salary_range:     str
    requirements:     str   # plain text of job requirements
    responsibilities: str
    preferred:        str   # nice-to-have skills
    full_text:        str   # complete JD body
    source_url:       str
    source_name:      str


class BaseJDCrawler(ABC):
    """Abstract base class for all JD crawlers.

    Subclasses implement ``_fetch_keyword`` for a specific platform.
    Common concerns (rate limiting, validation, serialisation) live here.
    """

    SOURCE_NAME: str = "unknown"

    def __init__(self, request_interval: float = cfg.collection.jd_request_interval) -> None:
        self._interval = request_interval
        self._snapshot_date = _SNAPSHOT_DATE

    def crawl(
        self,
        keywords: list[str],
        max_per_keyword: int = cfg.collection.jd_max_per_keyword,
        output_dir: Optional[Path] = None,
    ) -> int:
        """Crawl JDs for each keyword and save to the raw zone.

        Returns the total number of records saved.
        """
        out_dir = output_dir or (cfg.paths.raw_jd / self.SOURCE_NAME)
        ensure_dir(out_dir)

        total = 0
        for keyword in keywords:
            logger.info("[%s] keyword=%r limit=%d", self.SOURCE_NAME, keyword, max_per_keyword)
            try:
                jds = self._fetch_keyword(keyword, max_per_keyword)
            except Exception as exc:
                logger.warning("[%s] keyword=%r failed: %s", self.SOURCE_NAME, keyword, exc)
                continue

            saved = 0
            for jd in jds:
                if not self._validate(jd):
                    continue
                doc = self._to_raw_doc(jd, total + saved)
                save_raw_doc(out_dir, doc["doc_id"], doc)
                saved += 1

            logger.info("[%s] keyword=%r saved=%d", self.SOURCE_NAME, keyword, saved)
            total += saved
            time.sleep(self._interval)

        self._update_catalog(out_dir, total)
        return total

    @abstractmethod
    def _fetch_keyword(self, keyword: str, limit: int) -> list[RawJD]:
        """Fetch up to *limit* JDs matching *keyword*.  Implemented by subclasses."""
        ...

    @staticmethod
    def _validate(jd: RawJD) -> bool:
        if not jd.job_title.strip():
            return False
        if not jd.requirements.strip() and not jd.full_text.strip():
            return False
        return True

    def _to_raw_doc(self, jd: RawJD, seq: int) -> dict[str, Any]:
        slug = self.SOURCE_NAME.replace("_", "")
        doc_id = f"jd_{slug}_{seq:06d}"
        content: dict[str, Any] = {
            "job_title":        jd.job_title,
            "company_name":     jd.company_name,
            "location":         jd.location,
            "salary_range":     jd.salary_range,
            "requirements":     jd.requirements,
            "responsibilities": jd.responsibilities,
            "preferred":        jd.preferred,
            "full_text":        jd.full_text,
        }
        payload: dict[str, Any] = {
            "doc_id":        doc_id,
            "source_name":   self.SOURCE_NAME,
            "source_url":    jd.source_url,
            "snapshot_time": f"{self._snapshot_date}T00:00:00Z",
            "language":      "zh",
            "license_note":  "Public web scraping; verify platform ToS before redistribution.",
            "doc_type":      "jd",
            "content":       content,
        }
        payload["sha256"] = dict_sha256(content)
        return payload

    def _update_catalog(self, out_dir: Path, count: int) -> None:
        catalog.upsert(CatalogEntry(
            source_id     = f"jd_{self.SOURCE_NAME}_v1",
            source_name   = f"JD Crawler — {self.SOURCE_NAME}",
            source_type   = "jd_crawler",
            source_url    = self._base_url,
            license_note  = "Public web scraping",
            snapshot_date = self._snapshot_date,
            record_count  = count,
            local_path    = str(out_dir.relative_to(cfg.paths.raw_root.parent)),
            description   = f"CS job postings scraped from {self.SOURCE_NAME}",
            tags          = ["jd", "recruitment", "zh"],
        ))

    @property
    def _base_url(self) -> str:
        return ""


class LagouSeleniumCrawler(BaseJDCrawler):
    """拉勾网 JD crawler using Selenium (headless Chrome).

    Selenium is required because 拉勾 renders job lists client-side and
    blocks plain HTTP requests without valid session cookies.

    Prerequisites
    -------------
    1. ``pip install selenium webdriver-manager``
    2. Chrome must be installed on the machine.
    3. Log in to https://www.lagou.com at least once so the profile stores
       valid session cookies (use ``--chrome-profile``).
    """

    SOURCE_NAME = "lagou"
    _SEARCH_URL = "https://www.lagou.com/wn/jobs?kd={keyword}&pn={page}"

    def __init__(
        self,
        chrome_profile: Optional[str] = None,
        headless: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._chrome_profile = chrome_profile
        self._headless = headless
        self._driver: Any = None

    def _get_driver(self) -> Any:
        if self._driver is not None:
            return self._driver
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from webdriver_manager.chrome import ChromeDriverManager
            from selenium.webdriver.chrome.service import Service
        except ImportError:
            logger.error("selenium / webdriver-manager not installed.  Run: pip install selenium webdriver-manager")
            raise

        opts = Options()
        if self._headless:
            opts.add_argument("--headless=new")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument(f"user-agent={random.choice(_USER_AGENTS)}")
        if self._chrome_profile:
            opts.add_argument(f"--user-data-dir={self._chrome_profile}")

        # Anti-detection: hide Selenium / ChromeDriver fingerprints.
        # Without these, sites detect navigator.webdriver=true and block the session.
        opts.add_experimental_option("excludeSwitches", ["enable-automation"])
        opts.add_experimental_option("useAutomationExtension", False)
        opts.add_argument("--disable-blink-features=AutomationControlled")

        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=opts)

        # Patch navigator.webdriver via CDP so it reads as undefined at runtime
        driver.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {"source": "Object.defineProperty(navigator,'webdriver',{get:()=>undefined})"},
        )
        self._driver = driver
        return self._driver

    def _fetch_keyword(self, keyword: str, limit: int) -> list[RawJD]:
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        driver = self._get_driver()
        results: list[RawJD] = []
        page = 1

        while len(results) < limit:
            url = self._SEARCH_URL.format(keyword=keyword, page=page)
            driver.get(url)
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, ".job-name, .position-name"))
                )
            except Exception:
                logger.debug("Timeout waiting for job list on page %d", page)
                break

            cards = driver.find_elements(By.CSS_SELECTOR, "li.item")
            if not cards:
                break

            for card in cards:
                jd = self._parse_card(card, keyword)
                if jd:
                    results.append(jd)
                if len(results) >= limit:
                    break

            page += 1
            time.sleep(self._interval)

        if self._driver:
            self._driver.quit()
            self._driver = None

        return results[:limit]

    @staticmethod
    def _parse_card(card: Any, keyword: str) -> Optional[RawJD]:
        """Extract a RawJD from a single search-result card element."""
        try:
            title = card.find_element("css selector", ".job-name").text.strip()
            company = card.find_element("css selector", ".company-name").text.strip()
            location = card.find_element("css selector", ".work-addr").text.strip()
            salary = card.find_element("css selector", ".money").text.strip()
            tags_els = card.find_elements("css selector", ".li-tag")
            full_text = " ".join(el.text.strip() for el in tags_els)
            url = card.find_element("css selector", "a").get_attribute("href") or ""
            return RawJD(
                job_title=title, company_name=company, location=location,
                salary_range=salary, requirements=full_text,
                responsibilities="", preferred="", full_text=full_text,
                source_url=url, source_name="lagou",
            )
        except Exception:
            return None

    @property
    def _base_url(self) -> str:
        return "https://www.lagou.com"


class CsvImportCrawler(BaseJDCrawler):
    """Import JDs from a locally downloaded CSV file.

    This is the most reliable fallback when live scraping is blocked.
    The CSV must contain at least ``job_title`` and one of
    ``requirements`` / ``description`` / ``full_text`` columns.
    Other columns are stored verbatim in ``content``.

    Compatible CSV formats
    ----------------------
    - Kaggle "Job Descriptions" datasets
    - Manually exported search results
    - Any tab- or comma-separated file with a header row
    """

    SOURCE_NAME = "csv_import"

    def __init__(self, csv_path: Path, source_label: str = "csv_import", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._csv_path = csv_path
        self.SOURCE_NAME = source_label  # type: ignore[misc]

    def _fetch_keyword(self, keyword: str, limit: int) -> list[RawJD]:
        """Yield rows whose title or text contains *keyword*."""
        results: list[RawJD] = []
        kw_lower = keyword.lower()
        for row in self._iter_csv():
            text_fields = " ".join(str(v) for v in row.values()).lower()
            if kw_lower not in text_fields:
                continue
            jd = self._row_to_jd(row)
            if jd:
                results.append(jd)
            if len(results) >= limit:
                break
        return results

    def _iter_csv(self) -> Iterator[dict[str, str]]:
        suffix = self._csv_path.suffix.lower()
        if suffix in (".xlsx", ".xls"):
            yield from self._iter_xlsx()
        else:
            delimiter = self._detect_delimiter()
            with self._csv_path.open(encoding="utf-8-sig") as f:
                yield from csv.DictReader(f, delimiter=delimiter)

    def _iter_xlsx(self) -> Iterator[dict[str, str]]:
        try:
            import openpyxl  # type: ignore[import]
        except ImportError:
            logger.error("openpyxl not installed.  Run: pip install openpyxl")
            raise
        wb = openpyxl.load_workbook(self._csv_path, read_only=True)
        ws = wb.active
        headers: list[str] = []
        for i, row in enumerate(ws.iter_rows(values_only=True)):
            if i == 0:
                headers = [str(c) if c is not None else f"col_{j}" for j, c in enumerate(row)]
                continue
            yield {h: (str(v) if v is not None else "") for h, v in zip(headers, row)}
        wb.close()

    def _detect_delimiter(self) -> str:
        with self._csv_path.open(encoding="utf-8-sig") as f:
            sample = f.read(4096)
        return "\t" if sample.count("\t") > sample.count(",") else ","

    @staticmethod
    def _row_to_jd(row: dict[str, str]) -> Optional[RawJD]:
        def get(*keys: str) -> str:
            for k in keys:
                v = row.get(k, "").strip()
                if v:
                    return v
            return ""

        # Support 51job / zhilian column naming as well as generic names
        title = get(
            "job_name", "job_title", "title", "position",
            "职位名称", "岗位名称", "职位", "岗位",
        )
        if not title:
            return None

        full_text = get(
            "require_content", "full_text", "description", "jd_text",
            "职位描述", "岗位描述",
        )
        requirements = get(
            "requirements", "job_requirements", "require_content",
            "任职要求",
        ) or full_text

        return RawJD(
            job_title        = title,
            company_name     = get("company_name", "company", "公司名称"),
            location         = get("city", "location", "work_place", "工作地点", "工作城市"),
            salary_range     = get("salary", "salary_range", "薪资", "薪酬"),
            requirements     = requirements,
            responsibilities = get("responsibilities", "job_description", "工作职责"),
            preferred        = get("walfare", "tag", "preferred", "welfare", "福利", "加分项"),
            full_text        = full_text,
            source_url       = get("url", "source_url", "link"),
            source_name      = "csv_import",
        )

    @property
    def _base_url(self) -> str:
        return self._csv_path.resolve().as_uri()


_CRAWLER_REGISTRY: dict[str, type[BaseJDCrawler]] = {
    "lagou":      LagouSeleniumCrawler,
    "csv_import": CsvImportCrawler,
}


def get_crawler(source: str, **kwargs: Any) -> BaseJDCrawler:
    """Instantiate a crawler by source name."""
    cls = _CRAWLER_REGISTRY.get(source)
    if cls is None:
        raise ValueError(f"Unknown source '{source}'. Available: {', '.join(_CRAWLER_REGISTRY)}")
    return cls(**kwargs)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="lagou", choices=list(_CRAWLER_REGISTRY),
                   help="Crawl source (default: lagou)")
    p.add_argument("--max", type=int, default=cfg.collection.jd_max_per_keyword,
                   help="Max JDs per keyword")
    p.add_argument("--keywords", nargs="*", default=None,
                   help="Custom keyword list (default: config value)")
    p.add_argument("--csv-path", type=Path, default=None,
                   help="Local CSV file path (for csv_import source)")
    p.add_argument("--chrome-profile", default=None,
                   help="Chrome user-data-dir for reusing login session")
    p.add_argument("--no-headless", action="store_true",
                   help="Show browser window (useful for debugging login issues)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    keywords = args.keywords or list(cfg.collection.jd_target_keywords)

    kwargs: dict[str, Any] = {}
    if args.source == "lagou":
        kwargs["chrome_profile"] = args.chrome_profile
        kwargs["headless"] = not args.no_headless
    elif args.source == "csv_import":
        if not args.csv_path:
            raise SystemExit("--csv-path is required for csv_import source")
        kwargs["csv_path"] = args.csv_path

    crawler = get_crawler(args.source, **kwargs)
    total = crawler.crawl(keywords=keywords, max_per_keyword=args.max)
    print(f"Done: {total} JDs saved (source: {args.source}).")
