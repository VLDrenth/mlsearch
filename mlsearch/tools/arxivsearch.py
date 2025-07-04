from __future__ import annotations

import html
import re
import time
import logging
from dataclasses import dataclass, asdict
from typing import List
import requests
from xml.etree import ElementTree as ET

_ARXIV_API = "https://export.arxiv.org/api/query"  # 90 req/min fair-use

@dataclass
class Paper:
    """Minimal metadata we expose to the LLM."""
    title: str
    authors: List[str]
    year: int
    pdf_url: str
    summary: str


def _parse_entry(entry_el: ET.Element) -> Paper:
    """Convert one <entry/> XML element to Paper."""
    ns = {"a": "http://www.w3.org/2005/Atom"}
    title = html.unescape(entry_el.findtext("a:title", "", ns)).strip()
    summary = html.unescape(entry_el.findtext("a:summary", "", ns)).strip()
    authors = [a.findtext("a:name", "", ns) for a in entry_el.findall("a:author", ns)]
    pdf_url = next(
        (l.attrib["href"] for l in entry_el.findall("a:link", ns)
         if l.attrib.get("type") == "application/pdf"),
        ""
    )

    # Year = first 4 digits in <published>
    m = re.match(r"(\d{4})-", entry_el.findtext("a:published", "", ns))
    year = int(m.group(1)) if m else 0
    return Paper(title=title, authors=authors, year=year, pdf_url=pdf_url, summary=summary)

def arxiv_search(query: str, *, limit: int = 100) -> List[dict]:
    """
    Search arXiv and return the newest *limit* papers.

    Parameters
    ----------
    query : str
        Search expression (arXiv API syntax).
    limit : int, default 100
        Maximum number of papers to return (1 ≤ limit ≤ 30000).

    Returns
    -------
    list[dict]
        JSON-serialisable list of paper metadata. Keys:
        ``title, authors, year, pdf_url, summary``.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"🔍 Searching arXiv for: '{query}' (limit: {limit})")
    
    # Use the query directly as provided by the model
    search_query = query
    
    limit = max(1, min(int(limit), 30000))
    params = {
        "search_query": search_query,
        "sortBy": "relevance",
        "sortOrder": "descending", 
        "start": 0,
        "max_results": limit,
        "ti": int(time.time()),  # poor-man cache-buster
    }

    logger.info(f"📡 Making API request to arXiv...")
    resp = requests.get(_ARXIV_API, params=params, timeout=10)
    resp.raise_for_status()

    feed = ET.fromstring(resp.text)
    papers = [_parse_entry(e) for e in feed.findall("{http://www.w3.org/2005/Atom}entry")]
    
    result = [asdict(p) for p in papers]
    
    logger.info(f"✅ Found {len(result)} papers")
    if result:
        logger.info(f"📄 Latest paper: '{result[0]['title']}' ({result[0]['year']})")
    
    return result

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    query = "active learning AND uncertainty sampling"
    results = arxiv_search(query, limit=20)

    for paper in results:
        print(f"{paper['title']} ({paper['year']}) by {', '.join(paper['authors'])}")
        print(f"PDF: {paper['pdf_url']}")
        print(f"Summary: {paper['summary'][:100]}...")
        print()