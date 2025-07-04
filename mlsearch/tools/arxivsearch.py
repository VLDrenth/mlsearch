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

def _optimize_arxiv_query(original_query: str) -> str:
    """Use LLM to optimize ArXiv search query for better relevance."""
    from mlsearch.core.llmclient import LLMClient
    
    query_optimizer = LLMClient(model_name="gpt-4o")
    
    optimization_prompt = f"""
You are an expert at constructing ArXiv search queries to find highly relevant academic papers.

Original Query: "{original_query}"

Your task: Optimize this query to find the most relevant papers while avoiding irrelevant results.

ArXiv Search Syntax:
- Use "AND" to require all terms: "bayesian" AND "active learning"  
- Use "OR" for alternatives: "transformer" OR "attention mechanism"
- Use quotes for exact phrases: "active learning"
- Use field searches: ti:"active learning" (title), abs:"bayesian methods" (abstract)
- Use parentheses for grouping: ("active learning" OR "selective sampling") AND "bayesian"

Guidelines:
1. For multi-concept queries, ensure ALL key concepts are required (use AND)
2. Use exact phrases in quotes for technical terms
3. Consider synonyms and alternative terms with OR
4. Avoid overly broad terms that could match irrelevant papers
5. Focus on the core concepts that make a paper relevant

Examples:
Bad: "machine learning" (too broad, returns everything)
Good: "active learning" AND ("uncertainty sampling" OR "query strategy")

Bad: "bayesian active learning methods" (might match papers with just "bayesian" or just "learning")  
Good: "active learning" AND ("bayesian" OR "probabilistic") AND ("uncertainty" OR "query selection")

For the query "{original_query}", construct an optimized ArXiv search query that will find papers specifically about this topic.

Return only the optimized query string, no explanation.
    """
    
    try:
        optimized = query_optimizer.generate(optimization_prompt).strip()
        # Remove any quotes around the entire response
        if optimized.startswith('"') and optimized.endswith('"'):
            optimized = optimized[1:-1]
        return optimized
    except Exception as e:
        logging.getLogger(__name__).warning(f"Query optimization failed: {e}, using original query")
        return original_query

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
    limit : int, default 10
        Maximum number of papers to return (1 ≤ limit ≤ 100).

    Returns
    -------
    list[dict]
        JSON-serialisable list of paper metadata. Keys:
        ``title, authors, year, pdf_url, summary``.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"🔍 Searching arXiv for: '{query}' (limit: {limit})")
    
    # Let LLM optimize the ArXiv query for better relevance
    optimized_query = _optimize_arxiv_query(query)
    
    # Add ML/AI category filter to the optimized query
    ml_categories = "cat:cs.AI OR cat:cs.LG OR cat:cs.CL OR cat:stat.ML"
    filtered_query = f"({optimized_query}) AND ({ml_categories})"
    logger.info(f"🎯 Optimized query: '{filtered_query}'")
    
    limit = max(1, min(int(limit), 100))
    params = {
        "search_query": filtered_query,
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
    # Example usage
    logging.basicConfig(level=logging.INFO)
    query = "active learning AND uncertainty sampling"
    results = arxiv_search(query, limit=20)

    for paper in results:
        print(f"{paper['title']} ({paper['year']}) by {', '.join(paper['authors'])}")
        print(f"PDF: {paper['pdf_url']}")
        print(f"Summary: {paper['summary'][:100]}...")
        print()