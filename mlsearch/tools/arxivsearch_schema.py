ARXIV_SEARCH_SCHEMA = {
    "name": "arxiv_search",
    "description": (
        "Search arXiv.org and return metadata for the most recent papers "
        "matching the query."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "arXiv query string, e.g. 'cat:cs.CL AND \"transformer\"'"
                ),
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 30000,
                "default": 100,
                "description": "Number of papers to return.",
            },
        },
        "required": ["query"],
    },
}
