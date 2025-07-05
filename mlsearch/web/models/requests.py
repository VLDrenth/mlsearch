"""
Request models for MLSearch web API.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """Request model for paper search."""
    query: str = Field(..., description="Natural language search query", min_length=1, max_length=500)
    max_results: int = Field(default=100, description="Maximum results per search", ge=1, le=30000)
    search_strategies: Optional[List[str]] = Field(
        default=None,
        description="Specific search strategies to use",
        example=["Recent Advances", "Foundational Literature"]
    )
    arxiv_categories: Optional[List[str]] = Field(
        default=None,
        description="Specific ArXiv categories to focus on",
        example=["cs.LG", "stat.ML", "cs.AI"]
    )
    date_range: Optional[dict] = Field(
        default=None,
        description="Date range filter",
        example={"start": "2020-01-01", "end": "2024-12-31"}
    )


class SearchStatusRequest(BaseModel):
    """Request model for checking search status."""
    search_id: str = Field(..., description="Unique search identifier")


class PaperDetailRequest(BaseModel):
    """Request model for getting paper details."""
    paper_id: str = Field(..., description="ArXiv paper ID")
    include_related: bool = Field(default=False, description="Include related papers")