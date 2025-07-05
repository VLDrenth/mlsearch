"""
Response models for MLSearch web API.
"""
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class SearchStatus(str, Enum):
    """Search status enumeration."""
    QUEUED = "queued"
    PLANNING = "planning"
    SEARCHING = "searching"
    ANALYZING = "analyzing"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentStatus(str, Enum):
    """Agent status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Paper(BaseModel):
    """Paper model for API responses."""
    id: str = Field(..., description="ArXiv paper ID")
    title: str = Field(..., description="Paper title")
    authors: List[str] = Field(..., description="List of authors")
    abstract: str = Field(..., description="Paper abstract")
    published: datetime = Field(..., description="Publication date")
    categories: List[str] = Field(..., description="ArXiv categories")
    url: str = Field(..., description="ArXiv URL")
    pdf_url: str = Field(..., description="PDF download URL")
    relevance_score: Optional[float] = Field(default=None, description="Relevance score (0-1)")
    found_by_agent: Optional[str] = Field(default=None, description="Which agent found this paper")
    search_strategy: Optional[str] = Field(default=None, description="Search strategy used")


class AgentProgress(BaseModel):
    """Agent progress model."""
    agent_id: str = Field(..., description="Agent identifier")
    agent_type: str = Field(..., description="Type of agent")
    status: AgentStatus = Field(..., description="Current status")
    search_strategy: str = Field(..., description="Assigned search strategy")
    focus_description: str = Field(..., description="Agent's focus area")
    arxiv_categories: List[str] = Field(..., description="Assigned ArXiv categories")
    search_terms: List[str] = Field(..., description="Search terms being used")
    papers_found: int = Field(default=0, description="Total papers found")
    relevant_papers: int = Field(default=0, description="Relevant papers found")
    searches_completed: int = Field(default=0, description="Number of searches completed")
    progress_percentage: float = Field(default=0.0, description="Progress percentage (0-100)")
    current_activity: Optional[str] = Field(default=None, description="Current activity description")


class SearchProgress(BaseModel):
    """Search progress model for real-time updates."""
    search_id: str = Field(..., description="Search identifier")
    status: SearchStatus = Field(..., description="Overall search status")
    query: str = Field(..., description="Original search query")
    started_at: datetime = Field(..., description="Search start time")
    updated_at: datetime = Field(..., description="Last update time")
    agents: List[AgentProgress] = Field(..., description="Agent progress details")
    total_papers_found: int = Field(default=0, description="Total papers found across all agents")
    total_relevant_papers: int = Field(default=0, description="Total relevant papers found")
    estimated_completion: Optional[datetime] = Field(default=None, description="Estimated completion time")
    progress_percentage: float = Field(default=0.0, description="Overall progress percentage")


class SearchResult(BaseModel):
    """Complete search result model."""
    search_id: str = Field(..., description="Search identifier")
    query: str = Field(..., description="Original search query")
    status: SearchStatus = Field(..., description="Search status")
    started_at: datetime = Field(..., description="Search start time")
    completed_at: Optional[datetime] = Field(default=None, description="Search completion time")
    duration_seconds: Optional[float] = Field(default=None, description="Total duration in seconds")
    papers: List[Paper] = Field(..., description="Found papers")
    total_papers: int = Field(..., description="Total number of papers")
    agents_used: List[str] = Field(..., description="Agents that participated")
    search_strategies: List[str] = Field(..., description="Search strategies employed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class SearchInitResponse(BaseModel):
    """Response when starting a new search."""
    search_id: str = Field(..., description="Unique search identifier")
    status: SearchStatus = Field(..., description="Initial status")
    message: str = Field(..., description="Status message")
    websocket_url: str = Field(..., description="WebSocket URL for real-time updates")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")