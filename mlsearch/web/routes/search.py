"""
Search API endpoints for MLSearch web interface.
"""
import asyncio
import logging
import uuid
from datetime import datetime
from typing import Dict, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.websockets import WebSocket

from ..models.requests import SearchRequest, SearchStatusRequest
from ..models.responses import (
    SearchInitResponse, SearchResult, SearchProgress, SearchStatus,
    AgentProgress, AgentStatus, Paper, ErrorResponse
)

# Import the orchestrator and tools
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.simple_orchestrator import SimpleOrchestrator
from tools.arxivsearch import arxiv_search

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory storage for search results (in production, use Redis or database)
active_searches: Dict[str, SearchProgress] = {}
completed_searches: Dict[str, SearchResult] = {}

def create_search_id() -> str:
    """Generate a unique search ID."""
    return str(uuid.uuid4())

async def run_search_orchestrator(search_id: str, request: SearchRequest):
    """Run the search orchestrator in the background."""
    try:
        # Update search status
        if search_id in active_searches:
            active_searches[search_id].status = SearchStatus.PLANNING
            active_searches[search_id].updated_at = datetime.now()
        
        # Create orchestrator with tools
        def arxiv_search_with_limit(query: str, limit: int = None) -> list:
            if limit is None:
                limit = request.max_results
            return arxiv_search(query, limit=min(limit, request.max_results))
        
        tools = {"arxiv_search": arxiv_search_with_limit}
        orchestrator = SimpleOrchestrator(tools)
        
        # Update status to searching
        if search_id in active_searches:
            active_searches[search_id].status = SearchStatus.SEARCHING
            active_searches[search_id].updated_at = datetime.now()
        
        # Run the orchestrator
        result = await orchestrator.run(request.query)
        
        # Update status to analyzing
        if search_id in active_searches:
            active_searches[search_id].status = SearchStatus.ANALYZING
            active_searches[search_id].updated_at = datetime.now()
        
        # Extract ranked papers with relevance reasoning from orchestrator
        papers = []
        logger.info("Extracting ranked papers from orchestrator...")
        
        try:
            # Use the orchestrator's method to get ranked papers with reasoning
            ranked_papers_data = orchestrator.get_ranked_papers_with_reasoning()
            logger.info(f"Found {len(ranked_papers_data)} ranked papers from orchestrator")
            
            if not ranked_papers_data:
                # Fallback to all papers if ranking didn't work
                logger.warning("No ranked papers found, falling back to all papers")
                all_papers_data = orchestrator.get_all_papers()
                logger.info(f"Fallback: Found {len(all_papers_data)} papers total")
                
                for paper_data in all_papers_data[:10]:  # Limit to top 10 as fallback
                    try:
                        paper = Paper(
                            id=paper_data.get('id', 'unknown'),
                            title=paper_data.get('title', 'Unknown Title'),
                            authors=paper_data.get('authors', []) if isinstance(paper_data.get('authors', []), list) else [paper_data.get('authors', '')],
                            abstract=paper_data.get('summary', paper_data.get('abstract', '')),
                            published=datetime(paper_data.get('year', 2024), 1, 1),
                            categories=paper_data.get('categories', []),
                            url=paper_data.get('url', ''),
                            pdf_url=paper_data.get('pdf_url', ''),
                            found_by_agent=paper_data.get('found_by_agent', 'unknown'),
                            search_strategy=paper_data.get('search_strategy', 'Unknown'),
                            relevance_score=0.5  # Default relevance
                        )
                        papers.append(paper)
                    except Exception as e:
                        logger.error(f"Error converting fallback paper: {e}")
            else:
                # Use the ranked papers with relevance reasoning
                for paper_data in ranked_papers_data:
                    try:
                        # Convert ranked paper data to our Paper model
                        authors = paper_data.get('authors', '')
                        if isinstance(authors, str):
                            authors = [authors] if authors else []
                        
                        # Generate ArXiv URL from title (this is a simplification)
                        title_slug = paper_data.get('title', '').lower().replace(' ', '-').replace(',', '').replace(':', '')[:50]
                        paper_url = paper_data.get('pdf_url', '') or f"https://arxiv.org/search/?query={title_slug}&searchtype=title"
                        pdf_url = paper_data.get('pdf_url', '') or f"https://arxiv.org/pdf/{title_slug}.pdf"
                        
                        paper = Paper(
                            id=f"ranked_{paper_data.get('rank', 'unknown')}",
                            title=paper_data.get('title', 'Unknown Title'),
                            authors=authors,
                            abstract=paper_data.get('summary', '') + (f"\n\n**Why relevant:** {paper_data.get('relevance_reasoning', '')}" if paper_data.get('relevance_reasoning') else ''),
                            published=datetime(paper_data.get('year', 2024), 1, 1),
                            categories=[paper_data.get('agent_focus', 'Unknown')],
                            url=paper_url,
                            pdf_url=pdf_url,
                            found_by_agent=paper_data.get('agent_type', 'unknown'),
                            search_strategy=paper_data.get('agent_focus', 'Unknown'),
                            relevance_score=paper_data.get('relevance_score', 0.5)
                        )
                        papers.append(paper)
                        
                    except Exception as e:
                        logger.error(f"Error converting ranked paper data: {e}")
                        logger.error(f"Ranked paper data: {paper_data}")
            
        except Exception as e:
            logger.error(f"Error extracting ranked papers from orchestrator: {e}")
        
        logger.info(f"Successfully converted {len(papers)} relevant papers for web interface")
        
        # Create completed search result
        search_result = SearchResult(
            search_id=search_id,
            query=request.query,
            status=SearchStatus.COMPLETED,
            started_at=active_searches[search_id].started_at,
            completed_at=datetime.now(),
            papers=papers,
            total_papers=len(papers),
            agents_used=[agent.get('id', 'unknown') for agent in getattr(orchestrator, 'agent_outputs', [])],
            search_strategies=list(set([paper.search_strategy for paper in papers if hasattr(paper, 'search_strategy')])),
            metadata={"raw_result": str(result)[:1000]}  # Truncated raw result
        )
        
        # Calculate duration
        if search_result.completed_at and search_result.started_at:
            duration = search_result.completed_at - search_result.started_at
            search_result.duration_seconds = duration.total_seconds()
        
        # Move from active to completed
        completed_searches[search_id] = search_result
        if search_id in active_searches:
            del active_searches[search_id]
        
        logger.info(f"Search {search_id} completed successfully with {len(papers)} papers")
        
    except Exception as e:
        logger.error(f"Search {search_id} failed: {e}")
        # Update search status to failed
        if search_id in active_searches:
            active_searches[search_id].status = SearchStatus.FAILED
            active_searches[search_id].updated_at = datetime.now()

@router.post("/", response_model=SearchInitResponse)
async def start_search(request: SearchRequest, background_tasks: BackgroundTasks):
    """Start a new paper search."""
    try:
        search_id = create_search_id()
        
        # Create initial search progress
        search_progress = SearchProgress(
            search_id=search_id,
            status=SearchStatus.QUEUED,
            query=request.query,
            started_at=datetime.now(),
            updated_at=datetime.now(),
            agents=[],  # Will be populated when orchestrator starts
            total_papers_found=0,
            total_relevant_papers=0,
            progress_percentage=0.0
        )
        
        active_searches[search_id] = search_progress
        
        # Start the search in the background
        background_tasks.add_task(run_search_orchestrator, search_id, request)
        
        return SearchInitResponse(
            search_id=search_id,
            status=SearchStatus.QUEUED,
            message="Search initiated successfully",
            websocket_url=f"/ws/{search_id}"
        )
        
    except Exception as e:
        logger.error(f"Failed to start search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{search_id}/status", response_model=SearchProgress)
async def get_search_status(search_id: str):
    """Get the current status of a search."""
    # Check active searches first
    if search_id in active_searches:
        return active_searches[search_id]
    
    # Check completed searches
    if search_id in completed_searches:
        completed = completed_searches[search_id]
        return SearchProgress(
            search_id=completed.search_id,
            status=completed.status,
            query=completed.query,
            started_at=completed.started_at,
            updated_at=completed.completed_at or completed.started_at,
            agents=[],  # Not tracking agent details in completed searches
            total_papers_found=completed.total_papers,
            total_relevant_papers=completed.total_papers,
            progress_percentage=100.0
        )
    
    raise HTTPException(status_code=404, detail="Search not found")

@router.get("/{search_id}/results", response_model=SearchResult)
async def get_search_results(search_id: str):
    """Get the results of a completed search."""
    if search_id in completed_searches:
        return completed_searches[search_id]
    
    if search_id in active_searches:
        raise HTTPException(status_code=202, detail="Search still in progress")
    
    raise HTTPException(status_code=404, detail="Search not found")

@router.delete("/{search_id}")
async def cancel_search(search_id: str):
    """Cancel an active search."""
    if search_id in active_searches:
        del active_searches[search_id]
        return {"message": "Search cancelled successfully"}
    
    raise HTTPException(status_code=404, detail="Active search not found")

@router.get("/")
async def list_searches():
    """List all searches (active and completed)."""
    return {
        "active_searches": list(active_searches.keys()),
        "completed_searches": list(completed_searches.keys()),
        "total_active": len(active_searches),
        "total_completed": len(completed_searches)
    }