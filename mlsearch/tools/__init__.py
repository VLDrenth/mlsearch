"""
Tools package initialization with centralized registry setup.
"""

from .arxivsearch import arxiv_search
from .biorxivsearch import biorxiv_search
from .arxivsearch_schema import ARXIV_SEARCH_SCHEMA, ARXIV_SEARCH_FUNCTION
from .biorxivsearch_schema import BIORXIV_SEARCH_SCHEMA, BIORXIV_SEARCH_FUNCTION
try:
    from ..core.tool_registry import get_tool_registry, register_from_schema
except ImportError:
    # Handle when running as script or from different context
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.tool_registry import get_tool_registry, register_from_schema

def initialize_tools():
    """Initialize and register all available tools with the global registry."""
    registry = get_tool_registry()
    
    # Clear any existing tools
    registry.clear()
    
    # Register ArXiv search tool
    register_from_schema(ARXIV_SEARCH_SCHEMA, arxiv_search)
    
    # Register BioRxiv search tool (following CLAUDE.md guidance to focus on ArXiv)
    # We'll keep this available but the orchestrator should prefer ArXiv
    register_from_schema(BIORXIV_SEARCH_SCHEMA, biorxiv_search)

def get_default_tools():
    """Get default tools for backward compatibility."""
    return {
        "arxiv_search": arxiv_search,
        "biorxiv_search": biorxiv_search,
    }

# Legacy registry for backward compatibility
TOOL_REGISTRY = {
    "arxiv_search": arxiv_search,
}

# Initialize tools when module is imported
initialize_tools()

__all__ = [
    "arxiv_search", 
    "biorxiv_search",
    "ARXIV_SEARCH_SCHEMA",
    "BIORXIV_SEARCH_SCHEMA", 
    "ARXIV_SEARCH_FUNCTION",
    "BIORXIV_SEARCH_FUNCTION",
    "initialize_tools",
    "get_default_tools",
    "TOOL_REGISTRY"
]