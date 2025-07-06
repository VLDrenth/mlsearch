#!/usr/bin/env python3
"""
ML Search CLI - Search for ML papers on ArXiv using natural language queries.
"""
import asyncio
import sys
import os
import logging
import argparse
from pathlib import Path

# Add the mlsearch package to the path
sys.path.insert(0, str(Path(__file__).parent / "mlsearch"))

from agents.simple_orchestrator import SimpleOrchestrator
from tools import initialize_tools, get_default_tools
from tools.arxivsearch import arxiv_search
from core.tool_registry import get_tool_registry

async def main():
    """Main CLI function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='ML Search CLI - Search for ML papers on ArXiv using natural language queries.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py "time series MIDAS models"
  python main.py --max-results 50 "active learning methods"
  python main.py -m 500 "transformer architectures for NLP"
  python main.py -m 5000 "deep learning optimization techniques"
        """
    )
    parser.add_argument('query', nargs='*', help='Search query for ML papers')
    parser.add_argument('-m', '--max-results', type=int, default=100, 
                        help='Maximum number of papers to retrieve per search (default: 100, max: 30000)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set it in a .env file or as an environment variable")
        sys.exit(1)
    
    # Get user query
    if args.query:
        query = " ".join(args.query)
    else:
        query = input("Enter your ML paper search query: ")
    
    if not query.strip():
        print("Error: No query provided")
        parser.print_help()
        sys.exit(1)
    
    # Validate max_results
    max_results = max(1, min(args.max_results, 30000))  # Clamp between 1 and 30000
    if max_results != args.max_results:
        print(f"Warning: max-results clamped to {max_results} (valid range: 1-30000)")
    
    # Create a wrapper function for arxiv_search with the specified max_results
    def arxiv_search_with_limit(query: str, limit: int = None) -> list:
        """Wrapper for arxiv_search that uses the CLI-specified max_results as default."""
        if limit is None:
            limit = max_results
        return arxiv_search(query, limit=min(limit, max_results))
    
    # Initialize components with modern tool registry
    # Ensure tools are initialized
    initialize_tools()
    
    # The tool registry is already set up, but we can provide a legacy wrapper if needed
    tools = {
        "arxiv_search": arxiv_search_with_limit
    }
    
    # Create orchestrator - it will use tool registry first, then fall back to legacy tools
    orchestrator = SimpleOrchestrator(tools=tools)
    
    print(f"Researching: {query}")
    print(f"Max results per search: {max_results}")
    print("=" * 50)
    
    try:
        # Run the simplified orchestrator
        result = await orchestrator.run(query)
        print("\nResults:")
        print(result)
    except Exception as e:
        print(f"Error: {e}")
        logging.exception("Detailed error information:")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())