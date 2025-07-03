#!/usr/bin/env python3
"""
ML Search CLI - Search for ML papers on ArXiv using natural language queries.
"""
import asyncio
import sys
import os
from pathlib import Path

# Add the mlsearch package to the path
sys.path.insert(0, str(Path(__file__).parent / "mlsearch"))

from core.llmclient import LLMClient
from agents.orchestrator import Orchestrator
from tools.arxivsearch import arxiv_search

async def main():
    """Main CLI function."""
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set it in a .env file or as an environment variable")
        sys.exit(1)
    
    # Initialize components
    planner = LLMClient(model_name="gpt-4o-mini")
    tools = {
        "arxiv_search": arxiv_search
    }
    orchestrator = Orchestrator(planner, tools)
    
    # Get user query
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Enter your ML paper search query: ")
    
    if not query.strip():
        print("Error: No query provided")
        sys.exit(1)
    
    print(f"Searching for: {query}")
    print("=" * 50)
    
    try:
        # Run the orchestrator
        result = await orchestrator.run(query)
        print("\nResults:")
        print(result)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())