# MLSearch

A machine learning research search system that uses AI agents to find and analyze academic papers from ArXiv.

## What it does

MLSearch takes your research query and uses AI agents to search ArXiv for relevant papers, then synthesizes the findings into a comprehensive report. The system uses multiple autonomous agents that work together to provide thorough research coverage.

## Installation

```bash
pip install -r requirements.txt
```

Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

### Web Interface (Recommended)

**Live Demo**: https://mlsearch.onrender.com

Or run locally:
```bash
python -m mlsearch.web.app
```
Then open http://localhost:8000

### Command Line
```bash
python main.py "your search query"

# With options
python main.py --max-results 200 --verbose "deep learning time series"
```

## Examples

- "Latest developments in transformer architectures"
- "Federated learning privacy techniques"
- "Time series forecasting with deep learning"