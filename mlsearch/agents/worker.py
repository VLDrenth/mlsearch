from __future__ import annotations
from typing import TYPE_CHECKING
import logging
import asyncio

if TYPE_CHECKING:
    from agents.orchestrator import Orchestrator

class Worker:
    """Base worker class for specialized agents."""
    
    def __init__(self, orchestrator: Orchestrator) -> None:
        self.orchestrator = orchestrator
        self.output = ""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info(f"🤖 {self.__class__.__name__} initialized")
    
    def get_output(self) -> str:
        """Return the worker's output."""
        self.logger.info(f"📤 Returning output: {self.output[:50]}{'...' if len(self.output) > 50 else ''}")
        return self.output
    
    async def work(self, task: str) -> str:
        """Override this method in subclasses to implement specific work logic."""
        self.logger.info(f"💭 Starting work on task: {task}")
        self.output = f"Work completed on: {task}"
        return self.output
    
    async def _call_tool(self, tool_name: str, args: dict) -> str:
        """Call a tool through the orchestrator."""
        self.logger.info(f"🔧 Worker calling tool: {tool_name} with args: {args}")
        return await self.orchestrator._call_tool(tool_name, args)

class ResearchWorker(Worker):
    """Worker specialized for research tasks."""
    
    def __init__(self, orchestrator: Orchestrator) -> None:
        super().__init__(orchestrator)
        self.output = "Research completed"
        self.logger.info("🔬 ResearchWorker ready for research tasks")
    
    async def work(self, task: str) -> str:
        """Perform research work with iterative search strategy based on intermediate results."""
        self.logger.info(f"🔬 ResearchWorker analyzing task: {task}")
        
        # Create a reasoning LLM client
        from core.llmclient import LLMClient
        reasoning_llm = LLMClient(model_type="reasoning")
        
        # Start with initial search strategy
        self.logger.info("🤔 Planning initial search strategy...")
        
        all_results = []
        search_history = []
        max_searches = 3  # Limit to avoid infinite loops
        
        try:
            import json
            
            # Initial search planning
            initial_prompt = f"""
            You are a research execution assistant. Plan the first search for this research direction:
            
            Research Direction: {task}
            
            Plan a focused initial search to understand the current state of this research area.
            
            Respond with JSON:
            {{
                "query": "specific search terms for arxiv",
                "limit": 10,
                "focus": "what this initial search aims to discover"
            }}
            """
            
            for search_round in range(max_searches):
                self.logger.info(f"🔍 Starting search round {search_round + 1}/{max_searches}")
                
                if search_round == 0:
                    # First search - use initial planning
                    search_response = reasoning_llm.generate(initial_prompt)
                else:
                    # Subsequent searches - analyze previous results
                    analysis_prompt = f"""
                    You are analyzing search results to plan the next search.
                    
                    Research Direction: {task}
                    
                    Previous searches and results:
                    {self._format_search_history(search_history)}
                    
                    Current findings summary:
                    - Found {len(all_results)} papers total
                    - {len([r for r in all_results if r.get('year', 0) >= 2023])} papers from 2023+
                    
                    Recent papers found:
                    {self._format_recent_papers(all_results)}
                    
                    Based on what you've found, what specific aspect needs more exploration?
                    What search terms would find relevant papers you haven't discovered yet?
                    
                    Respond with JSON:
                    {{
                        "query": "new search terms targeting unexplored aspects",
                        "limit": 10,
                        "focus": "what gap this search will fill",
                        "analysis": "what the previous results showed"
                    }}
                    """
                    search_response = reasoning_llm.generate(analysis_prompt)
                
                # Parse search plan
                search_response = search_response.strip()
                if search_response.startswith("```json"):
                    search_response = search_response[7:]
                if search_response.endswith("```"):
                    search_response = search_response[:-3]
                search_response = search_response.strip()
                
                search_plan = json.loads(search_response)
                
                # Execute the search
                query = search_plan.get("query", "")
                limit = search_plan.get("limit", 10)
                focus = search_plan.get("focus", "")
                
                self.logger.info(f"🎯 Search {search_round + 1}: {query}")
                self.logger.info(f"📋 Focus: {focus}")
                
                # Call the arxiv_search tool
                search_results = await self._call_tool("arxiv_search", {
                    "query": query,
                    "limit": limit
                })
                
                # Parse results
                if isinstance(search_results, str):
                    try:
                        search_results = eval(search_results)
                    except:
                        search_results = []
                
                if not isinstance(search_results, list):
                    search_results = [search_results] if search_results else []
                
                # Track search history
                search_history.append({
                    "round": search_round + 1,
                    "query": query,
                    "focus": focus,
                    "results_count": len(search_results),
                    "results": search_results
                })
                
                # Add new results (avoid duplicates by title)
                existing_titles = {r.get('title', '').lower() for r in all_results}
                new_results = [r for r in search_results if r.get('title', '').lower() not in existing_titles]
                all_results.extend(new_results)
                
                self.logger.info(f"📊 Search {search_round + 1} found {len(search_results)} papers, {len(new_results)} new")
                
                # Analyze if we should continue searching
                if search_round < max_searches - 1:
                    continue_prompt = f"""
                    Should we continue searching for more papers on: {task}?
                    
                    Current status:
                    - Total papers found: {len(all_results)}
                    - New papers from last search: {len(new_results)}
                    - Searches completed: {search_round + 1}
                    
                    Recent findings:
                    {self._format_recent_papers(search_results[:3])}
                    
                    Should we do one more search? Consider:
                    - Are there important gaps still unexplored?
                    - Are we finding substantially new relevant papers?
                    - Would another search angle be valuable?
                    
                    Respond with JSON:
                    {{
                        "continue": true/false,
                        "reason": "explanation of decision"
                    }}
                    """
                    
                    continue_response = reasoning_llm.generate(continue_prompt)
                    continue_response = continue_response.strip()
                    if continue_response.startswith("```json"):
                        continue_response = continue_response[7:]
                    if continue_response.endswith("```"):
                        continue_response = continue_response[:-3]
                    continue_response = continue_response.strip()
                    
                    try:
                        continue_decision = json.loads(continue_response)
                        if not continue_decision.get("continue", False):
                            self.logger.info(f"🏁 Stopping search: {continue_decision.get('reason', 'No reason given')}")
                            break
                    except:
                        # If parsing fails, continue with next search
                        pass
            
            # Generate comprehensive summary
            self.logger.info("📝 Generating comprehensive research summary...")
            summary_prompt = f"""
            Create a comprehensive research summary for: {task}
            
            Search process:
            {self._format_search_history(search_history)}
            
            Total papers found: {len(all_results)}
            
            Key papers by relevance and recency:
            {self._format_key_papers(all_results)}
            
            Provide a structured analysis covering:
            1. Current state of research in this area
            2. Key innovations and recent developments
            3. Major research directions and trends
            4. Gaps and opportunities for future work
            
            Focus on insights gained from the {len(all_results)} papers found across {len(search_history)} targeted searches.
            """
            
            summary = reasoning_llm.generate(summary_prompt)
            
            self.output = f"Iterative Research Analysis\n\n{summary}\n\nSearch History:\n{self._format_search_history(search_history)}\n\nDetailed Results:\n{str(all_results)}"
            
        except Exception as e:
            self.logger.error(f"❌ Error in iterative research: {e}")
            # Fallback to simple search
            self.logger.info("🔄 Falling back to simple search...")
            simple_results = await self._call_tool("arxiv_search", {
                "query": task,
                "limit": 10
            })
            self.output = f"Research completed with simple search: {simple_results}"
        
        return self.output
    
    def _format_search_history(self, history: list) -> str:
        """Format search history for display."""
        formatted = []
        for search in history:
            formatted.append(f"Search {search['round']}: {search['query']}")
            formatted.append(f"  Focus: {search['focus']}")
            formatted.append(f"  Found: {search['results_count']} papers")
        return "\n".join(formatted)
    
    def _format_recent_papers(self, papers: list) -> str:
        """Format recent papers for analysis."""
        if not papers:
            return "No papers found"
        
        formatted = []
        for i, paper in enumerate(papers[:5]):
            if isinstance(paper, dict):
                title = paper.get('title', 'Unknown')
                year = paper.get('year', 'Unknown')
                authors = ', '.join(paper.get('authors', [])[:2])
                if len(paper.get('authors', [])) > 2:
                    authors += " et al."
                formatted.append(f"{i+1}. {title} ({year}) - {authors}")
        return "\n".join(formatted)
    
    def _format_key_papers(self, papers: list) -> str:
        """Format key papers with more detail."""
        if not papers:
            return "No papers found"
        
        # Sort by year (recent first) and take top 10
        sorted_papers = sorted(papers, key=lambda p: p.get('year', 0), reverse=True)
        
        formatted = []
        for i, paper in enumerate(sorted_papers[:10]):
            if isinstance(paper, dict):
                title = paper.get('title', 'Unknown')
                year = paper.get('year', 'Unknown')
                authors = ', '.join(paper.get('authors', [])[:3])
                if len(paper.get('authors', [])) > 3:
                    authors += " et al."
                summary = paper.get('summary', 'No summary')[:150] + "..."
                formatted.append(f"{i+1}. {title} ({year})\n   Authors: {authors}\n   Summary: {summary}")
        
        return "\n\n".join(formatted)

class AnalysisWorker(Worker):
    """Worker specialized for analyzing and synthesizing research findings."""
    
    def __init__(self, orchestrator: Orchestrator) -> None:
        super().__init__(orchestrator)
        self.output = "Analysis completed"
        self.logger.info("📊 AnalysisWorker ready for analysis tasks")
    
    async def work(self, task: str) -> str:
        """Perform analysis work by reviewing existing research and synthesizing insights."""
        self.logger.info(f"📊 AnalysisWorker analyzing: {task}")
        
        from core.llmclient import LLMClient
        analysis_llm = LLMClient(model_type="analysis")
        
        # Focus on analysis rather than new searches
        analysis_prompt = f"""
        You are an expert research analyst. Your task is to analyze and synthesize research findings.
        
        Task: {task}
        
        Please provide a comprehensive analysis that includes:
        1. Key themes and patterns
        2. Recent developments and trends
        3. Gaps in current research
        4. Future research directions
        
        If you need to gather specific papers for analysis, determine the most targeted search strategy.
        
        Respond with a JSON plan:
        {{
            "analysis_focus": "what aspects you'll analyze",
            "searches_needed": [
                {{
                    "query": "specific search terms",
                    "limit": 5,
                    "purpose": "why this search is needed"
                }}
            ],
            "analysis_approach": "your analytical methodology"
        }}
        """
        
        self.logger.info("🔍 Planning analysis approach...")
        analysis_plan = analysis_llm.generate(analysis_prompt)
        
        try:
            import json
            # Clean the response
            analysis_plan = analysis_plan.strip()
            if analysis_plan.startswith("```json"):
                analysis_plan = analysis_plan[7:]
            if analysis_plan.endswith("```"):
                analysis_plan = analysis_plan[:-3]
            analysis_plan = analysis_plan.strip()
            
            plan = json.loads(analysis_plan)
            
            # Gather targeted research if needed
            research_data = []
            for search in plan.get("searches_needed", []):
                self.logger.info(f"📚 Gathering research: {search['query']}")
                results = await self._call_tool("arxiv_search", {
                    "query": search["query"],
                    "limit": search["limit"]
                })
                
                if isinstance(results, str):
                    try:
                        results = eval(results)
                    except:
                        pass
                
                if isinstance(results, list):
                    research_data.extend(results)
            
            # Perform analysis
            if research_data:
                analysis_prompt = f"""
                Based on the research data below, provide a comprehensive analysis for: {task}
                
                Research Data: {len(research_data)} papers found
                
                Analysis Framework: {plan.get('analysis_approach', 'Systematic analysis')}
                
                Research Papers:
                """
                
                for i, paper in enumerate(research_data[:10]):  # Top 10 papers
                    if isinstance(paper, dict):
                        analysis_prompt += f"\n{i+1}. {paper.get('title', 'Unknown')} ({paper.get('year', 'N/A')})\n   {paper.get('summary', 'No summary')[:300]}...\n"
                
                analysis_prompt += f"\n\nPlease provide a detailed analysis covering:\n1. Key innovations and breakthroughs\n2. Methodological trends\n3. Performance improvements\n4. Limitations and challenges\n5. Future research directions\n\nFocus on: {plan.get('analysis_focus', task)}"
                
                self.logger.info("🧠 Generating comprehensive analysis...")
                analysis_result = analysis_llm.generate(analysis_prompt)
                
                self.output = f"Analysis Focus: {plan.get('analysis_focus', task)}\n\n{analysis_result}\n\nBased on {len(research_data)} research papers."
            else:
                self.output = f"Analysis completed for: {task}\n\nNo additional research data was needed for this analysis."
        
        except Exception as e:
            self.logger.error(f"❌ Analysis error: {e}")
            self.output = f"Analysis completed with basic approach for: {task}"
        
        return self.output

class SurveyWorker(Worker):
    """Worker specialized for conducting comprehensive literature surveys."""
    
    def __init__(self, orchestrator: Orchestrator) -> None:
        super().__init__(orchestrator)
        self.output = "Survey completed"
        self.logger.info("📋 SurveyWorker ready for survey tasks")
    
    async def work(self, task: str) -> str:
        """Perform comprehensive literature survey work."""
        self.logger.info(f"📋 SurveyWorker conducting survey: {task}")
        
        from core.llmclient import LLMClient
        survey_llm = LLMClient(model_type="worker")
        
        # Plan comprehensive survey strategy
        survey_prompt = f"""
        You are conducting a comprehensive literature survey.
        
        Task: {task}
        
        Plan a systematic survey approach with multiple search strategies to ensure comprehensive coverage:
        
        {{
            "survey_scope": "scope and boundaries of the survey",
            "search_strategies": [
                {{
                    "query": "broad foundational search terms",
                    "limit": 15,
                    "category": "foundational"
                }},
                {{
                    "query": "recent developments search terms",
                    "limit": 15,
                    "category": "recent"
                }},
                {{
                    "query": "applications search terms",
                    "limit": 10,
                    "category": "applications"
                }}
            ],
            "survey_methodology": "how you'll organize and present findings"
        }}
        """
        
        self.logger.info("📊 Planning comprehensive survey...")
        survey_plan = survey_llm.generate(survey_prompt)
        
        try:
            import json
            # Clean the response
            survey_plan = survey_plan.strip()
            if survey_plan.startswith("```json"):
                survey_plan = survey_plan[7:]
            if survey_plan.endswith("```"):
                survey_plan = survey_plan[:-3]
            survey_plan = survey_plan.strip()
            
            plan = json.loads(survey_plan)
            
            # Execute comprehensive searches
            all_papers = []
            categories = {}
            
            for strategy in plan.get("search_strategies", []):
                category = strategy.get("category", "general")
                self.logger.info(f"🔍 Survey search ({category}): {strategy['query']}")
                
                results = await self._call_tool("arxiv_search", {
                    "query": strategy["query"],
                    "limit": strategy["limit"]
                })
                
                if isinstance(results, str):
                    try:
                        results = eval(results)
                    except:
                        pass
                
                if isinstance(results, list):
                    categories[category] = results
                    all_papers.extend(results)
            
            # Generate comprehensive survey
            survey_prompt = f"""
            Create a comprehensive literature survey for: {task}
            
            Survey Scope: {plan.get('survey_scope', 'Comprehensive review')}
            
            Papers found: {len(all_papers)} across {len(categories)} categories
            
            Categories surveyed:
            """
            
            for category, papers in categories.items():
                survey_prompt += f"\n{category.upper()}: {len(papers)} papers"
                for i, paper in enumerate(papers[:3]):  # Top 3 per category
                    if isinstance(paper, dict):
                        survey_prompt += f"\n  • {paper.get('title', 'Unknown')} ({paper.get('year', 'N/A')})"
            
            survey_prompt += f"\n\nPlease provide a structured survey with:\n1. Overview of the field\n2. Key themes and categories\n3. Chronological development\n4. Major contributions by category\n5. Current state of research\n6. Future directions\n\nMethodology: {plan.get('survey_methodology', 'Systematic review')}"
            
            self.logger.info("📝 Generating comprehensive survey...")
            survey_result = survey_llm.generate(survey_prompt)
            
            self.output = f"Literature Survey: {plan.get('survey_scope', task)}\n\n{survey_result}\n\nBased on {len(all_papers)} papers across {len(categories)} research categories."
        
        except Exception as e:
            self.logger.error(f"❌ Survey error: {e}")
            self.output = f"Survey completed with basic approach for: {task}"
        
        return self.output

# Registry of available worker types
worker_registry = {
    "ResearchWorker": ResearchWorker,
    "AnalysisWorker": AnalysisWorker,
    "SurveyWorker": SurveyWorker,
    "CodeWorker": ResearchWorker,  # For now, use ResearchWorker
    "WriterWorker": ResearchWorker,  # For now, use ResearchWorker
}