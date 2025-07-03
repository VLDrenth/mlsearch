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
        """Perform research work with reasoning about search strategy."""
        self.logger.info(f"🔬 ResearchWorker analyzing task: {task}")
        
        # Create a reasoning LLM client
        from core.llmclient import LLMClient
        reasoning_llm = LLMClient(model_name="gpt-4o-mini")
        
        # Execute the research direction assigned by orchestrator
        # Focus on tactical search strategy, not high-level research planning
        reasoning_prompt = f"""
        You are a research execution assistant. The orchestrator has assigned you a specific research direction.
        
        Assigned Research Direction: {task}
        
        Your job is to execute this research direction efficiently by determining:
        1. What are the best search terms to find papers for this specific direction?
        2. What variations of search terms will maximize coverage for this angle?
        3. How many results should you fetch for each search?
        4. What specific keywords and phrases are most relevant to this research angle?
        
        Focus on EXECUTING the assigned research direction, not planning what to research.
        
        Respond with a JSON plan:
        {{
            "searches": [
                {{
                    "query": "specific search terms for arxiv",
                    "limit": 10,
                    "focus": "how this search supports the assigned research direction"
                }}
            ],
            "reasoning": "explanation of your search execution strategy"
        }}
        """
        
        self.logger.info("🤔 Reasoning about search strategy...")
        reasoning_response = reasoning_llm.generate(reasoning_prompt)
        
        self.logger.info(f"💡 Search strategy: {reasoning_response[:100]}...")
        
        try:
            import json
            # Clean the response - sometimes LLMs wrap JSON in markdown
            reasoning_response = reasoning_response.strip()
            if reasoning_response.startswith("```json"):
                reasoning_response = reasoning_response[7:]
            if reasoning_response.endswith("```"):
                reasoning_response = reasoning_response[:-3]
            reasoning_response = reasoning_response.strip()
            
            search_plan = json.loads(reasoning_response)
            
            # Execute the planned searches
            all_results = []
            for i, search in enumerate(search_plan.get("searches", [])):
                self.logger.info(f"🔍 Executing search {i+1}/{len(search_plan['searches'])}: {search['query']}")
                
                # Call the arxiv_search tool
                search_results = await self._call_tool("arxiv_search", {
                    "query": search["query"],
                    "limit": search["limit"]
                })
                
                # Parse the results (they come as a string representation of list)
                if isinstance(search_results, str):
                    try:
                        search_results = eval(search_results)  # Quick parse for now
                    except:
                        pass
                
                if isinstance(search_results, list):
                    all_results.extend(search_results)
                else:
                    all_results.append(search_results)
            
            # Summarize findings
            self.logger.info(f"📊 Found {len(all_results)} total results across all searches")
            
            # Create a summary
            summary_prompt = f"""
            Based on the following research results for the task "{task}", create a comprehensive summary:
            
            Search Strategy Used: {search_plan.get('reasoning', 'Multiple targeted searches')}
            
            Results Found: {len(all_results)} papers
            
            Key Findings:
            """
            
            # Add top results to summary
            for i, result in enumerate(all_results[:5]):  # Top 5 results
                if isinstance(result, dict):
                    summary_prompt += f"\n{i+1}. {result.get('title', 'Unknown')} ({result.get('year', 'Unknown')})\n   Authors: {', '.join(result.get('authors', []))}\n   Summary: {result.get('summary', 'No summary')[:200]}...\n"
            
            summary_prompt += "\n\nPlease provide a structured summary highlighting the key innovations and trends found."
            
            self.logger.info("📝 Generating research summary...")
            summary = reasoning_llm.generate(summary_prompt)
            
            self.output = f"Research Analysis: {search_plan.get('reasoning', '')}\n\n{summary}\n\nDetailed Results:\n{str(all_results)}"
            
        except Exception as e:
            self.logger.error(f"❌ Error in research analysis: {e}")
            # Fallback to simple search
            self.logger.info("🔄 Falling back to simple search...")
            simple_results = await self._call_tool("arxiv_search", {
                "query": task,
                "limit": 10
            })
            self.output = f"Research completed with simple search: {simple_results}"
        
        return self.output

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
        analysis_llm = LLMClient(model_name="gpt-4o-mini")
        
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
        survey_llm = LLMClient(model_name="gpt-4o-mini")
        
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