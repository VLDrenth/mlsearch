from __future__ import annotations
import logging
from typing import Dict, Callable, Any
from core.llmclient import LLMClient


class Agent:
    """Base agent class with direct tool access and reasoning capabilities."""
    
    def __init__(self, tools: Dict[str, Callable], llm_client: LLMClient = None) -> None:
        self.tools = tools
        self.llm = llm_client or LLMClient(model_name="gpt-4o-mini")
        self.output = ""
        self.relevant_papers = []  # Memory of relevant papers found
        self.search_notes = []     # Notes about search progress
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info(f"🤖 {self.__class__.__name__} initialized with {len(tools)} tools")
    
    async def work(self, task: str) -> str:
        """Override this method in subclasses to implement specific agent behavior."""
        self.logger.info(f"💭 Starting work on task: {task}")
        self.output = f"Work completed on: {task}"
        return self.output
    
    def get_output(self) -> str:
        """Return the agent's current output."""
        return self.output
    
    async def call_tool(self, tool_name: str, **kwargs) -> Any:
        """Call a tool directly with the given arguments."""
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not available. Available tools: {list(self.tools.keys())}")
        
        tool_func = self.tools[tool_name]
        self.logger.info(f"🔧 Calling tool: {tool_name} with args: {kwargs}")
        
        # Handle async tools
        import asyncio
        if asyncio.iscoroutinefunction(tool_func):
            result = await tool_func(**kwargs)
        else:
            result = tool_func(**kwargs)
        
        self.logger.info(f"✅ Tool {tool_name} completed")
        return result
    
    def _evaluate_papers_relevance(self, task: str, papers: List[dict]) -> List[dict]:
        """Evaluate which papers are relevant to the task and add them to memory."""
        if not papers:
            return []
        
        # Prepare papers for evaluation
        papers_for_eval = []
        for i, paper in enumerate(papers):
            if isinstance(paper, dict):
                title = paper.get('title', 'Unknown')
                authors = ', '.join(paper.get('authors', [])[:2])
                if len(paper.get('authors', [])) > 2:
                    authors += " et al."
                summary = paper.get('summary', 'No summary')[:200] + "..."
                papers_for_eval.append(f"{i+1}. {title} ({paper.get('year', 'Unknown')}) - {authors}\n   {summary}")
        
        evaluation_prompt = f"""
        Task: {task}
        
        Papers found in search:
        {chr(10).join(papers_for_eval)}
        
        Your job is to identify which papers are DIRECTLY relevant to the task "{task}".
        
        Criteria for relevance:
        - Does the paper directly address the topic or closely related concepts?
        - Does it provide methods, results, or insights relevant to the task?
        - Would this paper be useful for someone researching this topic?
        
        Be selective - only keep papers that are truly relevant.
        
        Respond with:
        RELEVANT_PAPERS: [comma-separated list of paper numbers that are relevant]
        NOTES: [brief notes about what you found and what might be missing]
        
        Example:
        RELEVANT_PAPERS: 1, 3, 7, 9
        NOTES: Found good papers on MIDAS methodology but need more on practical applications
        """
        
        evaluation = self.llm.generate(evaluation_prompt)
        
        # Parse evaluation
        relevant_indices = []
        notes = ""
        
        for line in evaluation.split('\n'):
            if line.startswith('RELEVANT_PAPERS:'):
                relevant_str = line.replace('RELEVANT_PAPERS:', '').strip()
                if relevant_str and relevant_str != "NONE":
                    try:
                        relevant_indices = [int(x.strip()) for x in relevant_str.split(',')]
                        relevant_indices = [x for x in relevant_indices if 1 <= x <= len(papers)]
                    except:
                        pass
            elif line.startswith('NOTES:'):
                notes = line.replace('NOTES:', '').strip()
        
        # Add relevant papers to memory
        newly_relevant = []
        for idx in relevant_indices:
            paper = papers[idx - 1]  # Convert to 0-based index
            # Check if we already have this paper (by title)
            paper_title = paper.get('title', '').lower().strip()
            already_have = any(existing.get('title', '').lower().strip() == paper_title 
                             for existing in self.relevant_papers)
            
            if not already_have:
                self.relevant_papers.append(paper)
                newly_relevant.append(paper)
        
        # Add notes to memory
        if notes:
            self.search_notes.append(notes)
        
        self.logger.info(f"📝 Found {len(newly_relevant)} new relevant papers (total: {len(self.relevant_papers)})")
        if notes:
            self.logger.info(f"📝 Notes: {notes}")
        
        return newly_relevant
    
    def _format_memory_papers(self) -> str:
        """Format papers in memory for display."""
        if not self.relevant_papers:
            return "No relevant papers found yet."
        
        formatted = []
        for i, paper in enumerate(self.relevant_papers[:10]):  # Show up to 10
            title = paper.get('title', 'Unknown')
            year = paper.get('year', 'Unknown')
            authors = ', '.join(paper.get('authors', [])[:2])
            if len(paper.get('authors', [])) > 2:
                authors += " et al."
            formatted.append(f"{i+1}. {title} ({year}) - {authors}")
        
        if len(self.relevant_papers) > 10:
            formatted.append(f"... and {len(self.relevant_papers) - 10} more papers")
        
        return "\n".join(formatted)


class ResearchAgent(Agent):
    """Agent specialized for research tasks with autonomous tool usage."""
    
    async def work(self, task: str) -> str:
        """Perform research work with complete autonomy over search strategy."""
        self.logger.info(f"🔬 ResearchAgent working on: {task}")
        
        # Give the agent complete autonomy to decide how to use tools
        research_prompt = f"""
        You are an autonomous research agent. Your task is to research: {task}
        
        You have access to the following tools:
        - arxiv_search(query, limit): Search academic papers on ArXiv
        
        IMPORTANT: ArXiv Search Syntax - Use proper syntax to get good results:
        - Use quotes for exact phrases: "active learning"
        - Use AND/OR for logic: "bayesian" AND "active learning"
        - Use parentheses for grouping: ("active learning" OR "selective sampling") AND "bayesian"
        - Use field searches: ti:"active learning" (title), abs:"bayesian methods" (abstract)
        - Use category filters: cat:cs.AI OR cat:cs.LG OR cat:stat.ML
        
        Examples of good queries:
        - "mutual information" AND "dependence measures"
        - ti:"correlation" AND (abs:"alternatives" OR abs:"nonlinear")
        - ("distance correlation" OR "maximal information coefficient") AND cat:stat.ML
        
        Your goal is to find relevant papers and provide comprehensive insights.
        You can decide:
        - How many searches to perform
        - What search terms to use (use proper arXiv syntax!)
        - How to analyze and synthesize results
        - When you have enough information
        
        You should reason through your research strategy and execute it autonomously.
        
        Think step by step:
        1. What specific aspects of this topic should I research?
        2. What search queries would be most effective? (use proper arXiv syntax)
        3. How should I analyze the results?
        4. Do I need additional searches based on what I find?
        
        Please provide your research plan and then execute it.
        """
        
        # Let the agent plan its own research approach
        self.logger.info("🧠 Agent planning research strategy...")
        plan = self.llm.generate(research_prompt)
        
        # The agent should now execute its plan by calling tools
        # For now, we'll implement a simple autonomous research pattern
        await self._execute_autonomous_research(task, plan)
        
        return self.output
    
    async def _execute_autonomous_research(self, task: str, plan: str) -> None:
        """Execute research autonomously based on the agent's plan."""
        try:
            # Start with an initial search
            self.logger.info("🔍 Starting autonomous research execution...")
            
            # Let the agent decide on initial search terms
            search_prompt = f"""
            Based on the task: {task}
            
            What should be the first arXiv search query to start researching this topic?
            
            Use proper arXiv syntax:
            - Use quotes for exact phrases: "active learning"
            - Use AND/OR for logic: "bayesian" AND "active learning"
            - Use parentheses for grouping: ("active learning" OR "selective sampling") AND "bayesian"
            - Use field searches: ti:"active learning" (title), abs:"bayesian methods" (abstract)
            - Use category filters: cat:cs.AI OR cat:cs.LG OR cat:stat.ML
            
            Provide only the search query string using proper arXiv syntax, no explanation.
            """
            
            initial_query = self.llm.generate(search_prompt).strip()
            if initial_query.startswith('"') and initial_query.endswith('"'):
                initial_query = initial_query[1:-1]
            
            self.logger.info(f"🎯 Initial search query: {initial_query}")
            
            # Execute the initial search
            results = await self.call_tool("arxiv_search", query=initial_query, limit=100)
            
            # Evaluate papers and add relevant ones to memory
            search_results = results if isinstance(results, list) else [results]
            papers_found = len([r for r in search_results if r is not None])
            
            newly_relevant = self._evaluate_papers_relevance(task, search_results)
            self.logger.info(f"📊 Search results: {papers_found} total, {len(newly_relevant)} relevant")
            
            analysis_prompt = f"""
            You are an autonomous research agent. You just searched for: "{initial_query}"
            
            CURRENT STATUS:
            - Papers found in this search: {papers_found}
            - Relevant papers from this search: {len(newly_relevant)}
            - Total relevant papers in memory: {len(self.relevant_papers)}
            - Your research notes: {'; '.join(self.search_notes) if self.search_notes else 'None yet'}
            
            Your task is: {task}
            
            MEMORY - Relevant papers you've found so far:
            {self._format_memory_papers()}
            
            DECISION CRITERIA:
            - Do you have enough relevant papers to provide a good answer? (typically 5-15 GOOD papers)
            - Are there important aspects of the topic not yet covered?
            - Are your search notes suggesting missing areas?
            - Would additional searches likely find more relevant papers?
            
            IMPORTANT: Use proper arXiv syntax for any new search queries:
            - Use quotes for exact phrases: "active learning"
            - Use AND/OR for logic: "bayesian" AND "active learning"
            - Use parentheses for grouping: ("active learning" OR "selective sampling") AND "bayesian"
            - Use field searches: ti:"active learning" (title), abs:"bayesian methods" (abstract)
            - Use category filters: cat:cs.AI OR cat:cs.LG OR cat:stat.ML
            
            Based on your memory and notes, what should you do next?
            
            Respond with:
            DECISION: [CONTINUE_SEARCHING or FINALIZE_RESULTS]
            REASONING: [explain your decision based on what you have vs what might be missing]
            
            If you decided CONTINUE_SEARCHING, also provide:
            NEXT_SEARCHES: [comma-separated list of up to 3 new search queries using proper arXiv syntax]
            """
            
            decision_response = self.llm.generate(analysis_prompt)
            
            # Parse the agent's decision
            all_results = search_results.copy()
            
            # Extract decision and next searches
            decision = "FINALIZE_RESULTS"  # Default to finalize if unclear
            next_searches = []
            reasoning = ""
            
            lines = decision_response.split("\n")
            for line in lines:
                if line.startswith("DECISION:"):
                    decision = line.replace("DECISION:", "").strip()
                elif line.startswith("REASONING:"):
                    reasoning = line.replace("REASONING:", "").strip()
                elif line.startswith("NEXT_SEARCHES:"):
                    searches_text = line.replace("NEXT_SEARCHES:", "").strip()
                    if searches_text != "NONE" and searches_text:
                        next_searches = [q.strip().strip('"') for q in searches_text.split(",")]
                        next_searches = [q for q in next_searches if q and q != "NONE"]
            
            # Log the decision clearly
            self.logger.info(f"🤔 Agent decision: {decision}")
            if reasoning:
                self.logger.info(f"💭 Reasoning: {reasoning}")
            if decision == "CONTINUE_SEARCHING" and next_searches:
                self.logger.info(f"🔍 Next searches: {next_searches}")
            
            # Execute additional searches if the agent decided to continue
            if decision == "CONTINUE_SEARCHING":
                # If no specific searches provided, ask the agent to complete its response
                if not next_searches:
                    self.logger.info("🔄 Agent decided to continue but didn't provide searches. Asking agent to complete response...")
                    retry_prompt = f"""
                    You decided to CONTINUE_SEARCHING for: {task}
                    
                    Your reasoning was: {reasoning}
                    
                    However, you didn't provide the NEXT_SEARCHES field. Please complete your response by providing specific search queries.
                    
                    Use proper arXiv syntax and provide:
                    NEXT_SEARCHES: [comma-separated list of 2-3 new search queries using proper arXiv syntax]
                    
                    Example format:
                    NEXT_SEARCHES: "time series" AND "deep learning", "neural networks" AND cat:cs.LG, "forecasting methods"
                    """
                    
                    retry_response = self.llm.generate(retry_prompt).strip()
                    self.logger.info(f"🔄 Agent retry response: {retry_response}")
                    
                    # Parse the retry response
                    for line in retry_response.split("\n"):
                        if line.startswith("NEXT_SEARCHES:"):
                            searches_text = line.replace("NEXT_SEARCHES:", "").strip()
                            if searches_text != "NONE" and searches_text:
                                next_searches = [q.strip().strip('"') for q in searches_text.split(",")]
                                next_searches = [q for q in next_searches if q and q != "NONE"]
                                break
                    
                    if next_searches:
                        self.logger.info(f"🔍 Agent provided searches: {next_searches}")
                    else:
                        self.logger.warning("🔄 Agent still didn't provide searches, proceeding with analysis...")
                
                if next_searches:
                    # Execute additional searches and evaluate papers
                    for i, query in enumerate(next_searches[:3]):
                        if query:
                            self.logger.info(f"🔍 Agent-directed search {i+1}: {query}")
                            more_results = await self.call_tool("arxiv_search", query=query, limit=100)
                            
                            # Evaluate and store relevant papers from this search
                            search_batch = more_results if isinstance(more_results, list) else [more_results]
                            newly_relevant = self._evaluate_papers_relevance(task, search_batch)
                            self.logger.info(f"📊 Search {i+1} results: {len(search_batch)} total, {len(newly_relevant)} relevant")
                            
                            if isinstance(more_results, list):
                                all_results.extend(more_results)
                            else:
                                all_results.append(more_results)
            
            # If still no results after multiple attempts, let agent try one more strategy
            final_papers = [r for r in all_results if r is not None]
            if len(final_papers) == 0:
                self.logger.info("🔄 No papers found, letting agent try final strategy...")
                final_strategy_prompt = f"""
                You've tried multiple searches for: {task}
                
                Previous searches:
                1. {initial_query}
                {chr(10).join(f"{i+2}. {q}" for i, q in enumerate(next_searches))}
                
                All searches returned 0 papers. This suggests the search terms might be too specific,
                or the topic might be expressed differently in academic literature.
                
                Suggest ONE final, broader search strategy using different terminology.
                Think about:
                - What are the core concepts in different terms?
                - What broader field might this fall under?
                - What simpler keywords might work?
                
                Use proper arXiv syntax:
                - Use quotes for exact phrases: "active learning"
                - Use AND/OR for logic: "bayesian" AND "active learning"
                - Use parentheses for grouping: ("active learning" OR "selective sampling") AND "bayesian"
                - Use field searches: ti:"active learning" (title), abs:"bayesian methods" (abstract)
                - Use category filters: cat:cs.AI OR cat:cs.LG OR cat:stat.ML
                
                Provide only the search query string using proper arXiv syntax:
                """
                
                final_query = self.llm.generate(final_strategy_prompt).strip().strip('"')
                if final_query:
                    self.logger.info(f"🎯 Final strategy search: {final_query}")
                    final_results = await self.call_tool("arxiv_search", query=final_query, limit=100)
                    if isinstance(final_results, list):
                        all_results.extend(final_results)
                    else:
                        all_results.append(final_results)
            
            # Generate final output using relevant papers from memory
            self.logger.info(f"🎯 Finalizing results with {len(self.relevant_papers)} relevant papers")
            
            final_output = f"""# Research Results for: {task}

## Summary
Found {len(self.relevant_papers)} relevant papers through autonomous search and evaluation.

Research Notes:
{chr(10).join(f"- {note}" for note in self.search_notes) if self.search_notes else "- No specific notes recorded"}

## Relevant Papers Found

"""
            
            # Add relevant papers with metadata
            for i, paper in enumerate(self.relevant_papers, 1):
                title = paper.get('title', 'Unknown')
                year = paper.get('year', 'Unknown')
                authors = ', '.join(paper.get('authors', []))
                pdf_url = paper.get('pdf_url', '')
                summary = paper.get('summary', 'No summary available')[:300] + "..."
                
                final_output += f"### {i}. {title} ({year})\n"
                final_output += f"**Authors:** {authors}\n"
                if pdf_url:
                    final_output += f"**PDF:** {pdf_url}\n"
                final_output += f"**Summary:** {summary}\n\n"
            
            if not self.relevant_papers:
                final_output += "No relevant papers found. Consider trying different search terms or broadening the topic scope.\n"
            
            self.output = final_output
            
        except Exception as e:
            self.logger.error(f"❌ Error in autonomous research: {e}")
            self.output = f"Research completed with error: {str(e)}"
    
    def _format_papers(self, papers) -> str:
        """Format papers for analysis."""
        if not papers:
            return "No papers found"
        
        if not isinstance(papers, list):
            papers = [papers]
        
        formatted = []
        for i, paper in enumerate(papers[:8]):  # Show top 8 papers
            if isinstance(paper, dict):
                title = paper.get('title', 'Unknown')
                year = paper.get('year', 'Unknown')
                authors = ', '.join(paper.get('authors', [])[:2])
                if len(paper.get('authors', [])) > 2:
                    authors += " et al."
                formatted.append(f"{i+1}. {title} ({year}) - {authors}")
            else:
                formatted.append(f"{i+1}. {str(paper)}")
        
        return "\n".join(formatted)
    
    def _format_papers_with_citations(self, papers) -> str:
        """Format papers with citation numbers for analysis."""
        if not papers:
            return "No papers found"
        
        formatted = []
        for i, paper in enumerate(papers):
            if isinstance(paper, dict):
                title = paper.get('title', 'Unknown')
                year = paper.get('year', 'Unknown')
                authors = ', '.join(paper.get('authors', [])[:2])
                if len(paper.get('authors', [])) > 2:
                    authors += " et al."
                summary = paper.get('summary', 'No summary')[:200] + "..."
                formatted.append(f"[{i+1}] {title} ({year}) - {authors}\n    {summary}")
            else:
                formatted.append(f"[{i+1}] {str(paper)}")
        
        return "\n\n".join(formatted)
    
    def _generate_reference_list(self, papers) -> str:
        """Generate a formatted reference list with URLs."""
        if not papers:
            return "## References\n\nNo papers found."
        
        references = ["## References\n"]
        for i, paper in enumerate(papers):
            if isinstance(paper, dict):
                title = paper.get('title', 'Unknown')
                year = paper.get('year', 'Unknown')
                authors = ', '.join(paper.get('authors', []))
                pdf_url = paper.get('pdf_url', '')
                
                ref_line = f"[{i+1}] {authors} ({year}). {title}"
                if pdf_url:
                    ref_line += f" [PDF]({pdf_url})"
                references.append(ref_line)
            else:
                references.append(f"[{i+1}] {str(paper)}")
        
        return "\n".join(references)
    
    def _generate_paper_list(self, papers) -> str:
        """Generate a nicely formatted paper list (like in academic papers)."""
        if not papers:
            return "## Papers Analyzed\n\nNo papers found in the search."
        
        paper_list = ["## Papers Analyzed\n"]
        paper_list.append(f"This analysis is based on {len(papers)} papers from the academic literature:\n")
        
        # Group papers by year for better presentation
        papers_by_year = {}
        for paper in papers:
            if isinstance(paper, dict):
                year = paper.get('year', 'Unknown')
                if year not in papers_by_year:
                    papers_by_year[year] = []
                papers_by_year[year].append(paper)
        
        # Sort years in descending order (most recent first)
        sorted_years = sorted(papers_by_year.keys(), reverse=True)
        
        for year in sorted_years:
            if year != 'Unknown':
                paper_list.append(f"**{year}:**")
            else:
                paper_list.append("**Year Unknown:**")
                
            for paper in papers_by_year[year]:
                title = paper.get('title', 'Unknown')
                authors = paper.get('authors', [])
                
                # Format authors nicely
                if len(authors) == 0:
                    author_str = "Unknown authors"
                elif len(authors) == 1:
                    author_str = authors[0]
                elif len(authors) == 2:
                    author_str = f"{authors[0]} and {authors[1]}"
                else:
                    author_str = f"{authors[0]} et al."
                
                paper_list.append(f"- {author_str}. *{title}*")
            
            paper_list.append("")  # Empty line between years
        
        return "\n".join(paper_list)


class AnalysisAgent(Agent):
    """Agent specialized for analyzing and synthesizing research findings."""
    
    async def work(self, task: str) -> str:
        """Perform analysis work with autonomous tool usage."""
        self.logger.info(f"📊 AnalysisAgent working on: {task}")
        
        analysis_prompt = f"""
        You are an autonomous analysis agent. Your task is to analyze: {task}
        
        You have access to research tools and can gather papers if needed for your analysis.
        
        Your goal is to provide deep analytical insights, not just summaries.
        
        Think about:
        1. What specific analysis is needed?
        2. Do I need to gather additional papers for this analysis?
        3. What analytical framework should I use?
        4. What insights can I provide?
        
        Execute your analysis autonomously.
        """
        
        # Let the agent decide its analysis approach
        plan = self.llm.generate(analysis_prompt)
        
        # Agent can decide to gather papers if needed
        if "search" in plan.lower() or "papers" in plan.lower():
            # Agent wants to gather papers
            search_prompt = f"""
            For the analysis task: {task}
            
            What arXiv search query would help you gather the most relevant papers for analysis?
            
            Use proper arXiv syntax:
            - Use quotes for exact phrases: "active learning"
            - Use AND/OR for logic: "bayesian" AND "active learning"
            - Use parentheses for grouping: ("active learning" OR "selective sampling") AND "bayesian"
            - Use field searches: ti:"active learning" (title), abs:"bayesian methods" (abstract)
            - Use category filters: cat:cs.AI OR cat:cs.LG OR cat:stat.ML
            
            Provide only the search query string using proper arXiv syntax.
            """
            
            query = self.llm.generate(search_prompt).strip()
            if query.startswith('"') and query.endswith('"'):
                query = query[1:-1]
            
            self.logger.info(f"📚 Gathering papers for analysis: {query}")
            results = await self.call_tool("arxiv_search", query=query, limit=100)
            
            # Check if we need to try different search strategies
            search_results = results if isinstance(results, list) else [results]
            papers_found = len([r for r in search_results if r is not None])
            
            all_results = search_results.copy()
            
            # If no papers found, let agent try alternative searches
            if papers_found == 0:
                self.logger.info("🔄 No papers found, trying alternative search strategies...")
                
                alt_search_prompt = f"""
                You searched for: "{query}" but found 0 papers for analysis task: {task}
                
                Suggest 2-3 alternative search strategies using different terminology.
                Think about:
                - What are the core concepts in different terms?
                - What broader or narrower terms might work?
                - What related fields might have relevant work?
                
                Use proper arXiv syntax:
                - Use quotes for exact phrases: "active learning"
                - Use AND/OR for logic: "bayesian" AND "active learning"
                - Use parentheses for grouping: ("active learning" OR "selective sampling") AND "bayesian"
                - Use field searches: ti:"active learning" (title), abs:"bayesian methods" (abstract)
                - Use category filters: cat:cs.AI OR cat:cs.LG OR cat:stat.ML
                
                Provide comma-separated search queries using proper arXiv syntax:
                """
                
                alt_queries_response = self.llm.generate(alt_search_prompt).strip()
                alt_queries = [q.strip().strip('"') for q in alt_queries_response.split(",")]
                alt_queries = [q for q in alt_queries if q and len(q) > 3]
                
                for alt_query in alt_queries[:2]:
                    self.logger.info(f"🔍 Alternative search: {alt_query}")
                    alt_results = await self.call_tool("arxiv_search", query=alt_query, limit=100)
                    if isinstance(alt_results, list):
                        all_results.extend(alt_results)
                    else:
                        all_results.append(alt_results)
            
            # Perform analysis with gathered papers
            final_papers = [r for r in all_results if r is not None and isinstance(r, dict)]
            analysis_prompt = f"""
            Analysis task: {task}
            
            Papers found: {len(final_papers)}
            Papers for analysis:
            {self._format_papers_with_citations(final_papers)}
            
            Provide a comprehensive analysis focusing on:
            1. Key themes and patterns
            2. Methodological approaches
            3. Performance trends
            4. Limitations and challenges
            5. Future research directions
            
            CRITICAL CITATION REQUIREMENTS: 
            - You MUST include inline citations throughout your analysis using [1], [2], etc.
            - Every claim, finding, or method mentioned should be cited with the paper number
            - Use citations like: "Recent work on dependency measures [1] shows that..." or "Correlation methods [3, 5] provide..."
            - Cite multiple papers when relevant: [1, 2, 4]
            - Do NOT write an analysis without citations - this is mandatory
            
            Example format:
            "The analysis reveals that mutual information approaches [1, 2] consistently outperform traditional correlation measures. Distance correlation [3] provides a robust alternative for nonlinear dependencies..."
            
            If no papers were found, provide analysis based on your knowledge of the field.
            """
            
            analysis = self.llm.generate(analysis_prompt)
            
            # Add reference list and paper list
            reference_list = self._generate_reference_list(final_papers)
            paper_list = self._generate_paper_list(final_papers)
            
            self.output = f"Analysis Results\n\n{analysis}\n\n{reference_list}\n\n{paper_list}"
        else:
            # Agent doesn't need additional papers
            self.output = f"Analysis completed for: {task}\n\n{plan}"
        
        return self.output
    
    def _format_papers(self, papers) -> str:
        """Format papers for analysis."""
        if not papers:
            return "No papers found"
        
        if not isinstance(papers, list):
            papers = [papers]
        
        formatted = []
        for i, paper in enumerate(papers[:5]):
            if isinstance(paper, dict):
                title = paper.get('title', 'Unknown')
                year = paper.get('year', 'Unknown')
                summary = paper.get('summary', 'No summary')[:200] + "..."
                formatted.append(f"{i+1}. {title} ({year})\n   {summary}")
            else:
                formatted.append(f"{i+1}. {str(paper)}")
        
        return "\n\n".join(formatted)
    
    def _format_papers_with_citations(self, papers) -> str:
        """Format papers with citation numbers for analysis."""
        if not papers:
            return "No papers found"
        
        formatted = []
        for i, paper in enumerate(papers):
            if isinstance(paper, dict):
                title = paper.get('title', 'Unknown')
                year = paper.get('year', 'Unknown')
                authors = ', '.join(paper.get('authors', [])[:2])
                if len(paper.get('authors', [])) > 2:
                    authors += " et al."
                summary = paper.get('summary', 'No summary')[:200] + "..."
                formatted.append(f"[{i+1}] {title} ({year}) - {authors}\n    {summary}")
            else:
                formatted.append(f"[{i+1}] {str(paper)}")
        
        return "\n\n".join(formatted)
    
    def _generate_reference_list(self, papers) -> str:
        """Generate a formatted reference list with URLs."""
        if not papers:
            return "## References\n\nNo papers found."
        
        references = ["## References\n"]
        for i, paper in enumerate(papers):
            if isinstance(paper, dict):
                title = paper.get('title', 'Unknown')
                year = paper.get('year', 'Unknown')
                authors = ', '.join(paper.get('authors', []))
                pdf_url = paper.get('pdf_url', '')
                
                ref_line = f"[{i+1}] {authors} ({year}). {title}"
                if pdf_url:
                    ref_line += f" [PDF]({pdf_url})"
                references.append(ref_line)
            else:
                references.append(f"[{i+1}] {str(paper)}")
        
        return "\n".join(references)
    
    def _generate_paper_list(self, papers) -> str:
        """Generate a nicely formatted paper list (like in academic papers)."""
        if not papers:
            return "## Papers Analyzed\n\nNo papers found in the search."
        
        paper_list = ["## Papers Analyzed\n"]
        paper_list.append(f"This analysis is based on {len(papers)} papers from the academic literature:\n")
        
        # Group papers by year for better presentation
        papers_by_year = {}
        for paper in papers:
            if isinstance(paper, dict):
                year = paper.get('year', 'Unknown')
                if year not in papers_by_year:
                    papers_by_year[year] = []
                papers_by_year[year].append(paper)
        
        # Sort years in descending order (most recent first)
        sorted_years = sorted(papers_by_year.keys(), reverse=True)
        
        for year in sorted_years:
            if year != 'Unknown':
                paper_list.append(f"**{year}:**")
            else:
                paper_list.append("**Year Unknown:**")
                
            for paper in papers_by_year[year]:
                title = paper.get('title', 'Unknown')
                authors = paper.get('authors', [])
                
                # Format authors nicely
                if len(authors) == 0:
                    author_str = "Unknown authors"
                elif len(authors) == 1:
                    author_str = authors[0]
                elif len(authors) == 2:
                    author_str = f"{authors[0]} and {authors[1]}"
                else:
                    author_str = f"{authors[0]} et al."
                
                paper_list.append(f"- {author_str}. *{title}*")
            
            paper_list.append("")  # Empty line between years
        
        return "\n".join(paper_list)


# Registry of available agent types
agent_registry = {
    "ResearchAgent": ResearchAgent,
    "AnalysisAgent": AnalysisAgent,
}