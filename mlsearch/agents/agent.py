from __future__ import annotations
import logging
from typing import Dict, Callable, Any
from core.llmclient import LLMClient


class Agent:
    """Base agent class with direct tool access and reasoning capabilities."""
    
    def __init__(self, tools: Dict[str, Callable], llm_client: LLMClient = None) -> None:
        self.tools = tools
        self.llm = llm_client or LLMClient(model_type="agent")
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
    
    def _should_broaden_search(self, search_results: list, relevant_papers: list) -> bool:
        """Determine if search should be broadened based on results."""
        total_papers = len([r for r in search_results if r is not None])
        relevant_count = len(relevant_papers)
        
        # Broaden if we found very few total papers or very few relevant ones
        return total_papers < 10 or relevant_count < 3


class ResearchAgent(Agent):
    """Agent specialized for research tasks with autonomous tool usage."""
    
    async def work(self, task: str) -> str:
        """Perform research work with complete autonomy over search strategy."""
        self.logger.info(f"🔬 ResearchAgent working on: {task}")
        
        # Get relevant categories from orchestrator
        relevant_categories = getattr(self, 'relevant_categories', ['cs.LG', 'stat.ML'])  # fallback
        categories_str = ' OR '.join(f'cat:{cat}' for cat in relevant_categories)
        
        # Give the agent complete autonomy to decide how to use tools
        research_prompt = f"""
        You are an autonomous research agent. Your task is to research: {task}
        
        You have access to the following tools:
        - arxiv_search(query, limit): Search academic papers on ArXiv
        
        IMPORTANT: ArXiv Search Strategy - Use flexible approaches for best results:
        
        **General Strategy:** Use broader, concept-based searches rather than exact phrases:
        - Identify core concepts from your topic
        - Use related terms and synonyms rather than exact phrases
        - Focus on outcomes, effects, factors rather than exact terminology
        - Consider cross-disciplinary angles (economics, health, education, etc.)
        
        **Search Syntax:**
        - Use AND/OR for concepts: "[concept A]" AND "[concept B]" AND ("[related term 1]" OR "[related term 2]")
        - Use parentheses for grouping: ("[broader term]" OR "[specific term]") AND "[main concept]"
        - Use broader abstract searches: abs:"[core concept]" AND abs:"[related concept]"
        - Include category filters: {categories_str}
        
        **Search Approach:**
        1. **Start broad**: Use core concepts and related terms from YOUR topic
        2. **Include synonyms**: Think of different ways YOUR topic might be described
        3. **Cross-disciplinary**: Consider how YOUR topic might appear in different fields
        4. **Avoid overly specific**: Don't require exact phrase matches in titles
        
        Replace all bracketed placeholders with actual terms from your specific research topic.
        
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
            # Get relevant categories from orchestrator
            relevant_categories = getattr(self, 'relevant_categories', ['cs.LG', 'stat.ML'])  # fallback
            categories_str = ' OR '.join(f'cat:{cat}' for cat in relevant_categories)
            
            # Start with an initial search
            self.logger.info("🔍 Starting autonomous research execution...")
            
            # Generate initial search queries using orchestrator guidance
            self.logger.info("🎯 Generating search queries using orchestrator strategy...")
            
            # Use orchestrator-provided configuration if available
            search_strategy = getattr(self, 'search_strategy', 'Foundational Literature')
            search_terms = getattr(self, 'search_terms', task.split()[:3])
            query_patterns = getattr(self, 'query_patterns', [])
            
            self.logger.info(f"🎯 Search Strategy: {search_strategy}")
            self.logger.info(f"🎯 Search Terms: {search_terms}")
            
            # Generate diverse initial queries based on strategy
            initial_queries = []
            
            # Use orchestrator-provided query patterns first
            if query_patterns:
                initial_queries.extend(query_patterns[:2])  # Use first 2 provided patterns
            
            # Strategy-specific query generation
            if search_strategy == "Recent Advances":
                # Focus on recent papers with temporal filters
                recent_query = f"({' OR '.join([f'\"{term}\"' for term in search_terms[:2]])}) AND (2020 OR 2021 OR 2022 OR 2023 OR 2024)"
                initial_queries.append(recent_query)
            elif search_strategy == "Foundational Literature":
                # Focus on comprehensive coverage
                foundational_query = f"({' OR '.join([f'\"{term}\"' for term in search_terms])})"
                initial_queries.append(foundational_query)
            elif search_strategy == "Cross-Disciplinary":
                # Broader terms for cross-disciplinary search
                cross_query = f"({search_terms[0]} AND {search_terms[1]})" if len(search_terms) >= 2 else f'"{search_terms[0]}"'
                initial_queries.append(cross_query)
            elif search_strategy == "Method-Specific":
                # Focus on specific methods/algorithms
                method_query = f"({' AND '.join([f'\"{term}\"' for term in search_terms[:2]])})"
                initial_queries.append(method_query)
            elif search_strategy == "Application-Focused":
                # Focus on applications and case studies
                app_query = f"({search_terms[0]} AND (application OR applications OR case))"
                initial_queries.append(app_query)
            elif search_strategy == "Theoretical":
                # Focus on theoretical foundations
                theory_query = f"({search_terms[0]} AND (theory OR theoretical OR mathematical OR analysis))"
                initial_queries.append(theory_query)
            
            # Fallback: Generate from task if no strategy-specific queries
            if not initial_queries:
                task_words = task.lower().split()
                core_concepts = [w for w in task_words if len(w) > 3 and w not in ['with', 'using', 'for', 'and', 'the', 'that', 'this']][:3]
                if len(core_concepts) >= 2:
                    fallback_query = f'"{core_concepts[0]}" OR "{core_concepts[1]}"'
                    initial_queries.append(fallback_query)
                else:
                    initial_queries.append(f'"{task.split()[0]}"')
            
            # Remove duplicates and empty queries
            initial_queries = list(set([q for q in initial_queries if q and len(q) > 5]))
            
            self.logger.info(f"🎯 Generated {len(initial_queries)} strategy-specific queries:")
            for i, query in enumerate(initial_queries, 1):
                self.logger.info(f"  {i}. {query}")
            
            # Execute the first initial search
            primary_query = initial_queries[0] if initial_queries else task
            self.primary_query = primary_query  # Store for later reference
            results = await self.call_tool("arxiv_search", query=primary_query, limit=100)
            
            # Store remaining queries for progressive search
            self.remaining_initial_queries = initial_queries[1:] if len(initial_queries) > 1 else []
            
            # Evaluate papers and add relevant ones to memory
            search_results = results if isinstance(results, list) else [results]
            papers_found = len([r for r in search_results if r is not None])
            
            newly_relevant = self._evaluate_papers_relevance(task, search_results)
            self.logger.info(f"📊 Search results: {papers_found} total, {len(newly_relevant)} relevant")
            
            # Simple heuristic-based search termination logic
            all_results = search_results.copy()
            relevant_count = len(self.relevant_papers)
            total_searches = getattr(self, 'search_count', 0) + 1
            self.search_count = total_searches
            
            # Calculate relevance rate for this search
            last_search_relevance = len(newly_relevant) / max(papers_found, 1) if papers_found > 0 else 0
            
            # Simple decision criteria
            continue_searching = (
                relevant_count < 12 and  # Need more papers (target: 8-15)
                total_searches < 5 and   # Haven't exhausted search attempts
                (last_search_relevance > 0.3 or relevant_count < 3)  # Either finding good results or need minimum
            )
            
            # Log the decision logic
            self.logger.info(f"🤔 Search Decision Logic:")
            self.logger.info(f"  - Relevant papers found: {relevant_count}/12 target")
            self.logger.info(f"  - Total searches done: {total_searches}/5 max")
            self.logger.info(f"  - Last search relevance: {last_search_relevance:.2%}")
            self.logger.info(f"  - Decision: {'CONTINUE' if continue_searching else 'FINALIZE'}")
            
            # Progressive search refinement - generate next searches based on what worked
            next_searches = []
            if continue_searching:
                self.logger.info("🔄 Generating progressive search refinement...")
                
                # Extract keywords from successful papers for query refinement
                successful_keywords = []
                if self.relevant_papers:
                    for paper in self.relevant_papers[-3:]:  # Use last 3 relevant papers
                        if paper.get('title'):
                            # Extract key terms from titles (simple approach)
                            title_words = paper['title'].lower().split()
                            # Look for technical terms (longer words, avoid common words)
                            keywords = [w for w in title_words if len(w) > 4 and w not in ['paper', 'study', 'analysis', 'method', 'approach']]
                            successful_keywords.extend(keywords[:3])  # Take first 3 keywords
                
                # Generate progressive search queries
                progressive_queries = []
                
                # Strategy 1: Use remaining initial queries first
                if hasattr(self, 'remaining_initial_queries') and self.remaining_initial_queries:
                    progressive_queries.extend(self.remaining_initial_queries)
                    self.remaining_initial_queries = []  # Clear after use
                
                # Strategy 2: Use successful keywords from found papers
                if successful_keywords and len(successful_keywords) >= 2:
                    keyword_query = f'"{successful_keywords[0]}" OR "{successful_keywords[1]}"'
                    if len(successful_keywords) > 2:
                        keyword_query += f' OR "{successful_keywords[2]}"'
                    progressive_queries.append(keyword_query)
                
                # Strategy 3: Try different category combinations
                if total_searches >= 2:
                    # Expand to related categories
                    expanded_categories = relevant_categories + ['cs.AI', 'cs.CL', 'stat.AP']
                    category_query = f"({' OR '.join(task.split()[:2])}) AND ({' OR '.join([f'cat:{cat}' for cat in expanded_categories[:3]])})"
                    progressive_queries.append(category_query)
                
                # Strategy 4: Fallback to very broad search
                if total_searches >= 3:
                    broad_terms = task.split()[:2]  # Take first two words from task
                    fallback_query = f"{broad_terms[0]} AND {broad_terms[1] if len(broad_terms) > 1 else broad_terms[0]}"
                    progressive_queries.append(fallback_query)
                
                # Select up to 3 queries, prioritizing based on search history
                next_searches = progressive_queries[:3]
                
                if next_searches:
                    self.logger.info(f"🔍 Progressive search queries: {next_searches}")
                else:
                    self.logger.warning("🔄 No progressive queries generated, skipping additional searches...")
                
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
                1. {self.primary_query}
                {chr(10).join(f"{i+2}. {q}" for i, q in enumerate(next_searches))}
                
                All searches returned 0 papers. This suggests the search terms might be too specific,
                or the topic might be expressed differently in academic literature.
                
                Suggest ONE final, broader search strategy using different terminology.
                Think about:
                - What are the core concepts expressed in simpler, more general terms?
                - What broader academic fields might study this topic?
                - What are the effects, outcomes, or factors related to this topic?
                - How might this topic appear in interdisciplinary research?
                - What fundamental human behaviors or social phenomena does this relate to?
                
                Use proper arXiv syntax with terms relevant to YOUR topic:
                - Use quotes for exact phrases: "[exact phrase from task]"
                - Use AND/OR for logic: "[concept A]" AND "[concept B]"
                - Use parentheses for grouping: ("[method X]" OR "[method Y]") AND "[main topic]"
                - Use field searches: ti:"[title keyword]" (title), abs:"[abstract keyword]" (abstract)
                - Use category filters: {categories_str}
                
                Generate a search query using the actual concepts from your task.
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
        
        # Get relevant categories from orchestrator
        relevant_categories = getattr(self, 'relevant_categories', ['cs.LG', 'stat.ML'])  # fallback
        categories_str = ' OR '.join(f'cat:{cat}' for cat in relevant_categories)
        
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
            
            Use proper arXiv syntax with terms relevant to YOUR analysis task:
            - Use quotes for exact phrases: "[exact phrase from task]"
            - Use AND/OR for logic: "[concept A]" AND "[concept B]"
            - Use parentheses for grouping: ("[method X]" OR "[method Y]") AND "[main topic]"
            - Use field searches: ti:"[title keyword]" (title), abs:"[abstract keyword]" (abstract)
            - Use category filters: {categories_str}
            
            Generate a search query using actual concepts from your analysis task.
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
                
                Use proper arXiv syntax with terms relevant to YOUR task:
                - Use quotes for exact phrases: "[exact phrase from task]"
                - Use AND/OR for logic: "[concept A]" AND "[concept B]"
                - Use parentheses for grouping: ("[method X]" OR "[method Y]") AND "[main topic]"
                - Use field searches: ti:"[title keyword]" (title), abs:"[abstract keyword]" (abstract)
                - Use category filters: {categories_str}
                
                Generate search queries using actual concepts from your analysis task.
                
                IMPORTANT: Provide ONLY comma-separated search queries, no explanations or formatting.
                Example format: query1, query2, query3
                
                Comma-separated search queries:
                """
                
                alt_queries_response = self.llm.generate(alt_search_prompt).strip()
                
                # Try to extract just the queries from the response
                import re
                # Look for the last line that contains comma-separated queries
                lines = alt_queries_response.split('\n')
                query_line = alt_queries_response
                
                # Find lines that look like comma-separated queries (no markdown, bullets, etc.)
                for line in reversed(lines):
                    line = line.strip()
                    if ',' in line and not line.startswith(('*', '-', '#', '```')) and len(line) < 300:
                        query_line = line
                        break
                
                alt_queries = [q.strip().strip('"').strip('`') for q in query_line.split(",")]
                alt_queries = [q for q in alt_queries if q and len(q) > 10 and ('AND' in q or 'OR' in q or '"' in q)]
                
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