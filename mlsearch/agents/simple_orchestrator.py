from __future__ import annotations
import logging
import asyncio
from typing import Dict, Callable, List
from core.llmclient import LLMClient
from agents.agent import Agent, agent_registry


class SimpleOrchestrator:
    """Simplified orchestrator that spawns agents and performs intelligent merging."""
    
    def __init__(self, tools: Dict[str, Callable]) -> None:
        self.tools = tools
        self.llm = LLMClient(model_name="gpt-4o-mini")
        self.logger = logging.getLogger(__name__)
        self.agents: List[Agent] = []
    
    async def run(self, user_task: str) -> str:
        """Run the orchestrator with minimal coordination and intelligent merging."""
        self.logger.info(f"🚀 SimpleOrchestrator starting task: {user_task}")
        
        # Determine relevant arXiv categories for this task
        relevant_categories = await self._select_categories(user_task)
        
        # Determine what agents to spawn
        agent_plan = await self._plan_agents(user_task)
        
        # Spawn and run agents with category guidance
        agent_outputs = await self._run_agents(agent_plan, user_task, relevant_categories)
        
        # Intelligent merging with quality assessment
        final_result = await self._intelligent_merge(user_task, agent_outputs)
        
        self.logger.info("✅ SimpleOrchestrator completed successfully")
        return final_result
    
    async def _select_categories(self, task: str) -> List[str]:
        """Determine the most relevant arXiv categories for the given task."""
        self.logger.info("🎯 Selecting relevant arXiv categories...")
        
        category_prompt = f"""
        Task: {task}
        
        Select the most relevant arXiv categories for this research task. Consider which academic disciplines would likely contain papers relevant to this topic.
        
        Available arXiv categories:
        
        **Computer Science (cs):**
        - cs.AI (Artificial Intelligence)
        - cs.LG (Machine Learning)
        - cs.CL (Computation and Language)
        - cs.CV (Computer Vision)
        - cs.CR (Cryptography and Security)
        - cs.DB (Databases)
        - cs.DS (Data Structures and Algorithms)
        - cs.IR (Information Retrieval)
        - cs.NE (Neural and Evolutionary Computing)
        - cs.RO (Robotics)
        - cs.SY (Systems and Control)
        
        **Mathematics (math):**
        - math.ST (Statistics Theory)
        - math.PR (Probability)
        - math.OC (Optimization and Control)
        - math.NA (Numerical Analysis)
        - math.CO (Combinatorics)
        - math.AT (Algebraic Topology)
        - math.DS (Dynamical Systems)
        - math.NT (Number Theory)
        
        **Statistics (stat):**
        - stat.ML (Machine Learning)
        - stat.ME (Methodology)
        - stat.TH (Statistics Theory)
        - stat.AP (Applications)
        - stat.CO (Computation)
        
        **Physics (physics):**
        - quant-ph (Quantum Physics)
        - cond-mat (Condensed Matter)
        - physics.data-an (Data Analysis, Statistics and Probability)
        - physics.comp-ph (Computational Physics)
        - physics.soc-ph (Physics and Society)
        
        **Economics (econ):**
        - econ.EM (Econometrics)
        - econ.TH (Theoretical Economics)
        - econ.GN (General Economics)
        
        **Quantitative Finance (q-fin):**
        - q-fin.ST (Statistical Finance)
        - q-fin.RM (Risk Management)
        - q-fin.PM (Portfolio Management)
        - q-fin.CP (Computational Finance)
        
        **Quantitative Biology (q-bio):**
        - q-bio.QM (Quantitative Methods)
        - q-bio.GN (Genomics)
        - q-bio.NC (Neurons and Cognition) [use for psychology topics]
        
        **Electrical Engineering (eess):**
        - eess.SP (Signal Processing)
        - eess.SY (Systems and Control)
        - eess.IV (Image and Video Processing)
        - eess.AS (Audio and Speech Processing)
        
        IMPORTANT NOTES:
        - For psychology topics, use q-bio.NC (Neurons and Cognition)
        - For sociology/social topics, use physics.soc-ph (Physics and Society)
        - For economics topics, use econ.GN, econ.EM, or econ.TH
        - Only select from the exact category codes listed above
        
        Based on the task "{task}", select 2-5 most relevant categories. Consider interdisciplinary topics.
        
        Respond with only the category codes, comma-separated:
        Example: cs.LG, stat.ML, math.OC
        """
        
        response = self.llm.generate(category_prompt).strip()
        
        # Parse categories
        categories = [cat.strip() for cat in response.split(",")]
        
        # Validate against known arXiv categories
        valid_categories = {
            'cs.AI', 'cs.LG', 'cs.CL', 'cs.CV', 'cs.CR', 'cs.DB', 'cs.DS', 'cs.IR', 'cs.NE', 'cs.RO', 'cs.SY',
            'math.ST', 'math.PR', 'math.OC', 'math.NA', 'math.CO', 'math.AT', 'math.DS', 'math.NT',
            'stat.ML', 'stat.ME', 'stat.TH', 'stat.AP', 'stat.CO',
            'quant-ph', 'cond-mat', 'physics.data-an', 'physics.comp-ph', 'physics.soc-ph',
            'econ.EM', 'econ.TH', 'econ.GN',
            'q-fin.ST', 'q-fin.RM', 'q-fin.PM', 'q-fin.CP',
            'q-bio.QM', 'q-bio.GN', 'q-bio.NC',
            'eess.SP', 'eess.SY', 'eess.IV', 'eess.AS'
        }
        
        # Filter to only valid categories
        valid_selected = [cat for cat in categories if cat in valid_categories]
        
        # Map common invalid categories to valid ones
        category_mapping = {
            'psychology': 'q-bio.NC',
            'sociology': 'physics.soc-ph', 
            'social': 'physics.soc-ph',
            'economics': 'econ.GN',
            'finance': 'q-fin.ST',
            'biology': 'q-bio.QM',
            'physics': 'physics.soc-ph'
        }
        
        # Map any invalid categories
        for cat in categories:
            if cat not in valid_categories:
                for invalid, valid in category_mapping.items():
                    if invalid in cat.lower() and valid not in valid_selected:
                        valid_selected.append(valid)
                        break
        
        categories = valid_selected
        
        if not categories:
            # Fallback to broad ML/Stats categories
            categories = ["cs.LG", "stat.ML"]
            self.logger.warning("🔄 Category selection failed, using default ML/Stats categories")
        
        self.logger.info(f"🎯 Selected categories: {', '.join(categories)}")
        return categories
    
    def _get_search_strategy_templates(self) -> Dict[str, Dict]:
        """Get predefined search strategy templates for consistent agent deployment."""
        return {
            "Foundational Literature": {
                "description": "Core papers, surveys, seminal works (older, highly-cited)",
                "query_patterns": [
                    '"{concept}" AND (survey OR review OR overview)',
                    '"{concept}" AND (introduction OR tutorial OR foundations)',
                    '"{concept}" AND cat:{category}'
                ],
                "temporal_focus": "all_time",
                "paper_types": ["surveys", "reviews", "foundational works"]
            },
            "Recent Advances": {
                "description": "Latest developments, cutting-edge methods (2020+)",
                "query_patterns": [
                    '"{concept}" AND (2020 OR 2021 OR 2022 OR 2023 OR 2024)',
                    '"{concept}" AND (novel OR new OR recent OR latest)',
                    '"{concept}" AND cat:{category}'
                ],
                "temporal_focus": "2020_onwards",
                "paper_types": ["conference papers", "recent methods", "state-of-the-art"]
            },
            "Cross-Disciplinary": {
                "description": "Papers from related fields applying to this domain",
                "query_patterns": [
                    '"{concept}" AND ({related_field} OR {application_domain})',
                    '"{concept}" AND (interdisciplinary OR cross-domain)',
                    '"{concept}" AND cat:{alt_category}'
                ],
                "temporal_focus": "all_time",
                "paper_types": ["interdisciplinary work", "cross-domain applications"]
            },
            "Method-Specific": {
                "description": "Deep dive into specific algorithms/techniques",
                "query_patterns": [
                    '"{method}" AND (algorithm OR technique OR method)',
                    '"{method}" AND (implementation OR optimization)',
                    '"{method}" AND cat:{category}'
                ],
                "temporal_focus": "all_time",
                "paper_types": ["methodological papers", "algorithmic improvements"]
            },
            "Application-Focused": {
                "description": "Real-world applications and case studies",
                "query_patterns": [
                    '"{concept}" AND (application OR applications OR case)',
                    '"{concept}" AND (real-world OR practical OR industry)',
                    '"{concept}" AND (dataset OR benchmark OR evaluation)'
                ],
                "temporal_focus": "recent_preferred",
                "paper_types": ["application papers", "case studies", "empirical work"]
            },
            "Theoretical": {
                "description": "Mathematical foundations, theoretical analysis",
                "query_patterns": [
                    '"{concept}" AND (theory OR theoretical OR mathematical)',
                    '"{concept}" AND (analysis OR proof OR convergence)',
                    '"{concept}" AND (bounds OR complexity OR guarantees)'
                ],
                "temporal_focus": "all_time",
                "paper_types": ["theoretical papers", "mathematical analysis", "proofs"]
            }
        }
    
    async def _plan_agents(self, task: str) -> List[Dict]:
        """Determine what agents to spawn for the given task."""
        self.logger.info("🧠 Planning agent deployment...")
        
        planning_prompt = f"""
        You are a research orchestrator planning search strategies for: {task}
        
        Your job is to create specialized search agents with DISTINCT, NON-OVERLAPPING search strategies to maximize paper discovery.
        
        AVAILABLE SEARCH STRATEGIES:
        1. **Foundational Literature**: Core papers, surveys, seminal works (older, highly-cited)
           - Query patterns: survey/review papers, foundational works, introductory papers
           - Focus: Comprehensive coverage of established knowledge
        2. **Recent Advances**: Latest developments, cutting-edge methods (2020+)
           - Query patterns: temporal filters (2020+), novel/new/recent keywords
           - Focus: State-of-the-art methods and recent breakthroughs
        3. **Cross-Disciplinary**: Papers from related fields applying to this domain
           - Query patterns: interdisciplinary keywords, alternative categories
           - Focus: Applications from other domains, cross-field perspectives
        4. **Method-Specific**: Deep dive into specific algorithms/techniques
           - Query patterns: algorithm/technique/method keywords, implementation details
           - Focus: Detailed methodological papers and algorithmic improvements
        5. **Application-Focused**: Real-world applications and case studies
           - Query patterns: application/practical/industry keywords, datasets/benchmarks
           - Focus: Empirical work, case studies, practical implementations
        6. **Theoretical**: Mathematical foundations, theoretical analysis
           - Query patterns: theory/mathematical/analysis keywords, proofs/bounds/complexity
           - Focus: Mathematical foundations, theoretical guarantees, formal analysis
        
        ARXIV CATEGORY SPECIALIZATION:
        - cs.LG: Machine Learning
        - cs.AI: Artificial Intelligence  
        - stat.ML: Statistics - Machine Learning
        - cs.CL: Computational Linguistics
        - cs.CV: Computer Vision
        - cs.IR: Information Retrieval
        - stat.AP: Statistics - Applications
        - cs.DS: Data Structures and Algorithms
        - econ.EM: Econometrics
        - q-fin.ST: Statistical Finance
        
        TASK COMPLEXITY ASSESSMENT:
        
        **1 Agent (Simple, focused queries):**
        - Single well-defined concept
        - Basic "what is X?" questions
        - Limited scope literature review
        
        **2-3 Agents (Moderate complexity):**
        - Comparison between 2-3 methods/approaches
        - Topic spanning 2-3 domains
        - Need both breadth and depth
        
        **3-4 Agents (Complex, multi-faceted):**
        - Broad topics with multiple distinct aspects
        - Cross-disciplinary research needed
        - Multiple methodological approaches
        - Need comprehensive coverage
        
        FOR EACH AGENT, SPECIFY:
        1. **search_strategy**: One of the 6 strategies above
        2. **arxiv_categories**: Specific ArXiv categories (2-3 max per agent)
        3. **search_terms**: Specific keywords/phrases for this agent's focus
        4. **focus_description**: Clear, actionable research focus
        5. **query_patterns**: Specific query patterns to use
        
        ANALYZE "{task}" and determine optimal agent deployment:
        
        - Core concepts: [identify 2-4 key concepts]
        - Research domains: [identify relevant fields]
        - Methodological approaches: [identify different methods/techniques]
        - Time sensitivity: [recent advances vs foundational work needed]
        
        Respond with ONLY a JSON array. Each ResearchAgent should have:
        {{
            "type": "ResearchAgent",
            "search_strategy": "Recent Advances",
            "arxiv_categories": ["cs.LG", "stat.ML"],
            "search_terms": ["deep learning", "neural networks", "transformer"],
            "focus_description": "Search for recent advances in deep learning architectures from 2020-2024",
            "query_patterns": ["(\\"deep learning\\" OR \\"neural networks\\") AND (2020 OR 2021 OR 2022 OR 2023 OR 2024)"]
        }}
        
        CRITICAL REQUIREMENTS:
        - Each agent must have DIFFERENT arxiv_categories (no overlap)
        - Each agent must have DIFFERENT search_terms (minimal overlap)
        - Each agent must have DIFFERENT search_strategy
        - Focus on COMPLEMENTARY rather than overlapping research
        - Use specific, actionable search instructions
        """
        
        response = self.llm.generate(planning_prompt)
        
        try:
            # Clean JSON response
            import json
            import re
            response = response.strip()
            self.logger.debug(f"🔍 Raw LLM response: {response}")
            
            # Extract JSON from response - handle various formats
            json_text = response
            
            # Look for JSON code blocks
            json_match = re.search(r'```json\s*(\[.*?\])\s*```', response, re.DOTALL)
            if json_match:
                json_text = json_match.group(1)
            else:
                # Look for JSON arrays anywhere in the response
                json_match = re.search(r'(\[.*?\])', response, re.DOTALL)
                if json_match:
                    json_text = json_match.group(1)
                else:
                    # Fallback: look for the last occurrence of [ to end of response
                    last_bracket = response.rfind('[')
                    if last_bracket != -1:
                        json_text = response[last_bracket:].strip()
            
            json_text = json_text.strip()
            self.logger.debug(f"🔍 Extracted JSON: {json_text}")
            
            agent_plan = json.loads(json_text)
            self.logger.info(f"📋 Planned {len(agent_plan)} agents:")
            
            # Validate and normalize agent specifications
            normalized_plan = []
            for i, agent_spec in enumerate(agent_plan, 1):
                # Handle different possible JSON structures
                if isinstance(agent_spec, str):
                    # Simple string format: ["ResearchAgent", "AnalysisAgent"]
                    agent_type = agent_spec
                    normalized_spec = {
                        "type": agent_type,
                        "focus": task,
                        "search_strategy": "Foundational Literature",
                        "arxiv_categories": ["cs.LG", "stat.ML"],
                        "search_terms": task.split()[:3],
                        "focus_description": f"General research on {task}",
                        "query_patterns": [f'"{task.split()[0]}" OR "{task.split()[1]}"' if len(task.split()) > 1 else f'"{task.split()[0]}"']
                    }
                elif isinstance(agent_spec, dict):
                    # Enhanced object format with search strategy details
                    agent_type = agent_spec.get('type', 'ResearchAgent')
                    normalized_spec = {
                        "type": agent_type,
                        "focus": agent_spec.get('focus', agent_spec.get('focus_description', task)),
                        "search_strategy": agent_spec.get('search_strategy', 'Foundational Literature'),
                        "arxiv_categories": agent_spec.get('arxiv_categories', ["cs.LG", "stat.ML"]),
                        "search_terms": agent_spec.get('search_terms', task.split()[:3]),
                        "focus_description": agent_spec.get('focus_description', agent_spec.get('focus', task)),
                        "query_patterns": agent_spec.get('query_patterns', [f'"{task.split()[0]}"'])
                    }
                else:
                    # Fallback for unexpected formats
                    normalized_spec = {
                        "type": 'ResearchAgent',
                        "focus": task,
                        "search_strategy": "Foundational Literature",
                        "arxiv_categories": ["cs.LG", "stat.ML"],
                        "search_terms": task.split()[:3],
                        "focus_description": f"General research on {task}",
                        "query_patterns": [f'"{task.split()[0]}"']
                    }
                
                normalized_plan.append(normalized_spec)
                
                self.logger.info(f"  Agent {i}: {agent_type} - {normalized_spec['search_strategy']}")
                self.logger.info(f"    Categories: {normalized_spec['arxiv_categories']}")
                self.logger.info(f"    Search Terms: {normalized_spec['search_terms']}")
            
            return normalized_plan
            
        except Exception as e:
            self.logger.error(f"❌ Error parsing agent plan: {e}")
            self.logger.error(f"❌ Failed JSON extraction: {json_text if 'json_text' in locals() else 'N/A'}")
            self.logger.error(f"❌ Full response was: {response}")
            # Fallback to simple research agent
            return [{"type": "ResearchAgent", "focus": task}]
    
    async def _run_agents(self, agent_plan: List[Dict], task: str, categories: List[str]) -> List[Dict]:
        """Spawn and run agents concurrently with category guidance."""
        self.logger.info(f"🤖 Spawning {len(agent_plan)} agents...")
        
        # Create agents
        agents = []
        for i, spec in enumerate(agent_plan):
            agent_type = spec.get("type", "ResearchAgent")
            focus = spec.get("focus", task)
            
            if agent_type in agent_registry:
                agent = agent_registry[agent_type](self.tools)
                
                # Set enhanced agent configuration
                agent.relevant_categories = spec.get("arxiv_categories", categories)
                agent.search_strategy = spec.get("search_strategy", "Foundational Literature")
                agent.search_terms = spec.get("search_terms", task.split()[:3])
                agent.focus_description = spec.get("focus_description", focus)
                agent.query_patterns = spec.get("query_patterns", [f'"{task.split()[0]}"'])
                
                agents.append({"agent": agent, "focus": focus, "id": f"agent_{i+1}"})
                self.logger.info(f"✅ Spawned {agent_type} with strategy: {agent.search_strategy}")
                self.logger.info(f"    Categories: {agent.relevant_categories}")
                self.logger.info(f"    Search Terms: {agent.search_terms}")
            else:
                self.logger.error(f"❌ Unknown agent type: {agent_type}")
        
        # Run agents concurrently
        self.logger.info("🏃 Running agents concurrently...")
        tasks = []
        for agent_info in agents:
            task_coro = agent_info["agent"].work(agent_info["focus"])
            tasks.append(task_coro)
        
        # Wait for all agents to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect outputs
        agent_outputs = []
        for i, (agent_info, result) in enumerate(zip(agents, results)):
            if isinstance(result, Exception):
                self.logger.error(f"❌ Agent {agent_info['id']} failed: {result}")
                agent_outputs.append({
                    "id": agent_info["id"],
                    "type": agent_info["agent"].__class__.__name__,
                    "focus": agent_info["focus"],
                    "output": f"Agent failed: {str(result)}",
                    "success": False
                })
            else:
                self.logger.info(f"✅ Agent {agent_info['id']} completed successfully")
                agent_outputs.append({
                    "id": agent_info["id"],
                    "type": agent_info["agent"].__class__.__name__,
                    "focus": agent_info["focus"],
                    "output": result,
                    "success": True
                })
        
        return agent_outputs
    
    async def _intelligent_merge(self, original_task: str, agent_outputs: List[Dict]) -> str:
        """Intelligently merge agent outputs with quality assessment."""
        self.logger.info("🧠 Performing intelligent merge with quality assessment...")
        
        # Prepare agent outputs for analysis
        outputs_summary = []
        successful_outputs = []
        
        for output in agent_outputs:
            if output["success"]:
                outputs_summary.append(f"Agent {output['id']} ({output['type']}) - Focus: {output['focus']}")
                successful_outputs.append(output)
            else:
                outputs_summary.append(f"Agent {output['id']} - FAILED: {output['output']}")
        
        if not successful_outputs:
            return "No agents completed successfully. Please try again."
        
        # Extract all papers from agents and rank them
        paper_ranking = self._extract_and_rank_papers(original_task, successful_outputs)
        
        return paper_ranking
    
    def _format_agent_outputs(self, agent_outputs: List[Dict]) -> str:
        """Format agent outputs for synthesis."""
        formatted = []
        
        for output in agent_outputs:
            formatted.append(f"=== {output['type']} (Focus: {output['focus']}) ===")
            formatted.append(output['output'])
            formatted.append("")  # Empty line
        
        return "\n".join(formatted)
    
    def _extract_and_rank_papers(self, original_task: str, agent_outputs: List[Dict]) -> str:
        """Extract all papers from agents and rank them by relevance."""
        import re
        
        # Extract all paper data from agent outputs
        all_papers = []
        
        for output in agent_outputs:
            agent_output = output.get('output', '')
            agent_focus = output.get('focus', '')
            agent_type = output.get('type', '')
            
            # Look for the new format: "### N. Title (Year)"
            if '## Relevant Papers Found' in agent_output:
                lines = agent_output.split('\n')
                in_papers_section = False
                current_paper = {}
                
                for line in lines:
                    line = line.strip()
                    if line == '## Relevant Papers Found':
                        in_papers_section = True
                        continue
                    elif line.startswith('##') and in_papers_section and not line.startswith('### '):
                        break
                    elif in_papers_section:
                        # Parse paper entries
                        if line.startswith('### '):
                            # Save previous paper if exists
                            if current_paper.get('title'):
                                all_papers.append(current_paper)
                            
                            # Start new paper: "### 1. Title (Year)"
                            match = re.match(r'### \d+\.\s*(.+?)\s*\((\d+)\)', line)
                            if match:
                                title, year = match.groups()
                                current_paper = {
                                    'title': title.strip(),
                                    'year': int(year) if year.isdigit() else 0,
                                    'authors': '',
                                    'pdf_url': '',
                                    'summary': '',
                                    'agent_focus': agent_focus,
                                    'agent_type': agent_type
                                }
                        elif line.startswith('**Authors:**'):
                            current_paper['authors'] = line.replace('**Authors:**', '').strip()
                        elif line.startswith('**PDF:**'):
                            current_paper['pdf_url'] = line.replace('**PDF:**', '').strip()
                        elif line.startswith('**Summary:**'):
                            current_paper['summary'] = line.replace('**Summary:**', '').strip()
                
                # Don't forget the last paper
                if current_paper.get('title'):
                    all_papers.append(current_paper)
        
        # Remove duplicates based on title similarity
        unique_papers = self._deduplicate_papers(all_papers)
        
        # Rank papers by relevance
        ranked_papers = self._rank_papers_by_relevance(original_task, unique_papers)
        
        return ranked_papers
    
    def _deduplicate_papers(self, papers: List[Dict]) -> List[Dict]:
        """Remove duplicate papers based on title similarity."""
        unique_papers = []
        seen_titles = set()
        
        for paper in papers:
            title_lower = paper['title'].lower().strip()
            # Simple deduplication - could be more sophisticated
            if title_lower not in seen_titles:
                seen_titles.add(title_lower)
                unique_papers.append(paper)
        
        return unique_papers
    
    def _rank_papers_by_relevance(self, task: str, papers: List[Dict]) -> str:
        """Use LLM to rank papers by relevance and determine top N."""
        if not papers:
            return f"# Search Results for: {task}\n\nNo relevant papers found."
        
        # Prepare paper list for ranking
        papers_for_ranking = []
        for i, paper in enumerate(papers):
            papers_for_ranking.append(f"{i+1}. {paper['title']} ({paper['year']}) - {paper['authors']}")
        
        ranking_prompt = f"""
        Task: {task}
        
        Papers found by research agents:
        {chr(10).join(papers_for_ranking)}
        
        Your job is to:
        1. Rank these papers by relevance to the task "{task}"
        2. Determine how many papers are truly most relevant (top N)
        3. Provide reasoning for why the top papers are relevant
        
        Consider:
        - How directly does the paper address the task topic?
        - Is this a key/seminal paper in the field?
        - Does it provide methods, results, or insights relevant to the task?
        - Recent papers may be more relevant for current state of field
        
        Respond with:
        TOP_N: [number of truly most relevant papers, typically 5-15]
        RANKING: [comma-separated list of paper numbers in order of relevance]
        
        For each of the top papers, provide reasoning:
        REASONING_1: [why paper ranked #1 is most relevant]
        REASONING_2: [why paper ranked #2 is relevant]
        REASONING_3: [why paper ranked #3 is relevant]
        [continue for all top N papers]
        
        Example:
        TOP_N: 3
        RANKING: 7, 2, 5
        REASONING_1: Paper 7 directly addresses MIDAS models with comprehensive methodology
        REASONING_2: Paper 2 provides foundational time series analysis techniques relevant to MIDAS
        REASONING_3: Paper 5 offers practical applications of mixed-frequency data sampling
        """
        
        ranking_response = self.llm.generate(ranking_prompt)
        
        # Parse ranking response
        top_n = 10  # default
        ranking = list(range(1, min(len(papers) + 1, 11)))  # default ranking
        reasoning_dict = {}  # Store reasoning for each rank
        
        for line in ranking_response.split('\n'):
            line = line.strip()
            if line.startswith('TOP_N:'):
                try:
                    top_n = int(line.replace('TOP_N:', '').strip())
                    top_n = min(max(top_n, 1), len(papers))  # Clamp between 1 and total papers
                except:
                    pass
            elif line.startswith('RANKING:'):
                try:
                    ranking_str = line.replace('RANKING:', '').strip()
                    ranking = [int(x.strip()) for x in ranking_str.split(',')]
                    # Filter valid indices
                    ranking = [x for x in ranking if 1 <= x <= len(papers)]
                except:
                    pass
            elif line.startswith('REASONING_'):
                # Parse reasoning: "REASONING_1: explanation here"
                try:
                    colon_pos = line.find(':')
                    if colon_pos > 0:
                        reasoning_key = line[:colon_pos].strip()
                        reasoning_text = line[colon_pos + 1:].strip()
                        # Extract rank number from REASONING_N
                        rank_num = int(reasoning_key.replace('REASONING_', ''))
                        reasoning_dict[rank_num] = reasoning_text
                except:
                    pass
        
        # Format final output
        result = [f"# Most Relevant Papers for: {task}\n"]
        result.append(f"Found {len(papers)} papers total. Showing top {min(top_n, len(ranking))} most relevant:\n")
        
        for i, paper_idx in enumerate(ranking[:top_n]):
            if paper_idx <= len(papers):
                paper = papers[paper_idx - 1]  # Convert to 0-based index
                rank_num = i + 1
                
                result.append(f"## {rank_num}. {paper['title']} ({paper['year']})")
                result.append(f"**Authors:** {paper['authors']}")
                if paper['pdf_url']:
                    result.append(f"**PDF:** {paper['pdf_url']}")
                result.append(f"**Found by:** {paper['agent_type']} (Focus: {paper['agent_focus']})")
                
                # Add reasoning if available
                if rank_num in reasoning_dict:
                    result.append(f"**Why relevant:** {reasoning_dict[rank_num]}")
                
                result.append("")
        
        result.append(f"---")
        result.append(f"Search completed by {len(self.agents)} agents")
        
        return "\n".join(result)