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
    
    async def _plan_agents(self, task: str) -> List[Dict]:
        """Determine what agents to spawn for the given task."""
        self.logger.info("🧠 Planning agent deployment...")
        
        planning_prompt = f"""
        You are an orchestrator planning how to research this task: {task}
        
        Available agent types:
        - ResearchAgent: Autonomous research with paper search and analysis
        - AnalysisAgent: Deep analysis and synthesis of findings
        
        TASK COMPLEXITY ASSESSMENT:
        Analyze the task and determine how many agents are needed:
        
        **1 Agent (Simple tasks):**
        - Single, well-defined concept to research
        - Straightforward literature review
        - Basic comparison of 2-3 methods
        Example: "What is active learning?"
        
        **2 Agents (Moderate complexity):**
        - Multi-faceted topic requiring research + analysis
        - Comparison across different domains/approaches
        - Need both breadth of research and depth of analysis
        Example: "Compare machine learning optimization methods"
        
        **3+ Agents (Complex tasks):**
        - Broad topic with multiple distinct subtopics that can be researched in parallel
        - Topic spans multiple domains/disciplines
        - Need specialized research on different aspects simultaneously
        - Complex synthesis required across domains
        Examples: 
        - "Alternative measures for variable dependence across statistics, ML, and information theory"
        - "Time series forecasting methods across economics, neural networks, and statistical modeling"
        - "Machine learning optimization techniques: gradient-based, evolutionary, and Bayesian approaches"
        
        DECISION PROCESS:
        1. Is this a simple, single-concept question? → 1 agent
        2. Does it need both research and specialized analysis? → 2 agents  
        3. Are there multiple distinct subtopics that can be researched in parallel? → 3+ agents
        4. Does the topic span multiple domains/disciplines? → 3+ agents
        5. Would parallel specialized research improve coverage? → 3+ agents
        
        Based on your analysis of "{task}", determine the optimal number and type of agents.
        
        Respond with ONLY a JSON list. Examples for different complexities:
        
        1 Agent example:
        ["ResearchAgent"]
        
        2 Agent example:
        ["ResearchAgent", "AnalysisAgent"]
        
        3+ Agent example:
        [
            {{"type": "ResearchAgent", "focus": "statistical dependency measures"}},
            {{"type": "ResearchAgent", "focus": "machine learning correlation methods"}},
            {{"type": "ResearchAgent", "focus": "information-theoretic measures"}},
            {{"type": "AnalysisAgent", "focus": "comparative analysis across all approaches"}}
        ]
        
        IMPORTANT: 
        - Return ONLY the JSON array, no explanation text
        - DO NOT default to 2 agents - analyze the task complexity first
        - For broad topics covering multiple domains: USE 3+ AGENTS with parallel research
        - Each ResearchAgent should focus on a distinct subtopic/domain
        - Complex tasks benefit from parallel specialized research
        
        TASK ANALYSIS FOR "{task}":
        - How many distinct subtopics or domains does this cover?
        - Would parallel research on different aspects improve coverage?
        - Is this broad enough to warrant multiple specialized researchers?
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
                    agent_focus = task  # Use main task as focus
                elif isinstance(agent_spec, dict):
                    # Object format: {"type": "ResearchAgent", "focus": "..."}
                    agent_type = agent_spec.get('type', 'ResearchAgent')
                    agent_focus = agent_spec.get('focus', agent_spec.get('description', task))
                else:
                    # Fallback for unexpected formats
                    agent_type = 'ResearchAgent'
                    agent_focus = task
                
                normalized_spec = {"type": agent_type, "focus": agent_focus}
                normalized_plan.append(normalized_spec)
                
                self.logger.info(f"  Agent {i}: {agent_type} - {agent_focus}")
            
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
                # Set categories on the agent
                agent.relevant_categories = categories
                agents.append({"agent": agent, "focus": focus, "id": f"agent_{i+1}"})
                self.logger.info(f"✅ Spawned {agent_type} with focus: {focus}")
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