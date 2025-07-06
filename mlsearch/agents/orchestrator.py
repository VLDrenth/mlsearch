from core.llmclient import LLMClient
from core.types import Plan
from core.tool_registry import get_tool_registry
from prompts.planner import build_planner_prompts
from agents.worker import Worker, worker_registry
import json, asyncio, logging
from typing import Dict, Callable, Optional

class Orchestrator:
    def __init__(self, planner: LLMClient, tools: Optional[Dict[str, Callable]] = None) -> None:
        self.planner = planner          # LLM used only for the plan
        self.legacy_tools = tools or {} # Legacy tools for backward compatibility
        self.tool_registry = get_tool_registry()  # Modern tool registry
        self.pool    = {}               # Worker instance
        self.logger = logging.getLogger(__name__)
        
        # Log tool availability
        registry_tools = len(self.tool_registry.list_tools())
        legacy_tools = len(self.legacy_tools)
        self.logger.info(f"🎼 Orchestrator initialized with {max(registry_tools, legacy_tools)} tools (registry: {registry_tools}, legacy: {legacy_tools})")

    async def run(self, user_task: str) -> str:
        self.logger.info(f"🚀 Starting orchestration for task: {user_task}")
        
        self.logger.info("🧠 Analyzing task and planning research angles...")
        research_angles = await self._plan_research_angles(user_task)
        
        self.logger.info("🤖 Asking planner to generate execution plan...")
        plan_json = self._ask_planner(user_task, research_angles)
        plan = Plan.model_validate_json(plan_json)
        
        self.logger.info(f"📋 Generated plan with {len(plan.steps)} steps")
        for i, step in enumerate(plan.steps, 1):
            self.logger.info(f"  Step {i}: {step.action} - {step.id}")

        # simple topological execution (no cycles in tiny PoC)
        results: dict[str, str] = {}
        for step in plan.steps:
            await self._execute_step(step, results)

        final_result = results[plan.steps[-1].id]
        self.logger.info(f"✅ Orchestration completed successfully")
        return final_result

    async def _execute_step(self, step, results):
        self.logger.info(f"⚙️  Executing step: {step.id} ({step.action})")

        # resolve dependencies first
        for dep in step.depends_on:
            if dep not in results:
                raise RuntimeError(f"Missing dependency {dep}")

        if step.action == "spawn":
            self.logger.info(f"🏗️  Spawning worker: {step.worker}")
            
            # Determine worker type - if step.worker contains underscore, extract type before underscore
            if "_" in step.worker:
                worker_type = step.worker.split("_")[0]
            else:
                worker_type = step.worker
                
            if worker_type not in worker_registry:
                self.logger.error(f"❌ Unknown worker type: {worker_type}")
                results[step.id] = f"Error: Unknown worker type {worker_type}"
                return
                
            # Use the full step.worker as the unique ID in the pool
            self.pool[step.worker] = worker_registry[worker_type](self)
            results[step.id] = f"spawned {step.worker}"
            self.logger.info(f"✅ Worker {step.worker} spawned successfully")

        elif step.action == "tool":
            self.logger.info(f"🔧 Calling tool: {step.tool} with args: {step.args}")
            out = await self._call_tool(step.tool, step.args)
            results[step.id] = out
            result_preview = str(out)[:100] + "..." if len(str(out)) > 100 else str(out)
            self.logger.info(f"✅ Tool {step.tool} completed. Result: {result_preview}")

        elif step.action == "work":
            self.logger.info(f"🔨 Executing work step: {step.worker} on task: {step.task}")
            if step.worker and step.worker in self.pool:
                worker = self.pool[step.worker]
                work_result = await worker.work(step.task or "No task specified")
                results[step.id] = work_result
                self.logger.info(f"✅ Work completed by {step.worker}")
            else:
                self.logger.error(f"❌ Worker {step.worker} not found in pool")
                results[step.id] = f"Error: Worker {step.worker} not available"

        elif step.action == "merge":
            self.logger.info(f"🔄 Merging results from dependencies: {step.depends_on}")
            # Collect results from dependencies
            dep_results = []
            for dep in step.depends_on:
                if dep in results:
                    dep_results.append(results[dep])
            
            # Collect outputs from spawned workers
            worker_outputs = []
            for worker_id, worker in self.pool.items():
                if worker_id in step.depends_on:
                    worker_outputs.append(worker.get_output())
            
            # Combine all results
            all_results = dep_results + worker_outputs
            if all_results:
                # Join non-empty results with newlines
                filtered_results = [str(result) for result in all_results if result and str(result).strip()]
                results[step.id] = "\n".join(filtered_results) if filtered_results else "No results to merge"
            else:
                results[step.id] = "No results to merge"
            
            self.logger.info(f"✅ Merged {len(all_results)} results")

    async def _plan_research_angles(self, task: str) -> list[str]:
        """Plan specific research angles to minimize redundancy between workers."""
        self.logger.info("🎯 Breaking down task into specific research angles...")
        
        from core.llmclient import LLMClient
        planning_llm = LLMClient(model_type="planning")
        
        planning_prompt = f"""
        You are a research orchestrator. Break down this research task into 3-4 distinct, non-overlapping research angles that different workers can explore in parallel.

        Task: {task}
        
        Each angle should:
        1. Focus on a specific aspect of the topic
        2. Be clearly distinct from other angles (minimize overlap)
        3. Be substantial enough to warrant dedicated research
        4. Together with other angles, provide comprehensive coverage
        
        Respond with a JSON list of research angles:
        {{
            "angles": [
                "Specific research angle 1: Focus on X aspect",
                "Specific research angle 2: Focus on Y aspect", 
                "Specific research angle 3: Focus on Z aspect"
            ]
        }}
        
        Example for "transformer attention mechanisms":
        {{
            "angles": [
                "Recent attention mechanism innovations (2023-2025): New architectures, efficiency improvements, novel attention patterns",
                "Foundational attention theory and mathematical frameworks: Core principles, theoretical analysis, mathematical foundations",
                "Real-world applications and performance benchmarks: Practical implementations, performance comparisons"
            ]
        }}
        """
        
        response = planning_llm.generate(planning_prompt)
        
        try:
            # Clean the response
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            
            import json
            angles_data = json.loads(response)
            angles = angles_data.get("angles", [])
            
            self.logger.info(f"📐 Planned {len(angles)} research angles:")
            for i, angle in enumerate(angles, 1):
                self.logger.info(f"  Angle {i}: {angle}")
            
            return angles
            
        except Exception as e:
            self.logger.error(f"❌ Error parsing research angles: {e}")
            # Fallback to generic angles
            return [
                f"Recent developments and innovations in {task}",
                f"Foundational concepts and theory related to {task}",
                f"Practical applications and performance aspects of {task}"
            ]

    def _ask_planner(self, task: str, research_angles: list[str]) -> str:
        system_prompt, user_prompt = build_planner_prompts(task, research_angles)
        self.planner.set_system_prompt(system_prompt)
        return self.planner.generate(user_prompt)

    async def _call_tool(self, name: str, args: dict) -> str:
        # Try tool registry first
        if self.tool_registry.get_tool(name):
            return await self.tool_registry.execute_tool(name, args)
        
        # Fallback to legacy tools
        if name not in self.legacy_tools:
            available_tools = list(self.tool_registry.list_tools()) + list(self.legacy_tools.keys())
            raise KeyError(f"Unknown tool {name}. Available tools: {available_tools}")
        
        fn = self.legacy_tools[name]
        if asyncio.iscoroutinefunction(fn):
            return await fn(**args)
        return fn(**args)
