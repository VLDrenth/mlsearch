from core.llmclient import LLMClient
from core.types import Plan
from prompts.planner import build_planner_prompts
from agents.worker import Worker, worker_registry
import json, asyncio

class Orchestrator:
    def __init__(self, planner: LLMClient, tools: dict) -> None:
        self.planner = planner          # LLM used only for the plan
        self.tools   = tools            # {'web_search': fn, 'python': fn}
        self.pool    = {}               # Worker instance

    async def run(self, user_task: str) -> str:
        plan_json = self._ask_planner(user_task)
        plan = Plan.model_validate_json(plan_json)

        # simple topological execution (no cycles in tiny PoC)
        results: dict[str, str] = {}
        for step in plan.steps:
            await self._execute_step(step, results)

        return results[plan.steps[-1].id]

    async def _execute_step(self, step, results):

        # resolve dependencies first
        for dep in step.depends_on:
            if dep not in results:
                raise RuntimeError(f"Missing dependency {dep}")

        if step.action == "spawn":
            self.pool[step.worker] = worker_registry[step.worker](self)
            results[step.id] = f"spawned {step.worker}"

        elif step.action == "tool":
            out = await self._call_tool(step.tool, step.args)
            results[step.id] = out

        elif step.action == "merge":
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

    def _ask_planner(self, task: str) -> str:
        system_prompt, user_prompt = build_planner_prompts(task)
        self.planner.set_system_prompt(system_prompt)
        return self.planner.generate(user_prompt)

    async def _call_tool(self, name: str, args: dict) -> str:
        if name not in self.tools:
            raise KeyError(f"Unknown tool {name}")
        fn = self.tools[name]
        if asyncio.iscoroutinefunction(fn):
            return await fn(**args)
        return fn(**args)
