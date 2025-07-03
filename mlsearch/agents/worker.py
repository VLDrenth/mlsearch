from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.orchestrator import Orchestrator

class Worker:
    """Base worker class for specialized agents."""
    
    def __init__(self, orchestrator: Orchestrator) -> None:
        self.orchestrator = orchestrator
        self.output = ""
    
    def get_output(self) -> str:
        """Return the worker's output."""
        return self.output

class ResearchWorker(Worker):
    """Worker specialized for research tasks."""
    
    def __init__(self, orchestrator: Orchestrator) -> None:
        super().__init__(orchestrator)
        self.output = "Research completed"

# Registry of available worker types
worker_registry = {
    "ResearchWorker": ResearchWorker,
    "CodeWorker": ResearchWorker,  # For now, use ResearchWorker
    "WriterWorker": ResearchWorker,  # For now, use ResearchWorker
}