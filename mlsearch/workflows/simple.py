from __future__ import annotations

from core.llmclient import LLMClient


class SimpleWorkflow:
    """
    Run a single LLM call.

    Parameters
    ----------
    client :
        Instance of :class:`core.llm_client.LLMClient`.
    """

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    def run(self, prompt: str) -> str:
        """Return the model’s answer to *prompt*."""
        return self.client.generate(prompt)
