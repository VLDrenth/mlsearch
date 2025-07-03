from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletion

load_dotenv()


class LLMClient:
    """
    Vendor-agnostic LLM wrapper (currently OpenAI-only).

    Parameters
    ----------
    model_name : str | None, optional
        Identifier of the model to use.  If *None*, the provider-specific
        default in :pyattr:`MODEL_DEFAULT` is used.
    system_prompt : str | None, optional
        Message injected as the first *system* role.
    provider : {'openai'}, optional
        LLM provider.  Additional providers can be added later.
    """

    MODEL_DEFAULT = {
        "openai": "gpt-4o-mini",  # July 2025 smallest GPT-4-class model
    }

    def __init__(
        self,
        model_name: Optional[str] = None,
        *,
        system_prompt: Optional[str] = None,
        provider: str = "openai",
    ) -> None:
        self.provider = provider.lower()
        self.model_name = model_name or self.MODEL_DEFAULT[self.provider]
        self.system_prompt = system_prompt
        self.client: OpenAI | None = None


    # public API
    def generate(self, prompt: str, *, stream: bool = False) -> str:
        """
        Return the model’s answer to *prompt*.

        Parameters
        ----------
        prompt : str
            The user message.
        stream : bool, default False
            If *True*, stream tokens and return the collected text.

        Returns
        -------
        str
            The model’s response (fully concatenated if ``stream`` is True).
        """
        if self.client is None:
            self._set_client()

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=stream,
        )

        if stream:
            chunks: list[str] = [
                part.choices[0].delta.content or ""  # delta may be None
                for part in completion
            ]
            return "".join(chunks).strip()

        return completion.choices[0].message.content.strip()

    def set_system_prompt(self, system_prompt: str) -> None:
        """Overwrite the existing system prompt."""
        self.system_prompt = system_prompt

    # private helpers
    def _set_client(self) -> None:
        """Instantiate the provider SDK client (lazy)."""
        if self.provider == "openai":
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            raise ValueError(f"Unknown provider '{self.provider}'")
