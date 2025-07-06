from __future__ import annotations

import os
from typing import Optional, List, Dict, Any, Union

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage, ChatCompletionMessageToolCall

from mlsearch.config import get_model_name

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

    def __init__(
        self,
        model_name: Optional[str] = None,
        *,
        system_prompt: Optional[str] = None,
        provider: str = "openai",
        model_type: str = "default",
    ) -> None:
        self.provider = provider.lower()
        self.model_name = model_name or get_model_name(model_type)
        self.system_prompt = system_prompt
        self.client: OpenAI | None = None
        self._conversation_history: List[Dict[str, Any]] = []


    # public API
    def generate(self, prompt: str, *, stream: bool = False, max_tokens: Optional[int] = None) -> str:
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

        request_params = {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
        }
        
        if max_tokens is not None:
            request_params["max_tokens"] = max_tokens
            
        completion = self.client.chat.completions.create(**request_params)

        if stream:
            chunks: list[str] = [
                part.choices[0].delta.content or ""  # delta may be None
                for part in completion
            ]
            return "".join(chunks).strip()

        return completion.choices[0].message.content.strip()

    def generate_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        *,
        tool_choice: Optional[str] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate a response with tool calling support.

        Parameters
        ----------
        prompt : str
            The user message.
        tools : List[Dict[str, Any]]
            List of tool definitions in OpenAI format.
        tool_choice : str, optional
            Whether to force tool usage ("auto", "none", or specific tool name).
        stream : bool, default False
            If True, stream tokens.

        Returns
        -------
        Dict[str, Any]
            Response containing either text content or tool calls.
        """
        if self.client is None:
            self._set_client()

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Create request parameters
        request_params = {
            "model": self.model_name,
            "messages": messages,
            "tools": tools,
            "stream": stream,
        }

        if tool_choice:
            request_params["tool_choice"] = tool_choice

        completion = self.client.chat.completions.create(**request_params)

        if stream:
            # Handle streaming response
            chunks = []
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    chunks.append(chunk.choices[0].delta.content)
            
            return {
                "content": "".join(chunks).strip(),
                "tool_calls": None,
                "finish_reason": "stop"
            }

        # Handle non-streaming response
        message = completion.choices[0].message
        
        result = {
            "content": message.content.strip() if message.content else "",
            "tool_calls": [],
            "finish_reason": completion.choices[0].finish_reason
        }

        # Parse tool calls if present
        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tool_call in message.tool_calls:
                result["tool_calls"].append({
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                })

        return result

    def continue_conversation(
        self,
        tool_results: List[Dict[str, Any]],
        *,
        stream: bool = False,
    ) -> str:
        """
        Continue conversation after tool calls with their results.

        Parameters
        ----------
        tool_results : List[Dict[str, Any]]
            List of tool call results.
        stream : bool, default False
            If True, stream tokens.

        Returns
        -------
        str
            The model's response to the tool results.
        """
        if self.client is None:
            self._set_client()

        # Add tool results to conversation history
        for tool_result in tool_results:
            self._conversation_history.append({
                "role": "tool",
                "tool_call_id": tool_result["tool_call_id"],
                "content": str(tool_result["result"])
            })

        # Generate follow-up response
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=self._conversation_history,
            stream=stream,
        )

        if stream:
            chunks = [
                part.choices[0].delta.content or ""
                for part in completion
            ]
            return "".join(chunks).strip()

        return completion.choices[0].message.content.strip()

    def set_system_prompt(self, system_prompt: str) -> None:
        """Overwrite the existing system prompt."""
        self.system_prompt = system_prompt

    def reset_conversation(self) -> None:
        """Clear conversation history."""
        self._conversation_history = []

    # private helpers
    def _set_client(self) -> None:
        """Instantiate the provider SDK client (lazy)."""
        if self.provider == "openai":
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            raise ValueError(f"Unknown provider '{self.provider}'")
