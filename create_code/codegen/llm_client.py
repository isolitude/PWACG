#!/usr/bin/env python3
# coding: utf-8
"""OpenAI-compatible LLM client for DeepSeek V4 Flash.

Wraps the OpenAI SDK with retry logic, error handling, and structured output.
"""
from __future__ import annotations
import json
import os
import re
from typing import Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore


DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_MAX_TOKENS = 32768
DEFAULT_TEMPERATURE = 0.0


def _extract_python_code(content: str) -> str:
    """Extract Python code from LLM response.

    Tries in order:
    1. JSON with 'file_content' key (legacy)
    2. Markdown ```python ... ``` code block
    3. Markdown ``` ... ``` code block (no language)
    4. Raw content as-is
    """
    # 1. Try JSON
    try:
        data = json.loads(content)
        if "file_content" in data:
            return data["file_content"]
    except (json.JSONDecodeError, TypeError):
        pass

    # 2. Try ```python ... ```
    match = re.search(r'```python\n(.*?)```', content, re.DOTALL)
    if match:
        return match.group(1).rstrip()

    # 3. Try ``` ... ``` without language spec
    match = re.search(r'```\n(.*?)```', content, re.DOTALL)
    if match:
        return match.group(1).rstrip()

    # 4. Return raw content
    return content.strip()


class LLMClient:
    """Client for DeepSeek API via OpenAI-compatible interface."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ):
        if OpenAI is None:
            raise RuntimeError("openai package not installed. Run: pip install openai")
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self.base_url = base_url or DEFAULT_BASE_URL
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client: Optional[OpenAI] = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self._client

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        response_format: Optional[dict] = None,
        max_retries: int = 3,
    ) -> dict:
        """Generate structured output from LLM.

        Args:
            system_prompt: The system prompt (code style, rules, context)
            user_prompt: The user prompt (IR JSON + task description)
            response_format: Optional {"type": "json_object"} for structured output
            max_retries: Number of retries on failure

        Returns:
            Parsed JSON dict with file_content, imports, exported_symbols, etc.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if response_format:
            kwargs["response_format"] = response_format

        last_error: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                content = resp.choices[0].message.content
                if content is None:
                    raise RuntimeError("LLM returned empty content")
                file_content = _extract_python_code(content)
                if not file_content:
                    raise RuntimeError("Could not extract Python code from LLM response")
                return {"file_content": file_content}
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2 ** attempt)  # exponential backoff
                continue

        raise RuntimeError(f"LLM generation failed after {max_retries} attempts: {last_error}")
