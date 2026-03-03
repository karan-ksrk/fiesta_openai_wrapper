import asyncio
from datetime import datetime, timezone
from typing import AsyncGenerator, Dict, Iterable, List, Optional, Any, Union
import os

from openai_schema import ChatCompletionRequest, Message
from providers.fiesta_provider import FiestaProvider


def _coerce_content_to_text(content: Union[str, List[Any], None]) -> str:
    """
    Normalize OpenAI-style message content to a plain string.

    Supports:
      - plain string content
      - list of content parts like [{"type": "text", "text": "..."}]
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content

    parts: List[str] = []
    for part in content:
        # Common OpenAI-style shape: {"type": "text", "text": "..."}
        if isinstance(part, dict):
            text = part.get("text")
            if isinstance(text, str):
                parts.append(text)
                continue
        # Fallback: just coerce to string
        parts.append(str(part))
    return "".join(parts)


def _format_message(role: str, content: str) -> str:
    if role == "system":
        return f"System: {content}"
    if role == "user":
        return f"User: {content}"
    if role == "assistant":
        return f"Assistant: {content}"
    return f"{role}: {content}"


def _messages_to_prompt(messages: Iterable[Message], max_chars: int = 20000) -> str:
    """
    Convert OpenAI-style messages into a single text prompt suitable for Fiesta.
    """
    # Build the full prompt first (useful for short conversations).
    formatted: List[str] = []
    msgs = list(messages)
    for msg in msgs:
        formatted.append(_format_message(msg.role, _coerce_content_to_text(msg.content)))

    prompt = "\n\n".join(formatted)
    if len(prompt) <= max_chars:
        return prompt

    # Fiesta enforces a hard prompt length limit (20k chars). When the conversation
    # grows too large (e.g. long tool/system messages), keep the most recent
    # messages and drop older ones until we fit.
    kept: List[str] = []
    total = 0

    # Always prefer keeping the latest messages (iterate backwards).
    for msg in reversed(msgs):
        line = _format_message(msg.role, _coerce_content_to_text(msg.content))
        extra = len(line) + (2 if kept else 0)  # account for "\n\n"
        if total + extra > max_chars:
            continue
        kept.append(line)
        total += extra

    if not kept:
        # Extreme edge case: a single message exceeds the limit.
        # Keep the tail end so the user’s most recent instruction is preserved.
        return prompt[-max_chars:]

    kept.reverse()
    return "\n\n".join(kept)


class FiestaOpenAIClient:
    """
    Lightweight OpenAI-compatible wrapper around FiestaProvider.

    Usage (sync):

        from fiesta_openai import FiestaOpenAIClient, ChatCompletionRequest, Message

        client = FiestaOpenAIClient()
        req = ChatCompletionRequest(
            model="claude-sonnet-4",
            messages=[Message(role="user", content="Hello!")],
        )
        resp = client.create(req)
        print(resp["choices"][0]["message"]["content"])
    """

    def __init__(self) -> None:
        self._provider = FiestaProvider()

    def get_token(self) -> str:
        return self._provider.get_token()

    async def acreate(
        self, request: ChatCompletionRequest, stream: bool = False
    ) -> Dict:
        """
        Async version of create.

        If stream=False, returns an OpenAI-style dict:
            {
              "id": ...,
              "object": "chat.completion",
              "model": request.model,
              "choices": [
                {
                  "index": 0,
                  "message": {"role": "assistant", "content": "<text>"},
                  "finish_reason": "stop",
                }
              ],
            }

        If stream=True, this returns the same dict but the provider itself
        will already have streamed tokens to stdout.
        """
        max_prompt_chars = int(os.environ.get("FIESTA_MAX_PROMPT_CHARS", "20000"))
        prompt = _messages_to_prompt(request.messages, max_chars=max_prompt_chars)

        print(f"[DEBUG] Prompt length: {len(prompt)}")
        print(f"[DEBUG] Prompt preview: {prompt[:200]}")

        content = await self._provider.generate(prompt, stream=stream)

        print(f"[DEBUG] Raw content returned: {repr(content)}")  # ← key line

        # Build a minimal OpenAI-compatible response object.
        now = datetime.now(timezone.utc)
        resp: Dict = {
            "id": f"chatcmpl-fiesta-{int(now.timestamp())}",
            "object": "chat.completion",
            "created": int(now.timestamp()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "finish_reason": "stop",
                }
            ],
        }
        return resp

    def create(self, request: ChatCompletionRequest, stream: bool = False) -> Dict:
        """
        Synchronous helper that wraps `acreate` with `asyncio.run`, to look
        similar to OpenAI's ChatCompletion.create.
        """
        return asyncio.run(self.acreate(request, stream=stream))


__all__ = ["FiestaOpenAIClient", "ChatCompletionRequest", "Message"]
