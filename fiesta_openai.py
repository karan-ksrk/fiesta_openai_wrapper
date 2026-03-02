import asyncio
from datetime import datetime, timezone
from typing import AsyncGenerator, Dict, Iterable, List, Optional, Any, Union

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


def _messages_to_prompt(messages: Iterable[Message]) -> str:
    """
    Convert OpenAI-style messages into a single text prompt suitable for Fiesta.
    """
    lines: List[str] = []
    for msg in messages:
        role = msg.role
        content = _coerce_content_to_text(msg.content)
        if role == "system":
            lines.append(f"System: {content}")
        elif role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
        else:
            lines.append(f"{role}: {content}")
    return "\n\n".join(lines)


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
        prompt = _messages_to_prompt(request.messages)
        content = await self._provider.generate(prompt, stream=stream)

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
