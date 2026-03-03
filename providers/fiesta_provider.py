from datetime import datetime, timezone
from providers.auth_utils import run_interactive_otp_flow
from providers.token_utils import get_jwt_expiry, get_jwt_expires_in
import asyncio
import json
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

from providers.base import BaseProvider


BASE_DIR = Path(__file__).resolve().parent
FIESTA_URL = "https://api.aifiesta.ai/api/chats/v3/completions"


def _print_stream_chunk(content: str) -> None:
    """Print content immediately, flushing for real-time console output."""
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    safe = content.encode(enc, errors="replace").decode(enc, errors="replace")
    print(safe, end="", flush=True)


def _extract_stream_text(body: str) -> str:
    """Extract concatenated text from Fiesta's event-stream style response."""
    parts: list[str] = []
    for chunk in body.split("-E"):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            event = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        if event.get("event") == "chat:stream":
            payload = event.get("payload") or {}
            content = payload.get("content")
            if content:
                parts.append(content)
    return "".join(parts)


class FiestaProvider(BaseProvider):

    def __init__(self):
        self.token = self.get_token()
        if not self.token:
            raise RuntimeError("Failed to get Fiesta token")
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    def get_token(self) -> str:
        load_dotenv(Path("providers/.env"))
        self.token = os.environ.get("FIESTA_SESSION")

        use_env_token = False
        if self.token:
            try:
                expiry = get_jwt_expiry(self.token)
                if expiry and expiry > datetime.now(timezone.utc):
                    use_env_token = True
            except Exception as e:
                print(f"Error checking token expiry: {e}")
                use_env_token = False

        if not use_env_token:
            # Only run OTP flow if no token or it is expired
            email = input("Enter your email: ")
            send_payload = {"email": f"{email}"}
            verify_base_payload = {"email": f"{email}"}
            self.token = run_interactive_otp_flow(send_payload, verify_base_payload)

            # Persist the fresh token into providers/.env as FIESTA_SESSION
            env_path = BASE_DIR / ".env"

            env_path.write_text(f"FIESTA_SESSION={self.token}\n", encoding="utf-8")
        return self.token

    async def generate(self, prompt: str, stream: bool = False):
        payload = {
            "models": [
                {"model": "claude", "version": "claude-sonnet-4"},
            ],
            "assetIds": [],
            "type": "text",
            "prompt": prompt,
            "tools": [],
        }

        if stream:
            return await self._generate_stream(self.headers, payload)
        return await self._generate_full(self.headers, payload)

    async def _generate_stream(self, headers: dict, payload: dict) -> str:
        buffer = ""
        full_text: list[str] = []

        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream(
                "POST", FIESTA_URL, json=payload, headers=headers
            ) as response:
                if not 200 <= response.status_code < 300:
                    body = await response.aread()
                    try:
                        detail = json.loads(body)
                    except json.JSONDecodeError:
                        detail = body.decode(errors="replace")
                    raise RuntimeError(
                        f"Fiesta call failed with status {response.status_code}: {detail}"
                    )

                async for chunk in response.aiter_text():
                    buffer += chunk
                    while "-E" in buffer:
                        part, _, buffer = buffer.partition("-E")
                        part = part.strip()
                        if not part:
                            continue
                        try:
                            event = json.loads(part)
                        except json.JSONDecodeError:
                            continue
                        if event.get("event") == "chat:stream":
                            content = (event.get("payload") or {}).get("content")
                            if content:
                                full_text.append(content)
                                _print_stream_chunk(content)

        return "".join(full_text)

    async def _generate_full(self, headers: dict, payload: dict) -> str:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                FIESTA_URL, json=payload, headers=headers, timeout=60
            )

        if not 200 <= response.status_code < 300:
            try:
                detail = response.json()
            except ValueError:
                detail = response.text
            raise RuntimeError(
                f"Fiesta call failed with status {response.status_code}: {detail}"
            )

        content_type = response.headers.get("content-type", "")

        # Fiesta almost always returns event-stream format — check this first
        if "text/event-stream" in content_type or "-E" in response.text:
            return _extract_stream_text(response.text)

        # Plain JSON fallback — extract text from known response shapes
        try:
            data = response.json()
            # Try common Fiesta/OpenAI-style response paths
            if isinstance(data, dict):
                # OpenAI-style
                if "choices" in data:
                    return data["choices"][0]["message"]["content"]
                # Fiesta-style flat response
                if "content" in data:
                    return data["content"]
                if "text" in data:
                    return data["text"]
                if "response" in data:
                    return data["response"]
                # Last resort: dump it so you can see the shape
                print(f"[DEBUG] Unknown JSON response shape: {data}")
                return str(data)
            return str(data)
        except ValueError:
            return _extract_stream_text(response.text)


def main():
    provider = FiestaProvider()
    print("Expires at:", get_jwt_expiry(provider.token))
    print("Expires in:", get_jwt_expires_in(provider.token))

    result = asyncio.run(provider.generate("which model are you using?", stream=True))
    print()  # newline after stream


if __name__ == "__main__":
    main()
