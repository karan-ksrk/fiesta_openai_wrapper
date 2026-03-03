from fastapi.responses import StreamingResponse, JSONResponse
from fastapi import FastAPI, Request, HTTPException
from fiesta_openai import FiestaOpenAIClient, ChatCompletionRequest
from pydantic import ValidationError
import json

app = FastAPI()
client = FiestaOpenAIClient()


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    stream = body.get("stream", False)  # ← read stream flag from request

    try:
        req = ChatCompletionRequest(**body)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())

    if stream:
        return StreamingResponse(
            _stream_response(req),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",  # important for nginx proxies
            }
        )

    resp = await client.acreate(req, stream=False)
    return JSONResponse(content=resp)


async def _stream_response(req: ChatCompletionRequest):
    """Wrap the full response as a single SSE chunk so OpenClaw renders it."""
    resp = await client.acreate(req, stream=False)
    content = resp["choices"][0]["message"]["content"]

    # SSE chunk — OpenAI delta format
    chunk = {
        "id": resp["id"],
        "object": "chat.completion.chunk",
        "created": resp["created"],
        "model": resp["model"],
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant", "content": content},
            "finish_reason": None,
        }]
    }
    yield f"data: {json.dumps(chunk)}\n\n"

    # Final chunk signals end of stream
    end_chunk = {
        "id": resp["id"],
        "object": "chat.completion.chunk",
        "created": resp["created"],
        "model": resp["model"],
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop",
        }]
    }
    yield f"data: {json.dumps(end_chunk)}\n\n"
    yield "data: [DONE]\n\n"
