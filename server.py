from fastapi import FastAPI, Request, HTTPException
from fiesta_openai import FiestaOpenAIClient, ChatCompletionRequest
from pydantic import ValidationError

app = FastAPI()
client = FiestaOpenAIClient()


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    OpenAI-compatible Chat Completions endpoint.

    We parse the raw JSON body ourselves so we can log it and
    see precise validation errors instead of FastAPI's generic 422.
    """
    body = await request.json()

    try:
        req = ChatCompletionRequest(**body)
    except ValidationError as e:
        # Log detailed validation errors and return them in the response
        print("Validation error while parsing ChatCompletionRequest:", e.errors())
        raise HTTPException(status_code=422, detail=e.errors())

    resp = await client.acreate(req, stream=False)
    print(resp)
    return resp
