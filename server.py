from fastapi import FastAPI
from fiesta_openai import FiestaOpenAIClient, ChatCompletionRequest

app = FastAPI()
client = FiestaOpenAIClient()


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    # OpenClaw sends the same JSON as OpenAI ChatCompletion
    resp = await client.acreate(req, stream=False)
    return resp
