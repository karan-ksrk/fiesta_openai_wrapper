from fiesta_openai import FiestaOpenAIClient, ChatCompletionRequest, Message

client = FiestaOpenAIClient()

req = ChatCompletionRequest(
    model="claude-sonnet-4",
    messages=[Message(role="user", content="python program to convert any photo to character art")],
)

resp = client.create(req, stream=False)
print(resp["choices"][0]["message"]["content"])
