from pydantic import BaseModel, ConfigDict
from typing import List, Optional


class Message(BaseModel):
    model_config = ConfigDict(extra="ignore")  # 👈

    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")  # 👈

    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 500
