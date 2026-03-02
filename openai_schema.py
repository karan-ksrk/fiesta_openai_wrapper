from pydantic import BaseModel, ConfigDict
from typing import List, Optional, Union, Any


class Message(BaseModel):
    model_config = ConfigDict(extra="ignore")  # 👈

    role: str
    # OpenAI-style content can be either a plain string or
    # a list of content parts (e.g. [{"type": "text", "text": "..."}]).
    content: Union[str, List[Any]]


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")  # 👈

    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 500
