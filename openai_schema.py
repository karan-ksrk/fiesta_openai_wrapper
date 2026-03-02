from pydantic import BaseModel, ConfigDict
from typing import List, Optional, Union, Any


class Message(BaseModel):
    model_config = ConfigDict(extra="ignore")  # 👈

    role: str
    # OpenAI-style content can be either a plain string or
    # a list of content parts (e.g. [{"type": "text", "text": "..."}]).
    # Some clients also send null content (e.g. tool/function-call messages).
    content: Optional[Union[str, List[Any]]] = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")  # 👈

    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 500
