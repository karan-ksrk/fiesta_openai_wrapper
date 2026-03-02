from abc import ABC, abstractmethod


class BaseProvider(ABC):
    @abstractmethod
    async def generate(self, prompt, temperature, max_tokens):
        pass
