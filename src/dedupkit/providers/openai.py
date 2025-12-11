from enum import Enum
from .base import EmbeddingProvider
from .embedding import EmbeddingModel
from openai import OpenAI

class OpenAIEmbeddingModels(Enum):
    SMALL = EmbeddingModel("text-embedding-3-small", 1536)
    LARGE = EmbeddingModel("text-embedding-3-large", 3072)

class OpenAIEmbeddingProvider(EmbeddingProvider):

    def __init__(self, api_key: str, model: EmbeddingModel = OpenAIEmbeddingModels.SMALL.value):
        self.model = model
        self.client = OpenAI(api_key=api_key)

    @property
    def dimensions(self) -> int:
        return self.model.dimensions

    def embed(self, text: str) -> list[float]:
        response = self.client.embeddings.create(model=self.model.name, input=text)
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.model.name, input=texts)
        return [item.embedding for item in response.data]
