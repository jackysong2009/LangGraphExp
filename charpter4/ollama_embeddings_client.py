from typing import List
import httpx
from langchain_core.embeddings import Embeddings

class OllamaEmbeddings(Embeddings):
    def __init__(self, model: str, base_url: str = "http://localhost:11434", timeout: float = 120.0):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        r = httpx.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": texts},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["embeddings"]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]