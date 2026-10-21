from typing import Protocol

from fastapi import Request


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedEmbedder:
    """Local ONNX embeddings: no API key, no GPU, no per-request cost."""

    def __init__(self, model_name: str):
        from fastembed import TextEmbedding  # imported lazily so tests don't load the model

        self._model = TextEmbedding(model_name=model_name)
        self.dim = len(next(iter(self._model.embed(["dimension probe"]))))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.passage_embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        # BGE models expect an instruction prefix on queries; query_embed applies it.
        return next(iter(self._model.query_embed(text))).tolist()


def get_embedder(request: Request) -> Embedder:
    return request.app.state.embedder
