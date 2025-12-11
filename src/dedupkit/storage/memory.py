from .base import StorageBackend, SearchHit
import numpy as np

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    a_arr = np.array(a)
    b_arr = np.array(b)

    dot_product = np.dot(a_arr, b_arr.T)
    magnitude_a = np.linalg.norm(a_arr)
    magnitude_b = np.linalg.norm(b_arr)

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


class MemoryStorage(StorageBackend):

    def __init__(self):
        self._storage: dict[str, dict] = {}

    def store(self, item_id: str, embedding: list[float], metadata: dict | None) -> None:
        self._storage[item_id] = { "embedding": embedding, "metadata": metadata }

    def search(self, embedding: list[float], top_k: int) -> list[SearchHit]:
        if not self._storage:
            return []

        results = []
        for embed_id, data in self._storage.items():
            similarity = cosine_similarity(embedding, data["embedding"])
            results.append(SearchHit(id=embed_id, similarity=similarity, metadata=data["metadata"]))

        results.sort(key=lambda x: x.similarity, reverse=True)

        return results[:top_k]

    def delete(self, item_id: str) -> bool:
        if not self._storage:
            return False
        deleted = self._storage.pop(item_id, None)

        return deleted is not None

    def count(self) -> int:
        return len(self._storage)
