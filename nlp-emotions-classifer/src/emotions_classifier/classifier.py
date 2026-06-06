import numpy as np

from .model import ONNXModel


def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


class EmotionsClassifier:
    def __init__(self, cache_dir: str | None = None, use_quantized: bool = True):
        self._model = ONNXModel(cache_dir=cache_dir)

    @property
    def labels(self) -> list[str]:
        return self._model.labels

    def classify(
        self,
        texts: str | list[str],
        threshold: float = 0.5,
    ) -> list[list[tuple[str, float]]]:
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        logits = self._model.predict(texts)
        probs = softmax(logits)
        results = []
        for row in probs:
            emotions = [
                (self._model.labels[i], float(row[i]))
                for i in np.where(row >= threshold)[0]
            ]
            emotions.sort(key=lambda x: x[1], reverse=True)
            results.append(emotions)
        return results[0] if single else results

    def classify_top_k(
        self,
        texts: str | list[str],
        k: int = 3,
    ) -> list[list[tuple[str, float]]]:
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        logits = self._model.predict(texts)
        probs = softmax(logits)
        results = []
        for row in probs:
            top_k_idx = np.argsort(row)[-k:][::-1]
            emotions = [
                (self._model.labels[i], float(row[i])) for i in top_k_idx
            ]
            results.append(emotions)
        return results[0] if single else results
