from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

MODEL_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "models"
    / "SchuylerH"
    / "bert-multilingual-go-emtions"
    / "onnx"
)

EMOTION_LABELS_ZH: dict[str, str] = {
    "admiration": "讚賞",
    "amusement": "有趣",
    "anger": "憤怒",
    "annoyance": "煩惱",
    "approval": "認可",
    "caring": "關心",
    "confusion": "困惑",
    "curiosity": "好奇",
    "desire": "慾望",
    "disappointment": "失望",
    "disapproval": "不認可",
    "disgust": "厭惡",
    "embarrassment": "尷尬",
    "excitement": "興奮",
    "fear": "恐懼",
    "gratitude": "感激",
    "grief": "悲痛",
    "joy": "喜悅",
    "love": "愛",
    "nervousness": "緊張",
    "optimism": "樂觀",
    "pride": "自豪",
    "realization": "領悟",
    "relief": "寬慰",
    "remorse": "自責",
    "sadness": "悲傷",
    "surprise": "驚訝",
    "neutral": "中性",
}


class ONNXModel:
    def __init__(self, cache_dir: str | None = None):
        model_path = MODEL_DIR / "model.onnx"
        tokenizer_path = MODEL_DIR / "tokenizer.json"
        config_path = MODEL_DIR / "config.json"

        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))

        import json
        with open(config_path) as f:
            config = json.load(f)
        self.labels = [config["id2label"][str(i)] for i in range(len(config["id2label"]))]
        self._setup_tokenizer()

    def _setup_tokenizer(self):
        self.tokenizer.enable_truncation(512)
        self.tokenizer.enable_padding(pad_id=0)

    def predict(self, texts: list[str]) -> np.ndarray:
        encodings = self.tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        token_type_ids = np.zeros_like(input_ids)
        ort_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
        logits = self.session.run(None, ort_inputs)[0]
        return logits


def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


class EmotionsClassifier:
    def __init__(self):
        self._model = ONNXModel()

    @property
    def labels(self) -> list[str]:
        return self._model.labels

    def classify_all(
        self,
        texts: list[str],
        threshold: float = 0.3,
    ) -> list[dict[str, float]]:
        logits = self._model.predict(texts)
        probs = softmax(logits)
        results = []
        for row in probs:
            emotions = {
                self._model.labels[i]: float(row[i])
                for i in np.where(row >= threshold)[0]
            }
            results.append(emotions)
        return results
