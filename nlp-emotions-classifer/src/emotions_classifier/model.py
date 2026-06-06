import json
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


class ONNXModel:
    def __init__(self, cache_dir: str | None = None, use_quantized: bool = True):
        model_path = MODEL_DIR / "model.onnx"
        tokenizer_path = MODEL_DIR / "tokenizer.json"
        config_path = MODEL_DIR / "config.json"

        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))

        with open(config_path) as f:
            config = json.load(f)
        self.labels = [config["id2label"][str(i)] for i in range(len(config["id2label"]))]
        self._setup_tokenizer()

    def _setup_tokenizer(self):
        self.tokenizer.enable_truncation(512)
        self.tokenizer.enable_padding(pad_id=0)

    def predict(self, texts: list[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
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
