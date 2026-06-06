import json
from pathlib import Path

import onnx
import torch
import transformers


MODEL_ID = "SchuylerH/bert-multilingual-go-emtions"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "models" / MODEL_ID


class BertWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).logits


def export():
    hf_model = transformers.AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)

    hf_model.eval()
    model = BertWrapper(hf_model)

    onnx_dir = OUTPUT_DIR / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    dummy = tokenizer(["example sentence"], return_tensors="pt")

    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"], dummy["token_type_ids"]),
        onnx_dir / "model.onnx",
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "token_type_ids": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
        },
        opset_version=18,
        dynamo=False,
    )

    # Embed external data
    onnx_model = onnx.load(onnx_dir / "model.onnx")
    data_file = onnx_dir / "model.onnx.data"
    if data_file.exists():
        onnx.save_model(onnx_model, onnx_dir / "model.onnx")
        data_file.unlink()

    tokenizer.save_pretrained(onnx_dir)

    id2label = hf_model.config.id2label
    label_config = {
        "id2label": {str(k): v for k, v in id2label.items()},
        "model_type": hf_model.config.model_type,
        "max_position_embeddings": hf_model.config.max_position_embeddings,
    }
    with open(onnx_dir / "config.json", "w") as f:
        json.dump(label_config, f, indent=2)

    labels = [id2label[i] for i in sorted(id2label)]
    print(f"ONNX model exported to: {onnx_dir}")
    print(f"Labels ({len(labels)}): {labels}")


if __name__ == "__main__":
    export()
