import argparse
import json
import sys

from src.emotions_classifier import EmotionsClassifier


def main():
    parser = argparse.ArgumentParser(
        description="Classify emotions in short text using ONNX CPU inference."
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("file", nargs="?", help="Input file (one sentence per line)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold")
    parser.add_argument("--top-k", type=int, default=0, help="Return top K emotions (overrides threshold)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    classifier = EmotionsClassifier()

    if args.file:
        with open(args.file) as f:
            lines = [line.rstrip("\n") for line in f if line.strip()]
    else:
        lines = [line.rstrip("\n") for line in sys.stdin if line.strip()]

    if not lines:
        return

    results = []
    for i in range(0, len(lines), args.batch_size):
        batch = lines[i : i + args.batch_size]
        if args.top_k > 0:
            batch_results = classifier.classify_top_k(batch, k=args.top_k)
        else:
            batch_results = classifier.classify(batch, threshold=args.threshold)
        for text, emotions in zip(batch, batch_results):
            results.append({"text": text, "emotions": emotions})

    json.dump(results, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
