#!/usr/bin/env python3
import argparse
import json
import sys
from collections import Counter

from src.hci_analysis.danmaku import download_danmaku, parse_time, ts_format
from src.hci_analysis.emotions import EmotionsClassifier, EMOTION_LABELS_ZH


def ts_to_dsec(ts: str) -> int:
    parts = [int(x) for x in ts.replace(".", ":").split(":")]
    if len(parts) == 3:
        return parts[0] * 36000 + parts[1] * 600 + parts[2] * 10
    if len(parts) == 4:
        return parts[0] * 36000 + parts[1] * 600 + parts[2] * 10 + parts[3]
    raise ValueError(f"Invalid timestamp: {ts}")


def main():
    parser = argparse.ArgumentParser(description="動畫瘋彈幕情緒分析工具")
    parser.add_argument("sn", type=int, help="影片 SN 碼")
    parser.add_argument("-o", "--output", help="輸出路徑 (預設: stdout)")
    parser.add_argument("--start-range", help="過濾時間範圍，格式: MM:SS~MM:SS")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="情緒分類信心門檻 (預設: 0.3)",
    )
    parser.add_argument(
        "--segment",
        type=int,
        default=0,
        help="時間窗聚合秒數 (例如 8)，輸出每時段內彈幕情緒分佈",
    )
    parser.add_argument(
        "--no-classify",
        action="store_true",
        help="只下載 JSONL，不進行情緒分類",
    )
    args = parser.parse_args()

    start_range = None
    if args.start_range:
        parts = args.start_range.split("~")
        if len(parts) != 2:
            print("錯誤: --start-range 格式應為 MM:SS~MM:SS", file=sys.stderr)
            sys.exit(1)
        start_range = (parse_time(parts[0]), parse_time(parts[1]))

    records = download_danmaku(
        sn=args.sn,
        output_path=args.output if args.no_classify else None,
        start_range=start_range,
    )

    if args.no_classify:
        return

    classifier = EmotionsClassifier()
    texts = [r["text"] for r in records]
    emotions_batch = classifier.classify_all(texts, threshold=args.threshold)

    for record, emotions in zip(records, emotions_batch):
        record["emotions"] = {
            EMOTION_LABELS_ZH.get(k, k): round(v, 4)
            for k, v in sorted(emotions.items(), key=lambda x: -x[1])
        }

    output = args.output or sys.stdout
    is_file = isinstance(output, str)

    if args.segment > 0:
        window_dsec = args.segment * 10
        segments: dict[int, list[dict]] = {}
        for record in records:
            dsec = ts_to_dsec(record["start"])
            bin_start = (dsec // window_dsec) * window_dsec
            segments.setdefault(bin_start, []).append(record)

        fout = open(output, "w", encoding="utf-8") if is_file else output
        for bin_start in sorted(segments):
            danmakus = segments[bin_start]
            all_emotions: list[str] = []
            for d in danmakus:
                all_emotions.extend(d["emotions"].keys())
            total = len(all_emotions)
            distribution = {
                k: round(v / total, 4) for k, v in Counter(all_emotions).items()
            } if total else {}

            bin_end = bin_start + window_dsec
            seg = {
                "segment_start": ts_format(bin_start),
                "segment_end": ts_format(bin_end),
                "danmaku_count": len(danmakus),
                "emotion_distribution": distribution,
            }
            fout.write(json.dumps(seg, ensure_ascii=False) + "\n")
        if is_file:
            fout.close()
            print(f"時間窗聚合 JSONL 已儲存: {output}")
    else:
        fout = open(output, "w", encoding="utf-8") if is_file else output
        for record in records:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        if is_file:
            fout.close()
            print(f"情緒標記 JSONL 已儲存: {output}")


if __name__ == "__main__":
    main()
