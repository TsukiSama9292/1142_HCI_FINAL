#!/usr/bin/env python3
import argparse
import json
import os
import sys
from collections import Counter

from src.hci_analysis.danmaku import download_danmaku, parse_time, ts_format
from src.hci_analysis.emotions import EmotionsClassifier, EMOTION_CODES
from src.hci_analysis.preprocess import preprocess


def ts_to_dsec(ts: str) -> int:
    parts = [int(x) for x in ts.replace(".", ":").split(":")]
    if len(parts) == 3:
        return parts[0] * 36000 + parts[1] * 600 + parts[2] * 10
    if len(parts) == 4:
        return parts[0] * 36000 + parts[1] * 600 + parts[2] * 10 + parts[3] // 10
    raise ValueError(f"Invalid timestamp: {ts}")



def write_records(path: str, records: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def label_path(output_path: str) -> str:
    base, ext = os.path.splitext(output_path)
    return f"{base}_label{ext}"

def ana_path(output_path: str) -> str:
    base, ext = os.path.splitext(output_path)
    return f"{base}_label_ana{ext}"


def build_segments(
    n_records: list[dict],
    window_sec: int,
    clip_start_dsec: int | None = None,
) -> dict[int, list[dict]]:
    window_dsec = window_sec * 10
    segments: dict[int, list[dict]] = {}
    for record in n_records:
        dsec = ts_to_dsec(record["start"])
        if clip_start_dsec is not None:
            offset = dsec - clip_start_dsec
            bin_start = clip_start_dsec + (offset // window_dsec) * window_dsec
        else:
            bin_start = (dsec // window_dsec) * window_dsec
        segments.setdefault(bin_start, []).append(record)
    return segments


def _fill_window_gaps(
    segments: dict[int, list[dict]],
    window_dsec: int,
    start_dsec: int | None = None,
    end_dsec: int | None = None,
) -> list[tuple[int, list[dict]]]:
    if not segments and start_dsec is None:
        return []
    if start_dsec is None:
        start_dsec = min(segments.keys())
    if end_dsec is None:
        end_dsec = max(segments.keys()) + window_dsec
    bins: list[tuple[int, list[dict]]] = []
    bin_start = start_dsec
    while bin_start < end_dsec:
        bins.append((bin_start, segments.get(bin_start, [])))
        bin_start += window_dsec
    return bins


def process_sn(sn: int, start_range_str: str | None, output: str | None,
               threshold: float, segment: int, no_classify: bool, no_preprocess: bool):
    start_range = None
    if start_range_str:
        parts = start_range_str.split("~")
        if len(parts) != 2:
            print(f"錯誤: sn={sn} --start-range 格式應為 MM:SS~MM:SS", file=sys.stderr)
            return
        start_range = (parse_time(parts[0]), parse_time(parts[1]))

    if no_classify:
        output_dl = output or f"logs/{sn}/nlp/output.jsonl"
        os.makedirs(os.path.dirname(output_dl), exist_ok=True)
        records = download_danmaku(sn=sn, output_path=output_dl, start_range=start_range)
        return
    else:
        records = download_danmaku(sn=sn, output_path=None, start_range=start_range)

    classifier = EmotionsClassifier()

    if no_preprocess:
        texts = [r["text"] for r in records]
        emotions_batch = classifier.classify_all(texts, threshold=threshold)
        for record, emotions in zip(records, emotions_batch):
            record["emotions"] = {
                EMOTION_CODES.get(k, k): round(v, 4)
                for k, v in sorted(emotions.items(), key=lambda x: -x[1])
            }
    else:
        nlp_texts: list[str] = []
        nlp_indices: list[int] = []
        for i, r in enumerate(records):
            text, rule_label, action = preprocess(r["text"])
            if action is not None:
                r["_skip"] = True
            elif rule_label is not None:
                r["emotions"] = {EMOTION_CODES.get(rule_label, rule_label): 1.0}
                r["_source"] = "rule"
            else:
                nlp_texts.append(text)
                nlp_indices.append(i)

        if nlp_texts:
            emotions_batch = classifier.classify_all(nlp_texts, threshold=threshold)
            for idx, emotions in zip(nlp_indices, emotions_batch):
                records[idx]["emotions"] = {
                    EMOTION_CODES.get(k, k): round(v, 4)
                    for k, v in sorted(emotions.items(), key=lambda x: -x[1])
                }
                records[idx]["_source"] = "nlp"

    for r in records:
        if r.get("_skip"):
            continue
        top_code = max(r["emotions"], key=r["emotions"].get) if r["emotions"] else "@C"
        r["_emotion_code"] = top_code

    n_records = [r for r in records if not r.get("_skip")]
    output_path = output or f"logs/{sn}/nlp/output.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if segment > 0:
        window_dsec = segment * 10
        segments: dict[int, list[dict]] = {}
        clip_start_dsec = start_range[0] if start_range else None
        for record in n_records:
            dsec = ts_to_dsec(record["start"])
            if clip_start_dsec is not None:
                offset = dsec - clip_start_dsec
                bin_start = clip_start_dsec + (offset // window_dsec) * window_dsec
            else:
                bin_start = (dsec // window_dsec) * window_dsec
            segments.setdefault(bin_start, []).append(record)

        if start_range:
            sr_start, sr_end = start_range
        else:
            all_starts = sorted(segments.keys())
            sr_start = all_starts[0] if all_starts else 0
            sr_end = (all_starts[-1] + window_dsec) if all_starts else 0

        seg_records_full: list[dict] = []
        seg_records_label: list[dict] = []
        for bin_start, danmakus in _fill_window_gaps(segments, window_dsec, sr_start, sr_end):
            all_emotions: list[str] = []
            for d in danmakus:
                all_emotions.extend(d["emotions"].keys())
            total = len(all_emotions)
            distribution = dict(
                sorted(
                    ((k, round(v / total, 4)) for k, v in Counter(all_emotions).items()),
                    key=lambda x: -x[1],
                )
            ) if total else {}
            codes = [d["_emotion_code"] for d in danmakus]
            code_counts = Counter(codes)
            code_total = sum(code_counts.values())
            code_distribution = dict(
                sorted(
                    ((k, round(v / code_total, 4)) for k, v in code_counts.items()),
                    key=lambda x: -x[1],
                )
            ) if code_total else {}

            seg_start_ts = ts_format(bin_start)
            seg_end_ts = ts_format(bin_start + window_dsec)
            seg_full = {
                "segment_start": seg_start_ts,
                "segment_end": seg_end_ts,
                "danmaku_count": len(danmakus),
                "emotion_distribution": distribution,
            }
            seg_label = {
                "segment_start": seg_start_ts,
                "segment_end": seg_end_ts,
                "danmaku_count": len(danmakus),
                "emotion": code_distribution,
            }
            seg_records_full.append(seg_full)
            seg_records_label.append(seg_label)

        write_records(output_path, seg_records_full)
        label_out = label_path(output_path)
        write_records(label_out, seg_records_label)
        print(f"時間窗聚合 JSONL 已儲存: {output_path}")
        print(f"情緒編碼 JSONL 已儲存: {label_out}")
    else:
        full_records: list[dict] = []
        label_records: list[dict] = []
        for r in n_records:
            out = {k: v for k, v in r.items() if not k.startswith("_")}
            full_records.append(out)
            label_records.append({
                "start": r["start"],
                "text": r["text"],
                "emotion": r["_emotion_code"],
            })

        write_records(output_path, full_records)
        label_out = label_path(output_path)
        write_records(label_out, label_records)
        print(f"情緒標記 JSONL 已儲存: {output_path}")
        print(f"情緒編碼 JSONL 已儲存: {label_out}")

        ana_segments = build_segments(n_records, 8, clip_start_dsec=start_range[0] if start_range else None)
        if start_range:
            ana_sr_start, ana_sr_end = start_range
        else:
            ana_keys = sorted(ana_segments.keys())
            ana_sr_start = ana_keys[0] if ana_keys else 0
            ana_sr_end = (ana_keys[-1] + 80) if ana_keys else 0

        ana_records: list[dict] = []
        for bin_start, danmakus in _fill_window_gaps(ana_segments, 80, ana_sr_start, ana_sr_end):
            codes = [d["_emotion_code"] for d in danmakus]
            code_counts = Counter(codes)
            code_total = sum(code_counts.values())
            code_distribution = dict(
                sorted(
                    ((k, round(v / code_total, 4)) for k, v in code_counts.items()),
                    key=lambda x: -x[1],
                )
            ) if code_total else {}
            ana_records.append({
                "segment_start": ts_format(bin_start),
                "segment_end": ts_format(bin_start + 80),
                "danmaku_count": len(danmakus),
                "emotion": code_distribution,
            })
        ana_out = ana_path(output_path)
        write_records(ana_out, ana_records)
        print(f"情緒編碼統計 JSONL 已儲存: {ana_out}")


def main():
    parser = argparse.ArgumentParser(description="動畫瘋彈幕情緒分析工具")
    parser.add_argument("sn", nargs="?", type=int, default=None, help="影片 SN 碼")
    parser.add_argument("--from-config", help="從群組設定檔批次處理所有 SN")
    parser.add_argument("-o", "--output", help="輸出路徑 (預設: logs/<sn>/nlp/output.jsonl)")
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
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="略過前處理 Pipeline (過濾/正規化/規則標記)",
    )
    args = parser.parse_args()

    if args.from_config:
        with open(args.from_config, encoding="utf-8") as f:
            config = json.load(f)
        for group_name, group_info in config["groups"].items():
            print(f"\n=== 群組: {group_name} ===", file=sys.stderr)
            for sn_entry in group_info["sns"]:
                sn = sn_entry["sn"]
                sn_range = sn_entry.get("range")
                print(f"  處理 SN={sn} ({sn_entry.get('short', sn_entry.get('title', ''))}) range={sn_range}", file=sys.stderr)
                process_sn(
                    sn=sn,
                    start_range_str=sn_range,
                    output=None,
                    threshold=args.threshold,
                    segment=args.segment,
                    no_classify=args.no_classify,
                    no_preprocess=args.no_preprocess,
                )
        return

    if args.sn is None:
        parser.print_help()
        sys.exit(1)

    process_sn(
        sn=args.sn,
        start_range_str=args.start_range,
        output=args.output,
        threshold=args.threshold,
        segment=args.segment,
        no_classify=args.no_classify,
        no_preprocess=args.no_preprocess,
    )


if __name__ == "__main__":
    main()
