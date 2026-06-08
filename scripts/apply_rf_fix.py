#!/usr/bin/env python3
"""
將研究人員手動修正 (rf_fix.txt) 合併到 output_label.jsonl，
產生 final_label.jsonl 與 final_label_ana.jsonl。

用法:
  python3 scripts/apply_rf_fix.py <sn> [--fix-path path]
  python3 scripts/apply_rf_fix.py --all

rf_fix.txt 格式: 每行 "row_index emotion_code" (1-based)
"""
import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

SN_RANGES = {
    11351: {"start_sec": 9 * 60 + 10, "end_sec": 14 * 60 + 46},
    14464: {"start_sec": 11 * 60 + 52, "end_sec": 17 * 60 + 28},
    27444: {"start_sec": 2 * 60 + 17, "end_sec": 7 * 60 + 53},
    31622: {"start_sec": 10 * 60, "end_sec": 15 * 60 + 36},
    7296:  {"start_sec": 6 * 60 + 17, "end_sec": 11 * 60 + 53},
    8649:  {"start_sec": 10 * 60 + 47, "end_sec": 16 * 60 + 23},
}


def ts_to_dsec(ts: str) -> int:
    parts = [int(x) for x in ts.replace(".", ":").split(":")]
    if len(parts) == 3:
        return parts[0] * 36000 + parts[1] * 600 + parts[2] * 10
    if len(parts) == 4:
        return parts[0] * 36000 + parts[1] * 600 + parts[2] * 10 + parts[3] // 10
    raise ValueError(f"Invalid timestamp: {ts}")


def ts_format(dsec: int) -> str:
    m = (dsec // 600) % 60
    s = (dsec // 10) % 60
    cs = dsec % 10
    return f"0:{m:02d}:{s:02d}.{cs}0"


def build_ana_records(records: list[dict], sn: int) -> list[dict]:
    info = SN_RANGES.get(sn)
    if info:
        clip_start_dsec = info["start_sec"] * 10
        clip_end_dsec = info["end_sec"] * 10
    else:
        dsecs = [ts_to_dsec(r["start"]) for r in records]
        clip_start_dsec = min(dsecs) if dsecs else 0
        clip_end_dsec = max(dsecs) if dsecs else 0

    window_dsec = 80
    bins: dict[int, list[str]] = defaultdict(list)
    for r in records:
        dsec = ts_to_dsec(r["start"])
        offset = dsec - clip_start_dsec
        bin_start = clip_start_dsec + (offset // window_dsec) * window_dsec
        bins[bin_start].append(r["emotion"])

    ana = []
    bin_start = clip_start_dsec
    while bin_start < clip_end_dsec:
        emotions = bins.get(bin_start, [])
        total = len(emotions)
        if total:
            counts = Counter(emotions)
            distribution = dict(
                sorted(
                    ((k, round(v / total, 4)) for k, v in counts.items()),
                    key=lambda x: -x[1],
                )
            )
        else:
            distribution = {}
        ana.append({
            "segment_start": ts_format(bin_start),
            "segment_end": ts_format(bin_start + window_dsec),
            "danmaku_count": total,
            "emotion": distribution,
        })
        bin_start += window_dsec
    return ana


def process_sn(sn: int, fix_path: str | None = None):
    nlp_dir = f"logs/{sn}/nlp"
    label_path = f"{nlp_dir}/output_label.jsonl"

    if not os.path.exists(label_path):
        print(f"[SKIP] sn={sn}: {label_path} not found", file=sys.stderr)
        return

    if fix_path is None:
        fix_path = f"{nlp_dir}/rf_fix.txt"

    # Read output_label.jsonl
    with open(label_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    # Apply fixes
    fixes_applied = 0
    if os.path.exists(fix_path):
        with open(fix_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(None, 1)
                if len(parts) < 2:
                    continue
                try:
                    idx = int(parts[0]) - 1  # convert to 0-based
                except ValueError:
                    continue
                code = parts[1].strip()
                if 0 <= idx < len(records):
                    records[idx]["emotion"] = code
                    fixes_applied += 1
                else:
                    print(f"  [WARN] sn={sn}: row {idx+1} out of range (max {len(records)})",
                          file=sys.stderr)
        print(f"  sn={sn}: applied {fixes_applied} fixes from {fix_path}", file=sys.stderr)
    else:
        print(f"  sn={sn}: no fix file at {fix_path}, copying as-is", file=sys.stderr)

    # Write final_label.jsonl
    final_path = f"{nlp_dir}/final_label.jsonl"
    with open(final_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  sn={sn}: wrote {final_path} ({len(records)} records)", file=sys.stderr)

    # Build and write final_label_ana.jsonl
    ana_records = build_ana_records(records, sn)
    ana_path = f"{nlp_dir}/final_label_ana.jsonl"
    with open(ana_path, "w", encoding="utf-8") as f:
        for r in ana_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  sn={sn}: wrote {ana_path} ({len(ana_records)} windows)", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="合併手動修正標籤到 output_label.jsonl")
    parser.add_argument("sn", nargs="?", type=int, default=None, help="影片 SN 碼")
    parser.add_argument("--fix-path", help="rf_fix.txt 路徑 (預設: logs/<sn>/nlp/rf_fix.txt)")
    parser.add_argument("--all", action="store_true", help="處理所有有 nlp 目錄的 SN")
    args = parser.parse_args()

    if args.all:
        import glob
        for nlp_dir in sorted(glob.glob("logs/*/nlp/")):
            sn_str = nlp_dir.split("/")[1]
            try:
                sn = int(sn_str)
            except ValueError:
                continue
            print(f"\n=== SN={sn} ===", file=sys.stderr)
            process_sn(sn)
        return

    if args.sn is None:
        parser.print_help()
        sys.exit(1)
    process_sn(args.sn, args.fix_path)


if __name__ == "__main__":
    main()
