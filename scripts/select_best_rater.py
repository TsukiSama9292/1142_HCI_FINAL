#!/usr/bin/env python3
"""
為每個動漫找出最佳評分者配對（最高 pairwise Cohen's Kappa），
若該配對 Kappa > 0.6 則選取其中一位研究員作為範本，
產生 protagonist.jsonl，並輸出完整的 Kappa 摘要報告。
"""
import argparse
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.hci_analysis.lsa import (
    load_rater_emotions,
    align_raters,
    get_emotion_codes,
    cohen_kappa,
)

CODES = get_emotion_codes()


def load_group_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def select_best_rater(protagonist_dir: str, config: dict, output_dir: str) -> dict:
    result = {}
    for group_name, group_info in config["groups"].items():
        for sn_entry in group_info["sns"]:
            sn = sn_entry["sn"]
            sn_dir = os.path.join(protagonist_dir, str(sn))
            rater_data = load_rater_emotions(sn_dir)
            rater_names = sorted(rater_data.keys())

            if len(rater_data) < 2:
                print(f"  [SKIP] sn={sn}: only {len(rater_data)} raters", file=sys.stderr)
                continue

            aligned = align_raters(rater_data)
            if len(aligned) < 2:
                print(f"  [SKIP] sn={sn}: no overlapping windows", file=sys.stderr)
                continue

            n_windows = len(next(iter(aligned.values())))

            pairs = list(itertools.combinations(rater_names, 2))
            pairwise_results = []
            for r1, r2 in pairs:
                k = cohen_kappa(aligned[r1], aligned[r2], CODES)
                pairwise_results.append({
                    "rater_a": r1, "rater_b": r2, **k
                })

            pairwise_results.sort(key=lambda x: -x["kappa"])
            best_pair = pairwise_results[0]
            best_kappa = best_pair["kappa"]

            selected_rater = None
            qualified = best_kappa > 0.6
            if qualified:
                selected_rater = sorted([best_pair["rater_a"], best_pair["rater_b"]])[0]

            print(
                f"  sn={sn} ({sn_entry.get('short','')}): "
                f"best pair={best_pair['rater_a']}-{best_pair['rater_b']}, "
                f"kappa={best_kappa:.4f}, "
                f"qualified={qualified}, "
                f"selected={selected_rater}",
                file=sys.stderr,
            )

            result[f"sn_{sn}"] = {
                "sn": sn,
                "title": sn_entry.get("short", sn_entry.get("title", "")),
                "group": group_name,
                "n_windows": n_windows,
                "n_raters": len(rater_names),
                "raters": rater_names,
                "pairwise_kappa": pairwise_results,
                "best_pair": {
                    "rater_a": best_pair["rater_a"],
                    "rater_b": best_pair["rater_b"],
                    "kappa": best_kappa,
                },
                "kappa_threshold": 0.6,
                "qualified": qualified,
                "selected_rater": selected_rater,
            }

            if qualified and selected_rater:
                proto_path = os.path.join(sn_dir, "protagonist.jsonl")
                src_path = os.path.join(sn_dir, "rf", f"{selected_rater}.jsonl")
                with open(src_path, encoding="utf-8") as f:
                    records = [json.loads(line) for line in f if line.strip()]
                with open(proto_path, "w", encoding="utf-8") as f:
                    for r in records:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                print(f"    -> generated {proto_path} from rater {selected_rater}", file=sys.stderr)

    return result


def main():
    parser = argparse.ArgumentParser(description="選擇最佳評分者作為主角情緒範本")
    parser.add_argument("--group-config", default="config/analysis_groups.json")
    parser.add_argument("--protagonist-dir", default="logs")
    parser.add_argument("--output", "-o", default="results")
    args = parser.parse_args()

    config = load_group_config(args.group_config)
    os.makedirs(args.output, exist_ok=True)

    print("Selecting best rater per video...", file=sys.stderr)
    report = select_best_rater(args.protagonist_dir, config, args.output)

    out_path = os.path.join(args.output, "best_rater_selection.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
