#!/usr/bin/env python3
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.hci_analysis.lsa import (
    load_protagonist_emotions,
    load_danmaku_distributions,
    align_sequences,
    get_emotion_codes,
    transition_matrix,
    z_scores,
    adjusted_residuals,
    cosine_similarities,
    resonance_ratio,
    matrix_to_df,
    chi_square_test,
    pool_aligned,
    dominant_emotion,
)


CODES = get_emotion_codes()


def load_video(sn: int, protagonist_dir: str, danmaku_dir: str):
    p_path = os.path.join(protagonist_dir, str(sn), "protagonist.jsonl")
    d_path = os.path.join(danmaku_dir, str(sn), "nlp", "output_label_ana.jsonl")
    if not os.path.exists(p_path):
        print(f"  [SKIP] protagonist not found: {p_path}", file=sys.stderr)
        return None
    if not os.path.exists(d_path):
        print(f"  [SKIP] danmaku not found: {d_path}", file=sys.stderr)
        return None
    p_records = load_protagonist_emotions(p_path)
    d_records = load_danmaku_distributions(d_path)
    aligned = align_sequences(p_records, d_records)
    if not aligned:
        print(f"  [SKIP] no aligned windows: sn={sn}", file=sys.stderr)
        return None
    print(f"  loaded sn={sn}: {len(p_records)} protagonist, {len(aligned)} aligned", file=sys.stderr)
    return aligned


def analyze_group(aligned: list, label: str, output_dir: str, sn_label: str = ""):
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"  Group: {label}  ({sn_label})", file=sys.stderr)
    print(f"  Windows: {len(aligned)}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    group_dir = os.path.join(output_dir, _safe_name(label))
    os.makedirs(group_dir, exist_ok=True)

    code_counts = Counter(a[0] for a in aligned)
    d_distribs = []
    for _, distrib in aligned:
        d_distribs.extend(distrib.keys())
    danmaku_counts = Counter(d_distribs)

    proto_df = pd.DataFrame({
        "code": list(code_counts.keys()),
        "count": list(code_counts.values()),
    }).sort_values("count", ascending=False)
    proto_df.to_json(os.path.join(group_dir, "protagonist_distribution.jsonl"),
                     orient="records", force_ascii=False, lines=True)

    danmaku_df = pd.DataFrame({
        "code": list(danmaku_counts.keys()),
        "count": list(danmaku_counts.values()),
    }).sort_values("count", ascending=False)
    danmaku_df.to_json(os.path.join(group_dir, "danmaku_distribution.jsonl"),
                       orient="records", force_ascii=False, lines=True)

    sims = cosine_similarities(aligned, CODES)
    res_ratio, res_n, dis_n = resonance_ratio(sims)
    sim_df = pd.DataFrame({
        "cosine_similarity": sims,
        "resonance": ["resonance" if s > 0.5 else "dissonance" for s in sims],
    })
    sim_df.to_json(os.path.join(group_dir, "cosine_similarities.jsonl"),
                   orient="records", force_ascii=False, lines=True)

    summary = {
        "group": label,
        "total_windows": len(aligned),
        "resonance_ratio": round(res_ratio, 4),
        "resonance_count": res_n,
        "dissonance_count": dis_n,
        "mean_cosine": round(float(np.mean(sims)), 4),
        "std_cosine": round(float(np.std(sims)), 4),
    }
    with open(os.path.join(group_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    lags = [0, 1, 2]
    for lag in lags:
        mat = transition_matrix(aligned, CODES, lag=lag)
        z = z_scores(mat)
        ar = adjusted_residuals(mat)

        mat_df = matrix_to_df(mat, CODES, fmt="d")
        mat_df.to_csv(os.path.join(group_dir, f"transition_matrix_lag{lag}.csv"))

        z_df = matrix_to_df(z, CODES)
        z_df.to_csv(os.path.join(group_dir, f"z_scores_lag{lag}.csv"))

        ar_df = matrix_to_df(ar, CODES)
        ar_df.to_csv(os.path.join(group_dir, f"adjusted_residuals_lag{lag}.csv"))

        sig = extract_significant_paths(mat, z, CODES)
        sig_path = os.path.join(group_dir, f"significant_paths_lag{lag}.jsonl")
        with open(sig_path, "w", encoding="utf-8") as f:
            for row in sig:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return summary, sims


def extract_significant_paths(
    mat: np.ndarray, z: np.ndarray, codes: list[str], threshold: float = 1.96
) -> list[dict]:
    paths = []
    for i, p_code in enumerate(codes):
        for j, d_code in enumerate(codes):
            if z[i, j] > threshold and mat[i, j] > 0:
                paths.append({
                    "protagonist_emotion": p_code,
                    "danmaku_emotion": d_code,
                    "observed_count": int(mat[i, j]),
                    "z_score": round(float(z[i, j]), 4),
                })
    paths.sort(key=lambda x: -x["z_score"])
    return paths


def _safe_name(s: str) -> str:
    return s.replace("/", "_").replace(" ", "_")


def load_group_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="主角情緒 vs 彈幕情緒 LSA 分析")
    parser.add_argument("--group-config", required=True, help="群組設定 JSON 路徑")
    parser.add_argument("--protagonist-dir", default="logs",
                        help="主角情緒 JSONL 根目錄 (預設: logs)")
    parser.add_argument("--danmaku-dir", default="logs",
                        help="彈幕分析 JSONL 根目錄 (預設: logs)")
    parser.add_argument("--output", "-o", default="results",
                        help="輸出目錄 (預設: results)")
    args = parser.parse_args()

    config = load_group_config(args.group_config)
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    all_aligned: dict[str, list] = {}
    meta: dict[str, dict] = {}

    for group_name, group_info in config["groups"].items():
        print(f"\nLoading group: {group_name}", file=sys.stderr)
        group_aligned = []
        for sn_entry in group_info["sns"]:
            sn = sn_entry["sn"]
            result = load_video(sn, args.protagonist_dir, args.danmaku_dir)
            if result is not None:
                group_aligned.extend(result)

        if group_aligned:
            all_aligned[group_name] = group_aligned
            meta[group_name] = group_info.get("meta", {})
        else:
            print(f"  [WARN] no data for group: {group_name}", file=sys.stderr)

    if not all_aligned:
        print("No data loaded. Exiting.", file=sys.stderr)
        sys.exit(1)

    group_summaries = {}
    all_sims = []

    for group_name, aligned in all_aligned.items():
        summary, sims = analyze_group(aligned, group_name, output_dir)
        group_summaries[group_name] = summary
        all_sims.append((group_name, sims))

    if len(all_aligned) >= 2:
        pooled = pool_aligned(all_aligned)
        analyze_group(pooled, "全部 (All)", output_dir, sn_label="pooled")

        print(f"\n{'='*60}", file=sys.stderr)
        print("  Cross-group comparisons", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        cross_dir = os.path.join(output_dir, "cross_group")
        os.makedirs(cross_dir, exist_ok=True)

        group_names = list(all_aligned.keys())
        for i in range(len(group_names)):
            for j in range(i + 1, len(group_names)):
                g1 = group_names[i]
                g2 = group_names[j]
                mat1 = transition_matrix(all_aligned[g1], CODES, lag=0)
                mat2 = transition_matrix(all_aligned[g2], CODES, lag=0)

                stacked = np.array([mat1.sum(), mat2.sum()])
                if stacked.min() > 0:
                    chi2_test = chi_square_test(np.vstack([
                        mat1.sum(axis=1),
                        mat2.sum(axis=1),
                    ]))
                else:
                    chi2_test = {"chi2": None, "p": None, "dof": None}

                cross_record = {
                    "group_a": g1,
                    "group_b": g2,
                    "chi_square": chi2_test,
                }
                cross_path = os.path.join(cross_dir, f"{_safe_name(g1)}_vs_{_safe_name(g2)}.json")
                with open(cross_path, "w", encoding="utf-8") as f:
                    json.dump(cross_record, f, ensure_ascii=False, indent=2)

    summary_path = os.path.join(output_dir, "all_summaries.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(group_summaries, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Results in: {output_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
