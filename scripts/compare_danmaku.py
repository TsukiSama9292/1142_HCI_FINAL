#!/usr/bin/env python3
"""
跨作品彈幕情緒分佈比較工具

比較同類型 vs 不同類型作品之間，觀眾彈幕情緒分佈的相似性。
使用 Jensen-Shannon Divergence、Cosine Similarity 與 Mann-Whitney U 檢定。
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.hci_analysis.lsa import (
    load_danmaku_distributions,
    get_emotion_codes,
    aggregate_work_distribution,
    pairwise_jensenshannon,
    pairwise_cosine,
    within_vs_between_test,
)


def load_group_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="跨作品彈幕情緒分佈比較"
    )
    parser.add_argument("--group-config", required=True, help="群組設定 JSON 路徑")
    parser.add_argument("--danmaku-dir", default="logs",
                        help="彈幕分析 JSONL 根目錄 (預設: logs)")
    parser.add_argument("--output", "-o", default="results",
                        help="輸出目錄 (預設: results)")
    args = parser.parse_args()

    config = load_group_config(args.group_config)
    codes = get_emotion_codes()

    works: dict[str, dict[str, float]] = {}
    work_meta: dict[str, dict] = {}
    group_map: dict[str, str] = {}
    group_info: dict[str, dict] = {}

    for group_name, group in config["groups"].items():
        group_info[group_name] = group.get("meta", {})
        for sn_entry in group["sns"]:
            sn = sn_entry["sn"]
            label = sn_entry.get("short", sn_entry.get("title", f"sn_{sn}"))
            d_path = os.path.join(
                args.danmaku_dir, str(sn), "nlp", "output_label_ana.jsonl"
            )
            if not os.path.exists(d_path):
                print(f"  [SKIP] danmaku not found: {d_path}", file=sys.stderr)
                continue
            records = load_danmaku_distributions(d_path)
            dist = aggregate_work_distribution(records, codes)
            works[label] = dist
            work_meta[label] = {
                "sn": sn,
                "title": sn_entry.get("title", ""),
                "type": group_name,
            }
            group_map[label] = group_name
            print(f"  loaded {label} (sn={sn}): {len(records)} windows, "
                  f"{sum(records[i].get('danmaku_count', 0) for i in range(len(records)))} danmaku",
                  file=sys.stderr)

    if len(works) < 2:
        print("Need at least 2 works to compare. Exiting.", file=sys.stderr)
        sys.exit(1)

    out_dir = os.path.join(args.output, "danmaku_comparison")
    os.makedirs(out_dir, exist_ok=True)

    js_matrix = pairwise_jensenshannon(works, codes)
    cos_matrix = pairwise_cosine(works, codes)

    test_result = within_vs_between_test(works, group_map, codes)

    work_names = list(works.keys())

    per_work = []
    for name in work_names:
        entry = {"work": name, "type": group_map[name]}
        entry["distribution"] = works[name]
        entry.update(work_meta.get(name, {}))
        per_work.append(entry)

    pairwise_entries = []
    for i, w1 in enumerate(work_names):
        for j, w2 in enumerate(work_names):
            if i >= j:
                continue
            pairwise_entries.append({
                "work_a": w1,
                "work_b": w2,
                "type_a": group_map[w1],
                "type_b": group_map[w2],
                "same_type": group_map[w1] == group_map[w2],
                "js_divergence": js_matrix[w1][w2],
                "cosine_similarity": cos_matrix[w1][w2],
            })

    output = {
        "n_works": len(works),
        "per_work_distributions": per_work,
        "pairwise": pairwise_entries,
        "js_divergence_matrix": js_matrix,
        "cosine_similarity_matrix": cos_matrix,
        "within_vs_between": test_result,
        "group_info": group_info,
    }

    out_path = os.path.join(out_dir, "danmaku_comparison.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nSaved: {out_path}", file=sys.stderr)

    if test_result.get("mann_whitney_u"):
        mw = test_result["mann_whitney_u"]
        wt = test_result["within_type"]
        bt = test_result["between_type"]
        print(f"  Within-type mean JS: {wt['mean_divergence']}", file=sys.stderr)
        print(f"  Between-type mean JS: {bt['mean_divergence']}", file=sys.stderr)
        print(f"  Mann-Whitney U: U={mw['U_statistic']}, p={mw['p_value']}, "
              f"significant={mw['significant']}", file=sys.stderr)
        if test_result.get("effect_size"):
            print(f"  Cohen's d: {test_result['effect_size']['cohens_d']}", file=sys.stderr)

    print(f"\nDone. Results in: {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
