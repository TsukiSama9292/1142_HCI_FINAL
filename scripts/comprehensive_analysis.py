#!/usr/bin/env python3
"""Comprehensive analysis: per-SN, per-group, and pooled valence/path analysis."""

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.hci_analysis.lsa import (
    load_protagonist_emotions,
    load_danmaku_distributions,
    align_sequences,
    get_emotion_codes,
    transition_matrix,
    z_scores,
    cosine_similarities,
    extract_emotion_paths,
)

POS = set("ABCDEFGHIJKL")
NEG = set("MNOPQRSTUVW")
AMB = set("XYZ@A")

CODE_ZH = {
    "A": "讚賞", "B": "有趣", "C": "認可", "D": "關心", "E": "慾望",
    "F": "興奮", "G": "感激", "H": "喜悅", "I": "愛", "J": "樂觀",
    "K": "自豪", "L": "寬慰", "M": "憤怒", "N": "煩惱", "O": "失望",
    "P": "不認可", "Q": "厭惡", "R": "尷尬", "S": "恐懼", "T": "悲痛",
    "U": "緊張", "V": "自責", "W": "悲傷", "X": "困惑", "Y": "好奇",
    "Z": "領悟", "@A": "驚訝", "@B": "不適用", "@C": "中性",
}

VLIST = ["positive", "negative", "ambiguous", "neutral"]
VMAP = {v: i for i, v in enumerate(VLIST)}


def valence(code):
    if code in POS:
        return "positive"
    if code in NEG:
        return "negative"
    if code in AMB:
        return "ambiguous"
    return "neutral"


def analyze_per_sn(sn, label, group_name, protagonist_dir, danmaku_dir):
    p_path = os.path.join(protagonist_dir, str(sn), "protagonist.jsonl")
    d_path = os.path.join(danmaku_dir, str(sn), "nlp", "final_label_ana.jsonl")
    if not (os.path.exists(p_path) and os.path.exists(d_path)):
        print(f"  [SKIP] sn={sn}: missing files", file=sys.stderr)
        return None

    pr = load_protagonist_emotions(p_path)
    dr = load_danmaku_distributions(d_path)
    aligned = align_sequences(pr, dr)
    if not aligned:
        return None

    CODES = get_emotion_codes()

    # Protagonist distribution
    pdist = Counter(r["protagonist_emotion"] for r in pr)
    pt = sum(pdist.values())
    pct = {k: round(v / pt * 100, 1) for k, v in sorted(pdist.items())}

    # Danmaku distribution
    ddist = Counter()
    dt = 0
    for r in dr:
        for code, prop in r["emotion"].items():
            n = r["danmaku_count"]
            ddist[code] += n * prop
            dt += n * prop
    dct_sorted = sorted(ddist.items(), key=lambda x: -x[1]) if dt > 0 else []
    dct = {k: round(v / dt * 100, 1) for k, v in dct_sorted[:10]}

    # Valence confusion matrix
    cm = np.zeros((4, 4), dtype=int)
    for pc, dd in aligned:
        pv = valence(pc)
        dc = max(dd, key=dd.get) if dd else "@C"
        dv = valence(dc)
        cm[VMAP[pv], VMAP[dv]] += 1
    vacc = float(np.trace(cm) / cm.sum()) if cm.sum() > 0 else 0

    # Cosine similarity
    sims = cosine_similarities(aligned, CODES)
    mcos = float(np.mean(sims))
    median_cos = float(np.median(sims))

    # Entropy
    codes = [r["protagonist_emotion"] for r in pr]
    edist = Counter(codes)
    etotal = sum(edist.values())
    entropy = -sum((c / etotal) * np.log2(c / etotal) for c in edist.values()) if etotal > 0 else 0

    # Significant paths
    mat = transition_matrix(aligned, CODES, lag=0)
    z = z_scores(mat)
    paths = extract_emotion_paths(mat, z, CODES)
    sig_paths = []
    for p in paths:
        if p["significant"]:
            sig_paths.append({
                "protagonist_code": p["protagonist_code"],
                "protagonist_zh": p["protagonist_zh"],
                "danmaku_code": p["danmaku_code"],
                "danmaku_zh": p["danmaku_zh"],
                "count": p["observed_count"],
                "z_score": round(p["z_score"], 2),
                "row_proportion": round(p["row_proportion"], 4),
            })

    return {
        "sn": sn,
        "label": label,
        "group": group_name,
        "windows": len(aligned),
        "protagonist_distribution": pct,
        "danmaku_top10": dct,
        "valence_accuracy": round(vacc, 4),
        "confusion_matrix": cm.tolist(),
        "mean_cosine": round(mcos, 4),
        "median_cosine": round(median_cos, 4),
        "entropy": round(entropy, 3),
        "unique_emotions": len(edist),
        "dominant_emotion": edist.most_common(1)[0][0],
        "dominant_ratio": round(edist.most_common(1)[0][1] / etotal, 4),
        "significant_paths": sig_paths,
    }


def analyze_group(aligned_list, group_name):
    if not aligned_list:
        return None
    cm = np.zeros((4, 4), dtype=int)
    for pc, dd in aligned_list:
        pv = valence(pc)
        dc = max(dd, key=dd.get) if dd else "@C"
        dv = valence(dc)
        cm[VMAP[pv], VMAP[dv]] += 1
    vacc = float(np.trace(cm) / cm.sum()) if cm.sum() > 0 else 0
    return {
        "group": group_name,
        "windows": len(aligned_list),
        "confusion_matrix": cm.tolist(),
        "valence_accuracy": round(vacc, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Comprehensive valence and path analysis")
    parser.add_argument("--group-config", required=True, help="群組設定 JSON 路徑")
    parser.add_argument("--protagonist-dir", default="logs", help="主角情緒 JSONL 根目錄")
    parser.add_argument("--danmaku-dir", default="logs", help="彈幕分析 JSONL 根目錄")
    parser.add_argument("--output", "-o", default="results", help="輸出目錄")
    args = parser.parse_args()

    config = json.load(open(args.group_config, encoding="utf-8"))
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    results = {}

    # Per-SN analysis
    print("Per-SN analysis:", file=sys.stderr)
    for group_name, group_info in config["groups"].items():
        for sn_entry in group_info["sns"]:
            sn = sn_entry["sn"]
            label = sn_entry.get("short", str(sn))
            r = analyze_per_sn(sn, label, group_name, args.protagonist_dir, args.danmaku_dir)
            if r:
                results[label] = r
                print(f"  {label}: entropy={r['entropy']:.3f}, valence_acc={r['valence_accuracy']:.3f}, "
                      f"cosine={r['mean_cosine']:.4f}, sig_paths={len(r['significant_paths'])}",
                      file=sys.stderr)

    # Group-level analysis
    print("\nGroup-level analysis:", file=sys.stderr)
    all_aligned = []
    group_aligned = {g: [] for g in config["groups"]}
    for gn, gi in config["groups"].items():
        for se in gi["sns"]:
            pp = os.path.join(args.protagonist_dir, str(se["sn"]), "protagonist.jsonl")
            dp = os.path.join(args.danmaku_dir, str(se["sn"]), "nlp", "final_label_ana.jsonl")
            if not (os.path.exists(pp) and os.path.exists(dp)):
                continue
            pr = load_protagonist_emotions(pp)
            dr = load_danmaku_distributions(dp)
            aligned = align_sequences(pr, dr)
            group_aligned[gn].extend(aligned)
            all_aligned.extend(aligned)

    for gn, ga in group_aligned.items():
        gr = analyze_group(ga, gn)
        if gr:
            results[f"__group__{gn}"] = gr
            print(f"  {gn}: windows={gr['windows']}, valence_acc={gr['valence_accuracy']:.3f}",
                  file=sys.stderr)

    # Pooled analysis
    print("\nPooled analysis:", file=sys.stderr)
    if all_aligned:
        CODES = get_emotion_codes()
        mat = transition_matrix(all_aligned, CODES, lag=0)
        z = z_scores(mat)
        paths = extract_emotion_paths(mat, z, CODES)
        pooled_sig = []
        for p in paths:
            if p["significant"]:
                pooled_sig.append({
                    "protagonist_code": p["protagonist_code"],
                    "protagonist_zh": p["protagonist_zh"],
                    "danmaku_code": p["danmaku_code"],
                    "danmaku_zh": p["danmaku_zh"],
                    "count": p["observed_count"],
                    "z_score": round(p["z_score"], 2),
                    "row_proportion": round(p["row_proportion"], 4),
                })
        results["__pooled_sig_paths"] = pooled_sig
        print(f"  Pooled significant paths: {len(pooled_sig)}", file=sys.stderr)
        for ps in pooled_sig:
            print(f"    {ps['protagonist_code']}({ps['protagonist_zh']}) -> "
                  f"{ps['danmaku_code']}({ps['danmaku_zh']}): count={ps['count']}, "
                  f"Z={ps['z_score']}, row%={ps['row_proportion']:.1%}", file=sys.stderr)

        # Pooled valence accuracy
        cm_pooled = np.zeros((4, 4), dtype=int)
        for pc, dd in all_aligned:
            pv = valence(pc)
            dc = max(dd, key=dd.get) if dd else "@C"
            dv = valence(dc)
            cm_pooled[VMAP[pv], VMAP[dv]] += 1
        pooled_vacc = float(np.trace(cm_pooled) / cm_pooled.sum()) if cm_pooled.sum() > 0 else 0
        results["__pooled"] = {
            "windows": len(all_aligned),
            "valence_accuracy": round(pooled_vacc, 4),
        }

    out_path = os.path.join(output_dir, "detailed_analysis.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
