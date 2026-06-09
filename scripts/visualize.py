#!/usr/bin/env python3
"""Generate visualizations: confusion matrix heatmaps, cosine box plots, Sankey diagram."""

import json
import os
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# Register Chinese font
font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(font_path)
prop = fm.FontProperties(fname=font_path)
plt.rcParams["font.family"] = prop.get_name()
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.hci_analysis.lsa import (
    load_protagonist_emotions,
    load_danmaku_distributions,
    align_sequences,
    get_emotion_codes,
    cosine_similarities,
)

RESULTS_DIR = "results"
VIZ_DIR = "docs/figures"
os.makedirs(VIZ_DIR, exist_ok=True)

POS = set("ABCDEFGHIJKL")
NEG = set("MNOPQRSTUVW")
AMB = set("XYZ@A")
VLIST = ["正面", "負面", "模糊", "中性"]
VMAP = {"positive": 0, "negative": 1, "ambiguous": 2, "neutral": 3}

CODE_ZH_SHORT = {
    "A": "讚賞", "B": "有趣", "C": "認可", "D": "關心", "E": "慾望",
    "F": "興奮", "G": "感激", "H": "喜悅", "I": "愛", "J": "樂觀",
    "K": "自豪", "L": "寬慰", "M": "憤怒", "N": "煩惱", "O": "失望",
    "P": "不認可", "Q": "厭惡", "R": "尷尬", "S": "恐懼", "T": "悲痛",
    "U": "緊張", "V": "自責", "W": "悲傷", "X": "困惑", "Y": "好奇",
    "Z": "領悟", "@A": "驚訝", "@B": "不適用", "@C": "中性",
}

def valence(code):
    if code in POS: return "positive"
    if code in NEG: return "negative"
    if code in AMB: return "ambiguous"
    return "neutral"

def load_group_config(path="config/analysis_groups.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

# ─── Chart 1: Valence Confusion Matrix Heatmaps ───
def plot_confusion_heatmaps():
    with open(os.path.join(RESULTS_DIR, "detailed_analysis.json"), encoding="utf-8") as f:
        data = json.load(f)

    groups = ["外掛爽感型", "心理折磨型", "搞笑解構型"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, group in zip(axes, groups):
        key = f"__group__{group}"
        if key not in data:
            ax.set_title(f"{group}\n(no data)")
            continue
        cm = np.array(data[key]["confusion_matrix"])
        acc = data[key]["valence_accuracy"]
        im = ax.imshow(cm, cmap="YlOrRd", vmin=0, aspect="equal")
        ax.set_xticks(range(4))
        ax.set_xticklabels(VLIST, fontsize=9)
        ax.set_yticks(range(4))
        ax.set_yticklabels(VLIST, fontsize=9)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=10, fontweight="bold")
        ax.set_xlabel("彈幕主導價性", fontsize=9, labelpad=2)
        ax.set_ylabel("主角情緒價性", fontsize=9, labelpad=2)
        ax.set_title(f"{group}\n準確率={acc:.1%}", fontsize=10, pad=4)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5, pad=0.02)
    plt.subplots_adjust(wspace=0.35, left=0.04, right=0.92, bottom=0.12, top=0.85)
    path = os.path.join(VIZ_DIR, "confusion_heatmap.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")

# ─── Chart 2: Cosine Similarity Box Plot ───
def plot_cosine_boxplot():
    config = load_group_config()
    group_data = {}
    for group_name in config["groups"]:
        group_dir = os.path.join(RESULTS_DIR, group_name.replace(" ", "_"))
        cos_path = os.path.join(group_dir, "cosine_similarities.jsonl")
        if not os.path.exists(cos_path):
            continue
        sims = [json.loads(l)["cosine_similarity"] for l in open(cos_path, encoding="utf-8")]
        group_data[group_name] = sims

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = list(group_data.keys())
    data = [group_data[l] for l in labels]
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="red"))
    colors = ["#ff9999", "#99ccff", "#99ff99"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("各類型情緒共鳴分數分佈")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="共鳴門檻 (0.5)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(VIZ_DIR, "cosine_boxplot.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")

# ─── Chart 3: Sankey-like Emotion Path Flow ───
def plot_emotion_sankey():
    """Simplified alluvia-style: group → protagonist → danmaku dominant."""
    config = load_group_config()
    CODES = get_emotion_codes()
    group_emotions = []
    all_links = []
    y_offset = 0
    node_names = []
    node_colors = []
    node_groups = []
    x_positions = []

    group_colors = {"外掛爽感型": "#ff9999", "心理折磨型": "#99ccff", "搞笑解構型": "#228B22"}
    spy = {"A": "red", "B": "orange", "C": "forestgreen", "D": "pink",
           "E": "goldenrod", "F": "gold", "G": "olive", "H": "pink",
           "I": "hotpink", "J": "skyblue", "K": "purple", "L": "teal",
           "M": "darkred", "N": "brown", "O": "saddlebrown", "P": "sienna",
           "Q": "darkgreen", "R": "orchid", "S": "darkblue", "T": "darkred",
           "U": "navy", "V": "crimson", "W": "blue", "X": "gray",
           "Y": "cyan", "Z": "lime", "@A": "cyan", "@B": "lightgray", "@C": "silver"}

    fig, ax = plt.subplots(figsize=(16, 10))

    group_ylims = []

    for gi, (group_name, group_info) in enumerate(config["groups"].items()):
        aligned_list = []
        for se in group_info["sns"]:
            pp = os.path.join("logs", str(se["sn"]), "protagonist.jsonl")
            dp = os.path.join("logs", str(se["sn"]), "nlp", "final_label_ana.jsonl")
            if not (os.path.exists(pp) and os.path.exists(dp)):
                continue
            pr = load_protagonist_emotions(pp)
            dr = load_danmaku_distributions(dp)
            aligned_list.extend(align_sequences(pr, dr))

        if not aligned_list:
            continue

        group_start_y = y_offset

        # Count protagonist emotions
        p_counts = Counter(a[0] for a in aligned_list)
        total = sum(p_counts.values())
        top_p = p_counts.most_common(6)

        # For each protagonist emotion, count dominant danmaku emotions
        for p_code, p_count in top_p:
            d_counter = Counter()
            for pc, dd in aligned_list:
                if pc == p_code:
                    dc = max(dd, key=dd.get) if dd else "@C"
                    d_counter[dc] += 1
            d_total = sum(d_counter.values())
            top_d = d_counter.most_common(3)

            if d_total == 0:
                continue

            nx_p = gi * 400 + 150
            nx_d = gi * 400 + 300

            # Protagonist node
            ax.plot(nx_p, y_offset, "o", color=spy.get(p_code, "gray"),
                    markersize=12 + 8 * p_count / total, zorder=3)
            ax.text(nx_p + 20, y_offset, f"{CODE_ZH_SHORT.get(p_code, p_code)}({p_count})",
                    fontsize=7, va="center")

            for d_code, d_count in top_d:
                if d_count == 0:
                    continue
                ax.plot(nx_d, y_offset, "s", color=spy.get(d_code, "lightgray"),
                        markersize=8 + 6 * d_count / d_total, alpha=0.8, zorder=3)
                ax.text(nx_d + 20, y_offset, f"{CODE_ZH_SHORT.get(d_code, d_code)}({d_count})",
                        fontsize=6, va="center")
                ax.plot([nx_p + 8, nx_d - 8], [y_offset, y_offset], "-",
                        color=spy.get(p_code, "gray"), alpha=0.3 + 0.5 * d_count / d_total,
                        linewidth=0.5 + 2 * d_count / d_total, zorder=1)
                y_offset += 2

        group_end_y = y_offset
        group_center_y = (group_start_y + group_end_y) / 2
        nx_g = gi * 400 + 50
        ax.text(nx_g, group_center_y, group_name, fontsize=11, fontweight="bold",
                color=group_colors.get(group_name, "black"),
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=group_colors.get(group_name, "gray"), alpha=0.9))
        # Column separators
        ax.axvline(x=gi * 400 + 100, color="lightgray", linewidth=0.5, linestyle=":")
        group_ylims.append((group_name, group_start_y, group_end_y))

    ax.set_xlim(-50, (gi + 1) * 400 + 50)
    ax.set_ylim(-5, y_offset + 5)
    ax.axis("off")
    ax.set_title("敘事策略類型 → 主角情緒 → 彈幕主導情緒 流向", fontsize=13, pad=10)
    plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02)
    path = os.path.join(VIZ_DIR, "emotion_sankey.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")

if __name__ == "__main__":
    plot_confusion_heatmaps()
    plot_cosine_boxplot()
    plot_emotion_sankey()
    print(f"\nAll visualizations saved to {VIZ_DIR}/")
