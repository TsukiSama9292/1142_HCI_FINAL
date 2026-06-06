import json
import itertools
import math
import os
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, norm, mannwhitneyu
from scipy.spatial.distance import jensenshannon

from .emotions import EMOTION_CODES


CODE_TO_EN: dict[str, str] = {v: k for k, v in EMOTION_CODES.items()}
CODE_TO_ZH: dict[str, str] = {
    "A": "讚賞", "B": "有趣", "C": "認可", "D": "關心", "E": "慾望",
    "F": "興奮", "G": "感激", "H": "喜悅", "I": "愛", "J": "樂觀",
    "K": "自豪", "L": "寬慰", "M": "憤怒", "N": "煩惱", "O": "失望",
    "P": "不認可", "Q": "厭惡", "R": "尷尬", "S": "恐懼", "T": "悲痛",
    "U": "緊張", "V": "自責", "W": "悲傷", "X": "困惑", "Y": "好奇",
    "Z": "領悟", "@A": "驚訝", "@B": "不適用", "@C": "中性",
}


def sec_to_ts(sec: int) -> str:
    m = sec // 60
    s = sec % 60
    return f"0:{m:02d}:{s:02d}.00"


def parse_rf_grid(text: str) -> list[str]:
    codes = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        i = 0
        while i < len(line):
            if line[i] == '@' and i + 2 <= len(line):
                codes.append(line[i:i+2])
                i += 2
            else:
                codes.append(line[i])
                i += 1
    return codes


def convert_rf_txt_to_jsonl(
    txt_path: str,
    jsonl_path: str,
    start_sec: int = 600,
    window: int = 8,
):
    with open(txt_path, encoding="utf-8") as f:
        text = f.read()
    codes = parse_rf_grid(text)
    records = []
    for i, code in enumerate(codes):
        start = start_sec + i * window
        end = start_sec + (i + 1) * window
        records.append({
            "segment_start": sec_to_ts(start),
            "segment_end": sec_to_ts(end),
            "protagonist_emotion": code,
        })
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return records


def load_protagonist_emotions(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    records.sort(key=lambda x: x["segment_start"])
    return records


def load_danmaku_distributions(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    records.sort(key=lambda x: x["segment_start"])
    return records


def get_emotion_codes() -> list[str]:
    order = [chr(ord("A") + i) for i in range(26)] + ["@A", "@B", "@C"]
    return order


def align_sequences(
    protagonist: list[dict],
    danmaku: list[dict],
) -> list[tuple[str, dict[str, float]]]:
    danmaku_map = {r["segment_start"]: r["emotion"] for r in danmaku}
    aligned = []
    for r in protagonist:
        seg_start = r["segment_start"]
        p_emotion = r["protagonist_emotion"]
        d_distrib = danmaku_map.get(seg_start)
        if d_distrib is not None:
            aligned.append((p_emotion, d_distrib))
    return aligned


def one_hot(code: str, codes: list[str]) -> np.ndarray:
    arr = np.zeros(len(codes), dtype=float)
    if code in codes:
        arr[codes.index(code)] = 1.0
    return arr


def dominant_emotion(distrib: dict[str, float], fallback: str = "@C") -> str:
    if not distrib:
        return fallback
    return max(distrib, key=distrib.get)


def transition_matrix(
    aligned: list[tuple[str, dict[str, float]]],
    codes: list[str],
    lag: int = 0,
) -> np.ndarray:
    n = len(codes)
    matrix = np.zeros((n, n), dtype=np.int64)
    for i in range(len(aligned) - lag):
        p_code = aligned[i][0]
        if lag == 0:
            d_code = dominant_emotion(aligned[i][1])
        else:
            d_code = dominant_emotion(aligned[i + lag][1])
        if p_code in codes and d_code in codes:
            matrix[codes.index(p_code), codes.index(d_code)] += 1
    return matrix


def z_scores(matrix: np.ndarray) -> np.ndarray:
    total = matrix.sum()
    if total == 0:
        return np.zeros_like(matrix, dtype=float)
    row_sums = matrix.sum(axis=1, keepdims=True)
    col_sums = matrix.sum(axis=0, keepdims=True)
    expected = (row_sums @ col_sums) / total
    p = col_sums / total
    variance = expected * (1 - p)
    variance = np.where(variance > 0, variance, 1e-10)
    z = (matrix - expected) / np.sqrt(variance)
    return z.astype(float)


def adjusted_residuals(matrix: np.ndarray) -> np.ndarray:
    total = matrix.sum()
    if total == 0:
        return np.zeros_like(matrix, dtype=float)
    row_sums = matrix.sum(axis=1, keepdims=True)
    col_sums = matrix.sum(axis=0, keepdims=True)
    expected = (row_sums @ col_sums) / total
    variance = expected * (1 - row_sums / total) * (1 - col_sums / total)
    variance = np.where(variance > 0, variance, 1e-10)
    resid = (matrix - expected) / np.sqrt(variance)
    return resid.astype(float)


def extract_emotion_paths(
    mat: np.ndarray,
    z: np.ndarray,
    codes: list[str],
    significance_threshold: float = 1.96,
) -> list[dict]:
    paths = []
    total_obs = mat.sum()
    for i, p_code in enumerate(codes):
        row_total = mat[i].sum()
        if row_total == 0:
            continue
        for j, d_code in enumerate(codes):
            count = int(mat[i, j])
            if count == 0:
                continue
            z_val = float(z[i, j])
            paths.append({
                "protagonist_code": p_code,
                "protagonist_en": CODE_TO_EN.get(p_code, p_code),
                "protagonist_zh": CODE_TO_ZH.get(p_code, p_code),
                "danmaku_code": d_code,
                "danmaku_en": CODE_TO_EN.get(d_code, d_code),
                "danmaku_zh": CODE_TO_ZH.get(d_code, d_code),
                "observed_count": count,
                "row_proportion": round(count / row_total, 4),
                "total_proportion": round(count / total_obs, 4) if total_obs else 0,
                "z_score": round(z_val, 4),
                "significant": z_val > significance_threshold,
            })
    paths.sort(key=lambda x: -x["z_score"])
    return paths


def cosine_similarities(
    aligned: list[tuple[str, dict[str, float]]],
    codes: list[str],
) -> list[float]:
    sims = []
    for p_code, d_distrib in aligned:
        p_vec = one_hot(p_code, codes)
        d_vec = np.array([d_distrib.get(c, 0.0) for c in codes], dtype=float)
        norm_p = np.linalg.norm(p_vec)
        norm_d = np.linalg.norm(d_vec)
        if norm_p == 0 or norm_d == 0:
            sims.append(0.0)
        else:
            sims.append(float(np.dot(p_vec, d_vec) / (norm_p * norm_d)))
    return sims


def resonance_ratio(
    similarities: list[float],
    threshold: float = 0.5,
) -> tuple[float, int, int]:
    total = len(similarities)
    if total == 0:
        return 0.0, 0, 0
    resonant = sum(1 for s in similarities if s > threshold)
    return resonant / total, resonant, total - resonant


def matrix_to_df(
    matrix: np.ndarray,
    codes: list[str],
    fmt: str = ".4f",
) -> pd.DataFrame:
    return pd.DataFrame(matrix, index=codes, columns=codes)


def chi_square_test(matrix: np.ndarray) -> dict:
    if matrix.sum() == 0:
        return {"chi2": None, "p": None, "dof": None}
    try:
        chi2, p, dof, expected = chi2_contingency(matrix)
        return {"chi2": float(chi2), "p": float(p), "dof": int(dof)}
    except ValueError as e:
        return {"chi2": None, "p": None, "dof": None, "error": str(e)}


def pool_aligned(
    video_data: dict[str, list[tuple[str, dict[str, float]]]],
) -> list[tuple[str, dict[str, float]]]:
    pooled = []
    for _sn, aligned in video_data.items():
        pooled.extend(aligned)
    return pooled


def load_rater_emotions(sn_dir: str, raters: list[str] = None) -> dict[str, list[dict]]:
    if raters is None:
        raters = ["a", "b", "c", "d"]
    result = {}
    import os
    for r in raters:
        path = os.path.join(sn_dir, "rf", f"{r}.jsonl")
        if os.path.exists(path):
            result[r] = load_protagonist_emotions(path)
    return result


def align_raters(rater_data: dict[str, list[dict]]) -> dict[str, list[str]]:
    keys = None
    for r, records in rater_data.items():
        cur = [rec["segment_start"] for rec in records]
        if keys is None:
            keys = set(cur)
        else:
            keys &= set(cur)
    if not keys:
        return {}
    aligned: dict[str, list[str]] = {}
    for r, records in rater_data.items():
        lookup = {rec["segment_start"]: rec["protagonist_emotion"] for rec in records}
        aligned[r] = [lookup[k] for k in sorted(keys)]
    return aligned


def cohen_kappa(r1: list[str], r2: list[str], codes: list[str] | None = None) -> dict:
    n = len(r1)
    if n == 0:
        return {"kappa": 0.0, "z": 0.0, "p": 1.0, "n": 0, "agreement": 0.0}
    observed = sum(1 for a, b in zip(r1, r2) if a == b) / n
    if codes is None:
        codes = sorted(set(r1) | set(r2))
    proportions = []
    for c in codes:
        p1 = sum(1 for a in r1 if a == c) / n
        p2 = sum(1 for a in r2 if a == c) / n
        proportions.append(p1 * p2)
    expected = sum(proportions)
    denom = 1 - expected
    if denom == 0:
        return {"kappa": 0.0, "z": 0.0, "p": 1.0, "n": n, "agreement": round(observed, 4)}
    kappa = (observed - expected) / denom
    var = observed * (1 - observed) / (n * denom * denom)
    se = math.sqrt(var) if var > 0 else 1e-10
    z = kappa / se
    p = 2 * (1 - norm.cdf(abs(z)))
    return {
        "kappa": round(kappa, 4),
        "z": round(z, 4),
        "p": round(p, 4),
        "n": n,
        "agreement": round(observed, 4),
    }


def fleiss_kappa(ratings: list[list[str]], codes: list[str] | None = None) -> dict:
    n_subjects = len(ratings)
    if n_subjects == 0:
        return {"kappa": 0.0, "z": 0.0, "p": 1.0, "n": 0}
    n_raters = len(ratings[0])
    if codes is None:
        codes = sorted(set(c for row in ratings for c in row))
    k = len(codes)
    code_index = {c: i for i, c in enumerate(codes)}
    matrix = np.zeros((n_subjects, k), dtype=np.int64)
    for i, row in enumerate(ratings):
        for c in row:
            if c in code_index:
                matrix[i, code_index[c]] += 1

    p_i = (np.sum(matrix ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_bar = np.mean(p_i)
    p_j = matrix.sum(axis=0) / (n_subjects * n_raters)
    p_e = np.sum(p_j ** 2)
    denom = 1 - p_e
    if denom == 0:
        return {"kappa": 0.0, "z": 0.0, "p": 1.0, "n": n_subjects}
    kappa = (p_bar - p_e) / denom
    sum_pj2 = np.sum(p_j ** 2)
    sum_pj3 = np.sum(p_j ** 3)
    var = 2 * (sum_pj2 - (2 * n_raters - 3) * sum_pj3 + 2 * (n_raters - 2) * sum_pj2 ** 2)
    var /= n_subjects * n_raters * (n_raters - 1) * denom ** 2
    se = math.sqrt(var) if var > 0 else 1e-10
    z = kappa / se
    p = 2 * (1 - norm.cdf(abs(z)))
    return {
        "kappa": round(float(kappa), 4),
        "z": round(float(z), 4),
        "p": round(float(p), 4),
        "n": n_subjects,
        "n_raters": n_raters,
    }


def aggregate_work_distribution(
    records: list[dict],
    codes: list[str],
) -> dict[str, float]:
    counts = {c: 0.0 for c in codes}
    total = 0.0
    for r in records:
        n = r.get("danmaku_count", 0)
        for code, prop in r.get("emotion", {}).items():
            counts[code] = counts.get(code, 0.0) + n * prop
        total += n
    if total == 0:
        return {c: 0.0 for c in codes}
    return {c: round(counts[c] / total, 6) for c in codes}


def pairwise_jensenshannon(
    distributions: dict[str, dict[str, float]],
    codes: list[str],
) -> dict[str, dict[str, float]]:
    works = list(distributions.keys())
    result: dict[str, dict[str, float]] = {w: {} for w in works}
    for w1, w2 in itertools.combinations(works, 2):
        p = np.array([distributions[w1].get(c, 0.0) for c in codes], dtype=float)
        q = np.array([distributions[w2].get(c, 0.0) for c in codes], dtype=float)
        js = float(jensenshannon(p, q, base=2))
        result[w1][w2] = round(js, 6)
        result[w2][w1] = round(js, 6)
    for w in works:
        result[w][w] = 0.0
    return result


def pairwise_cosine(
    distributions: dict[str, dict[str, float]],
    codes: list[str],
) -> dict[str, dict[str, float]]:
    works = list(distributions.keys())
    result: dict[str, dict[str, float]] = {w: {} for w in works}
    for w1, w2 in itertools.combinations(works, 2):
        p = np.array([distributions[w1].get(c, 0.0) for c in codes], dtype=float)
        q = np.array([distributions[w2].get(c, 0.0) for c in codes], dtype=float)
        n1 = np.linalg.norm(p)
        n2 = np.linalg.norm(q)
        if n1 == 0 or n2 == 0:
            cos = 0.0
        else:
            cos = float(np.dot(p, q) / (n1 * n2))
        result[w1][w2] = round(cos, 6)
        result[w2][w1] = round(cos, 6)
    for w in works:
        result[w][w] = 1.0
    return result


def within_vs_between_test(
    distributions: dict[str, dict[str, float]],
    group_map: dict[str, str],
    codes: list[str],
) -> dict:
    works = list(distributions.keys())
    within_divs: list[float] = []
    between_divs: list[float] = []
    within_pairs: list[tuple[str, str]] = []
    between_pairs: list[tuple[str, str]] = []
    for w1, w2 in itertools.combinations(works, 2):
        p = np.array([distributions[w1].get(c, 0.0) for c in codes], dtype=float)
        q = np.array([distributions[w2].get(c, 0.0) for c in codes], dtype=float)
        js = float(jensenshannon(p, q, base=2))
        if group_map.get(w1) == group_map.get(w2):
            within_divs.append(js)
            within_pairs.append((w1, w2))
        else:
            between_divs.append(js)
            between_pairs.append((w1, w2))
    result: dict = {
        "within_type": {
            "pairs": [
                {"work_a": a, "work_b": b, "js_divergence": v}
                for (a, b), v in zip(within_pairs, within_divs)
            ],
            "mean_divergence": round(float(np.mean(within_divs)), 6) if within_divs else None,
            "std_divergence": round(float(np.std(within_divs)), 6) if within_divs else None,
        },
        "between_type": {
            "pairs": [
                {"work_a": a, "work_b": b, "js_divergence": v}
                for (a, b), v in zip(between_pairs, between_divs)
            ],
            "mean_divergence": round(float(np.mean(between_divs)), 6) if between_divs else None,
            "std_divergence": round(float(np.std(between_divs)), 6) if between_divs else None,
        },
    }
    if within_divs and between_divs:
        u_stat, p_val = mannwhitneyu(within_divs, between_divs, alternative="less")
        result["mann_whitney_u"] = {
            "U_statistic": int(u_stat),
            "p_value": round(float(p_val), 6),
            "alternative": "within < between",
            "significant": bool(p_val < 0.05),
        }
        cohens_d = (
            float(np.mean(between_divs) - np.mean(within_divs))
            / float(np.sqrt((np.std(within_divs) ** 2 + np.std(between_divs) ** 2) / 2))
        )
        result["effect_size"] = {"cohens_d": round(float(cohens_d), 4)}
    return result
