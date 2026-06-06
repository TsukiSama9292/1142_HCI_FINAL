import json
import itertools
import math
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, norm

from .emotions import EMOTION_CODES


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
    chi2, p, dof, expected = chi2_contingency(matrix)
    return {"chi2": float(chi2), "p": float(p), "dof": int(dof)}


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
