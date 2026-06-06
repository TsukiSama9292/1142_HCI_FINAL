import json
import math
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

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
