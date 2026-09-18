"""Normalization helpers required by inference."""

from __future__ import annotations

import os

import numpy as np
import torch


_ORIGIN_DATA_CACHE: np.ndarray | None = None
ORIGIN_DATA_PATH = os.environ.get(
    "SCANLENS_ORIGIN_CSV", "./data/normalization_reference.csv"
)
TEST_DATA_PATH = os.environ.get("SCANLENS_TEST_CSV", "")
MAX_SURFACES = 13
N_WAVELENGTHS = 3


def load_origin_data() -> np.ndarray:
    global _ORIGIN_DATA_CACHE
    if _ORIGIN_DATA_CACHE is None:
        _ORIGIN_DATA_CACHE = np.loadtxt(ORIGIN_DATA_PATH, delimiter=",", dtype=float)
        if _ORIGIN_DATA_CACHE.ndim == 1:
            _ORIGIN_DATA_CACHE = _ORIGIN_DATA_CACHE.reshape(1, -1)
    return _ORIGIN_DATA_CACHE


def load_test_data(csv_path: str | None = None):
    path = csv_path or os.environ.get("SCANLENS_TEST_CSV") or TEST_DATA_PATH
    if not path:
        raise ValueError("No inference CSV was provided.")
    data = np.loadtxt(path, delimiter=",", dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    feature_end = 3 + MAX_SURFACES * N_WAVELENGTHS
    x_sys = normalize_dataSys(data[:, :3])
    x_bgr = normalize_dataBGR(data[:, 3:feature_end])
    x_type = data[:, feature_end:]
    return x_sys, x_bgr, x_type


def normalize_dataSys(data: np.ndarray, LB: float = 0, UB: float = 1) -> np.ndarray:
    reference = load_origin_data()[:, :2]
    data_min = np.min(reference, axis=0)
    data_max = np.max(reference, axis=0)
    denominator = data_max - data_min
    denominator[denominator == 0] = 1
    normalized = data.copy()
    normalized[:, :2] = (
        (UB - LB) * (data[:, :2] - data_min) / denominator + LB
    )
    return normalized


def normalize_dataBGR(data: np.ndarray, LB: float = 0, UB: float = 1) -> np.ndarray:
    feature_end = 3 + MAX_SURFACES * N_WAVELENGTHS
    reference = load_origin_data()[:, 3:feature_end]
    data_min = np.min(reference, axis=0)
    data_max = np.max(reference, axis=0)
    denominator = data_max - data_min
    denominator[denominator == 0] = 1
    normalized = (UB - LB) * (data - data_min) / denominator + LB
    normalized[:, (data_min == 0) & (data_max == 0)] = -1
    normalized[:, (data_min == 1) & (data_max == 1)] = 0
    return normalized


def convert2real_dataSys(
    normalized: torch.Tensor, LB: float = 0, UB: float = 1
) -> torch.Tensor:
    reference = torch.as_tensor(
        load_origin_data()[:, :2],
        dtype=normalized.dtype,
        device=normalized.device,
    )
    minimum = reference.amin(dim=0)
    maximum = reference.amax(dim=0)
    return (normalized - LB) * (maximum - minimum) / (UB - LB) + minimum


def convert2real_dataBGR(
    normalized: torch.Tensor, LB: float = 0, UB: float = 1
) -> torch.Tensor:
    feature_end = 3 + MAX_SURFACES * N_WAVELENGTHS
    reference = torch.as_tensor(
        load_origin_data()[:, 3:feature_end],
        dtype=normalized.dtype,
        device=normalized.device,
    )
    minimum = reference.amin(dim=0)
    maximum = reference.amax(dim=0)
    real = (normalized - LB) * (maximum - minimum) / (UB - LB) + minimum
    constant = torch.isclose(minimum, maximum)
    real[:, constant] = minimum[constant]
    return real
