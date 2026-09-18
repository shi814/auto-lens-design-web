"""Small inference-only utility surface used by the web application."""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch

from models import inference_model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_SEED = 2026


def set_random_seed(seed: int = DEFAULT_SEED, deterministic: bool = True) -> None:
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_torch_generator(seed: int = DEFAULT_SEED, offset: int = 0) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(int(seed) + int(offset))
    return generator


def set_parser():
    """Parse only options required by the deployed inference path."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_name", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--origin_csv", default="data/normalization_reference.csv")
    parser.add_argument("--material_csv", default="glass/material_catalog.csv")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_seq_length", type=int, default=13)
    parser.add_argument("--seq_lengths", default="7,9,11,13")
    parser.add_argument("--ri_atol", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--disable_rms_filter", dest="enable_rms_filter", action="store_false")
    parser.add_argument("--disable_efl_filter", dest="enable_efl_filter", action="store_false")
    parser.add_argument("--no_export_zmx_json", dest="export_zmx_json", action="store_false")
    parser.set_defaults(
        enable_rms_filter=True,
        enable_efl_filter=True,
        export_zmx_json=False,
    )
    opt = parser.parse_args()

    # Fixed architecture and optical-evaluation settings matching the weights.
    defaults = {
        "epochs": 5000,
        "input_size": 256,
        "hidden_size": 512,
        "output_size": 3,
        "glass_head_mode": "classification",
        "num_layers": 6,
        "num_heads": 8,
        "sys_dim": 2,
        "nWL": 3,
        "nRayDensity": 11,
        "nField": 3,
        "EPD": 4.0,
        "w_rms": 1.0,
        "w_distortion": 10.0,
        "w_tele": 1.0,
        "distortion_mode": "zemax_ftan",
        "distortion_ref_angle_deg": 0.01,
        "max_rmsSpotR": 0.04,
        "spot_weight": 0,
        "test_efl_error_threshold": 0.1,
        "enable_efl_first_order_control": True,
        "efl_loss_mode": "trace",
        "efl_loss_tolerance": 0.1,
        "efl_slope_floor": 1e-6,
        "efl_cap_factor": 10.0,
        "efl_first_order_weight": 0.5,
        "efl_first_order_tolerance": 0.1,
        "skip_missing_ri": False,
        "export_row": -1,
        "export_out_dir": "",
        "export_glass_matching_csv": "",
        "gt_match_report": False,
    }
    for name, value in defaults.items():
        setattr(opt, name, value)
    return opt


def get_OTS_CT():
    ri_atol = float(os.environ.get("SCANLENS_RI_ATOL", "1e-5"))
    material_csv = os.environ.get(
        "SCANLENS_MATERIAL_CSV", "./glass/material_catalog.csv"
    )
    material = np.loadtxt(material_csv, delimiter=",", dtype=float)
    tensor = torch.as_tensor(material, dtype=torch.float32, device=device)
    refractive_indices = tensor[:, :3]
    radii = tensor[:, 3:5]
    radii = torch.where(
        torch.isclose(radii, torch.zeros_like(radii), atol=1e-12),
        torch.zeros_like(radii),
        radii,
    )
    curvatures = torch.where(
        radii != 0, 1.0 / radii, torch.zeros_like(radii)
    )
    thickness = tensor[:, 5:6]

    keys = torch.round(refractive_indices / ri_atol).to(torch.long)
    unique_keys, inverse = torch.unique(keys, dim=0, return_inverse=True)
    counts = torch.bincount(inverse, minlength=unique_keys.size(0))
    max_candidates = int(counts.max().item())
    groups = unique_keys.size(0)

    group_c = torch.zeros(groups, max_candidates, 2, device=device)
    group_t = torch.zeros(groups, max_candidates, 1, device=device)
    group_mask = torch.zeros(
        groups, max_candidates, dtype=torch.bool, device=device
    )
    for group in range(groups):
        indices = (inverse == group).nonzero(as_tuple=True)[0]
        count = indices.numel()
        group_c[group, :count] = curvatures[indices]
        group_t[group, :count] = thickness[indices]
        group_mask[group, :count] = True
    return group_mask, group_c, group_t, unique_keys


def load_dict(model, checkpoint):
    model_state = model.state_dict()
    compatible = {key: value for key, value in checkpoint.items() if key in model_state}
    model_state.update(compatible)
    model.load_state_dict(model_state)
    return model


def create_transformer_val(opt, load_name):
    group_mask, group_c, group_t, unique_keys = get_OTS_CT()
    model = inference_model.LensTransformer(
        opt, group_mask, group_c, group_t, unique_keys
    )
    checkpoint = torch.load(load_name, map_location="cpu", weights_only=False)
    load_dict(model, checkpoint)
    return model


class LensBatch:
    __slots__ = ("X", "N_bgr", "CT")

    def __init__(self, X, N_bgr, CT):
        self.X = X
        self.N_bgr = N_bgr
        self.CT = CT
