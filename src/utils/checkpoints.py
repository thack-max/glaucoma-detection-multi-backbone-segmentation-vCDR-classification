"""
Unified checkpoint save/load helpers.

Every GlaucoMamba training script saves its own checkpoint format
(`{"model": ...}`, `{"classifier": ...}`, `{"mamba": ..., "forecaster": ...}`),
and each inference script has its own ad-hoc loading heuristic. This is a
deployment friction point — the one-way arrow from training to inference
shouldn't require string matching on dictionary keys.

These helpers define a single canonical checkpoint schema:

    {
        "format_version": "glaucomamba-ckpt-v1",
        "component": "screening_classifier"   # or "boundary_former", ...
        "state_dicts": { "<module_name>": <state_dict>, ... },
        "metadata": { "epoch": N, "val_metrics": {...}, "args": {...} },
    }

`save_checkpoint` writes this schema. `load_checkpoint` accepts both the
new schema AND any of the legacy formats the training scripts produced,
so existing checkpoints keep working.

Why this matters for the original goal:
GlaucoMamba ships as a two-tier pipeline that must survive the hop from
a training cluster to a rural clinic device or a hospital GPU server. A
checkpoint that only loads if the training and inference scripts agree
on a string key is a latent bug waiting to surface at deployment time.
A single schema + a backwards-compatible loader removes that risk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


CHECKPOINT_FORMAT_VERSION = "glaucomamba-ckpt-v1"


def save_checkpoint(
    path: str | Path,
    component: str,
    modules: dict[str, nn.Module],
    metadata: dict[str, Any] | None = None,
) -> None:
    """Write a canonical checkpoint file.

    Args:
        path: Destination .pth path.
        component: Short string identifying what this checkpoint is for
            (e.g. "screening_classifier", "boundary_former",
            "specialist_joint"). Consumers can use this for sanity checks.
        modules: Dict of {name: nn.Module} whose state dicts should be saved.
        metadata: Optional dict of auxiliary info (epoch, val metrics,
            hyperparameters). Must be JSON-serializable.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "component": component,
        "state_dicts": {name: m.state_dict() for name, m in modules.items()},
        "metadata": metadata or {},
    }
    torch.save(state, path)


def load_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint and return a normalized dict.

    Accepts three input shapes:
      1. New format: {"format_version": "...", "state_dicts": {...}, ...}
      2. Legacy dict-of-state-dicts: {"model": sd, "classifier": sd, ...}
      3. Raw state_dict (OrderedDict of tensors).

    Returns:
        {
            "state_dicts": { "<module_name>": <state_dict>, ... },
            "metadata": { ... },
            "component": "<component_name>" or None,
            "format_version": "..." or None,
        }
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    raw = torch.load(path, map_location=map_location, weights_only=False)

    # Case 1: new canonical format
    if isinstance(raw, dict) and raw.get("format_version", "").startswith("glaucomamba-ckpt"):
        return {
            "state_dicts": raw.get("state_dicts", {}),
            "metadata": raw.get("metadata", {}),
            "component": raw.get("component"),
            "format_version": raw.get("format_version"),
        }

    # Case 3: raw state_dict
    if isinstance(raw, dict) and all(
        isinstance(v, torch.Tensor) for v in raw.values()
    ):
        return {
            "state_dicts": {"model": raw},
            "metadata": {},
            "component": None,
            "format_version": None,
        }

    # Case 2: legacy dict-of-state-dicts — split the tensor-valued entries
    # from the metadata entries.
    if isinstance(raw, dict):
        state_dicts: dict[str, dict[str, torch.Tensor]] = {}
        metadata: dict[str, Any] = {}
        for key, value in raw.items():
            if isinstance(value, dict) and value and all(
                isinstance(v, torch.Tensor) for v in value.values()
            ):
                state_dicts[key] = value
            else:
                metadata[key] = value
        if not state_dicts:
            # A dict whose only entries are metadata — nothing to load.
            state_dicts = {}
        return {
            "state_dicts": state_dicts,
            "metadata": metadata,
            "component": None,
            "format_version": None,
        }

    raise ValueError(f"Unrecognized checkpoint format at {path}: {type(raw).__name__}")


def load_into(
    module: nn.Module,
    checkpoint: dict[str, Any],
    key_candidates: list[str] | None = None,
    strict: bool = False,
) -> bool:
    """Load weights from a normalized checkpoint into a module.

    Tries each key in `key_candidates` until one matches a state_dict in
    the checkpoint. If none match, falls back to the first available
    state_dict. Returns True on success, False if no compatible state_dict
    was found.

    Args:
        module: The nn.Module to populate.
        checkpoint: Output of load_checkpoint().
        key_candidates: Preferred state_dict keys, tried in order. Defaults
            to a broad set of conventions the training scripts use.
        strict: Passed through to module.load_state_dict.

    Returns:
        True if a state_dict was loaded, False otherwise.
    """
    if key_candidates is None:
        key_candidates = [
            "model",
            "classifier",
            "segmenter",
            "encoder",
            "mamba",
            "temporal",
            "forecaster",
            "fusion",
            "vf_net",
            "equity",
        ]

    sds = checkpoint.get("state_dicts", {})
    for key in key_candidates:
        if key in sds:
            module.load_state_dict(sds[key], strict=strict)
            return True

    # Last-resort fallback: if there is exactly one state_dict, use it.
    if len(sds) == 1:
        module.load_state_dict(next(iter(sds.values())), strict=strict)
        return True

    return False
