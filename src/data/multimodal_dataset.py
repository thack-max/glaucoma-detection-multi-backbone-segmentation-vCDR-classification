"""
Multimodal longitudinal dataset for Tier 2 progression forecasting.

The dataset groups observations by patient. Each patient has T visits with
possibly multiple modalities (fundus, OCT, VF, clinical) and a forecast
target (progression label, future RNFL thickness).

Manifest format (CSV at dataset_root/manifest.csv):

    patient_id,visit_index,timestamp_days,fundus_path,oct_path,vf_path,
    rnfl_thickness,iop,age,sex,group_id,progression_label

Groups are produced by iterating unique patient IDs; each __getitem__ returns
a full patient trajectory padded to `observation_window` visits.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.fundus_dataset import _read_image


@dataclass
class Visit:
    """A single longitudinal visit for a patient."""

    timestamp_days: float
    fundus_path: Path | None = None
    oct_path: Path | None = None
    vf_path: Path | None = None
    rnfl_thickness: float | None = None
    clinical: dict[str, float] = field(default_factory=dict)


@dataclass
class PatientTrajectory:
    """All visits for a single patient."""

    patient_id: str
    visits: list[Visit]
    progression_label: int | None = None
    group_id: int | None = None


def _parse_optional_float(val: str) -> float | None:
    if val in ("", "nan", "NaN", None):
        return None
    try:
        return float(val)
    except ValueError:
        return None


def _parse_optional_int(val: str) -> int | None:
    f = _parse_optional_float(val)
    return int(f) if f is not None else None


class MultimodalGlaucomaDataset(Dataset):
    """Longitudinal multimodal dataset for progression forecasting.

    Each item is a patient trajectory padded/truncated to a fixed number of
    visits. Returned tensors are batch-friendly: stacking the dataset yields
    [B, T, ...] for each modality.
    """

    def __init__(
        self,
        root: str | Path,
        observation_window: int = 4,
        manifest_name: str = "manifest.csv",
        transform: Any | None = None,
        load_oct: bool = True,
        load_vf: bool = True,
    ) -> None:
        """
        Args:
            root: Dataset root containing manifest.csv and modality folders.
            observation_window: Number of visits per trajectory.
            manifest_name: Filename of the manifest CSV.
            transform: Callable applied to per-visit fundus images.
            load_oct: If False, skip loading OCT volumes (saves time/memory).
            load_vf: If False, skip loading VF data.
        """
        self.root = Path(root)
        self.observation_window = observation_window
        self.transform = transform
        self.load_oct = load_oct
        self.load_vf = load_vf
        self.patients: list[PatientTrajectory] = []

        manifest = self.root / manifest_name
        if not manifest.exists():
            return

        by_patient: dict[str, PatientTrajectory] = {}
        with open(manifest) as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = row["patient_id"]

                if pid not in by_patient:
                    by_patient[pid] = PatientTrajectory(
                        patient_id=pid,
                        visits=[],
                        progression_label=_parse_optional_int(row.get("progression_label", "")),
                        group_id=_parse_optional_int(row.get("group_id", "")),
                    )

                visit = Visit(
                    timestamp_days=float(row.get("timestamp_days", 0.0)),
                    fundus_path=(self.root / row["fundus_path"]) if row.get("fundus_path") else None,
                    oct_path=(self.root / row["oct_path"]) if row.get("oct_path") else None,
                    vf_path=(self.root / row["vf_path"]) if row.get("vf_path") else None,
                    rnfl_thickness=_parse_optional_float(row.get("rnfl_thickness", "")),
                    clinical={
                        "iop": _parse_optional_float(row.get("iop", "")) or 0.0,
                        "age": _parse_optional_float(row.get("age", "")) or 0.0,
                        "sex": _parse_optional_float(row.get("sex", "")) or 0.0,
                    },
                )
                by_patient[pid].visits.append(visit)

        # Sort each patient's visits by timestamp
        for traj in by_patient.values():
            traj.visits.sort(key=lambda v: v.timestamp_days)
            self.patients.append(traj)

    def __len__(self) -> int:
        return len(self.patients)

    def _pad_or_truncate(self, visits: list[Visit]) -> list[Visit]:
        """Ensure exactly observation_window visits.

        Truncation keeps the most recent visits (clinically most relevant).
        Padding repeats the earliest visit to the left.
        """
        T = self.observation_window
        if len(visits) >= T:
            return visits[-T:]
        if not visits:
            return [Visit(timestamp_days=0.0)] * T
        pad = [visits[0]] * (T - len(visits))
        return pad + visits

    def __getitem__(self, idx: int) -> dict[str, Any]:
        traj = self.patients[idx]
        visits = self._pad_or_truncate(traj.visits)

        timestamps = torch.tensor(
            [v.timestamp_days for v in visits], dtype=torch.float32,
        )

        # Load fundus images per visit (optional)
        fundus_tensors: list[torch.Tensor] = []
        for v in visits:
            if v.fundus_path is not None and v.fundus_path.exists():
                img = _read_image(v.fundus_path)
                if self.transform is not None:
                    img_t = self.transform(img)["image"]
                else:
                    img_t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
                fundus_tensors.append(img_t)
            else:
                # Placeholder zero tensor
                fundus_tensors.append(torch.zeros(3, 224, 224))
        fundus = torch.stack(fundus_tensors)  # [T, 3, H, W]

        out: dict[str, Any] = {
            "patient_id": traj.patient_id,
            "fundus": fundus,
            "timestamps": timestamps,
            "clinical": torch.tensor(
                [[v.clinical["iop"], v.clinical["age"], v.clinical["sex"]] for v in visits],
                dtype=torch.float32,
            ),
        }

        if self.load_oct:
            octs = []
            for v in visits:
                if v.oct_path is not None and v.oct_path.exists():
                    octs.append(torch.from_numpy(np.load(v.oct_path)).float())
                else:
                    octs.append(torch.zeros(1, 32, 64, 64))
            out["oct"] = torch.stack(octs)

        if self.load_vf:
            vfs = []
            for v in visits:
                if v.vf_path is not None and v.vf_path.exists():
                    vfs.append(torch.from_numpy(np.load(v.vf_path)).float())
                else:
                    vfs.append(torch.zeros(54, 3))
            out["vf"] = torch.stack(vfs)

        # Targets
        rnfl = [v.rnfl_thickness for v in visits if v.rnfl_thickness is not None]
        out["rnfl_target"] = torch.tensor(rnfl[-1] if rnfl else 0.0, dtype=torch.float32)

        if traj.progression_label is not None:
            out["progression_label"] = torch.tensor(traj.progression_label, dtype=torch.long)
        if traj.group_id is not None:
            out["group_id"] = torch.tensor(traj.group_id, dtype=torch.long)

        return out
