"""
Prepare a longitudinal multimodal dataset manifest for Tier 2 training.

Targets datasets such as DIGS/ADAGES (Diagnostic Innovations in Glaucoma
Study and African Descent and Glaucoma Evaluation Study), where each
patient has multiple visits with fundus photos, OCT volumes, VF tests,
and clinical measurements over 2+ years.

Since the exact on-disk layout varies by distribution, this script ingests
a metadata CSV that you provide — often exported from the study's REDCap
database or the researcher's own spreadsheet — and normalizes it into the
manifest.csv format consumed by MultimodalGlaucomaDataset.

Expected input CSV columns (flexible names — see --column-map):

    patient_id       -- required
    visit_date       -- ISO YYYY-MM-DD; used to compute timestamp_days
    fundus_path      -- relative to --data-root
    oct_path         -- optional
    vf_path          -- optional
    rnfl_thickness   -- global average RNFL in microns (optional)
    iop              -- intraocular pressure (optional)
    age
    sex              -- M/F or 1/0
    race             -- mapped to group_id via --race-map

Progression label:
    If --progression-slope is set, the script fits a linear trend to each
    patient's RNFL thickness across visits. Patients with a slope below the
    given threshold (microns/year, negative number) are labeled progressing.

Usage:
    python scripts/prepare_longitudinal.py \
        --metadata /data/DIGS/visits.csv \
        --data-root /data/DIGS \
        --output data/longitudinal \
        --progression-slope -1.5
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


# Canonical destination columns for the manifest CSV.
MANIFEST_COLUMNS = [
    "patient_id", "visit_index", "timestamp_days",
    "fundus_path", "oct_path", "vf_path",
    "rnfl_thickness", "iop", "age", "sex",
    "group_id", "progression_label",
]

# Default source-column aliases we accept out-of-the-box.
DEFAULT_COLUMN_MAP: dict[str, list[str]] = {
    "patient_id": ["patient_id", "subject_id", "patientid", "id"],
    "visit_date": ["visit_date", "date", "exam_date", "visitdate"],
    "fundus_path": ["fundus_path", "fundus", "cfp", "image_path"],
    "oct_path": ["oct_path", "oct", "oct_volume"],
    "vf_path": ["vf_path", "vf", "visual_field"],
    "rnfl_thickness": ["rnfl_thickness", "rnfl", "rnfl_avg", "rnfl_average"],
    "iop": ["iop", "pressure", "intraocular_pressure"],
    "age": ["age", "age_at_visit"],
    "sex": ["sex", "gender"],
    "race": ["race", "ethnicity", "demographic_group"],
}

DEFAULT_RACE_MAP = {
    "asian": 0,
    "black": 1,
    "african american": 1,
    "african-american": 1,
    "white": 2,
    "caucasian": 2,
}


@dataclass
class VisitRow:
    patient_id: str
    visit_date: datetime | None
    fundus_path: str = ""
    oct_path: str = ""
    vf_path: str = ""
    rnfl_thickness: float | None = None
    iop: float | None = None
    age: float | None = None
    sex: int | None = None
    race: str | None = None


@dataclass
class PatientVisits:
    patient_id: str
    visits: list[VisitRow] = field(default_factory=list)
    progression_label: int | None = None
    group_id: int | None = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a longitudinal manifest")
    parser.add_argument("--metadata", type=str, required=True, help="Source metadata CSV")
    parser.add_argument("--data-root", type=str, required=True, help="Root directory the file paths are relative to")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--progression-slope",
        type=float,
        default=None,
        help="RNFL slope (microns/year) below which a patient is labeled progressing.",
    )
    parser.add_argument("--min-visits", type=int, default=2, help="Minimum visits per patient")
    parser.add_argument(
        "--column-map",
        type=str,
        default=None,
        help="JSON dict overriding the default column aliases, e.g. '{\"patient_id\": [\"subject\"]}'",
    )
    return parser.parse_args()


def _pick_column(row: dict[str, str], candidates: list[str]) -> str | None:
    for c in candidates:
        for key in row:
            if key.strip().lower() == c.lower() and row[key].strip():
                return row[key].strip()
    return None


def _parse_date(s: str | None) -> datetime | None:
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _parse_sex(s: str | None) -> int | None:
    if s is None:
        return None
    s = s.strip().lower()
    if s in {"m", "male", "1"}:
        return 1
    if s in {"f", "female", "0"}:
        return 0
    return None


def _parse_float(s: str | None) -> float | None:
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _load_visits(metadata_path: Path, column_map: dict[str, list[str]]) -> dict[str, PatientVisits]:
    patients: dict[str, PatientVisits] = {}

    with open(metadata_path, newline="") as f:
        reader = csv.DictReader(f)
        for raw_row in reader:
            pid = _pick_column(raw_row, column_map["patient_id"])
            if not pid:
                continue

            visit = VisitRow(
                patient_id=pid,
                visit_date=_parse_date(_pick_column(raw_row, column_map["visit_date"])),
                fundus_path=_pick_column(raw_row, column_map["fundus_path"]) or "",
                oct_path=_pick_column(raw_row, column_map["oct_path"]) or "",
                vf_path=_pick_column(raw_row, column_map["vf_path"]) or "",
                rnfl_thickness=_parse_float(_pick_column(raw_row, column_map["rnfl_thickness"])),
                iop=_parse_float(_pick_column(raw_row, column_map["iop"])),
                age=_parse_float(_pick_column(raw_row, column_map["age"])),
                sex=_parse_sex(_pick_column(raw_row, column_map["sex"])),
                race=_pick_column(raw_row, column_map["race"]),
            )
            patients.setdefault(pid, PatientVisits(pid)).visits.append(visit)

    return patients


def _compute_progression_label(
    patient: PatientVisits,
    slope_threshold: float,
) -> int | None:
    """Fit a line to RNFL thickness vs. years and compare slope to threshold.

    Returns None if there are fewer than 2 RNFL measurements.
    """
    points = [
        (v.visit_date, v.rnfl_thickness)
        for v in patient.visits
        if v.visit_date is not None and v.rnfl_thickness is not None
    ]
    if len(points) < 2:
        return None

    points.sort(key=lambda p: p[0])
    t0 = points[0][0]
    xs = [(p[0] - t0).days / 365.25 for p in points]
    ys = [p[1] for p in points]

    # Ordinary least squares slope
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0:
        return None
    slope = num / den
    return 1 if slope <= slope_threshold else 0


def _resolve_group_id(patient: PatientVisits) -> int | None:
    races = [v.race for v in patient.visits if v.race]
    if not races:
        return None
    return DEFAULT_RACE_MAP.get(races[0].lower(), None)


def _load_column_map(override: str | None) -> dict[str, list[str]]:
    column_map = {k: list(v) for k, v in DEFAULT_COLUMN_MAP.items()}
    if override:
        import json

        user_map = json.loads(override)
        for key, val in user_map.items():
            if key in column_map and isinstance(val, list):
                column_map[key] = val + column_map[key]
    return column_map


def main() -> None:
    args = _parse_args()
    metadata_path = Path(args.metadata).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    data_root = Path(args.data_root).expanduser().resolve()

    column_map = _load_column_map(args.column_map)
    patients = _load_visits(metadata_path, column_map)

    # Drop patients below the minimum visit count
    patients = {
        pid: pv for pid, pv in patients.items() if len(pv.visits) >= args.min_visits
    }

    # Compute progression labels + group IDs
    for pv in patients.values():
        if args.progression_slope is not None:
            pv.progression_label = _compute_progression_label(pv, args.progression_slope)
        pv.group_id = _resolve_group_id(pv)

    # Emit manifest sorted by (patient_id, visit_date)
    manifest_path = output / "manifest.csv"
    total_visits = 0
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()

        for pid, pv in sorted(patients.items()):
            visits_sorted = sorted(pv.visits, key=lambda v: v.visit_date or datetime.min)
            if not visits_sorted:
                continue
            baseline = visits_sorted[0].visit_date

            for idx, visit in enumerate(visits_sorted):
                timestamp_days = 0.0
                if baseline is not None and visit.visit_date is not None:
                    timestamp_days = (visit.visit_date - baseline).days

                # Resolve paths relative to data_root (handles both absolute
                # and data-root-relative inputs gracefully).
                def _resolve(p: str) -> str:
                    if not p:
                        return ""
                    pth = Path(p)
                    if not pth.is_absolute():
                        pth = data_root / pth
                    return str(pth)

                writer.writerow(
                    {
                        "patient_id": pid,
                        "visit_index": idx,
                        "timestamp_days": f"{timestamp_days:.1f}",
                        "fundus_path": _resolve(visit.fundus_path),
                        "oct_path": _resolve(visit.oct_path),
                        "vf_path": _resolve(visit.vf_path),
                        "rnfl_thickness": f"{visit.rnfl_thickness:.2f}" if visit.rnfl_thickness is not None else "",
                        "iop": f"{visit.iop:.1f}" if visit.iop is not None else "",
                        "age": f"{visit.age:.1f}" if visit.age is not None else "",
                        "sex": str(visit.sex) if visit.sex is not None else "",
                        "group_id": str(pv.group_id) if pv.group_id is not None else "",
                        "progression_label": str(pv.progression_label) if pv.progression_label is not None else "",
                    }
                )
                total_visits += 1

    # Summary
    progressing = sum(1 for pv in patients.values() if pv.progression_label == 1)
    stable = sum(1 for pv in patients.values() if pv.progression_label == 0)
    unknown = sum(1 for pv in patients.values() if pv.progression_label is None)

    print(f"Wrote {total_visits} visits for {len(patients)} patients to {manifest_path}")
    print(f"  progressing: {progressing}")
    print(f"  stable:      {stable}")
    if unknown:
        print(f"  unlabeled:   {unknown}")


if __name__ == "__main__":
    main()
