# Glaucoma Detection: Two-Tier Screening and Specialist Pipeline

> A two-tier clinical system for automated glaucoma diagnosis and progression
> forecasting, grounded exclusively in verified peer-reviewed research.

## Overview

This project addresses two distinct clinical needs with separate pipelines:

- **Tier 1 (Screening):** A single fundus photo produces a risk score and referral
  decision. Designed for rural clinics with a $500 camera and no OCT machine.
- **Tier 2 (Specialist):** Multimodal longitudinal data (fundus + OCT + VF + clinical)
  produces a 2-4 year progression forecast. For tertiary hospitals with OCT.

Tier 1 referrals feed directly into Tier 2 via a JSON-based referral packet.

### Architecture

```
 TIER 1: SCREENING                    TIER 2: SPECIALIST
 (Rural clinic, CPU)                  (Hospital, GPU)

 Fundus photo                         Fundus + OCT + VF + Clinical
      │                                      │
 FunduSAM segmentation               FunduSAM + MedSAM-2 segmentation
      │                                      │
 RETFound-CFP embedding              RETFound features (CFP + OCT)
      │                                      │
 Classifier head → risk score        ConvNeXt-V2 + BiLSTM forecasting
      │                                      │
 Referral decision                    2-4 year progression forecast
      │                                      │
      └──── referral.json ──────────>  Specialist assessment
```

### Baseline Comparison

The baseline paper's approach (MaskFormer + vCDR > 0.5 from PMC12469222) is retained
as a comparison mode in Tier 1 via `--baseline` flag. This allows direct A/B testing
of the old threshold-based method against the new RETFound-based classification on
every single screening case.

## Verified Research Basis

Every component is anchored to published, peer-reviewed work.

### Baseline Under Revision

| Paper | Reference |
|-------|-----------|
| Alasmari et al., "Explainable Transformer-Based Framework for Glaucoma Detection from Fundus Images Using Multi-Backbone Segmentation and vCDR-Based Classification" | *Diagnostics*, Sept 2025. [PMC12469222](https://pmc.ncbi.nlm.nih.gov/articles/PMC12469222/) |

Baseline metrics (MaskFormer + Swin-Base on REFUGE):
- Dice OD: 93.83%, OC: 93.71%
- IoU OD: 88.29%, OC: 91.09%
- Classification accuracy: 84.03% (vCDR > 0.5 threshold)

### Segmentation (Both Tiers)

| Model | Paper | Verified Metrics (REFUGE) |
|-------|-------|--------------------------|
| **FunduSAM** | [arXiv:2502.06220](https://arxiv.org/abs/2502.06220), Feb 2025 | Dice OD: 96.1%, OC: 86.7%; IoU OD: 88.2%, OC: 78.9% |
| **MedSAM-2** | [arXiv:2408.00874](https://arxiv.org/abs/2408.00874), 2024 | Video-tracking paradigm for 3D OCT volume segmentation |

Note: FunduSAM optic cup Dice (86.7%) is lower than MaskFormer (93.71%).
Cup boundary segmentation remains an open problem.

### Foundation Model (Both Tiers)

| Model | Paper | Key Facts |
|-------|-------|-----------|
| **RETFound** | Zhou et al., *Nature* 622, 156-163 (2023). [DOI](https://doi.org/10.1038/s41586-023-06555-x) | Pre-trained on ~1.6M retinal images via MAE on ViT-Large. AUC 0.83-0.95 for glaucoma. |

### Progression Forecasting (Tier 2 Only)

| Study | Reference | Key Results |
|-------|-----------|-------------|
| Mass Eye & Ear | [medRxiv, Nov 2025](https://www.medrxiv.org/content/10.1101/2025.10.31.25339266v1.full) | 10,864 patients. ConvNeXt-V2 + BiLSTM. AUC 0.97 for 2-4yr forecasting. |

Critical: RNFL OCT is the dominant signal (removing it: AUC 0.569).
VF data is largely redundant (removing it: AUC 0.975).

### VLM Explainability (Tier 2 Only, Experimental)

| Model | Reference | Verified Metrics |
|-------|-----------|-----------------|
| **LLaVA** | [PMC12641456](https://pmc.ncbi.nlm.nih.gov/articles/PMC12641456/), 2025 | AUC 0.92 glaucoma; MAE 1.79 dB for VF MD prediction |

**Experimental.** VLMs hallucinate medical findings. No FDA clearance.
Gated behind `--enable-vlm` flag, always marked as requiring human review.

### Fairness (Both Tiers)

| Resource | Reference |
|----------|-----------|
| **Harvard Glaucoma Fairness** dataset | [IEEE TMI, 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11251413/) |
| **Fair Identity Normalization** (FIN) | AUC for Black patients: 0.77 -> 0.82 |
| **retinIA** (human-AI synergy) | [Frontiers in Ophthalmology, 2025](https://www.frontiersin.org/journals/ophthalmology/articles/10.3389/fopht.2025.1581212/full). AI 88.6% vs residents 82.9%; combined sensitivity 80.4%. |

## Project Structure

```
├── configs/
│   ├── tier/                    # Deployment configs per tier
│   │   ├── screening.yaml       # Tier 1: FunduSAM + RETFound-CFP, CPU
│   │   └── specialist.yaml      # Tier 2: full multimodal, GPU
│   ├── data/                    # Dataset configs
│   └── model/                   # Per-component training configs
├── src/
│   ├── data/                    # Dataset classes, transforms
│   ├── models/
│   │   ├── segmentation/        # FunduSAM, MedSAM-2
│   │   ├── features/            # RETFound encoder
│   │   ├── progression/         # ConvNeXt-V2 + BiLSTM
│   │   ├── vlm/                 # LLaVA (experimental)
│   │   └── fairness/            # Fair Identity Normalization
│   ├── pipeline/
│   │   ├── common.py            # Shared: compute_vcdr, ReferralPacket, RiskLevel
│   │   ├── screening.py         # Tier 1: ScreeningPipeline
│   │   └── specialist.py        # Tier 2: SpecialistPipeline
│   ├── evaluation/              # Metrics, fairness auditing
│   └── utils/                   # Polar transforms, visualization
├── scripts/
│   ├── run_screening.py         # Tier 1 CLI entry point
│   ├── run_specialist.py        # Tier 2 CLI entry point
│   ├── train_screening_classifier.py  # Train Tier 1 classifier head
│   ├── train_segmentation.py    # Train FunduSAM
│   ├── train_progression.py     # Train BiLSTM forecaster
│   └── evaluate.py              # Tier-specific evaluation (--tier flag)
├── tests/                       # Unit tests
└── notebooks/                   # Exploration notebooks
```

## Quick Start

```bash
pip install -e ".[dev]"
```

### Tier 1: Screening

```bash
# Run screening on a fundus image
python scripts/run_screening.py --image fundus.png

# Compare against baseline (vCDR > 0.5 only)
python scripts/run_screening.py --image fundus.png --baseline

# Batch screening
python scripts/run_screening.py --image-dir images/ --output-dir results/
```

### Tier 2: Specialist

```bash
# Run specialist assessment
python scripts/run_specialist.py --patient-dir patient_data/

# Process a Tier 1 referral
python scripts/run_specialist.py --referral referral.json --oct scan.npy

# Enable experimental VLM reports
python scripts/run_specialist.py --patient-dir data/ --enable-vlm
```

### Evaluation

```bash
python scripts/evaluate.py --tier screening    # Tier 1 metrics
python scripts/evaluate.py --tier specialist   # Tier 2 metrics
python scripts/evaluate.py --tier both         # Full comparison
```

## Referral Flow

1. Tier 1 screening runs at a rural site, produces a `ScreeningOutput`
2. If referral is warranted, `create_referral()` produces a `ReferralPacket`
3. The packet is serialized to `referral.json` with artifact paths
4. Files are transmitted to the hospital (PACS, secure transfer, etc.)
5. Tier 2 loads the referral with `--referral referral.json`
6. Tier 2 skips redundant fundus processing, runs OCT + progression analysis

## Datasets

| Dataset | Size | Used By |
|---------|------|---------|
| REFUGE | 1,200 fundus images | Tier 1 + 2 segmentation |
| ORIGA | 650 fundus images | Tier 1 + 2 segmentation |
| G1020 | 1,020 fundus images | Tier 1 + 2 segmentation |
| DIGS/ADAGES | 27,610 OCT images | Tier 2 + VLM training |
| Harvard-GF | 3,300 subjects (2D + 3D) | Fairness evaluation |

## License

MIT
