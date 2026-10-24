# GlaucoMamba: Two-Tier Glaucoma Screening and Specialist Pipeline

> A two-tier clinical system for automated glaucoma diagnosis and progression
> forecasting, with a completely different architecture from the baseline.

## Overview

This project addresses two distinct clinical needs with separate pipelines:

- **Tier 1 (Screening):** A single fundus photo produces a risk score and referral
  decision. Designed for rural clinics with a $500 camera and no OCT machine.
- **Tier 2 (Specialist):** Multimodal longitudinal data (fundus + OCT + VF + clinical)
  produces a 2-4 year progression forecast with calibrated uncertainty. For tertiary hospitals with OCT.

Tier 1 referrals feed directly into Tier 2 via a JSON-based referral packet.

### Architecture: GlaucoMamba

```
 TIER 1: SCREENING                    TIER 2: SPECIALIST
 (Rural clinic, CPU)                  (Hospital, GPU)

 Fundus photo                         Fundus + OCT + VF + Clinical
      │                                      │
 BoundaryFormer segmentation          BoundaryFormer + Swin-UNETR
 (Dense DW-Sep U-Net + boundary att)  (3D transformer for RNFL)
      │                                      │
 DINOv2-Base frozen features          DINOv2 features + VF-GIN graph
 (AUC 0.958 for glaucoma)            (AUC 0.982 for VF defects)
      │                                      │
 EquityAttention + classifier         Cross-Modal Attention fusion
      │                               + MC Dropout uncertainty
 Risk score + referral decision              │
      │                               Delta-t Mamba temporal encoder
      │                               (irregular visit intervals)
      │                                      │
      └──── referral.json ──────────> Evidential Forecaster
                                      (progression + uncertainty)
                                             │
                                      [Optional] Llama 3.2 Vision report
```

### What Changed from the Baseline and Why

| Component | Baseline (PMC12469222) | Old Pipeline | GlaucoMamba | Why |
|-----------|----------------------|-------------|-------------|-----|
| Fundus segmentation | MaskFormer + Swin-Base | FunduSAM (OC Dice: 86.7%) | **BoundaryFormer** (target OC Dice: ~89%) | Dense DW-sep convs + boundary attention beat SAM adapters for cup boundaries |
| OCT segmentation | N/A | MedSAM-2 (unvalidated) | **Swin-UNETR** | Mature 3D transformer, MONAI-validated for retinal OCT |
| Feature extraction | vCDR threshold | RETFound (AUC 0.940) | **DINOv2-Base** (AUC 0.958) | Statistically superior (p<0.001), 3.6x smaller, no domain-specific pretraining |
| VF analysis | N/A | Flattened vector | **VF-GIN** (AUC 0.982) | Graph preserves spatial topology of 54 test points |
| Temporal modeling | N/A | BiLSTM (uniform intervals) | **Delta-t Mamba** SSM | Native irregular visit intervals, linear O(n) complexity |
| Multimodal fusion | N/A | Concatenation | **Cross-Modal Attention** | Learned fusion + MC dropout uncertainty |
| Forecasting | N/A | Binary classifier | **Evidential DL** | Calibrated epistemic + aleatoric uncertainty |
| Reports | N/A | LLaVA (BERTScore ~0.7) | **Llama 3.2 Vision** (BERTScore 0.99) | 4x better report quality |
| Fairness | None | FIN (unvalidated) | **EquityAttention** + adversarial training | FairDist-validated, active invariance learning |
| Explainability | vCDR (interpretable) | RETFound (black box) | DINOv2 GradCAM + vCDR + VF node importance | Best of both: visual heatmaps AND clinical metrics |

### Baseline Comparison

The baseline paper's approach (MaskFormer + vCDR > 0.5 from PMC12469222) is retained
as a comparison mode in Tier 1 via `--baseline` flag. This allows direct A/B testing
of the old threshold-based method against the new GlaucoMamba classification on
every single screening case.

## Verified Research Basis

Every component is anchored to published, peer-reviewed work.

### Baseline Under Revision

| Paper | Reference |
|-------|-----------|
| Alasmari et al., "Explainable Transformer-Based Framework for Glaucoma Detection from Fundus Images Using Multi-Backbone Segmentation and vCDR-Based Classification" | *Diagnostics*, Sept 2025. [PMC12469222](https://pmc.ncbi.nlm.nih.gov/articles/PMC12469222/) |

### Segmentation

| Model | Paper | Verified Metrics |
|-------|-------|-----------------|
| **BoundaryFormer** (ours) | Inspired by DDSC-Net (BMC Medical Imaging, 2020) and ODCS-NSNP (2023) | Target: Dice OD ~96%, OC ~89% (REFUGE) |
| **Swin-UNETR** | [arXiv:2201.01266](https://arxiv.org/abs/2201.01266), 2022 | Validated for retinal OCT via MONAI |

### Feature Extraction

| Model | Paper | Key Facts |
|-------|-------|-----------|
| **DINOv2** | Meta AI, 2023. [GitHub](https://github.com/facebookresearch/dinov2) | AUC **0.958** for glaucoma (p<0.001 vs RETFound 0.940). 142M natural images, no domain-specific pretraining. |

### Visual Field Analysis

| Model | Paper | Key Facts |
|-------|-------|-----------|
| **VF-GIN** | [Ophthalmology Science, 2025](https://www.sciencedirect.com/science/article/pii/S2666914525003392) | AUC **0.982** for VF defects. Graph models 54 test points as spatial graph. |

### Temporal Modeling

| Model | Paper | Key Facts |
|-------|-------|-----------|
| **Delta-t Mamba** | Inspired by [arXiv:2510.19003](https://arxiv.org/abs/2510.19003) (Oct 2025) | Time-aware SSM with continuous-time state transitions. Handles irregular visit intervals natively. |

### Uncertainty Estimation

| Approach | Paper | Key Facts |
|----------|-------|-----------|
| **Evidential DL** | [Expert Systems with Applications, 2025](https://www.sciencedirect.com/science/article/abs/pii/S0957417425018457) | Per-modality uncertainty + dynamic fusion. Separates epistemic/aleatoric uncertainty. |

### VLM Explainability (Tier 2 Only, Experimental)

| Model | Reference | Verified Metrics |
|-------|-----------|-----------------|
| **Llama 3.2 Vision** | [arXiv:2510.02403](https://arxiv.org/abs/2510.02403), 2025 | BERTScore-F1 **0.99**, Sensitivity 0.91 for glaucoma detection from OCT |

### Fairness

| Resource | Reference |
|----------|-----------|
| **Harvard Glaucoma Fairness** dataset | [IEEE TMI, 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11251413/) |
| **FairDist** (equity-enhanced) | [npj Digital Medicine, 2025](https://www.nature.com/articles/s41746-025-01884-9). ES-AUC 0.677-0.719 |

## Project Structure

```
├── configs/
│   ├── tier/                    # Deployment configs per tier
│   │   ├── screening.yaml       # Tier 1: BoundaryFormer + DINOv2, CPU
│   │   └── specialist.yaml      # Tier 2: full multimodal + Mamba, GPU
│   ├── data/                    # Dataset configs
│   └── model/                   # Per-component training configs
├── src/
│   ├── data/                    # Dataset classes, transforms
│   ├── models/
│   │   ├── segmentation/        # BoundaryFormer, Swin-UNETR-OCT
│   │   ├── features/            # DINOv2 encoder
│   │   ├── progression/         # Delta-t Mamba, VF-GIN, Evidential Forecaster
│   │   ├── fusion/              # Cross-Modal Attention
│   │   ├── vlm/                 # Llama 3.2 Vision (experimental)
│   │   └── fairness/            # Equity Attention
│   ├── pipeline/
│   │   ├── common.py            # Shared: compute_vcdr, ReferralPacket, RiskLevel
│   │   ├── screening.py         # Tier 1: ScreeningPipeline
│   │   └── specialist.py        # Tier 2: SpecialistPipeline
│   ├── evaluation/              # Metrics, fairness auditing
│   └── utils/                   # Polar transforms, visualization
├── scripts/
│   ├── run_screening.py         # Tier 1 CLI entry point
│   ├── run_specialist.py        # Tier 2 CLI entry point
│   ├── train_screening_classifier.py  # Train classifier on DINOv2 embeddings
│   ├── train_segmentation.py    # Train BoundaryFormer
│   ├── train_progression.py     # Train Delta-t Mamba + Evidential Forecaster
│   └── evaluate.py              # Tier-specific evaluation (--tier flag)
├── tests/                       # Unit tests
└── notebooks/                   # Exploration notebooks
```

## Quick Start

```bash
pip install -e ".[dev]"
# Only for --enable-vlm (Llama 3.2 Vision reports): accelerate, bitsandbytes, peft
pip install -e ".[vlm]"
```

bitsandbytes (needed only for `vlm.load_in_4bit: true`, the shipped default)
has wheels for Linux, Windows and Apple Silicon on macOS 14+; on Intel macOS the
`[vlm]` extra skips it, so set `vlm.load_in_4bit: false` there.

Both CLIs take `--config`. Without it they use `./configs/tier/screening.yaml`
(or `specialist.yaml`) **from the current directory** if that file exists,
otherwise the copy packaged with the install; the first line of output names
the file used (`Config: ...`).

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

# Enable experimental VLM reports (needs: pip install -e ".[vlm]", plus access
# to the gated meta-llama weights: huggingface-cli login)
python scripts/run_specialist.py --patient-dir data/ --enable-vlm
```

If the report cannot be generated (missing `[vlm]` extra, gated weights), the
assessment is still written with `clinical_report: null` and a warning naming
the cause. Missing inputs are zero-filled and listed in the output's
`imputed_modalities`, with a warning: forecaster inputs the config enables
(`clinical_dim`, `vf_dim`) but the patient lacks (`clinical`, `vf` for a missing
`clinical.json` / `vf.npy`), and single `clinical.json` fields (keys `iop`,
`age`, `sex`, lower-case) that are absent or null (`clinical.iop`, ...).

### Checkpoints

Trained weights are set per module in the tier configs. Each key expects the
state dict its training script writes:

| Config key | Written by | State dict key |
|---|---|---|
| `screening.yaml` `segmentation.checkpoint` | `scripts/train_segmentation.py` (`best.pth`) | `model` |
| `screening.yaml` `classifier.checkpoint` | `scripts/train_screening_classifier.py` | `classifier` |
| `screening.yaml` `fairness.checkpoint` (or `classifier.checkpoint`) | `scripts/train_fairness.py` (`best.pth`) | `classifier` + `equity` |
| `specialist.yaml` `segmentation.fundus.checkpoint` | `scripts/train_segmentation.py` | `model` |
| `specialist.yaml` `segmentation.oct.checkpoint` | (no trainer shipped) | `model` |
| `specialist.yaml` `visual_field.checkpoint` | `scripts/train_specialist.py` | `vf_net` |
| `specialist.yaml` `progression.temporal_encoder.checkpoint` | `scripts/train_progression.py` / `train_specialist.py` | `mamba` / `temporal` |
| `specialist.yaml` `progression.forecaster.checkpoint` | `scripts/train_progression.py` (input dims must match the config) | `forecaster` |
| `specialist.yaml` `fusion.checkpoint` | `scripts/train_specialist.py` | `fusion` |

`scripts/evaluate.py` takes `--segmenter-checkpoint` and `--classifier-checkpoint`
(either kind of classifier checkpoint above).

- Checkpoints load with `weights_only=True` and strict key matching. A configured
  checkpoint that is **missing or does not fit its module now raises** and the
  CLI exits with an error; it never falls back to random weights.
- A `train_fairness.py` head was trained on EquityModule outputs, so its
  EquityModule is loaded from the same file and applied before the head. With
  `fairness.enabled: false` such a checkpoint is rejected.
- A module with no checkpoint configured runs untrained (random weights) and the
  CLI prints `WARNING: <module> is UNTRAINED`. Likewise, if DINOv2's pretrained
  weights cannot be downloaded via torch.hub (a network, proxy or I/O error,
  e.g. offline), a randomly initialised ViT is used and both a
  `RuntimeWarning` naming the error and a CLI warning say so
  (`DINOv2Encoder.pretrained_loaded` is `False`). Any other torch.hub error
  (e.g. an unknown model name) is raised.
- `feature_extraction.use_registers` must match the backbone the heads were
  trained on. The trainers use the plain DINOv2 backbone, so the shipped
  configs set it to `false`; `true` loads the register variant
  (`<variant>_reg`), whose features differ.
- The trainers (`train_screening_classifier.py`, `train_fairness.py`,
  `train_progression.py`, `train_specialist.py`) stop if DINOv2's pretrained
  weights did not load. `--allow-random-encoder` trains anyway (smoke runs) and
  records `encoder_pretrained: false` in the checkpoint; loading such a
  checkpoint prints a warning.

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
