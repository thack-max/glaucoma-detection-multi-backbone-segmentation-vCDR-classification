# Glaucoma Detection: Multi-Backbone Segmentation and vCDR-Based Classification

> Revision and extension of deterministic segmentation approaches toward multimodal
> foundation models, spatiotemporal forecasting, and vision-language explainability
> for automated glaucoma diagnosis.

## Overview

This project builds a next-generation glaucoma diagnostic pipeline that moves beyond
static cup-to-disc ratio thresholds toward multimodal, longitudinal, and explainable
AI. It is structured as a 5-phase architecture grounded exclusively in verified,
peer-reviewed research.

### Architecture

```
Phase 1: Multimodal Data Ingestion
  Color Fundus + 3D OCT + Visual Fields + Clinical Priors
                          |
Phase 2: Anatomical Segmentation
  FunduSAM (fundus OD/OC) + MedSAM-2 (OCT volumes)
                          |
Phase 3: Foundation Feature Extraction
  RETFound (self-supervised retinal embeddings)
                          |
Phase 4: Spatiotemporal Progression Forecasting
  ConvNeXt-V2 / ViT + BiLSTM (2yr observation -> 2-4yr forecast)
                          |
Phase 5: Vision-Language Explainability
  Fine-tuned LLaVA (clinical report generation)
```

## Verified Research Basis

Every component is anchored to published, peer-reviewed work. No fabricated models
or inflated metrics.

### Baseline Under Revision

| Paper | Reference |
|-------|-----------|
| Alasmari et al., "Explainable Transformer-Based Framework for Glaucoma Detection from Fundus Images Using Multi-Backbone Segmentation and vCDR-Based Classification" | *Diagnostics*, Sept 2025. [PMC12469222](https://pmc.ncbi.nlm.nih.gov/articles/PMC12469222/) |

Baseline metrics (MaskFormer + Swin-Base on REFUGE):
- Dice OD: 93.83%, OC: 93.71%
- IoU OD: 88.29%, OC: 91.09%
- Classification accuracy: 84.03% (vCDR > 0.5 threshold)

### Phase 2 — Segmentation

| Model | Paper | Verified Metrics (REFUGE) |
|-------|-------|--------------------------|
| **FunduSAM** | [arXiv:2502.06220](https://arxiv.org/abs/2502.06220), Feb 2025 | Dice OD: 96.1%, OC: 86.7%; IoU OD: 88.2%, OC: 78.9% |
| **MedSAM-2** | [arXiv:2408.00874](https://arxiv.org/abs/2408.00874), 2024 | Video-tracking paradigm for 3D OCT volume segmentation |

Note: FunduSAM improves optic disc segmentation significantly over MaskFormer (+2.3%
Dice) but optic cup remains challenging (86.7% Dice). The cup boundary is inherently
ambiguous — this is an open problem.

### Phase 3 — Foundation Model

| Model | Paper | Key Facts |
|-------|-------|-----------|
| **RETFound** | Zhou et al., *Nature* 622, 156–163 (2023). [DOI](https://doi.org/10.1038/s41586-023-06555-x) | Pre-trained on ~1.6M retinal images (904K fundus + 736K OCT) via MAE on ViT-Large. AUC 0.83–0.95 for glaucoma across demographics. |

Limitations: Two separate models (RETFound-CFP and RETFound-OCT, not unified).
Pre-training data skewed toward UK Biobank. Does not fully solve cross-site domain
shift.

### Phase 4 — Progression Forecasting

| Study | Reference | Key Results |
|-------|-----------|-------------|
| Mass Eye & Ear longitudinal study | [medRxiv, Nov 2025](https://www.medrxiv.org/content/10.1101/2025.10.31.25339266v1.full) | 10,864 patients. ConvNeXt-V2 + BiLSTM. AUC 0.97 for 2-4yr forecasting. |

Critical finding: Removing RNFL OCT data crashed AUC to 0.569 while removing VF data
barely affected performance (AUC 0.975). **RNFL is the dominant signal for progression
prediction.**

### Phase 5 — Vision-Language Explainability

| Model | Reference | Verified Metrics |
|-------|-----------|-----------------|
| **LLaVA** (fine-tuned) | [PMC12641456](https://pmc.ncbi.nlm.nih.gov/articles/PMC12641456/), 2025 | AUC 0.92 glaucoma detection; MAE 1.79 dB for VF MD prediction |

Study used 27,610 OCT images from DIGS/ADAGES (1,025 participants). Best performance
when both vision encoder and MLP projector are fine-tuned. LLaVA outperformed
RETFound (AUC 0.91) and PaliGemma (AUC 0.90) on this task.

**Status: Experimental.** VLMs still hallucinate medical findings. No FDA clearance
exists for VLM-based diagnostic report generation.

### Fairness

| Resource | Reference |
|----------|-----------|
| **Harvard Glaucoma Fairness** dataset | [IEEE TMI, 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11251413/) |
| **Fair Identity Normalization** (FIN) | Same paper. AUC for Black patients improved 0.77 → 0.82. |
| **retinIA** (human-AI synergy) | [Frontiers in Ophthalmology, May 2025](https://www.frontiersin.org/journals/ophthalmology/articles/10.3389/fopht.2025.1581212/full). AI 88.6% vs residents 82.9%; combined sensitivity 80.4%. |

## Project Structure

```
├── configs/                  # Hydra/YAML configuration files
│   ├── data/                 # Dataset configs (REFUGE, ORIGA, DIGS-ADAGES)
│   └── model/                # Model configs per phase
├── src/
│   ├── data/                 # Dataset classes, transforms, preprocessing
│   ├── models/
│   │   ├── segmentation/     # FunduSAM, MedSAM-2 wrappers
│   │   ├── features/         # RETFound encoder
│   │   ├── progression/      # ConvNeXt-V2 + BiLSTM forecaster
│   │   ├── vlm/              # LLaVA report generator
│   │   └── fairness/         # Fair Identity Normalization
│   ├── pipeline/             # End-to-end phase orchestration
│   ├── evaluation/           # Metrics, fairness auditing
│   └── utils/                # Polar transforms, visualization
├── scripts/                  # Training and evaluation entry points
├── tests/                    # Unit tests
└── notebooks/                # Exploration and analysis notebooks
```

## Setup

```bash
pip install -e ".[dev]"
```

## Datasets

| Dataset | Size | Source |
|---------|------|--------|
| REFUGE | 1,200 fundus images | [Grand Challenge](https://refuge.grand-challenge.org/) |
| ORIGA | 650 fundus images | Request from authors |
| G1020 | 1,020 fundus images | Public |
| DIGS/ADAGES | 27,610 OCT images | [UCSD Hamilton Glaucoma Center](https://shileyeye.ucsd.edu/) |
| Harvard-GF | 3,300 subjects (2D + 3D) | [Harvard Ophthalmology AI Lab](https://ophai.hms.harvard.edu/) |

## License

MIT
