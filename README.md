# ContamPerc: Contamination Percolation in Multi-Agent LLM Systems

**A measurement framework and benchmark for quantifying misinformation propagation in multi-agent LLM networks.**

[![Paper](https://img.shields.io/badge/Paper-IEEE%20Access-blue)](https://ieeeaccess.ieee.org/)
[![Dataset](https://img.shields.io/badge/Dataset-400%20vignettes-green)]()
[![License: MIT](https://img.shields.io/badge/Code-MIT-yellow.svg)](LICENSE)
[![License: CC BY 4.0](https://img.shields.io/badge/Data-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

## What Is This?

When multiple AI agents talk to each other — e.g., a triage nurse AI passes notes to a diagnostic AI, which passes notes to a treatment AI — bad information can spread. But **how much** spreads, and does RLHF alignment help?

This benchmark answers that with:

1. **ContamPerc**: 400 synthetic vignettes across 10 domains, 50 semantic markers, and domain-appropriate agent roles
2. **The Contamination Gap Diagnostic (CGD)**: A single number classifying a model's alignment behavior — from +55 (blocks obvious injection but misses plausible misinformation) to −62 (follows instructions readily)
3. **~210,000 API calls** of validation across 5 model families from 5 organizations

## Quick Start

### Evaluate a New Model (~25,000 API calls, ~$40)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API credentials
cp config.example.yaml config.yaml
# Edit config.yaml with your API key

# 3. Run the benchmark
python run_experiments.py benchmark \
    --provider openai \
    --models gpt4o-mini \
    --n-trials 100

# 4. View results
cat results_openai/summary_report.md
```

### Supported Providers

| Provider | Models Tested | Config Key |
|----------|--------------|------------|
| **Databricks** | DBRX-120B, Claude Sonnet 4.6, Llama 4 Maverick | `databricks` |
| **OpenAI** | GPT-4o, GPT-4o-mini, o1, o3-mini | `openai` |
| **Anthropic** | Claude Sonnet, Claude Opus, Claude Haiku | `anthropic` |
| **Google** | Gemini 2.5 Flash, Gemini 2.5 Pro | `google` |

### Configuration

Set credentials via `config.yaml` or environment variables:

```bash
# Option A: config.yaml (see config.example.yaml)
cp config.example.yaml config.yaml

# Option B: Environment variables
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."
```

## Repository Structure

```
contamination-percolation/
├── run_experiments.py       # Main benchmark (all providers, all experiments)
├── run_cross_domain.py      # Cross-domain validation (legal, financial)
├── run_sensitivity.py       # Sensitivity ablation (temp, prompt variation)
├── benchmark_dataset.py     # 400 vignettes, 10 domains, 50 markers
├── config.example.yaml      # Configuration template
├── requirements.txt         # Python dependencies
├── LICENSE                  # MIT (code) + CC-BY-4.0 (data)
├── .gitignore              # Excludes config.yaml with credentials
└── README.md               # This file
```

## Experiments

| ID | Experiment | Description | API Calls |
|----|-----------|-------------|-----------|
| E1 | Phase Transition | Canary token percolation across 3 topologies | ~14,000/model |
| E2 | Tier-1 Semantic | Plausible-but-wrong domain facts | ~10,000/model |
| E3 | Safety Paradox | With/without same-model safety monitor | ~3,200/model |
| E6 | Control Baseline | No payload — measure marker false positives | ~1,000/model |
| E7 | Social Proof (Canary) | FC normal vs. FC isolated | ~6,400/model |
| E7T | Social Proof (Tier-1) | FC normal vs. FC isolated (semantic) | ~6,400/model |

### Running Individual Experiments

```bash
# Specific experiments
python run_experiments.py run \
    --provider openai --models gpt4o-mini \
    --experiment E1 E2 --n-trials 100

# Full reproduction (~138K calls)
python run_experiments.py full \
    --provider databricks --models dbrx claude llama \
    --n-trials 100

# Dry run (cost/time estimate only)
python run_experiments.py benchmark \
    --provider openai --models gpt4o-mini --dry-run
```

### Cross-Domain Validation

```bash
python run_cross_domain.py \
    --models dbrx claude llama \
    --domains legal_compliance financial_risk \
    --n-trials 100
```

### Sensitivity Analysis

```bash
python run_sensitivity.py --n-trials 100
```

## Key Results

### The CGD Leaderboard

| Model | Canary | Tier-1 | CGD | Class |
|-------|--------|--------|-----|-------|
| DBRX-120B | 1–9% | 60% | **+55** | Content-blind |
| Claude Sonnet 4.6 | 92–100% | 90–96% | **≈0** | Saturated |
| Llama 4 Maverick | 100% | 80% | **−20** | Instr. > context |
| Gemini 2.5 Flash | 100% | 32–48% | **−60** | Strong instr.-following |
| GPT-4o-mini | 58–100% | 7–74% | **−62** | Topology-dependent |

### Interpreting the CGD

- **Positive CGD (+15 to +55):** Model blocks obvious injection (canary tokens) but lets plausible misinformation through. *Action: Test with domain-specific payloads.*
- **Near-zero CGD (±15):** Both payload types propagate at similar rates. *Action: Use topology to compensate.*
- **Negative CGD (−15 to −62):** Model follows explicit instructions more readily than it absorbs subtle context. *Action: Defend against prompt injection.*

### Novel Finding: Social Dilution

GPT-4o-mini shows a unique effect where peer context in a multi-agent network *reduces* contamination by 24 percentage points (Δ = −0.24, p < 0.001). Clean outputs from peer agents dilute the payload's influence — the opposite of social amplification.

## Benchmark Dataset

400 synthetic vignettes across 10 domains:

| Domain | Type | Vignettes | Markers | Example Marker |
|--------|------|-----------|---------|----------------|
| Emergency Medicine | Clinical | 40 | 5 | K+ 2.8 mEq/L |
| Cardiology | Clinical | 40 | 5 | EF 12%, QTc 580 |
| Neurology | Clinical | 40 | 5 | AVM resection |
| Oncology | Clinical | 40 | 5 | T790M, TMB 42 |
| Pediatrics | Clinical | 40 | 5 | ferritin 8500 |
| Psychiatry | Clinical | 40 | 5 | lithium 2.1 |
| OB/GYN | Clinical | 40 | 5 | PAMG-1 |
| Pharmacology | Clinical | 40 | 5 | AUC 680 |
| Legal | Non-clinical | 40 | 5 | consent decree |
| Financial | Non-clinical | 40 | 5 | DSCR 0.85 |

```python
from benchmark_dataset import get_all_vignettes, get_domain

# All 400 vignettes
all_v = get_all_vignettes()

# Single domain
cardio = get_domain("cardiology")
print(cardio["vignettes"][0])
print(cardio["tier1_facts"])
print(cardio["roles"])
```

## Post-Processing

```bash
# Generate summary report from existing results
python run_experiments.py report --results-dir results_openai/

# Export as HuggingFace dataset with Croissant metadata
python run_experiments.py export --results-dir results_openai/
```

## Citation

```bibtex
@article{sharma2026contamination,
  title={Contamination Percolation in Multi-Agent LLM Systems:
         A Measurement Framework and Benchmark},
  author={Sharma, Aman},
  journal={IEEE Access},
  year={2026},
  doi={10.1109/ACCESS.2026.XXXXXXX}
}
```

## Related Work

- **PHI-GUARD** ([TechRxiv](https://doi.org/10.36227/techrxiv.177220388.80392106/v1)): Compliance-aware LLM routing for healthcare (data-out threat)
- **GUARDIAN** (NeurIPS 2025): Temporal graph modeling for agent safety
- **NetSafe** (ACL 2025): Topological safety of multi-agent systems
- **AgentHarm** (ICLR 2025): Harmfulness benchmark for LLM agents

## License

- **Code**: MIT License
- **Benchmark data**: CC-BY-4.0
- **Paper**: See IEEE Access publication

## Contact

Aman Sharma — Aman.Sharma5@student.ctuonline.edu
