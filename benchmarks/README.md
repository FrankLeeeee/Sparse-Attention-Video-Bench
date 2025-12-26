
This directory contains scripts for benchmarking video generation models with various sparse attention strategies. The pipeline consists of prompt preparation, bulk generation, and evaluation.

## Prerequisites

Install benchmark dependencies:

```bash
pip install -r requirements.txt
```
## Generation and Evaluation

### 1. Prepare Prompts

Fetch and optimize VBench prompts:

```bash
export OPENAI_API_KEY=sk-proj-...
python run.py prep --dimension overall_consistency
```

### 2. Generate Videos

Supported models: `hunyuan`, `wan1.3b`, `wan14b`
Supported strategies: `dense`, `svg`, `sparge`

```bash
# Syntax
python run.py gen --model <model> --strategy <strategy> --dimension <dim> [options]

# Example: Hunyuan + Dense
python run.py gen --model hunyuan --strategy dense --dimension overall_consistency
# Example: Wan 1.3B + SVG (auto-sets sparsity=0.75)
python run.py gen --model wan1.3b --strategy svg --dimension overall_consistency
# Example: Wan 14B + Sparge
python run.py gen --model wan14b --strategy sparge --dimension overall_consistency
```

### 3. Evaluation

#### Speed

Compare generation speeds:

```bash
# Compare specific folders
python run.py speed \
  --target dense   ../outputs/wan1.3b/dense \
  --target svg     ../outputs/wan1.3b/svg \
  --target sparge  ../outputs/wan1.3b/sparge

# Or recursive search
python run.py speed --recursive ../outputs/wan1.3b
```

#### Fidelity (PSNR / SSIM / LPIPS)

Measure pixel/perceptual alignment with baseline:

```bash
python run.py score \
  --reference-directory ./results/wan1.3b/dense/overall_consistency/seed_1024 \
  --generation-directory ./results/wan1.3b/svg/overall_consistency/seed_1024
```

#### Quality (VBench)

Evaluate quality dimensions (handles PATH issues automatically):

```bash
python run.py vbench \
    --dimension overall_consistency \
    --videos_path ./results/wan1.3b/dense/overall_consistency/seed_1024 \
    --mode=custom_input
```
