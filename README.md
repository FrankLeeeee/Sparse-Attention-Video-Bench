# Sparse Attention Video Bench

This repository serves as a benchmark for variou video sparse attention methods, all methods are collated to ensure fairness and consistent outputs.


## Get Started

```bash
cd Sparse-Attention-Video-Bench
python3 -m venv .venv
. .venv/bin/activate
## If your venv was created without pip (common in minimal docker images), bootstrap it:
python -m ensurepip --upgrade || true

python -m pip install -U pip
python -m pip install -U setuptools wheel
python -m pip install --no-cache-dir -v -e . --no-build-isolation
```

# Quick Runs

```bash
. .venv/bin/activate

bash ./examples/wan1.3b/run_wan1.3b_dense.sh
bash ./examples/wan1.3b/run_wan1.3b_svg.sh
bash ./examples/wan1.3b/run_wan1.3b_sparge.sh
```
