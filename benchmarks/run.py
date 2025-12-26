#!/usr/bin/env python3
"""
Sparse Attention Video Bench Unified Runner

Commands:
  - gen    : run model inference to generate videos
  - speed  : summarize generation speed from json logs
  - score  : evaluate PSNR/SSIM/LPIPS between reference and generated videos
  - prep   : prepare / optimize vbench prompts using OpenAI API
  - vbench : passthrough to `vbench evaluate ...` with PATH fix
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# =============================================================================
# Paths & Config
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "hunyuan": {
        "script": "scripts/hunyuan_t2v_inference.py",
        "default_args": {
            "height": 720,
            "width": 1280,
            "warmup_step": 1,
            "num_inference_steps": 50,
            "num_frames": 129,
        },
        "strategy_args": {
            "svg": {"sparsity": 0.25},
        },
        "seed_prefix": "SEED",
    },
    "wan1.3b": {
        "script": "scripts/wan_t2v_inference.py",
        "default_args": {
            "height": 720,
            "width": 1280,
            "warmup_step": 1,
            "num_inference_steps": 50,
            "num_frames": 81,
            "model_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        },
        "strategy_args": {
            "svg": {"sparsity": 0.75},
        },
        "seed_prefix": "seed",
    },
    "wan14b": {
        "script": "scripts/wan_t2v_inference.py",
        "default_args": {
            "height": 720,
            "width": 1280,
            "warmup_step": 1,
            "num_inference_steps": 50,
            "num_frames": 81,
            "model_id": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        },
        "strategy_args": {
            "svg": {"sparsity": 0.75},
        },
        "seed_prefix": "seed",
    },
}


def die(msg: str, code: int = 1) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(code)


# =============================================================================
# Speed Summary
# =============================================================================
@dataclass(frozen=True)
class RunSummary:
    label: str
    n: int
    mean_s: float
    median_s: float
    p90_s: float
    min_s: float
    max_s: float
    total_s: float
    settings: Dict[str, Any]


def quantile(sorted_values: List[float], q: float) -> float:
    # Quantile calculation expects sorted_values already sorted.
    if not sorted_values:
        return float("nan")
    if q <= 0:
        return sorted_values[0]
    if q >= 1:
        return sorted_values[-1]
    pos = (len(sorted_values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def iter_log_files(p: Path, recursive: bool) -> Iterable[Path]:
    if p.is_file():
        if p.suffix == ".log":
            yield p
        return
    if not p.exists():
        return
    if recursive:
        yield from p.rglob("*.log")
    else:
        yield from p.glob("*.log")


def load_json_log(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def pick_settings(logs: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = [
        "strategy",
        "height",
        "width",
        "num_frames",
        "num_inference_steps",
        "warmup_step",
        "sparsity",
    ]
    out: Dict[str, Any] = {}
    for k in keys:
        values = {d.get(k, None) for d in logs}
        if len(values) == 1:
            out[k] = next(iter(values))
        else:
            out[k] = sorted(list(values), key=lambda x: str(x))
    return out


def summarize_speed(path: Path, label: str, recursive: bool) -> Optional[RunSummary]:
    files = list(iter_log_files(path, recursive=recursive))
    logs: List[Dict[str, Any]] = []
    times: List[float] = []

    for f in files:
        d = load_json_log(f)
        if not d:
            continue
        t = d.get("generation_time", None)
        if t is None:
            continue
        try:
            t = float(t)
        except Exception:
            continue
        logs.append(d)
        times.append(t)

    if not times:
        return None

    times_sorted = sorted(times)
    return RunSummary(
        label=label,
        n=len(times_sorted),
        mean_s=sum(times_sorted) / len(times_sorted),
        median_s=statistics.median(times_sorted),
        p90_s=quantile(times_sorted, 0.90),
        min_s=times_sorted[0],
        max_s=times_sorted[-1],
        total_s=sum(times_sorted),
        settings=pick_settings(logs),
    )


def print_speed_table(rows: List[RunSummary]) -> None:
    def fmt(x: float) -> str:
        return "nan" if math.isnan(x) else f"{x:.3f}"

    header = [
        "Label",
        "N",
        "Mean(s)",
        "Median(s)",
        "P90(s)",
        "Min(s)",
        "Max(s)",
        "Strategy",
        "HxW",
        "Frames",
        "Sparsity",
    ]

    data_lines: List[List[str]] = []
    for r in rows:
        s = r.settings
        hw = f"{s.get('height','?')}x{s.get('width','?')}"
        sp_val = s.get("sparsity", None)
        sp = "-" if sp_val is None else str(sp_val)
        data_lines.append(
            [
                r.label,
                str(r.n),
                fmt(r.mean_s),
                fmt(r.median_s),
                fmt(r.p90_s),
                fmt(r.min_s),
                fmt(r.max_s),
                str(s.get("strategy", "?")),
                hw,
                str(s.get("num_frames", "?")),
                sp,
            ]
        )

    widths = [len(h) for h in header]
    for line in data_lines:
        for i, val in enumerate(line):
            widths[i] = max(widths[i], len(val))

    sep = "+" + "+".join(["-" * (w + 2) for w in widths]) + "+"
    print(sep)
    print("|" + "".join([f" {h:<{widths[i]}} |" for i, h in enumerate(header)]))
    print(sep)
    for line in data_lines:
        print(
            "|"
            + "".join(
                [
                    f" {val:{'>' if 1 <= i <= 6 else '<'}{widths[i]}} |"
                    for i, val in enumerate(line)
                ]
            )
        )
    print(sep)


def handle_speed(args: argparse.Namespace) -> None:
    targets: List[Tuple[str, Path]] = []
    if args.target:
        for label, p in args.target:
            targets.append((label, Path(p)))
    if args.legacy_paths:
        for p in args.legacy_paths:
            p_obj = Path(p)
            targets.append((p_obj.name, p_obj))

    if not targets:
        die("[Error] No paths provided. Use --target LABEL PATH or provide positional paths.", 1)

    rows: List[RunSummary] = []
    for label, p in targets:
        s = summarize_speed(p, label=label, recursive=args.recursive)
        if s:
            rows.append(s)
        else:
            print(f"[Warn] No valid logs found in: {p}")

    if not rows:
        die("No valid logs found.", 1)

    rows.sort(key=lambda r: r.mean_s)
    print_speed_table(rows)


# =============================================================================
# Score Eval
# =============================================================================
def handle_score(args: argparse.Namespace) -> None:
    # Lazy import heavy deps
    try:
        import torch
        from torcheval.metrics import PeakSignalNoiseRatio
        import pytorch_ssim
        import lpips
        from torchvision.io import read_video
        from tqdm import tqdm
    except ImportError as e:
        die(f"[Error] Missing dependency for scoring: {e}\nPlease run: pip install -r requirements.txt", 1)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    class Metric:
        name: str

        def update(self, img1: "torch.Tensor", img2: "torch.Tensor") -> None:
            raise NotImplementedError

        def compute(self) -> float:
            raise NotImplementedError

    class PSNR(Metric):
        def __init__(self) -> None:
            self.name = "PSNR"
            self.metric = PeakSignalNoiseRatio(device=device)

        def update(self, img1: "torch.Tensor", img2: "torch.Tensor") -> None:
            self.metric.update(img1, img2)

        def compute(self) -> float:
            return float(self.metric.compute().item())

    class SSIM(Metric):
        def __init__(self) -> None:
            self.name = "SSIM"
            self.values: List["torch.Tensor"] = []

        def update(self, img1: "torch.Tensor", img2: "torch.Tensor") -> None:
            # normalize to [-1, 1]
            img1 = img1.unsqueeze(0) / 255.0 * 2 - 1
            img2 = img2.unsqueeze(0) / 255.0 * 2 - 1
            self.values.append(pytorch_ssim.ssim(img1, img2))

        def compute(self) -> float:
            if not self.values:
                return float("nan")
            return float(torch.stack(self.values).mean().item())

    class LPIPS(Metric):
        def __init__(self) -> None:
            self.name = "LPIPS"
            self.metric = lpips.LPIPS(net="alex").to(device)
            self.values: List["torch.Tensor"] = []

        def update(self, img1: "torch.Tensor", img2: "torch.Tensor") -> None:
            img1 = img1.unsqueeze(0) / 255.0 * 2 - 1
            img2 = img2.unsqueeze(0) / 255.0 * 2 - 1
            self.values.append(self.metric(img1, img2))

        def compute(self) -> float:
            if not self.values:
                return float("nan")
            return float(torch.stack(self.values).mean().item())

    @torch.no_grad()
    def run_eval() -> None:
        ref_dir = Path(args.reference_directory)
        gen_dir = Path(args.generation_directory)

        ref_vids = sorted(ref_dir.glob("*.mp4"))
        gen_vids = sorted(gen_dir.glob("*.mp4"))

        gen_map = {v.stem: v for v in gen_vids}
        matched_pairs: List[Tuple[Path, Path]] = [(r, gen_map[r.stem]) for r in ref_vids if r.stem in gen_map]

        if not matched_pairs:
            die(f"[Error] No matching videos found between {ref_dir} and {gen_dir}", 1)

        metrics: List[Metric] = [PSNR(), LPIPS(), SSIM()]
        print(f"Evaluating {len(matched_pairs)} video pairs on device={device} ...")

        for ref_vid, gen_vid in tqdm(matched_pairs, desc="Evaluating"):
            ref_frames, _, _ = read_video(str(ref_vid), output_format="TCHW")
            gen_frames, _, _ = read_video(str(gen_vid), output_format="TCHW")

            min_len = min(len(ref_frames), len(gen_frames))
            if min_len == 0:
                continue

            ref_frames = ref_frames[:min_len]
            gen_frames = gen_frames[:min_len]

            # frame: (C,H,W)
            for rf, gf in zip(ref_frames, gen_frames):
                rf = rf.to(device=device, dtype=torch.float32)
                gf = gf.to(device=device, dtype=torch.float32)
                for m in metrics:
                    m.update(rf, gf)

        print("\nResults:")
        for m in metrics:
            print(f"  {m.name}: {m.compute():.4f}")

    run_eval()


# =============================================================================
# Prepare Prompts
# =============================================================================
def handle_prep(args: argparse.Namespace) -> None:
    try:
        from openai import OpenAI
        from tqdm import tqdm
    except ImportError:
        die("[Error] Missing 'openai' or 'tqdm'. Run: pip install openai tqdm", 1)

    benchmark_dir = SCRIPT_DIR
    vbench_dir = benchmark_dir / "VBench"

    instruction = (
        "Can you help me refine the following video caption for a video generation task? "
        "The caption is: {caption}. Please answer only with one sentence."
    )

    if not vbench_dir.exists():
        die(f"[Error] VBench directory not found at {vbench_dir}", 1)

    prompts_file = vbench_dir / "prompts" / "prompts_per_dimension" / f"{args.dimension}.txt"
    if not prompts_file.exists():
        die(f"[Error] Prompts file not found: {prompts_file}", 1)

    print(f"Reading prompts from: {prompts_file}")
    prompts = [l.strip() for l in prompts_file.read_text(encoding="utf-8").splitlines() if l.strip()]

    client = OpenAI()
    optimized: List[str] = []
    model_name = getattr(args, "openai_model", None) or "gpt-4"

    print(f"Optimizing {len(prompts)} prompts using OpenAI API (model={model_name}) ...")
    for prompt in tqdm(prompts):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": instruction.format(caption=prompt)}],
            )
            optimized.append(resp.choices[0].message.content.strip())
        except Exception as e:
            print(f"[Warn] OpenAI API failed for prompt: {e}")
            optimized.append(prompt)

    out_file = benchmark_dir / "prompts" / f"optimized_{args.dimension}.txt"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(optimized) + "\n", encoding="utf-8")
    print(f"Saved optimized prompts to: {out_file}")


# =============================================================================
# Generation
# =============================================================================
def handle_gen(args: argparse.Namespace) -> None:
    config = MODEL_CONFIGS[args.model]
    inference_script = ROOT_DIR / config["script"]
    if not inference_script.exists():
        die(f"[Error] Inference script not found: {inference_script}", 1)

    cmd: List[str] = [sys.executable, str(inference_script)]
    cmd += ["--strategy", args.strategy]
    cmd += ["--prompt_file", str(SCRIPT_DIR / "prompts" / f"optimized_{args.dimension}.txt")]
    cmd += ["--start-index", str(args.start_index)]
    cmd += ["--end-index", str(args.end_index)]
    cmd += ["--seed", str(args.seed)]

    for k, v in config["default_args"].items():
        cmd += [f"--{k}", str(v)]

    if args.strategy in config.get("strategy_args", {}):
        for k, v in config["strategy_args"][args.strategy].items():
            cmd += [f"--{k}", str(v)]

    seed_dirname = f"{config['seed_prefix']}_{args.seed}"
    output_dir = SCRIPT_DIR / "results" / args.model / args.strategy / args.dimension / seed_dirname
    cmd += ["--output_dir", str(output_dir)]

    print(f"\nRunning Generation: {args.model} | {args.strategy} | {args.dimension}")
    print(f"Output: {output_dir}")
    print(f"Command: {' '.join(cmd)}\n")

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise SystemExit(e.returncode)


# =============================================================================
# VBENCH passthrough
# =============================================================================
def handle_vbench(_: argparse.Namespace, extra_args: Sequence[str]) -> None:
    vbench_cmd = shutil.which("vbench") or str(Path.home() / ".local" / "bin" / "vbench")
    if not Path(vbench_cmd).exists():
        die("[Error] 'vbench' not found in PATH and not in ~/.local/bin/vbench", 127)

    cmd = [vbench_cmd, "evaluate", *extra_args]
    if not any(a.startswith("--mode") for a in extra_args):
        cmd.append("--mode=custom_input")

    print(f"Running: {' '.join(cmd)}")
    p = subprocess.run(cmd)
    if p.returncode != 0:
        raise SystemExit(p.returncode)

    # Auto-print results (best effort)
    eval_dir = SCRIPT_DIR / "evaluation_results"
    latest = max(eval_dir.glob("*_eval_results.json"), key=lambda x: x.stat().st_mtime, default=None)
    if not latest:
        return

    try:
        data = json.loads(latest.read_text(encoding="utf-8"))
        print("\n" + "=" * 60)
        print("VBench Results Summary")
        print(f"Source: {latest.name}")
        print("-" * 60)
        for dim, values in data.items():
            if isinstance(values, list) and values:
                score = values[0]
                if isinstance(score, (int, float)):
                    print(f"  {dim:<25}: {score:.4f}")
        print("=" * 60 + "\n")
    except Exception as e:
        print(f"\n[Warn] Could not auto-print results: {e}")


# =============================================================================
# CLI
# =============================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sparse Attention Video Bench Unified Runner",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # --- GEN ---
    p_gen = subparsers.add_parser("gen", help="Generate videos")
    p_gen.add_argument("--model", choices=MODEL_CONFIGS.keys(), required=True)
    p_gen.add_argument("--strategy", choices=["dense", "sparge", "svg"], required=True)
    p_gen.add_argument("--dimension", required=True)
    p_gen.add_argument("--seed", type=int, default=1024)
    p_gen.add_argument("--start-index", type=int, default=-1)
    p_gen.add_argument("--end-index", type=int, default=-1)
    p_gen.set_defaults(func=lambda a, _unknown: handle_gen(a))

    # --- SPEED ---
    p_speed = subparsers.add_parser("speed", help="Summarize speed from logs")
    p_speed.add_argument("--target", nargs=2, action="append", metavar=("LABEL", "PATH"), help="Target label and path")
    p_speed.add_argument("--recursive", action="store_true", help="Recursive search")
    p_speed.add_argument("legacy_paths", nargs="*", help="Positional paths (legacy)")
    p_speed.set_defaults(func=lambda a, _unknown: handle_speed(a))

    # --- SCORE ---
    p_score = subparsers.add_parser("score", help="Evaluate PSNR/SSIM/LPIPS")
    p_score.add_argument("-r", "--reference-directory", required=True)
    p_score.add_argument("-g", "--generation-directory", required=True)
    p_score.set_defaults(func=lambda a, _unknown: handle_score(a))

    # --- PREP ---
    p_prep = subparsers.add_parser("prep", help="Prepare VBench prompts")
    p_prep.add_argument("-d", "--dimension", required=True)
    p_prep.add_argument("--openai-model", dest="openai_model", default="gpt-4", help="OpenAI model name (default: gpt-4)")
    p_prep.set_defaults(func=lambda a, _unknown: handle_prep(a))

    # --- VBENCH ---
    p_vbench = subparsers.add_parser("vbench", help="Run VBench (passes all args)")
    p_vbench.set_defaults(func=handle_vbench)

    return parser


def main() -> None:
    parser = build_parser()
    # parse_known_args: let `vbench` receive arbitrary flags
    args, unknown = parser.parse_known_args()
    if hasattr(args, "func"):
        args.func(args, unknown)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
