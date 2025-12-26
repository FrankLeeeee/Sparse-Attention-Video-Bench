#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional, Tuple


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


def _quantile(sorted_values: List[float], q: float) -> float:
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


def _iter_log_files(p: Path, recursive: bool) -> Iterable[Path]:
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


def _load_log(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _pick_settings(logs: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = ["strategy", "height", "width", "num_frames", "num_inference_steps", "warmup_step", "sparsity"]
    out: Dict[str, Any] = {}
    for k in keys:
        values = {d.get(k, None) for d in logs}
        if len(values) == 1:
            out[k] = next(iter(values))
        else:
            out[k] = sorted(list(values), key=lambda x: str(x))
    return out


def summarize_path(path: Path, label: str, recursive: bool) -> Optional[RunSummary]:
    files = list(_iter_log_files(path, recursive=recursive))
    logs: List[Dict[str, Any]] = []
    times: List[float] = []
    for f in files:
        d = _load_log(f)
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
        p90_s=_quantile(times_sorted, 0.90),
        min_s=times_sorted[0],
        max_s=times_sorted[-1],
        total_s=sum(times_sorted),
        settings=_pick_settings(logs),
    )


def _fmt(x: float) -> str:
    if math.isnan(x):
        return "nan"
    return f"{x:.3f}"


def _print_table(rows: List[RunSummary]) -> None:
    header = ["Label", "N", "Mean(s)", "Median(s)", "P90(s)", "Min(s)", "Max(s)", "Strategy", "HxW", "Frames", "Sparsity"]
    data_lines = []
    
    for r in rows:
        s = r.settings
        hw = f"{s.get('height','?')}x{s.get('width','?')}"
        sp = str(s.get("sparsity", "")) if s.get("sparsity") is not None else "-"
        
        line = [
            r.label,
            str(r.n),
            _fmt(r.mean_s),
            _fmt(r.median_s),
            _fmt(r.p90_s),
            _fmt(r.min_s),
            _fmt(r.max_s),
            str(s.get("strategy", "?")),
            hw,
            str(s.get("num_frames", "?")),
            sp
        ]
        data_lines.append(line)

    widths = [len(h) for h in header]
    for line in data_lines:
        for i, val in enumerate(line):
            widths[i] = max(widths[i], len(val))

    sep = "+" + "+".join(["-" * (w + 2) for w in widths]) + "+"
    
    print(sep)
    header_str = "|"
    for i, h in enumerate(header):
        header_str += f" {h:<{widths[i]}} |"
    print(header_str)
    print(sep)
    
    for line in data_lines:
        row_str = "|"
        for i, val in enumerate(line):
            align = ">" if 1 <= i <= 6 else "<"
            row_str += f" {val:{align}{widths[i]}} |"
        print(row_str)
    print(sep)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize generation speed from *.log files.")
    parser.add_argument(
        "--target",
        nargs=2,
        action="append",
        metavar=("LABEL", "PATH"),
        help="A pair of label and path (e.g. --target dense ./logs/dense). Can be used multiple times.",
    )
    parser.add_argument(
        "legacy_paths",
        nargs="*",
        help="Positional paths (will use basename as label). Deprecated but supported for simple usage.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for *.log under given directories.",
    )
    args = parser.parse_args()

    targets: List[Tuple[str, Path]] = []
    
    if args.target:
        for label, p in args.target:
            targets.append((label, Path(p)))
            
    if args.legacy_paths:
        for p in args.legacy_paths:
            path_obj = Path(p)
            targets.append((path_obj.name, path_obj))

    if not targets:
        parser.print_help()
        raise SystemExit("\nError: No paths provided. Use --target LABEL PATH or provide positional paths.")

    rows: List[RunSummary] = []
    for label, p in targets:
        s = summarize_path(p, label=label, recursive=args.recursive)
        if s is None:
            print(f"[warn] no valid *.log with generation_time found under: {p}")
            continue
        rows.append(s)

    if not rows:
        raise SystemExit("No valid logs found.")

    # Sort by mean ascending
    rows.sort(key=lambda r: r.mean_s)
    _print_table(rows)

    merged: Dict[str, set] = {}
    for r in rows:
        for k, v in r.settings.items():
            merged.setdefault(k, set()).add(json.dumps(v, sort_keys=True, default=str))
    mismatched = [k for k, vs in merged.items() if len(vs) > 1]
    if mismatched:
        print("\n[warn] settings differ across compared groups (results may be incomparable): " + ", ".join(mismatched))


if __name__ == "__main__":
    main()


