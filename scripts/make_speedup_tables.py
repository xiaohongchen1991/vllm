#!/usr/bin/env python3
"""Parse bench_*.log pairs in benchmark_results/ and emit speedup markdown tables.

For each (model, TP) that has both a `_helion` and a `_none` bench log, compute
per-batch speedups:
  TTFT speedup  = mean TTFT (none)  / mean TTFT (helion)
  TPOT speedup  = mean TPOT (none)  / mean TPOT (helion)
  Throughput speedup = total token throughput (helion) / (none)
"""

import re
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "benchmark_results"
SKIP = {}
BATCHES = [1, 2, 4, 8, 16, 24, 32]

CONC_RE = re.compile(r"max-concurrency\s*=\s*(\d+)")
TTFT_RE = re.compile(r"Mean TTFT \(ms\):\s*([\d.]+)")
TPOT_RE = re.compile(r"Mean TPOT \(ms\):\s*([\d.]+)")
TPUT_RE = re.compile(r"Total token throughput \(tok/s\):\s*([\d.]+)")


def parse_log(path):
    """Return {concurrency: {"ttft":, "tpot":, "tput":}} from a bench log."""
    out = {}
    cur = None
    for line in path.read_text().splitlines():
        m = CONC_RE.search(line)
        if m:
            cur = int(m.group(1))
            out.setdefault(cur, {})
            continue
        if cur is None:
            continue
        m = TTFT_RE.search(line)
        if m:
            out[cur]["ttft"] = float(m.group(1))
        m = TPOT_RE.search(line)
        if m:
            out[cur]["tpot"] = float(m.group(1))
        m = TPUT_RE.search(line)
        if m:
            out[cur]["tput"] = float(m.group(1))
    return out


def base_key(path):
    """bench_<model>_<tag>.log -> <model>_<tag> stripped of _helion/_none."""
    name = path.name[len("bench_") : -len(".log")]
    for suffix in ("_helion", "_none"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return None


def pretty_name(key):
    """Turn RedHatAI_Qwen3-4B-FP8-dynamic_tp1 into a readable title."""
    name = key.split("_", 1)[-1] if key.startswith("RedHatAI_") else key
    return name


HEADERS = ["Batch", "TTFT speedup", "TPOT speedup",
           "Total token throughput speedup"]
WIDTHS = [7, 14, 14, 32]


def _row(cells):
    return "│" + "│".join(c.center(w) for c, w in zip(cells, WIDTHS)) + "│"


def _rule(left, mid, right):
    return left + mid.join("─" * w for w in WIDTHS) + right


def fmt_table(key, helion, none):
    lines = [pretty_name(key), ""]
    lines.append(_rule("┌", "┬", "┐"))
    lines.append(_row(HEADERS))
    lines.append(_rule("├", "┼", "┤"))
    rows = []
    for b in BATCHES:
        h, n = helion.get(b), none.get(b)
        if not h or not n:
            continue
        ttft = n["ttft"] / h["ttft"]
        tpot = n["tpot"] / h["tpot"]
        tput = h["tput"] / n["tput"]
        rows.append([str(b), f"{ttft:.3f}", f"{tpot:.3f}", f"{tput:.3f}"])
    for i, r in enumerate(rows):
        lines.append(_row(r))
        if i != len(rows) - 1:
            lines.append(_rule("├", "┼", "┤"))
    lines.append(_rule("└", "┴", "┘"))
    lines.append("")
    return "\n".join(lines)


def main():
    logs = list(RESULTS_DIR.glob("bench_*.log"))
    pairs = {}  # key -> {"helion": path, "none": path}
    for p in logs:
        key = base_key(p)
        if key is None:
            continue
        variant = "helion" if p.name.endswith("_helion.log") else "none"
        pairs.setdefault(key, {})[variant] = p

    out = []
    for key in sorted(pairs):
        if key in SKIP:
            continue
        pair = pairs[key]
        if "helion" not in pair or "none" not in pair:
            print(f"# skipping {key}: missing "
                  f"{'helion' if 'helion' not in pair else 'none'} log",
                  file=sys.stderr)
            continue
        helion = parse_log(pair["helion"])
        none = parse_log(pair["none"])
        out.append(fmt_table(key, helion, none))

    print("\n".join(out))


if __name__ == "__main__":
    main()
