#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pick the fastest pre-tuned config per input case for a Helion kernel.

For every input case the kernel's registered input generator produces, this
benchmarks the kernel once per config found in the given config files -- binding
that config directly instead of running the kernel's `pick_config` selection
-- and keeps the fastest. The winner per case is written as a config file in the
format the kernels load (`[{"key": ..., "config": ...}, ...]`), so it can
replace the platform file.

A case may carry several synthetic inputs (e.g. different routings); a config is
scored by the geometric mean of its latency over all of them, matching how the
autotuner scores multi-input cases. Configs that resolve to the same config once
the kernel's defaults are filled in -- the same entry in two files, a value
spelled out that the rest leave at its default, a key the kernel ignores -- are
benchmarked once.

Before a config is timed, its output is checked against the kernel's registered
reference, with the same comparison and tolerances `tools/benchmark_helion_kernels.py`
uses for its numerics check; a config that fails is reported and skipped instead
of timed. Pair this with the L2-clearing CUDA-graph timer so a config is only
trusted when it is both correct and fast.

Usage:
    # Re-select configs for a kernel from its own platform file
    python tools/benchmark_helion_configs.py --kernel mega_moe_block_mm

    # Consider every config file in a directory, plus one more file
    python tools/benchmark_helion_configs.py --kernel mega_moe_block_mm \\
        --config-dir vllm/kernels/helion/configs/mega_moe_block_mm \\
        --config-path /tmp/other.json --report /tmp/trials.json

    # Only re-time configs stored under the case's own key
    python tools/benchmark_helion_configs.py --kernel mega_moe_block_mm \\
        --exact-key --max-configs 4

    # List available kernels
    python tools/benchmark_helion_configs.py --list
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

try:
    import helion

    from vllm.benchmarks.lib.utils import default_vllm_config
    from vllm.config.logging import LoggingConfig
    from vllm.kernels.helion import (
        ConfigManager,
        get_kernel_by_name,
        get_registered_kernels,
    )
    from vllm.kernels.helion.case_key import CaseKey
    from vllm.kernels.helion.composite import (
        CompositeKernel,
        _joint_dict,
        is_composite_body,
    )
    from vllm.kernels.helion.inputs import normalize_arg_sets
    from vllm.kernels.helion.register import create_helion_decorated_kernel
    from vllm.kernels.helion.utils import get_config_gpu_name
    from vllm.logger import configure_logging, init_logger
    from vllm.triton_utils import triton
except ImportError as e:
    print(f"Error importing vLLM: {e}")
    print("Please ensure vLLM is installed and in your Python path")
    sys.exit(1)

# The benchmark methodology (CUDA graph capture, an L2 flush per call) lives in
# the kernel benchmark tool; reuse it so both report the same numbers.
_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))
from benchmark_helion_kernels import (  # noqa: E402
    check_correctness,
    check_requirements,
    cleanup_gpu_resources,
    do_bench_cudagraph_l2_clear,
    import_all_kernels,
    log_versions,
    make_eager_baseline,
)

logger = init_logger("vllm.scripts.benchmark_helion_configs")


@dataclass
class Candidate:
    """One config to try, and where it came from."""

    source: Path
    source_key: CaseKey
    config: dict[str, Any]

    @property
    def fingerprint(self) -> str:
        """Stable identity, so the same config in two files is tried once."""
        return json.dumps(self.config, sort_keys=True)


@dataclass
class Trial:
    candidate: Candidate
    per_input_ms: list[float]

    @property
    def score_ms(self) -> float:
        """Geometric mean over the case's inputs, as the autotuner scores them."""
        return geomean(self.per_input_ms)


@dataclass
class CaseResult:
    case: str
    key: CaseKey
    best: Trial | None = None
    trials: list[Trial] = field(default_factory=list)
    failed: int = 0


def geomean(values: list[float]) -> float:
    if not values or any(not math.isfinite(v) or v <= 0 for v in values):
        return math.inf
    return math.exp(math.fsum(math.log(v) for v in values) / len(values))


def is_joint_config(config: Any) -> bool:
    return isinstance(config, dict) and ("root" in config or "children" in config)


def resolve_config_paths(
    kernel_name: str, platform: str, config_dir: str | None, config_paths: list[str]
) -> list[Path]:
    paths = [Path(p) for p in config_paths]
    if config_dir is not None:
        directory = Path(config_dir)
        if not directory.is_dir():
            logger.error("--config-dir %s is not a directory", directory)
            return []
        paths.extend(sorted(directory.glob("*.json")))
    if not paths:
        # Default: the platform file the kernel loads at runtime.
        default_path = ConfigManager().get_config_file_path(kernel_name, platform)
        logger.info("No config file given; using %s", default_path)
        paths = [default_path]
    unique: list[Path] = []
    for path in paths:
        if path not in unique:
            unique.append(path)
    return unique


def load_candidates(paths: list[Path], composite: bool) -> list[Candidate]:
    """Read every entry of every file, skipping whatever does not parse."""
    candidates: list[Candidate] = []
    seen: set[str] = set()
    for path in paths:
        try:
            entries = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Skipping %s: cannot read config file (%s)", path, e)
            continue
        if not isinstance(entries, list):
            logger.warning("Skipping %s: expected a JSON array of entries", path)
            continue
        loaded = 0
        for index, entry in enumerate(entries):
            try:
                raw_key = entry.get("key") or {}
                key = CaseKey(raw_key) if raw_key else CaseKey.default()
                config = entry["config"]
                if not isinstance(config, dict):
                    raise TypeError("config is not an object")
            except (AttributeError, KeyError, TypeError, ValueError) as e:
                logger.warning(
                    "Skipping %s entry %d: malformed config entry (%s)",
                    path,
                    index,
                    e,
                )
                continue
            if is_joint_config(config) != composite:
                logger.warning(
                    "Skipping %s entry %d: kernel takes a %s config",
                    path,
                    index,
                    "root/children" if composite else "flat",
                )
                continue
            candidate = Candidate(source=path, source_key=key, config=config)
            if candidate.fingerprint in seen:
                continue
            seen.add(candidate.fingerprint)
            candidates.append(candidate)
            loaded += 1
        logger.info("Loaded %d config(s) from %s", loaded, path)
    return candidates


def normalize_config(value: Any, default: Any) -> Any:
    """Canonical form of a config value against the spec's default.

    Drops keys the spec does not know, fills missing ones with their default,
    and pads a short structure-length list with the defaults it would get
    anyway -- the three ways two entries can spell the same config. A list
    longer than the spec is left alone: which entries are live depends on the
    branch the config selects, so truncating could merge two different configs.
    """
    if isinstance(default, dict):
        provided = value if isinstance(value, dict) else {}
        return {k: normalize_config(provided.get(k, d), d) for k, d in default.items()}
    if (
        isinstance(value, list)
        and isinstance(default, list)
        and len(value) < len(default)
    ):
        return [*value, *default[len(value) :]]
    return default if value is None else value


class RunnerFactory:
    """Builds a callable per config, and the default config of a case's program."""

    def __init__(self, wrapper: Any):
        self.wrapper = wrapper
        self._composite: CompositeKernel | None = None
        self._kernel: Any = None
        if is_composite_body(wrapper.raw_kernel_func):
            # The composite owns its children's configs, so it compiles them
            # from the joint (root + children) dict.
            self._composite = CompositeKernel(
                wrapper.raw_kernel_func,
                settings=wrapper.helion_settings,
                stage_plan=wrapper.stage_plan,
            )
        else:
            self._kernel = create_helion_decorated_kernel(
                wrapper.raw_kernel_func, wrapper.helion_settings
            )

    def spec_default(self, args: tuple[Any, ...]) -> dict[str, Any]:
        """Default config of the program `args` traces."""
        if self._composite is not None:
            bound = self._composite.bind(args)
            return _joint_dict(bound.config_spec.default_config())
        return dict(self._kernel.bind(args).config_spec.default_config().config)

    def make(self, config: dict[str, Any]) -> Any:
        if self._composite is not None:
            return self._composite.configure(config)
        kernel = self._kernel
        compiled: list[Any] = []

        def run(*args: Any) -> Any:
            if not compiled:
                bound = kernel.bind(args)
                compiled.append(bound.compile_config(helion.Config.from_dict(config)))
            return compiled[0](*args)

        return run


def fingerprint(config: dict[str, Any], spec_default: dict[str, Any]) -> str:
    return json.dumps(normalize_config(config, spec_default), sort_keys=True)


class CheckedRunner:
    """A config's runner plus the settings the numerics check reads."""

    def __init__(self, run: Any, helion_settings: Any):
        self._run = run
        self.helion_settings = helion_settings

    def __call__(self, *args: Any) -> Any:
        return self._run(*args)


def benchmark_candidate(
    run: Any,
    arg_sets: list[tuple[Any, ...]],
    benchmark_fn: Any,
    repeat: int,
    return_mode: str,
) -> list[float]:
    return [
        benchmark_fn(lambda args=args: run(*args), rep=repeat, return_mode=return_mode)
        for args in arg_sets
    ]


def unique_candidates(
    case: str,
    candidates: list[Candidate],
    factory: RunnerFactory,
    arg_sets: list[tuple[Any, ...]],
    exact_key: CaseKey | None,
) -> list[Candidate]:
    """Drop configs that resolve to a config already in the list.

    Two entries can still be the same config after binding: a key the kernel
    ignores, a value spelled out that the rest leave at its default, or a
    structure-length list padded with defaults. Those are normalized against
    the program's default config before comparing, so each distinct config is
    benchmarked once.
    """
    try:
        spec_default = factory.spec_default(arg_sets[0])
    except Exception as e:  # noqa: BLE001 - fall back to exact comparison
        logger.warning(
            "%s: cannot derive the program's default config (%s); "
            "deduplicating on the raw configs",
            case,
            type(e).__name__,
        )
        spec_default = None

    unique: list[Candidate] = []
    seen: dict[str, Candidate] = {}
    considered = 0
    for candidate in candidates:
        if exact_key is not None and candidate.source_key != exact_key:
            continue
        considered += 1
        key = (
            candidate.fingerprint
            if spec_default is None
            else fingerprint(candidate.config, spec_default)
        )
        previous = seen.get(key)
        if previous is not None:
            logger.debug(
                "%s: %s (key=%s) duplicates %s (key=%s)",
                case,
                candidate.source,
                candidate.source_key,
                previous.source,
                previous.source_key,
            )
            continue
        seen[key] = candidate
        unique.append(candidate)
    if len(unique) != considered:
        logger.info(
            "%s: %d of %d config(s) duplicate an earlier one",
            case,
            considered - len(unique),
            considered,
        )
    return unique


def select_for_case(
    key: CaseKey,
    arg_sets: list[tuple[Any, ...]],
    candidates: list[Candidate],
    factory: RunnerFactory,
    benchmark_fn: Any,
    numerics_check: Any,
    args: argparse.Namespace,
) -> CaseResult:
    case = str(key)
    result = CaseResult(case=case, key=key)
    if args.dedupe:
        candidates = unique_candidates(
            case, candidates, factory, arg_sets, key if args.exact_key else None
        )
    for candidate in candidates:
        if args.max_configs and len(result.trials) >= args.max_configs:
            break
        try:
            run = factory.make(candidate.config)
            if numerics_check is not None:
                # One check per config, on the case's first input, like the
                # kernel benchmark tool. The check runs the reference on its own
                # copies, so the timed inputs below are untouched by it.
                numerics_check(run, arg_sets[0], case)
            per_input_ms = benchmark_candidate(
                run, arg_sets, benchmark_fn, args.repeat, args.return_mode
            )
        except Exception as e:  # noqa: BLE001 - a bad config is skipped, not fatal
            result.failed += 1
            first_line = str(e).splitlines()[0][:200] if str(e) else ""
            logger.warning(
                "%s: config from %s (key=%s) failed: %s: %s",
                case,
                candidate.source,
                candidate.source_key,
                type(e).__name__,
                first_line,
            )
            cleanup_gpu_resources()
            continue
        trial = Trial(candidate=candidate, per_input_ms=per_input_ms)
        result.trials.append(trial)
        logger.info(
            "%s: %s (key=%s) -> %.4f ms%s",
            case,
            candidate.source,
            candidate.source_key,
            trial.score_ms,
            f" (per input: {[round(t, 4) for t in per_input_ms]})"
            if len(per_input_ms) > 1
            else "",
        )
        if result.best is None or trial.score_ms < result.best.score_ms:
            result.best = trial
    return result


def print_table(results: list[CaseResult]) -> None:
    headers = ["case", "best_ms", "inputs", "config_file", "config_key"]
    data = []
    for result in results:
        if result.best is None:
            data.append([result.case, "n/a", "-", "-", "-"])
            continue
        best = result.best
        data.append(
            [
                result.case,
                f"{best.score_ms:.4f}",
                str(len(best.per_input_ms)),
                best.candidate.source.name,
                str(best.candidate.source_key),
            ]
        )
    cols = list(zip(*([headers] + data)))
    widths = [max(len(cell) for cell in col) for col in cols]

    def fmt(row: list[str]) -> str:
        return " | ".join(cell.ljust(w) for cell, w in zip(row, widths))

    print(fmt(headers))
    print("-+-".join("-" * w for w in widths))
    for row in data:
        print(fmt(row))


def write_config_file(path: Path, results: list[CaseResult]) -> None:
    pairs = [
        {"key": dict(result.key), "config": result.best.candidate.config}
        for result in results
        if result.best is not None
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(pairs, f, indent=2)
        f.write("\n")
    logger.info("Wrote %d config(s) to %s", len(pairs), path)


def write_report(path: Path, kernel: str, results: list[CaseResult]) -> None:
    report = {
        "kernel": kernel,
        "cases": [
            {
                "case": result.case,
                "key": dict(result.key),
                "best": (
                    None
                    if result.best is None
                    else {
                        "score_ms": result.best.score_ms,
                        "per_input_ms": result.best.per_input_ms,
                        "config_file": str(result.best.candidate.source),
                        "config_key": dict(result.best.candidate.source_key),
                        "config": result.best.candidate.config,
                    }
                ),
                "trials": [
                    {
                        "score_ms": trial.score_ms,
                        "per_input_ms": trial.per_input_ms,
                        "config_file": str(trial.candidate.source),
                        "config_key": dict(trial.candidate.source_key),
                    }
                    for trial in result.trials
                ],
                "failed_configs": result.failed,
            }
            for result in results
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    logger.info("Wrote trial report to %s", path)


def matches_case_filters(key: CaseKey, filters: list[str]) -> bool:
    return not filters or any(f in str(key) for f in filters)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pick the fastest pre-tuned config per input case",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[1] if "Usage:" in __doc__ else "",
    )
    parser.add_argument("--list", action="store_true", help="List kernels and exit")
    parser.add_argument(
        "--kernel", type=str, help="Name of the Helion kernel to benchmark"
    )
    parser.add_argument(
        "--config-path",
        nargs="+",
        default=[],
        metavar="PATH",
        help="Config file(s) to take candidate configs from",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        help="Directory whose *.json files are all used as candidate configs",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Where to write the selected configs "
        "(default: ./helion_config_<kernel>_<platform>.json)",
    )
    parser.add_argument(
        "--report",
        type=str,
        help="Optional path for a JSON report of every config tried",
    )
    parser.add_argument(
        "--platform",
        type=str,
        help="Platform name for reporting and default paths (default: detected)",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        metavar="SUBSTRING",
        help="Only benchmark cases whose key contains this text (repeatable)",
    )
    parser.add_argument(
        "--exact-key",
        action="store_true",
        help="Only try configs stored under the case's own key",
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        default=0,
        help="Stop after this many benchmarked configs per case (default: all)",
    )
    parser.add_argument(
        "--repeat",
        type=float,
        default=100,
        help="Benchmark repetitions (default: 100)",
    )
    parser.add_argument(
        "--return-mode",
        choices=["min", "max", "mean", "median"],
        default="mean",
        help="Statistic reported per benchmark sample (default: mean)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Log every candidate's latency (INFO level)",
    )
    parser.add_argument(
        "--no-cudagraph",
        dest="cudagraph",
        action="store_false",
        help="Disable CUDA graph mode (enabled by default)",
    )
    parser.add_argument(
        "--no-dedupe",
        dest="dedupe",
        action="store_false",
        help="Benchmark repeated configs instead of collapsing them",
    )
    parser.add_argument(
        "--no-correctness-check",
        dest="correctness_check",
        action="store_false",
        help="Time configs without checking their output against the reference",
    )
    args = parser.parse_args()

    if args.verbose:
        configure_logging(LoggingConfig(log_level="INFO"))
    log_versions()
    import_all_kernels()

    if args.list:
        kernels = get_registered_kernels()
        print("Available Helion kernels:")
        for name in sorted(kernels):
            print(f"  {name}")
        print(f"\nTotal: {len(kernels)} kernels")
        return

    if not args.kernel:
        parser.error("--kernel is required (or use --list to see available kernels)")
    kernel_name = args.kernel
    wrapper = get_kernel_by_name(kernel_name)
    if wrapper is None:
        logger.error("Kernel '%s' not found in registry.", kernel_name)
        logger.error("Available kernels: %s", sorted(get_registered_kernels()))
        sys.exit(1)
    if wrapper._disabled:
        logger.error(
            "Kernel '%s' is disabled: %s", kernel_name, wrapper._disabled_reason
        )
        sys.exit(1)
    if not check_requirements():
        sys.exit(1)

    numerics_check = None
    if args.correctness_check:
        try:
            baseline_fn = make_eager_baseline(kernel_name)
        except SystemExit:
            logger.warning(
                "Kernel '%s' registers no reference for the numerics check; "
                "timing configs without checking them",
                kernel_name,
            )
        else:
            helion_settings = wrapper.helion_settings

            def numerics_check(run, inputs, case):
                check_correctness(
                    CheckedRunner(run, helion_settings), baseline_fn, inputs, case
                )

    composite = is_composite_body(wrapper.raw_kernel_func)
    platform = args.platform or get_config_gpu_name(wrapper.use_variant_config)
    paths = resolve_config_paths(
        kernel_name, platform, args.config_dir, args.config_path
    )
    candidates = load_candidates(paths, composite)
    if not candidates:
        logger.error("No usable config in %s", [str(p) for p in paths])
        sys.exit(1)
    logger.info(
        "%d unique config(s) from %d file(s); platform '%s'; kernel '%s'%s",
        len(candidates),
        len(paths),
        platform,
        kernel_name,
        " (composite)" if composite else "",
    )

    if args.cudagraph:
        benchmark_fn = do_bench_cudagraph_l2_clear
    else:
        # triton's do_bench flushes L2 per call on its own.
        benchmark_fn = lambda fn, rep, return_mode: triton.testing.do_bench(  # noqa: E731
            fn, rep=int(rep), return_mode=return_mode
        )

    output = (
        Path(args.output)
        if args.output
        else Path.cwd() / f"helion_config_{kernel_name}_{platform}.json"
    )
    factory = RunnerFactory(wrapper)

    results: list[CaseResult] = []
    with default_vllm_config(), torch.inference_mode():
        cases = [
            (key, normalize_arg_sets(inputs))
            for key, inputs in wrapper.get_inputs().items()
            if matches_case_filters(key, args.case)
        ]
        if not cases:
            logger.error("No input case matched; nothing to benchmark")
            sys.exit(1)
        for index, (key, arg_sets) in enumerate(cases, start=1):
            logger.info(
                "Case %d/%d %s: %d input(s), %d candidate config(s)",
                index,
                len(cases),
                key,
                len(arg_sets),
                len(candidates),
            )
            results.append(
                select_for_case(
                    key,
                    arg_sets,
                    candidates,
                    factory,
                    benchmark_fn,
                    numerics_check,
                    args,
                )
            )
            cleanup_gpu_resources()

    print_table(results)

    write_config_file(output, results)
    if args.report:
        write_report(Path(args.report), kernel_name, results)

    missing = [result.case for result in results if result.best is None]
    if missing:
        logger.error("No config ran for %d case(s): %s", len(missing), missing)
        sys.exit(1)


if __name__ == "__main__":
    main()
