#!/usr/bin/env python3
import argparse
import json
import re
import sys
from copy import deepcopy
from pathlib import Path


LOG_PATTERN = re.compile(
    r"""
    \s+
    (?P<dst>\S+)\s*->\s*(?P<src>\S+)
    """,
    re.VERBOSE,
)


def load_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: JSON file not found: {path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse JSON file {path}: {e}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(data, dict):
        print("ERROR: Top-level JSON must be an object/dict.", file=sys.stderr)
        sys.exit(1)

    return data


def parse_log_remaps(path: Path) -> dict[str, str]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"ERROR: Log file not found: {path}", file=sys.stderr)
        sys.exit(1)

    remaps: dict[str, str] = {}
    for line_no, line in enumerate(text.splitlines(), start=1):
        match = LOG_PATTERN.search(line)
        if not match:
            continue

        dst = match.group("dst")
        src = match.group("src")
        remaps[dst] = src

    return remaps


def update_configs(
    original: dict,
    remaps: dict[str, str],
    strict: bool = False,
) -> tuple[dict, list[str], list[str]]:
    """
    Update a copy of the original config dict.

    Important:
    - Every replacement reads from 'original'
    - Every write goes to 'updated'
    - No chain resolution
    - No cycle detection needed
    """
    updated = deepcopy(original)
    applied = []
    skipped = []

    for dst, src in remaps.items():
        if dst not in original:
            msg = f"{dst}: destination key not found in JSON"
            if strict:
                raise KeyError(msg)
            skipped.append(msg)
            continue

        if src not in original:
            msg = f"{dst}: source key {src} not found in JSON"
            if strict:
                raise KeyError(msg)
            skipped.append(msg)
            continue

        if dst == src:
            continue

        updated[dst] = deepcopy(original[src])
        applied.append(f"{dst} -> {src}")

    return updated, applied, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update JSON configs based on benchmark log remap lines."
    )
    parser.add_argument("json_file", type=Path, help="Path to input JSON file")
    parser.add_argument("log_file", type=Path, help="Path to benchmark log file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Path to output JSON file (default: <input>.updated.json)",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input JSON file directly",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on missing keys",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indent for output JSON (default: 2)",
    )

    args = parser.parse_args()

    if args.in_place and args.output is not None:
        print("ERROR: Use either --in-place or --output, not both.", file=sys.stderr)
        sys.exit(1)

    original = load_json(args.json_file)
    remaps = parse_log_remaps(args.log_file)

    if not remaps:
        print("No remap lines found in log file.", file=sys.stderr)
        sys.exit(1)

    updated, applied, skipped = update_configs(
        original=original,
        remaps=remaps,
        strict=args.strict,
    )

    if args.in_place:
        out_path = args.json_file
    elif args.output is not None:
        out_path = args.output
    else:
        out_path = args.json_file.with_name(args.json_file.stem + ".updated.json")

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(updated, f, indent=args.indent, ensure_ascii=False)
        f.write("\n")

    print(f"Updated JSON written to: {out_path}")
    print(f"Applied remaps: {len(applied)}")
    for item in applied:
        print(f"  {item}")

    if skipped:
        print(f"Skipped: {len(skipped)}", file=sys.stderr)
        for item in skipped:
            print(f"  {item}", file=sys.stderr)


if __name__ == "__main__":
    main()
