# Copyright (C) 2026 Bader Alissaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Command-line interface: ``fhe-oracle check <model.py>``.

Runs :func:`~fhe_oracle.check.check` against a Python file that defines
``plaintext_fn``, ``fhe_fn``, and ``input_bounds`` at module level, and
prints the report. Optional module-level ``n_trials``/``threshold``/
``seed`` set defaults; CLI flags override them.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any, Optional

from .check import check

REQUIRED_ATTRS = ("plaintext_fn", "fhe_fn", "input_bounds")


def _load_model_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fhe-oracle",
        description="Adversarial precision testing for FHE programs.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser(
        "check", help="Run a precision check against a model file."
    )
    check_parser.add_argument(
        "model",
        type=Path,
        help="Python file defining plaintext_fn, fhe_fn, input_bounds.",
    )
    check_parser.add_argument("--n-trials", type=int, default=None)
    check_parser.add_argument("--threshold", type=float, default=None)
    check_parser.add_argument("--seed", type=int, default=None)
    check_parser.add_argument(
        "--format", choices=["markdown", "json"], default="markdown"
    )
    check_parser.add_argument(
        "--no-shrink", action="store_true", help="Skip auto-shrinking on FAIL."
    )
    return parser


def _run_check(args: argparse.Namespace) -> int:
    model_path: Path = args.model
    if not model_path.exists():
        print(f"fhe-oracle: model file not found: {model_path}", file=sys.stderr)
        return 2

    module = _load_model_module(model_path)
    missing = [attr for attr in REQUIRED_ATTRS if not hasattr(module, attr)]
    if missing:
        print(
            f"fhe-oracle: {model_path} is missing required attribute(s): "
            f"{', '.join(missing)}",
            file=sys.stderr,
        )
        return 2

    kwargs: dict[str, Any] = {}
    for cli_val, name in (
        (args.n_trials, "n_trials"),
        (args.threshold, "threshold"),
        (args.seed, "seed"),
    ):
        if cli_val is not None:
            kwargs[name] = cli_val
        elif hasattr(module, name):
            kwargs[name] = getattr(module, name)

    result = check(
        module.plaintext_fn,
        module.fhe_fn,
        input_bounds=module.input_bounds,
        shrink=not args.no_shrink,
        report_format=args.format,
        **kwargs,
    )
    print(result.report)
    return 0 if result.oracle_result.verdict == "PASS" else 1


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "check":
        try:
            return _run_check(args)
        except Exception as exc:  # noqa: BLE001 - top-level guard; exit 2 marks errors
            print(f"fhe-oracle: ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
