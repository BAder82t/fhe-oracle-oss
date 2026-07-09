# Copyright (C) 2026 Bader Alissaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Result formatting for FHEOracle.

Renders OracleResult instances as JSON or human-readable Markdown,
suitable for CI/CD artefacts and bug reports.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Optional

from .core import OracleResult


def to_json(
    result: OracleResult,
    indent: int = 2,
    diagnostics: Optional[dict[str, Any]] = None,
) -> str:
    """Serialise an OracleResult to a JSON string.

    Parameters
    ----------
    diagnostics : dict, optional
        Included under a top-level ``"diagnostics"`` key when non-empty.
        Shape is caller-defined.
    """
    payload = asdict(result)
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    if diagnostics:
        payload["diagnostics"] = diagnostics
    return json.dumps(payload, indent=indent, default=str)


def to_markdown(
    result: OracleResult, diagnostics: Optional[dict[str, Any]] = None
) -> str:
    """Render an OracleResult as a Markdown report.

    Parameters
    ----------
    diagnostics : dict, optional
        Rendered under ``## Diagnostics`` when non-empty and verdict
        is FAIL.
    """
    lines = [
        "# FHE Oracle Report",
        "",
        f"**Verdict:** {result.verdict}  ",
        f"**Max error:** {result.max_error:.6e}  ",
        f"**Threshold:** {result.threshold:.6e}  ",
        f"**Trials:** {result.n_trials}  ",
        f"**Elapsed:** {result.elapsed_seconds:.2f}s  ",
        f"**Scheme:** {result.scheme}  ",
        "",
        "## Worst input",
        "",
        "```",
        "[" + ", ".join(f"{v:.6f}" for v in result.worst_input) + "]",
        "```",
    ]
    if result.noise_state:
        lines += ["", "## Noise state", ""]
        for k, v in result.noise_state.items():
            lines.append(f"- **{k}:** {v}")
    if diagnostics and result.verdict == "FAIL":
        lines += ["", "## Diagnostics", ""]
        for k, v in diagnostics.items():
            lines.append(f"- **{k}:** {v}")
    return "\n".join(lines)
