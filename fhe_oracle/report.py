# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
# Patent pending: PCT/IB2026/053378
"""Result formatting for FHEOracle.

Renders OracleResult instances as JSON or human-readable Markdown,
suitable for CI/CD artefacts and bug reports.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone

from .core import OracleResult


def to_json(result: OracleResult, indent: int = 2) -> str:
    """Serialise an OracleResult to a JSON string."""
    payload = asdict(result)
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    return json.dumps(payload, indent=indent, default=str)


def to_markdown(result: OracleResult) -> str:
    """Render an OracleResult as a Markdown report."""
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
    return "\n".join(lines)
