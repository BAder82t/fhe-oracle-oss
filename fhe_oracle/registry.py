# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Plugin registry for fitness functions and heuristic seed generators.

Core ships with a minimal set of registrations (DivergenceFitness as
the default fitness, and non-patented fallback corner/random seed
generators). External packages -- notably ``fhe-oracle-pro`` -- can
register additional fitness functions (e.g. NoiseBudgetFitness) and
patented heuristic seed generators (Multiplication Magnifier, Depth
Seeker, Near-Threshold Explorer) via Python entry points declared in
their ``pyproject.toml``::

    [project.entry-points."fhe_oracle.fitness"]
    noise_budget = "fhe_oracle_pro.fitness:NoiseBudgetFitness"

    [project.entry-points."fhe_oracle.heuristics"]
    generate_seeds = "fhe_oracle_pro.heuristics:generate_seeds"

Entry points are loaded lazily on the first ``get_*`` call, so
importing Core has no side effects from plugins.

Core consumers call ``get_heuristic(name)`` and let the registry
either return a registered plugin or raise ``KeyError``. Fallbacks
(e.g. ``_fallback_corner_seeds``) are implemented by the consumer,
not by the registry.
"""

from __future__ import annotations

import importlib.metadata
from typing import Any, Callable, Optional

_ENTRY_POINT_GROUPS = {
    "fitness": "fhe_oracle.fitness",
    "heuristics": "fhe_oracle.heuristics",
}

_registered: dict[str, dict[str, Any]] = {
    "fitness": {},
    "heuristics": {},
}

_entry_points_loaded: dict[str, bool] = {
    "fitness": False,
    "heuristics": False,
}


def _load_entry_points(kind: str) -> None:
    """Load entry points for a category once; idempotent."""
    if _entry_points_loaded[kind]:
        return
    group = _ENTRY_POINT_GROUPS[kind]
    try:
        eps = importlib.metadata.entry_points(group=group)
    except TypeError:
        # Python < 3.10 fallback signature
        eps = importlib.metadata.entry_points().get(group, [])
    for ep in eps:
        try:
            obj = ep.load()
        except Exception:
            # Broken plugin — ignore rather than crash Core
            continue
        _registered[kind].setdefault(ep.name, obj)
    _entry_points_loaded[kind] = True


def register_fitness(name: str, factory: Any) -> None:
    """Programmatic registration (overrides entry-point registrations)."""
    _registered["fitness"][name] = factory


def register_heuristic(name: str, factory: Any) -> None:
    """Programmatic registration (overrides entry-point registrations)."""
    _registered["heuristics"][name] = factory


def get_fitness(name: str) -> Any:
    """Return the factory/class registered under ``name``. Raise KeyError."""
    _load_entry_points("fitness")
    try:
        return _registered["fitness"][name]
    except KeyError:
        raise KeyError(
            f"No fitness function registered under {name!r}. "
            f"Available: {sorted(_registered['fitness'])}. "
            f"Install fhe-oracle-pro for noise-budget-aware fitness."
        ) from None


def get_heuristic(name: str) -> Any:
    """Return the factory/class registered under ``name``. Raise KeyError."""
    _load_entry_points("heuristics")
    try:
        return _registered["heuristics"][name]
    except KeyError:
        raise KeyError(
            f"No heuristic registered under {name!r}. "
            f"Available: {sorted(_registered['heuristics'])}. "
            f"Install fhe-oracle-pro for patented heuristic seed "
            f"generators (PCT/IB2026/053378 Claim 5)."
        ) from None


def list_fitness() -> list[str]:
    _load_entry_points("fitness")
    return sorted(_registered["fitness"])


def list_heuristics() -> list[str]:
    _load_entry_points("heuristics")
    return sorted(_registered["heuristics"])


def has_fitness(name: str) -> bool:
    try:
        get_fitness(name)
        return True
    except KeyError:
        return False


def has_heuristic(name: str) -> bool:
    try:
        get_heuristic(name)
        return True
    except KeyError:
        return False


def _reset_for_tests() -> None:
    """Clear registry state. Test-only."""
    _registered["fitness"].clear()
    _registered["heuristics"].clear()
    _entry_points_loaded["fitness"] = False
    _entry_points_loaded["heuristics"] = False
