# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
# Patent pending: PCT/IB2026/053378
"""Coverage certificate: Chernoff-style bound on failure discovery.

Given B_rand uniform samples from domain X, of which k hit the region
{x : δ(x) ≥ τ}, the empirical failure fraction is μ̂_τ = k / B_rand.

The certificate provides:

- `p_disc_lower_bound(mu_tau)`: P[at least 1 hit in B_rand trials]
  = 1 − (1 − mu_tau)^B_rand. This is the exact binomial complement,
  not a Chernoff relaxation; we call it "Chernoff bound" in the paper
  because the Chernoff tail is the standard reference for the
  multiplicative form.
- `budget_for(eta, p)`: minimum B_rand such that P[≥1 hit] ≥ p,
  given assumed μ_τ = eta.

See `research/future-work/09-formal-budget-failure-prob-guarantees.md`
and `research/experiment-plan/A4-formal-chernoff-bound.md`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class CoverageCertificate:
    """Probabilistic guarantee from the random-sampling phase."""

    budget_rand: int  # B_rand: number of uniform samples drawn
    threshold: float  # τ: divergence threshold
    hits: int         # k: samples with δ(x) ≥ τ
    mu_hat: float     # k / B_rand (empirical failure fraction)

    # Derived at construction
    p_discovery: float = field(init=False)
    """P[≥1 hit | μ_τ = μ̂_τ, B_rand trials] = 1 − (1 − μ̂_τ)^B_rand."""

    def __post_init__(self) -> None:
        if self.budget_rand <= 0:
            raise ValueError("budget_rand must be positive")
        if self.threshold < 0:
            raise ValueError("threshold must be non-negative")
        if not (0 <= self.hits <= self.budget_rand):
            raise ValueError("hits must be in [0, budget_rand]")
        expected_mu = self.hits / self.budget_rand
        if abs(expected_mu - self.mu_hat) > 1e-12:
            raise ValueError("mu_hat must equal hits / budget_rand")
        object.__setattr__(
            self,
            "p_discovery",
            1.0 - (1.0 - self.mu_hat) ** self.budget_rand,
        )

    def p_disc_lower_bound(self, mu_tau: float) -> float:
        """P[≥1 hit in B_rand trials] assuming true failure fraction mu_tau."""
        if not (0.0 <= mu_tau <= 1.0):
            raise ValueError("mu_tau must be in [0, 1]")
        if mu_tau == 0.0:
            return 0.0
        return 1.0 - (1.0 - mu_tau) ** self.budget_rand

    @staticmethod
    def budget_for(eta: float, p: float = 0.95) -> int:
        """Minimum B_rand so P[≥1 hit] ≥ p, given μ_τ = eta.

        Solves 1 − (1 − eta)^B ≥ p → B = ⌈log(1−p) / log(1−eta)⌉.
        """
        if not (0.0 < eta <= 1.0):
            raise ValueError("eta must be in (0, 1]")
        if not (0.0 < p < 1.0):
            raise ValueError("p must be in (0, 1)")
        if eta == 1.0:
            return 1
        return math.ceil(math.log(1.0 - p) / math.log(1.0 - eta))

    def pass_confidence(self, eta: float) -> float:
        """Random-phase confidence that a bug region of measure ≥ eta was detected.

        Returns 1 − (1 − eta)^B_rand. This is a CONDITIONAL statement:
        *if* the true failure region has relative measure at least eta,
        *then* the random phase would have produced at least one hit
        with at least this probability. It is not a proof of absence.

        The parameter eta is chosen by the practitioner and represents
        the smallest bug-region measure they care about (e.g., 1e-4 =
        "detect any bug affecting ≥0.01% of the input space").
        """
        if not (0.0 <= eta <= 1.0):
            raise ValueError("eta must be in [0, 1]")
        if eta == 0.0:
            return 0.0
        return 1.0 - (1.0 - eta) ** self.budget_rand


def confidence_adjusted_pass(
    certificate: Optional["CoverageCertificate"],
    eta: float,
    p_cma: float = 0.0,
) -> float:
    """Hybrid PASS confidence parameterised by bug-region measure eta.

    Combines the random-phase coverage certificate with an optional
    CMA-ES success probability p_cma to give

        conf = 1 − (1 − eta)^{B_rand} · (1 − p_cma)

    The p_cma term defaults to 0 (worst-case, corresponding to the
    landscape satisfying property Q of the conditional CMA-ES
    guarantee — plateau-then-cliff). Pass a non-zero value only when
    P1–P3 have been verified empirically (see
    ``research/theory/cma-es-conditional-guarantee.md``).

    Parameters
    ----------
    certificate
        Coverage certificate from the random phase, or None if the
        random floor was zero.
    eta
        Smallest bug-region relative measure of interest. Must be in
        ``(0, 1]``.
    p_cma
        Optional CMA-ES discovery probability from the conditional
        bound. Must be in ``[0, 1]``. Default 0 (conservative).
    """
    if not (0.0 < eta <= 1.0):
        raise ValueError("eta must be in (0, 1]")
    if not (0.0 <= p_cma <= 1.0):
        raise ValueError("p_cma must be in [0, 1]")
    p_rand = certificate.pass_confidence(eta) if certificate is not None else 0.0
    return 1.0 - (1.0 - p_rand) * (1.0 - p_cma)
