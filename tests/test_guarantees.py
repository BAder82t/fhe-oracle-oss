# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for fhe_oracle.guarantees.CoverageCertificate."""

from __future__ import annotations

import math

import pytest

from fhe_oracle.guarantees import CoverageCertificate


# --- budget_for arithmetic ---

def test_budget_for_01_p95():
    """budget_for(eta=0.01, p=0.95) = ceil(log(0.05)/log(0.99)) = 299."""
    expected = math.ceil(math.log(0.05) / math.log(0.99))
    assert expected == 299
    assert CoverageCertificate.budget_for(eta=0.01, p=0.95) == 299


def test_budget_for_001_p99():
    """budget_for(eta=0.001, p=0.99) = ceil(log(0.01)/log(0.999)) = 4603."""
    expected = math.ceil(math.log(0.01) / math.log(0.999))
    assert CoverageCertificate.budget_for(eta=0.001, p=0.99) == expected


def test_budget_for_eta_one():
    """budget_for(eta=1.0, p=0.5) == 1 (one sample always hits)."""
    assert CoverageCertificate.budget_for(eta=1.0, p=0.5) == 1


def test_budget_for_validates_eta():
    with pytest.raises(ValueError):
        CoverageCertificate.budget_for(eta=0.0, p=0.5)
    with pytest.raises(ValueError):
        CoverageCertificate.budget_for(eta=-0.1, p=0.5)
    with pytest.raises(ValueError):
        CoverageCertificate.budget_for(eta=1.1, p=0.5)


def test_budget_for_validates_p():
    with pytest.raises(ValueError):
        CoverageCertificate.budget_for(eta=0.01, p=0.0)
    with pytest.raises(ValueError):
        CoverageCertificate.budget_for(eta=0.01, p=1.0)
    with pytest.raises(ValueError):
        CoverageCertificate.budget_for(eta=0.01, p=-0.1)


# --- p_disc_lower_bound ---

def test_p_disc_lower_bound_nominal():
    """B_rand=300, mu_tau=0.01: 1 - 0.99^300 ≈ 0.9510."""
    cert = CoverageCertificate(budget_rand=300, threshold=0.01, hits=3, mu_hat=0.01)
    assert cert.p_disc_lower_bound(0.01) == pytest.approx(0.9510, abs=1e-3)


def test_p_disc_lower_bound_mu_zero():
    cert = CoverageCertificate(budget_rand=100, threshold=0.01, hits=0, mu_hat=0.0)
    assert cert.p_disc_lower_bound(0.0) == 0.0


def test_p_disc_lower_bound_mu_one():
    cert = CoverageCertificate(budget_rand=100, threshold=0.01, hits=100, mu_hat=1.0)
    assert cert.p_disc_lower_bound(1.0) == 1.0


def test_p_disc_lower_bound_validates_mu():
    cert = CoverageCertificate(budget_rand=10, threshold=0.01, hits=0, mu_hat=0.0)
    with pytest.raises(ValueError):
        cert.p_disc_lower_bound(-0.1)
    with pytest.raises(ValueError):
        cert.p_disc_lower_bound(1.5)


# --- Constructor validation ---

def test_ctor_rejects_zero_budget():
    with pytest.raises(ValueError):
        CoverageCertificate(budget_rand=0, threshold=0.01, hits=0, mu_hat=0.0)


def test_ctor_rejects_negative_budget():
    with pytest.raises(ValueError):
        CoverageCertificate(budget_rand=-1, threshold=0.01, hits=0, mu_hat=0.0)


def test_ctor_rejects_negative_threshold():
    with pytest.raises(ValueError):
        CoverageCertificate(budget_rand=100, threshold=-0.1, hits=0, mu_hat=0.0)


def test_ctor_rejects_hits_exceeding_budget():
    with pytest.raises(ValueError):
        CoverageCertificate(budget_rand=100, threshold=0.01, hits=101, mu_hat=1.01)


def test_ctor_rejects_negative_hits():
    with pytest.raises(ValueError):
        CoverageCertificate(budget_rand=100, threshold=0.01, hits=-1, mu_hat=-0.01)


def test_ctor_rejects_mu_hat_inconsistent_with_hits():
    with pytest.raises(ValueError):
        CoverageCertificate(budget_rand=100, threshold=0.01, hits=5, mu_hat=0.10)


# --- p_discovery derived field ---

def test_p_discovery_derived_at_construction():
    """hits=5, budget_rand=100, mu_hat=0.05 → p_discovery ≈ 1 - 0.95^100."""
    cert = CoverageCertificate(
        budget_rand=100, threshold=0.01, hits=5, mu_hat=0.05
    )
    expected = 1.0 - 0.95 ** 100
    assert cert.p_discovery == pytest.approx(expected, rel=1e-9)
    assert cert.p_discovery == pytest.approx(0.9941, abs=1e-3)


def test_p_discovery_zero_hits_zero():
    cert = CoverageCertificate(budget_rand=100, threshold=0.01, hits=0, mu_hat=0.0)
    assert cert.p_discovery == 0.0


def test_ctor_frozen():
    """Dataclass is frozen; fields cannot be rebound."""
    cert = CoverageCertificate(budget_rand=10, threshold=0.01, hits=1, mu_hat=0.1)
    with pytest.raises(Exception):
        cert.hits = 2  # type: ignore[misc]
