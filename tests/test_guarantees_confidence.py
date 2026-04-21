# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for pass_confidence / confidence_adjusted_pass (Limitation 6, Direction 1)."""

from __future__ import annotations

import math

import pytest

from fhe_oracle.guarantees import CoverageCertificate, confidence_adjusted_pass


def test_pass_confidence_matches_rand_bound():
    cert = CoverageCertificate(budget_rand=150, threshold=0.01, hits=0, mu_hat=0.0)
    assert cert.pass_confidence(0.01) == pytest.approx(1.0 - 0.99 ** 150, rel=1e-12)


def test_pass_confidence_zero_eta():
    cert = CoverageCertificate(budget_rand=50, threshold=0.01, hits=0, mu_hat=0.0)
    assert cert.pass_confidence(0.0) == 0.0


def test_pass_confidence_full_eta():
    cert = CoverageCertificate(budget_rand=7, threshold=0.01, hits=7, mu_hat=1.0)
    assert cert.pass_confidence(1.0) == 1.0


def test_pass_confidence_validates_eta():
    cert = CoverageCertificate(budget_rand=10, threshold=0.01, hits=0, mu_hat=0.0)
    with pytest.raises(ValueError):
        cert.pass_confidence(-1e-9)
    with pytest.raises(ValueError):
        cert.pass_confidence(1.1)


def test_confidence_adjusted_pass_rand_only():
    cert = CoverageCertificate(budget_rand=150, threshold=0.01, hits=0, mu_hat=0.0)
    conf = confidence_adjusted_pass(cert, eta=0.01, p_cma=0.0)
    assert conf == pytest.approx(cert.pass_confidence(0.01), rel=1e-12)


def test_confidence_adjusted_pass_hybrid_increases():
    cert = CoverageCertificate(budget_rand=150, threshold=0.01, hits=0, mu_hat=0.0)
    rand = confidence_adjusted_pass(cert, eta=0.01, p_cma=0.0)
    hybrid = confidence_adjusted_pass(cert, eta=0.01, p_cma=0.5)
    assert hybrid > rand


def test_confidence_adjusted_pass_decomposes():
    """conf = 1 - (1 - p_rand)(1 - p_cma)."""
    cert = CoverageCertificate(budget_rand=100, threshold=0.01, hits=0, mu_hat=0.0)
    p_rand = cert.pass_confidence(1e-3)
    p_cma = 0.3
    expected = 1.0 - (1.0 - p_rand) * (1.0 - p_cma)
    got = confidence_adjusted_pass(cert, eta=1e-3, p_cma=p_cma)
    assert got == pytest.approx(expected, rel=1e-12)


def test_confidence_adjusted_pass_no_certificate():
    assert confidence_adjusted_pass(None, eta=1e-3, p_cma=0.8) == pytest.approx(0.8)
    assert confidence_adjusted_pass(None, eta=1e-3, p_cma=0.0) == 0.0


def test_confidence_adjusted_pass_validates_eta_and_p_cma():
    cert = CoverageCertificate(budget_rand=10, threshold=0.01, hits=0, mu_hat=0.0)
    with pytest.raises(ValueError):
        confidence_adjusted_pass(cert, eta=0.0, p_cma=0.0)
    with pytest.raises(ValueError):
        confidence_adjusted_pass(cert, eta=1.1, p_cma=0.0)
    with pytest.raises(ValueError):
        confidence_adjusted_pass(cert, eta=0.01, p_cma=-0.1)
    with pytest.raises(ValueError):
        confidence_adjusted_pass(cert, eta=0.01, p_cma=1.1)
