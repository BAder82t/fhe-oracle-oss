# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Smoke tests for adapter import paths.

Native FHE libraries are skipped when not installed — these tests only
verify that the adapter modules import cleanly and raise a helpful
error when the backend is missing.
"""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "fhe_oracle.adapters.base",
        "fhe_oracle.adapters.openfhe",
        "fhe_oracle.adapters.concrete",
        "fhe_oracle.adapters.seal",
    ],
)
def test_adapter_module_imports(module_name):
    mod = importlib.import_module(module_name)
    assert mod is not None


def test_openfhe_adapter_raises_without_backend():
    try:
        import openfhe  # noqa: F401
        pytest.skip("openfhe is installed; skip missing-backend test")
    except ImportError:
        pass
    from fhe_oracle.adapters.openfhe import OpenFHEAdapter

    with pytest.raises(RuntimeError, match="OpenFHE"):
        OpenFHEAdapter(fhe_fn=lambda cc, ct: ct, n_features=1)


def test_concrete_adapter_raises_without_backend():
    try:
        import concrete  # noqa: F401
        pytest.skip("concrete is installed; skip missing-backend test")
    except ImportError:
        pass
    from fhe_oracle.adapters.concrete import ConcreteAdapter

    with pytest.raises(RuntimeError, match="Concrete"):
        ConcreteAdapter(circuit=None)


def test_seal_adapter_raises_without_backend():
    try:
        import seal  # noqa: F401
        pytest.skip("seal is installed; skip missing-backend test")
    except ImportError:
        pass
    from fhe_oracle.adapters.seal import SealAdapter

    with pytest.raises(RuntimeError, match="seal"):
        SealAdapter(fhe_fn=lambda *a: None, n_features=1)
