# Copyright (C) 2026 Bader Alissaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the plugin registry (fhe_oracle.registry)."""

from __future__ import annotations

import pytest

from fhe_oracle import registry


@pytest.fixture(autouse=True)
def _clean_registry():
    registry._reset_for_tests()
    yield
    registry._reset_for_tests()


def test_register_and_get_fitness():
    sentinel = object()
    registry.register_fitness("my_fitness", sentinel)
    assert registry.get_fitness("my_fitness") is sentinel


def test_register_and_get_heuristic():
    sentinel = object()
    registry.register_heuristic("my_heuristic", sentinel)
    assert registry.get_heuristic("my_heuristic") is sentinel


def test_get_fitness_raises_keyerror_when_unregistered():
    with pytest.raises(KeyError):
        registry.get_fitness("does_not_exist")


def test_get_heuristic_raises_keyerror_when_unregistered():
    with pytest.raises(KeyError):
        registry.get_heuristic("does_not_exist")


def test_has_fitness_true_and_false():
    assert not registry.has_fitness("my_fitness")
    registry.register_fitness("my_fitness", object())
    assert registry.has_fitness("my_fitness")


def test_has_heuristic_true_and_false():
    assert not registry.has_heuristic("my_heuristic")
    registry.register_heuristic("my_heuristic", object())
    assert registry.has_heuristic("my_heuristic")


def test_list_fitness_and_heuristics_sorted():
    registry.register_fitness("b_fitness", object())
    registry.register_fitness("a_fitness", object())
    assert registry.list_fitness() == ["a_fitness", "b_fitness"]

    registry.register_heuristic("z_heuristic", object())
    registry.register_heuristic("a_heuristic", object())
    assert registry.list_heuristics() == ["a_heuristic", "z_heuristic"]


def test_register_fitness_overrides_existing():
    first, second = object(), object()
    registry.register_fitness("dup", first)
    registry.register_fitness("dup", second)
    assert registry.get_fitness("dup") is second
