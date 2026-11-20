"""Comprehensive verification test for all package __init__.py files, exports, and class constructors."""

import importlib
import inspect
from pathlib import Path

import pytest
import spxa


def test_top_level_subpackage_access():
    """Verify that all major subpackages are accessible from spxa directly."""
    for subpkg in [
        "core",
        "zoo",
        "sim",
        "vis",
        "copula",
        "control",
        "calibration",
        "analytics",
        "ops",
        "diff",
        "risk",
        "calculus",
        "discovery",
        "laws",
    ]:
        assert hasattr(spxa, subpkg), f"spxa should expose subpackage '{subpkg}'"
        mod = getattr(spxa, subpkg)
        assert inspect.ismodule(mod), f"spxa.{subpkg} should be a module"


def test_all_init_files_have_valid_exports():
    """Ensure every __init__.py in spxa defines a valid __all__ and exports exist."""
    init_files = list(Path(spxa.__file__).parent.glob("**/__init__.py"))
    assert len(init_files) >= 20

    for init_file in init_files:
        rel_parts = init_file.parent.relative_to(Path(spxa.__file__).parent.parent).parts
        mod_name = ".".join(rel_parts)

        mod = importlib.import_module(mod_name)
        assert hasattr(mod, "__all__"), f"{mod_name} must define __all__"
        all_exports = mod.__all__
        assert isinstance(all_exports, list), f"{mod_name}.__all__ must be a list"

        # Check for duplicates in __all__
        duplicates = [x for x in all_exports if all_exports.count(x) > 1]
        assert not duplicates, f"{mod_name} has duplicate __all__ entries: {set(duplicates)}"

        # Check every export actually exists in the module
        for export_name in all_exports:
            assert hasattr(mod, export_name), (
                f"{mod_name} lists '{export_name}' in __all__, but it does not exist in module"
            )


def test_diff_module_exports():
    """Verify spxa.diff exposes expected solvers, processes, Greeks, and NN tools."""
    pytest.importorskip("jax")
    import spxa.diff as diff

    expected = [
        "DifferentiableGBM",
        "DifferentiableHeston",
        "DifferentiableCIR",
        "DifferentiableSABR",
        "solve_sde",
        "solve_euler_maruyama",
        "solve_milstein",
        "solve_srk",
        "solve_reversible_heun",
        "compute_pathwise_greeks",
        "compute_lrm_greeks_gbm",
        "compute_malliavin_greeks_gbm",
        "compute_higher_order_greeks",
        "DifferentiableNeuralSDE",
        "simulate_deep_hedging_pnl",
    ]
    for symbol in expected:
        assert hasattr(diff, symbol), f"spxa.diff missing '{symbol}'"


def test_differentiable_sde_constructors():
    """Verify all differentiable process classes instantiate and report state/noise dims."""
    pytest.importorskip("jax")
    from spxa.diff.processes import (
        DifferentiableBates,
        DifferentiableBESQ,
        DifferentiableBessel,
        DifferentiableBilinear,
        DifferentiableCEV,
        DifferentiableCIR,
        DifferentiableGBM,
        DifferentiableHawkes,
        DifferentiableHeston,
        DifferentiableKou,
        DifferentiableMerton,
        DifferentiableOU,
        DifferentiableSABR,
        DifferentiableSchwartz1Factor,
        DifferentiableSchwartz2Factor,
        DifferentiableThreeHalves,
        DifferentiableVectorOU,
    )

    process_classes = [
        DifferentiableGBM,
        DifferentiableOU,
        DifferentiableCIR,
        DifferentiableBESQ,
        DifferentiableBessel,
        DifferentiableCEV,
        DifferentiableHeston,
        DifferentiableSABR,
        DifferentiableThreeHalves,
        DifferentiableBates,
        DifferentiableMerton,
        DifferentiableKou,
        DifferentiableSchwartz1Factor,
        DifferentiableSchwartz2Factor,
        DifferentiableVectorOU,
        DifferentiableHawkes,
        DifferentiableBilinear,
    ]

    for cls in process_classes:
        instance = cls()
        assert instance.state_dim >= 1
        assert instance.noise_dim >= 1
        assert isinstance(instance.has_jumps, bool)


def test_graph_module_exports():
    import spxa

    assert hasattr(spxa, "graph")
    assert hasattr(spxa.graph, "ComputationalGraph")
    assert hasattr(spxa.graph, "GradientTape")
    assert hasattr(spxa.graph, "build_black_scholes_pricing_graph")
    assert hasattr(spxa.graph, "GraphCalibrator")
    assert hasattr(spxa.graph, "PortfolioRiskGraph")
