"""The discovery search space."""

from __future__ import annotations

import pytest

from spxa.discovery.expr import Leaf
from spxa.discovery.families import BROWNIAN, DRIFT, GAMMA, VARIANCE_GAMMA
from spxa.discovery.grammar import (
    Grammar,
    canonical,
    default_subordinated_atoms,
    grammar_from_names,
    parse_atoms,
)


def test_default_atoms() -> None:
    g = Grammar()
    keys = [a.key() for a in g.atoms()]
    assert keys[:7] == ["BM", "VG", "NIG", "Meixner", "NormalJumps", "DoubleExpJumps", "CGMY"]
    assert {"Gamma", "-Gamma", "IG", "-IG", "Poisson", "-Poisson"} <= set(keys)
    assert "(Drift + BM) @ (Gamma + Poisson)" in keys
    assert "(VG) @ (IG)" in keys
    # the subordinations that coincide with families are not generated
    assert "(Drift + BM) @ (Gamma)" not in keys and "(Drift + BM) @ (IG)" not in keys
    assert len(keys) == 7 + 6 + 7


def test_structure_counts_and_uniqueness() -> None:
    g = Grammar(max_terms=2)
    structs = list(g.structures())
    n_atoms = len(g.atoms())
    assert len(structs) == n_atoms + n_atoms * (n_atoms - 1) // 2 == len(g)
    keys = [s.key() for s in structs]
    assert len(set(keys)) == len(keys)
    assert "BM + BM" not in keys and "BM + VG" in keys
    assert len(list(g.structures(max_terms=1))) == n_atoms


def test_extensions_are_canonical_and_skip_duplicates() -> None:
    g = grammar_from_names(["BM", "VG", "Gamma"], max_terms=3)
    ext = {e.key() for e in g.extensions(Leaf(VARIANCE_GAMMA))}
    assert ext == {"BM + VG", "VG + Gamma", "VG + -Gamma"}
    two = canonical(Leaf(VARIANCE_GAMMA) + Leaf(BROWNIAN))
    assert two.key() == "BM + VG"
    # two-sided families, then signed subordinators, then subordinated atoms
    sub = default_subordinated_atoms()[0]
    mixed = canonical(sub + -Leaf(GAMMA) + Leaf(VARIANCE_GAMMA) + Leaf(BROWNIAN))
    assert mixed.key() == f"BM + VG + -Gamma + {sub.key()}"
    assert canonical(Leaf(BROWNIAN)) is not None


def test_parsing_and_validation() -> None:
    atoms = parse_atoms(["VG", "-Gamma", "+IG"])
    assert [a.key() for a in atoms] == ["VG", "-Gamma", "IG"]
    with pytest.raises(ValueError):
        parse_atoms(["-VG"])
    with pytest.raises(KeyError):
        parse_atoms(["Heston"])
    with pytest.raises(ValueError):
        Grammar(max_terms=0)
    with pytest.raises(ValueError):
        Grammar(subordinator_families=(VARIANCE_GAMMA,))
    with pytest.raises(ValueError):
        Grammar(return_families=(DRIFT,))
    g = grammar_from_names(["BM", "Gamma"], subordinated=True)
    assert g.return_families == (BROWNIAN,) and g.subordinator_families == (GAMMA,)
    assert len(g.subordinated) == len(default_subordinated_atoms())
