r"""
The search space: which compositions of primitives discovery considers.

A :class:`Grammar` lists *atoms*, which are the building blocks of a
candidate. An atom is a single family, a signed subordinator such as
``-Gamma`` (a process with negative jumps only), or a subordinated
structure such as ``(Drift + BM) @ (Gamma + Poisson)``. Candidates are
independent sums of at most ``max_terms`` atoms. A drift leaf is always
appended at fitting time.

The grammar also prunes structures that are not identifiable or that
duplicate another candidate's law:

- ``BM + BM`` is ``BM``. A sum holds at most one copy of any atom whose
  family is closed under convolution with itself at fixed shape.
- ``(Drift + BM) @ Gamma`` is ``VG``, ``(Drift + BM) @ IG`` is ``NIG``, and
  ``(Drift + BM) @ Poisson`` is ``NormalJumps`` (Madan, Carr & Chang 1998;
  Barndorff-Nielsen 1997). Only the family form is generated.
- Sums are commutative, so each multiset of atoms appears once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations_with_replacement
from typing import Iterable, Iterator, Sequence

from spxa.discovery.expr import Leaf, Node, Scale, Subordinate, Sum
from spxa.discovery.families import (
    BROWNIAN,
    CGMY_FAMILY,
    DOUBLE_EXP_JUMPS,
    DRIFT,
    FAMILIES,
    GAMMA,
    INVERSE_GAUSSIAN,
    MEIXNER,
    NIG_FAMILY,
    NORMAL_JUMPS,
    POISSON,
    VARIANCE_GAMMA,
    Family,
)

#: Two-sided return families.
RETURN_FAMILIES: tuple[Family, ...] = (
    BROWNIAN,
    VARIANCE_GAMMA,
    NIG_FAMILY,
    MEIXNER,
    NORMAL_JUMPS,
    DOUBLE_EXP_JUMPS,
    CGMY_FAMILY,
)
#: One-sided families, used with both signs.
SUBORDINATOR_FAMILIES: tuple[Family, ...] = (GAMMA, INVERSE_GAUSSIAN, POISSON)


def signed(family: Family, sign: int) -> Node:
    """``Leaf(family)`` or its negation."""
    return Leaf(family) if sign > 0 else Scale(-1.0, Leaf(family))


def default_subordinated_atoms() -> list[Node]:
    """Time-changed atoms that are not themselves zoo families.

    - Brownian motion with drift run on a *two-component* clock. Gamma +
      Poisson gives VG with extra Gaussian jumps arriving on the clock.
    - VG and NIG run on a Gamma or IG clock: double subordination, which
      makes the tails heavier than either family alone.
    """
    bm = Leaf(DRIFT) + Leaf(BROWNIAN)
    atoms: list[Node] = []
    for c1, c2 in ((GAMMA, POISSON), (INVERSE_GAUSSIAN, POISSON), (GAMMA, INVERSE_GAUSSIAN)):
        atoms.append(Subordinate(bm, Leaf(c1) + Leaf(c2)))
    for base in (VARIANCE_GAMMA, NIG_FAMILY):
        for clock in (GAMMA, INVERSE_GAUSSIAN):
            atoms.append(Subordinate(Leaf(base), Leaf(clock)))
    return atoms


@dataclass(frozen=True)
class Grammar:
    """A finite space of candidate structures.

    Parameters
    ----------
    return_families :
        Two-sided families used as atoms.
    subordinator_families :
        One-sided families, used as ``+F`` and ``-F`` atoms.
    subordinated :
        Extra atoms with ``@`` (see :func:`default_subordinated_atoms`).
    max_terms :
        Largest number of atoms in a sum.
    """

    return_families: tuple[Family, ...] = RETURN_FAMILIES
    subordinator_families: tuple[Family, ...] = SUBORDINATOR_FAMILIES
    subordinated: tuple[Node, ...] = field(
        default_factory=lambda: tuple(default_subordinated_atoms())
    )
    max_terms: int = 2

    def __post_init__(self) -> None:
        if self.max_terms < 1:
            raise ValueError("max_terms must be at least 1")
        for f in self.subordinator_families:
            if not f.is_subordinator:
                raise ValueError(f"{f.name} is not a subordinator family")
        if DRIFT in self.return_families:
            raise ValueError("the drift is added automatically; do not list it as an atom")

    def atoms(self) -> list[Node]:
        out: list[Node] = [Leaf(f) for f in self.return_families]
        for f in self.subordinator_families:
            out += [signed(f, +1), signed(f, -1)]
        return out + list(self.subordinated)

    def structures(self, max_terms: int | None = None) -> Iterator[Node]:
        """Every admissible sum of 1..max_terms atoms, each multiset once."""
        atoms = self.atoms()
        top = self.max_terms if max_terms is None else max_terms
        for k in range(1, top + 1):
            for combo in combinations_with_replacement(range(len(atoms)), k):
                if len(set(combo)) < len(combo):
                    continue  # repeated atom: not identifiable
                nodes = [atoms[i] for i in combo]
                yield nodes[0] if k == 1 else Sum(nodes)

    def extensions(self, structure: Node) -> Iterator[Node]:
        """Structures with one more atom than ``structure`` (for beam search)."""
        present = {
            c.key() for c in (structure.children if isinstance(structure, Sum) else (structure,))
        }
        for atom in self.atoms():
            if atom.key() in present:
                continue
            yield canonical(structure + atom)

    def __len__(self) -> int:
        return sum(1 for _ in self.structures())


def _term_rank(term: Node) -> int:
    """0 for two-sided families, 1 for signed subordinators, 2 for anything else."""
    inner = term.child if isinstance(term, Scale) else term
    if isinstance(inner, Leaf):
        return 1 if inner.family.is_subordinator else 0
    return 2


def canonical(structure: Node) -> Node:
    """Order the terms of a top-level sum so that equal multisets compare equal.

    Two-sided families come first, then signed subordinators, then
    subordinated atoms; ties are broken by key. Rendered models then read
    like ``VG + 0.3 * Gamma``.
    """
    if not isinstance(structure, Sum):
        return structure
    return Sum(sorted(structure.children, key=lambda c: (_term_rank(c), c.key())))


def parse_atoms(names: Iterable[str]) -> list[Node]:
    """Atoms from short names: ``"VG"``, ``"-Gamma"``, ``"+IG"``."""
    out: list[Node] = []
    for raw in names:
        name = raw.strip()
        sign = -1 if name.startswith("-") else 1
        fam = FAMILIES[name.lstrip("+-")]
        if sign < 0 and not fam.is_subordinator:
            raise ValueError(f"{fam.name} is two-sided; its negation is a re-parametrisation")
        out.append(signed(fam, sign))
    return out


def grammar_from_names(
    names: Sequence[str], max_terms: int = 2, subordinated: bool = False
) -> Grammar:
    """A grammar over the named families (both signs for subordinators)."""
    fams = [FAMILIES[n.lstrip("+-")] for n in names]
    return Grammar(
        return_families=tuple(f for f in fams if not f.is_subordinator),
        subordinator_families=tuple(f for f in fams if f.is_subordinator),
        subordinated=tuple(default_subordinated_atoms()) if subordinated else (),
        max_terms=max_terms,
    )
