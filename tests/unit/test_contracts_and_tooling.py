"""Tooling/error-path regressions; mathematical identities live under exactness/."""

import json

import numpy as np
import pytest

from scripts.check_benchmark_regression import load_results, regressions
from scripts.check_exactness_markers import check_file
from scripts.extract_changelog import extract
from spxa import BrownianMotion, GammaProcess
from spxa.sim import brownian_bridge, brownian_bridge_interpolate, first_passage_time_exact_bm
from spxa.story.narrator import CompositionNode, NodeKind, narrate, narrate_latex, story_latex


@pytest.mark.parametrize("heading", ["## [1.2.3]", "## [1.2.3] — 2026-10-23", "## 1.2.3 - date"])
def test_release_parser_keeps_first_content_and_subheadings(tmp_path, heading):
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(
        f"# Changelog\n\n{heading}\n\nFirst content.\n\n"
        "### Fixed\n- Correct law.\n\n## [1.2.2]\nOlder.\n"
    )
    assert extract("v1.2.3", changelog) == "First content.\n\n### Fixed\n- Correct law."
    with pytest.raises(ValueError):
        extract("9.9.9", changelog)


@pytest.mark.parametrize(
    "source",
    [
        "import pytest\npytestmark=pytest.mark.skip(reason='x')\n",
        "import pytest\n@pytest.mark.xfail\nclass TestX: pass\n",
        "import pytest\ndef test_x(): pytest.skip('x')\n",
        "from pytest import xfail as hide\ndef test_x(): hide('x')\n",
        "import pytest\npytest.importorskip('foo')\n",
    ],
)
def test_exactness_guard_catches_non_function_decorators(tmp_path, source):
    path = tmp_path / "test_contract.py"
    path.write_text(source)
    assert check_file(path)


def test_exactness_guard_accepts_a_normal_mark(tmp_path):
    path = tmp_path / "test_good.py"
    path.write_text("import pytest\npytestmark=pytest.mark.exactness\ndef test_x(): assert True\n")
    assert not check_file(path)


def test_benchmark_comparison_is_explicit_and_validated(tmp_path):
    path = tmp_path / "b.json"
    path.write_text(
        json.dumps({"benchmarks": [{"name": "run", "fullname": "x::run", "stats": {"mean": 0.1}}]})
    )
    assert load_results(path) == {"x::run": 0.1}
    assert regressions({"x": 1}, {"x": 1.3}, 0.2)
    assert not regressions({"x": 1}, {"x": 1.1}, 0.2)
    with pytest.raises(ValueError):
        regressions({"x": 1}, {"y": 1}, 0.2)
    with pytest.raises(ValueError):
        regressions({"x": 1}, {"x": 1}, -0.1)
    path.write_text('{"benchmarks":[]}')
    with pytest.raises(ValueError):
        load_results(path)


@pytest.mark.parametrize("kind", list(NodeKind))
def test_frozen_story_nodes_still_render(kind):
    node = CompositionNode(
        kind=kind,
        name="example",
        children=[
            CompositionNode(kind=NodeKind.PRIMITIVE, name="BM"),
            CompositionNode(kind=NodeKind.PRIMITIVE, name="Gamma"),
        ],
    )
    assert isinstance(narrate(node), str)
    assert isinstance(narrate_latex(node), str)
    with pytest.raises((AttributeError, TypeError)):
        node.name = "changed"


def test_story_latex_for_exact_and_middle_capabilities():
    for law in [BrownianMotion(), BrownianMotion() + GammaProcess(a=1, b=2)]:
        assert "\\" in story_latex(law)


@pytest.mark.parametrize("kwargs", [{"t_end": 0}, {"sigma": -1}, {"x_end": np.nan}, {"n_paths": 0}])
def test_brownian_bridge_rejects_bad_parameters(kwargs):
    with pytest.raises(ValueError):
        brownian_bridge(**kwargs)


def test_bridge_interpolator_rejects_nonfinite_paths():
    with pytest.raises(ValueError):
        brownian_bridge_interpolate(
            np.array([[0, np.nan]]), np.array([0.0, 1.0]), np.array([0.0, 0.5, 1.0])
        )


@pytest.mark.parametrize(
    "kwargs", [{"sigma": 0}, {"sigma": -1}, {"level": np.inf}, {"n_samples": 0}]
)
def test_first_passage_invalid_domain(kwargs):
    with pytest.raises(ValueError):
        first_passage_time_exact_bm(**({"level": 1.0} | kwargs))


def test_tempered_stable_low_proposal_budget_raises_instead_of_approximating():
    from spxa.sim import sample_tempered_stable

    with pytest.raises(RuntimeError, match="budget exhausted"):
        sample_tempered_stable(
            alpha=0.5,
            C=0.4,
            lam=2.0,
            dt=1.0,
            n_steps=1,
            n_paths=100,
            rng=np.random.default_rng(42),
            n_rejection_attempts=1,
        )


@pytest.mark.parametrize("number", [1, 2, 3])
def test_notebook_generator_sources_and_ids_are_stable(tmp_path, number):
    from pathlib import Path

    from scripts.build_tutorials import nb1_cells, nb2_cells, nb3_cells, write_nb

    tutorial_dir = Path(__file__).resolve().parents[2] / "docs" / "tutorials"
    source = next(tutorial_dir.glob(f"0{number}_*.ipynb"))
    previous = json.loads(source.read_text())
    target = tmp_path / source.name
    target.write_text(source.read_text())
    write_nb([nb1_cells, nb2_cells, nb3_cells][number - 1], target)
    generated = json.loads(target.read_text())

    def normalize(cells):
        return [(cell["cell_type"], "".join(cell["source"])) for cell in cells]

    assert normalize(generated["cells"]) == normalize(previous["cells"])
    assert [c["id"] for c in generated["cells"]] == [c["id"] for c in previous["cells"]]


@pytest.mark.parametrize("H", [0.3, 0.5, 0.7])
def test_physics_tutorial_msd_curve_matches_the_one_dimensional_variance(H):
    import ast

    from scripts.build_tutorials import nb3_cells
    from spxa.zoo.beyond import FractionalBrownianMotion

    cell = next(c for c in nb3_cells if c["cell_type"] == "code" and "theory =" in c["source"])
    assignment = next(
        n
        for n in ast.walk(ast.parse(cell["source"]))
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "theory" for t in n.targets)
    )
    times = np.array([0.01, 0.1, 0.5, 1.0])
    curve = eval(
        compile(ast.Expression(assignment.value), "msd_reference", "eval"),
        {"__builtins__": {}},
        {"H": H, "t_vals": times},
    )
    expected = np.array([FractionalBrownianMotion(H=H, sigma=1).variance(t) for t in times])
    np.testing.assert_allclose(curve, expected, rtol=1e-14)


def test_first_passage_tutorial_histograms_keep_unconditional_and_censoring_mass():
    import ast

    from scripts.build_tutorials import nb3_cells

    cell = next(c for c in nb3_cells if "tau_exact = first_passage_time_exact_bm" in c["source"])
    calls = [
        n
        for n in ast.walk(ast.parse(cell["source"]))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "hist"
    ]
    values = {
        "np": np,
        "tau_exact": np.array([0.05, 0.15, 4.0, 20.0]),
        "tau_sim": np.array([0.05, 0.15, 3.0]),
        "n_grid_paths": 100,
        "bins": np.linspace(0.0, 8.0, 81),
        "bin_width": 0.1,
    }
    expected = {"tau_exact": 0.75, "tau_sim": 0.03}
    assert len(calls) == 2
    for call in calls:
        data = eval(
            compile(ast.Expression(call.args[0]), "histogram_data", "eval"),
            {"__builtins__": {}},
            values,
        )
        kwargs = {
            k.arg: eval(
                compile(ast.Expression(k.value), "histogram_option", "eval"),
                {"__builtins__": {}},
                values,
            )
            for k in call.keywords
            if k.arg in {"bins", "weights", "density"}
        }
        height, edges = np.histogram(data, **kwargs)
        assert np.dot(height, np.diff(edges)) == pytest.approx(expected[call.args[0].id])
