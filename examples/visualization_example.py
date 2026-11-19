"""
End-to-end demonstration of the spxa.vis visualization module.

Covers:
1. Lévy Triplet & Jump Density Inspector:
   - Triplet decomposition (b, σ², ν)
   - Finite vs Infinite activity classification
   - Blumenthal-Getoor index estimation
   - Dual-panel plotting: near-zero singularity zoom vs tail decay
   - Cross-model Lévy density comparison (Merton vs Kou vs VG vs CGMY)
2. Implied Volatility Surfaces:
   - 3D Surface generation (K, T, σ_IV) with wireframe and isoline contours
   - Cross-model volatility smile comparison at fixed maturity T
   - Implied volatility term structure across maturities
3. Story Composition DAG:
   - Export computation graph to ASCII tree
   - Export to Mermaid flowchart diagram
   - Export to Graphviz DOT format
   - Render publication-quality hierarchical DAG plot via NetworkX
"""

import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from spxa import (
    BrownianMotion,
    CGMY,
    CompoundPoisson,
    GammaProcess,
    HestonProcess,
    KouJumpDiffusion,
    MertonJumpDiffusion,
    NormalJump,
    SABRProcess,
    VarianceGamma,
)
from spxa.vis import (
    dag_to_ascii,
    dag_to_dot,
    dag_to_mermaid,
    inspect_levy_triplet,
    plot_dag,
    plot_iv_surface,
    plot_levy_comparison,
    plot_levy_density,
    plot_smile_comparison,
    plot_term_structure,
)


def run_levy_inspection_demo(output_dir: str = "/tmp/spxa_vis_demo") -> None:
    print("=" * 70)
    print("1. LÉVY TRIPLET INSPECTION & DENSITY VISUALIZATION")
    print("=" * 70)

    # A. Inspect finite activity jump-diffusion (Merton)
    merton = MertonJumpDiffusion(drift=0.05, sigma=0.15, rate=2.0, jump_mean=-0.08, jump_std=0.12)
    insp_merton = inspect_levy_triplet(merton)
    print(insp_merton.summary())
    print()

    # B. Inspect pure-jump infinite activity with finite variation (Variance Gamma)
    vg = VarianceGamma(sigma=0.2, nu=0.15, theta=-0.05)
    insp_vg = inspect_levy_triplet(vg)
    print(insp_vg.summary())
    print()

    # C. Inspect CGMY (tunable Blumenthal-Getoor index Y = 0.7)
    cgmy = CGMY(C=1.0, G=2.5, M=3.5, Y=0.7)
    insp_cgmy = inspect_levy_triplet(cgmy)
    print(insp_cgmy.summary())
    print()

    # D. Generate dual-panel plot for Variance Gamma
    vg_plot_path = os.path.join(output_dir, "vg_levy_density.png")
    fig_vg, _ = plot_levy_density(
        vg, x_range=(-1.5, 1.5), near_zero_range=0.08, save_path=vg_plot_path
    )
    plt.close(fig_vg)
    print(f"Saved Variance Gamma density plot: {vg_plot_path}")

    # E. Cross-model Lévy density comparison (Merton vs Kou vs VG vs CGMY)
    cmp_plot_path = os.path.join(output_dir, "levy_comparison.png")
    fig_cmp, _ = plot_levy_comparison(
        {
            "Merton Jump-Diffusion": merton,
            "Kou Double-Exponential": KouJumpDiffusion(rate=2.0),
            "Variance Gamma": vg,
            "CGMY (Y=0.7)": cgmy,
        },
        x_range=(-1.2, 1.2),
        near_zero_range=0.08,
        save_path=cmp_plot_path,
    )
    plt.close(fig_cmp)
    print(f"Saved Lévy comparison plot: {cmp_plot_path}")
    print()


def run_surface_demo(output_dir: str = "/tmp/spxa_vis_demo") -> None:
    print("=" * 70)
    print("2. IMPLIED VOLATILITY SURFACES & SMILE COMPARISONS")
    print("=" * 70)

    strikes = np.linspace(80, 120, 17)
    maturities = np.linspace(0.1, 2.0, 15)

    # A. 3D Implied Volatility Surface for Merton Jump-Diffusion
    merton = MertonJumpDiffusion(sigma=0.15, rate=1.5, jump_mean=-0.06, jump_std=0.14)
    surf_path_merton = os.path.join(output_dir, "merton_iv_surface.png")
    fig_surf, _ = plot_iv_surface(
        merton,
        strikes=strikes,
        maturities=maturities,
        S0=100.0,
        title="Merton Jump-Diffusion — 3D Implied Volatility Surface",
        save_path=surf_path_merton,
    )
    plt.close(fig_surf)
    print(f"Saved 3D IV Surface for Merton: {surf_path_merton}")

    # B. 3D Implied Volatility Surface for Heston Stochastic Volatility
    heston = HestonProcess(v0=0.04, kappa=2.5, theta=0.04, sigma_v=0.35, rho=-0.7, s0=100.0)
    surf_path_heston = os.path.join(output_dir, "heston_iv_surface.png")
    fig_h, _ = plot_iv_surface(
        heston,
        strikes=strikes,
        maturities=maturities,
        S0=100.0,
        cmap="plasma",
        title="Heston Stochastic Volatility — 3D Implied Volatility Surface",
        save_path=surf_path_heston,
    )
    plt.close(fig_h)
    print(f"Saved 3D IV Surface for Heston: {surf_path_heston}")

    # C. Cross-Model Smile Comparison at T = 1.0 yr
    smile_path = os.path.join(output_dir, "smile_comparison.png")
    fig_smile, _ = plot_smile_comparison(
        {
            "Merton Jump-Diffusion": merton,
            "Kou Jump-Diffusion": KouJumpDiffusion(rate=1.5),
            "Heston Stochastic Volatility": heston,
            "SABR Model": SABRProcess(f0=100.0, alpha=0.22, beta=0.7, nu=0.45, rho=-0.35),
        },
        strikes=np.linspace(85, 115, 13),
        T=1.0,
        S0=100.0,
        save_path=smile_path,
    )
    plt.close(fig_smile)
    print(f"Saved Volatility Smile comparison: {smile_path}")

    # D. Term Structure of Implied Volatility
    term_path = os.path.join(output_dir, "term_structure.png")
    fig_term, _ = plot_term_structure(
        {
            "Merton": merton,
            "Heston": heston,
        },
        maturities=np.linspace(0.1, 3.0, 12),
        strike=100.0,
        save_path=term_path,
    )
    plt.close(fig_term)
    print(f"Saved Term Structure plot: {term_path}")
    print()


def run_story_dag_demo(output_dir: str = "/tmp/spxa_vis_demo") -> None:
    print("=" * 70)
    print("3. PROCESS COMPOSITION DAG VISUALIZATION & EXPORT")
    print("=" * 70)

    # Build composed process: 2.5 * (BrownianMotion + CompoundPoisson)
    bm = BrownianMotion(mu=0.05, sigma=0.2)
    cp = CompoundPoisson(rate=1.2, jump_dist=NormalJump(mean=-0.02, std=0.1))
    composite_proc = 2.5 * (bm + cp)

    # A. ASCII / Unicode Tree
    print("[ASCII Tree Representation]")
    print(dag_to_ascii(composite_proc))
    print()

    # B. Mermaid Diagram
    print("[Mermaid Diagram Syntax]")
    print(dag_to_mermaid(composite_proc))
    print()

    # C. Graphviz DOT
    dot_str = dag_to_dot(composite_proc)
    dot_file = os.path.join(output_dir, "composition_dag.dot")
    with open(dot_file, "w") as f:
        f.write(dot_str)
    print(f"Saved Graphviz DOT file: {dot_file}")

    # D. Publication-quality Matplotlib Diagram via NetworkX
    dag_img_path = os.path.join(output_dir, "composition_dag.png")
    fig_dag, _ = plot_dag(
        composite_proc,
        title="Composition Graph: 2.5 * (BrownianMotion + CompoundPoisson)",
        save_path=dag_img_path,
    )
    plt.close(fig_dag)
    print(f"Saved Composition DAG graph: {dag_img_path}")

    # E. Subordinated process DAG: BrownianMotion @ GammaProcess
    vg_composed = bm @ GammaProcess(a=1.0, b=2.0)
    sub_img_path = os.path.join(output_dir, "subordination_dag.png")
    fig_sub, _ = plot_dag(
        vg_composed,
        title="Subordination Graph: BrownianMotion @ GammaProcess",
        save_path=sub_img_path,
    )
    plt.close(fig_sub)
    print(f"Saved Subordination DAG graph: {sub_img_path}")
    print()


def main() -> None:
    output_dir = "/tmp/spxa_vis_demo"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Running spxa.vis visualization suite (output: {output_dir})...\n")
    run_levy_inspection_demo(output_dir)
    run_surface_demo(output_dir)
    run_story_dag_demo(output_dir)
    print("=" * 70)
    print("All visualization demonstrations completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
