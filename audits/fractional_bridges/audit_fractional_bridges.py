r"""Mathematical Audit for Fractional Brownian Motion and Bridge Processes."""

import numpy as np

from audits.core.audit_harness import (
    ProcessAuditResult,
    verify_moment,
    verify_positivity,
)
from spxa.zoo.beyond.bessel_bridge import BesselBridge
from spxa.zoo.beyond.diffusion_bridge import DiffusionBridge
from spxa.zoo.beyond.fbm import FractionalBrownianMotion


def audit_fbm_process() -> ProcessAuditResult:
    result = ProcessAuditResult(process_name="FractionalBrownianMotion")

    H_vals = [0.35, 0.50, 0.70]
    sigma = 0.5
    t = 1.5

    for H in H_vals:
        fbm = FractionalBrownianMotion(H=H, sigma=sigma)
        theo_var = fbm.variance(t)

        n_samples = 15_000
        n_steps = 100
        rng = np.random.default_rng(1001 + int(H * 100))
        paths = fbm.simulate(n_steps=n_steps, n_paths=n_samples, T=t, rng=rng)
        term = paths[:, -1]

        verify_moment(result, f"fBM_H_{H}_Mean", theoretical=0.0, samples=term)
        verify_moment(result, f"fBM_H_{H}_Variance", theoretical=theo_var, samples=term)

    return result


def audit_bessel_bridge() -> ProcessAuditResult:
    result = ProcessAuditResult(process_name="BesselBridge")

    dim = 3.5
    x0 = 1.0
    xT = 2.5
    T = 2.0
    t_mid = 1.0

    bb = BesselBridge(dim=dim, x0=x0, xT=xT, T=T)

    n_samples = 20_000
    n_steps = 100
    paths = bb.simulate(num_steps=n_steps, num_paths=n_samples, seed=1002)

    # 1. Endpoints pinning
    pinned_start = bool(np.allclose(paths[:, 0], x0))
    pinned_end = bool(np.allclose(paths[:, -1], xT))

    result.add_check(
        name="BESQ_Bridge_Start_Pinning",
        theoretical=x0,
        empirical=float(np.mean(paths[:, 0])),
        passed=pinned_start,
        details=f"All paths start at x0={x0}",
    )
    result.add_check(
        name="BESQ_Bridge_End_Pinning",
        theoretical=xT,
        empirical=float(np.mean(paths[:, -1])),
        passed=pinned_end,
        details=f"All paths terminate at xT={xT}",
    )

    # 2. Positivity
    verify_positivity(result, samples=paths.flatten(), min_allowed=0.0)

    # 3. Intermediate conditional mean
    theo_mid_mean = bb.mean(t_mid)
    mid_samples = paths[:, n_steps // 2]
    verify_moment(result, "BESQ_Bridge_Mid_Mean", theoretical=theo_mid_mean, samples=mid_samples)

    return result


def audit_diffusion_bridge() -> ProcessAuditResult:
    result = ProcessAuditResult(process_name="DiffusionBridge")

    # 1. Brownian Bridge
    a_bm = 1.0
    b_bm = 4.0
    T = 2.0
    t_mid = 1.0
    sigma_bm = 0.8

    db_bm = DiffusionBridge(a=a_bm, b=b_bm, T=T, model="brownian", sigma=sigma_bm)
    paths_bm = db_bm.simulate(n_steps=100, n_paths=20_000, seed=1003)

    result.add_check(
        name="BM_Bridge_Start_Pinning",
        theoretical=a_bm,
        empirical=float(np.mean(paths_bm[:, 0])),
        passed=bool(np.allclose(paths_bm[:, 0], a_bm)),
        details=f"All paths start at a={a_bm}",
    )
    result.add_check(
        name="BM_Bridge_End_Pinning",
        theoretical=b_bm,
        empirical=float(np.mean(paths_bm[:, -1])),
        passed=bool(np.allclose(paths_bm[:, -1], b_bm)),
        details=f"All paths terminate at b={b_bm}",
    )

    theo_mid_mean_bm = db_bm.conditional_mean(t_mid)
    theo_mid_var_bm = db_bm.conditional_variance(t_mid)
    mid_bm = paths_bm[:, 50]

    verify_moment(result, "BM_Bridge_Mid_Mean", theoretical=theo_mid_mean_bm, samples=mid_bm)
    verify_moment(result, "BM_Bridge_Mid_Var", theoretical=theo_mid_var_bm, samples=mid_bm)

    # 2. OU Bridge
    a_ou = 2.0
    b_ou = 1.0
    kappa = 1.5
    theta = 1.5
    sigma_ou = 0.6

    db_ou = DiffusionBridge(
        a=a_ou,
        b=b_ou,
        T=T,
        model="ou",
        kappa=kappa,
        theta=theta,
        sigma=sigma_ou,
    )
    paths_ou = db_ou.simulate(n_steps=100, n_paths=20_000, seed=1004)

    result.add_check(
        name="OU_Bridge_Start_Pinning",
        theoretical=a_ou,
        empirical=float(np.mean(paths_ou[:, 0])),
        passed=bool(np.allclose(paths_ou[:, 0], a_ou)),
        details=f"All paths start at a={a_ou}",
    )
    result.add_check(
        name="OU_Bridge_End_Pinning",
        theoretical=b_ou,
        empirical=float(np.mean(paths_ou[:, -1])),
        passed=bool(np.allclose(paths_ou[:, -1], b_ou)),
        details=f"All paths terminate at b={b_ou}",
    )

    theo_mid_mean_ou = db_ou.conditional_mean(t_mid)
    mid_ou = paths_ou[:, 50]
    verify_moment(result, "OU_Bridge_Mid_Mean", theoretical=theo_mid_mean_ou, samples=mid_ou)

    return result


def run_all_fractional_bridge_audits() -> bool:
    audit_funcs = [
        ("Fractional Brownian Motion", audit_fbm_process),
        ("Bessel Bridge", audit_bessel_bridge),
        ("Diffusion Bridge", audit_diffusion_bridge),
    ]

    all_passed = True
    print("\n" + "=" * 88)
    print("AUDIT SUITE: FRACTIONAL & BRIDGE PROCESSES".center(88))
    print("=" * 88 + "\n")

    for name, fn in audit_funcs:
        res = fn()
        print(res.summary())
        print()
        if not res.passed:
            all_passed = False

    return all_passed


if __name__ == "__main__":
    success = run_all_fractional_bridge_audits()
    if not success:
        raise SystemExit(1)
