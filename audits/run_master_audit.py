r"""Master Audit Runner for SPXA Mathematical Zoo.

Executes all 8 audit suites across all 69 stochastic processes:
1. Lévy Processes (Brownian, Poisson, Gamma, IG, VG, NIG, Merton, Kou, CGMY, Meixner, GH, Stable, Tempered, NTS, NegBin)
2. Diffusions (OU, Vasicek, CIR, Bessel, CEV, Jacobi, Wright-Fisher, Kesten, Logistic, Quadratic, Hyperbolic)
3. Stochastic Volatility (Heston, SABR, 3/2, GARCH, Bates, BNS, Schöbel-Zhu, Double Heston, CEV-Heston, Heston-Nandi, Rough Heston)
4. Rates, Credit & Commodity (Hull-White, G2++, 2F-CIR, Shifted CIR, Chen, Black-Karasinski, Cheyette, Duffie-Singleton, Merton Firm, Brennan-Schwartz)
5. Jumps & Point Processes (Hawkes 1D & Multi, MMPP, JCIR, Jump-OU, Kou-SV, CBI, Telegraph, VGSA)
6. Fractional & Bridge Processes (Fractional Brownian Motion, Bessel Bridge, Diffusion Bridges)
7. Commodities & Regime Switching (Schwartz 1-Factor, Schwartz 2-Factor, RS-GBM, Bilinear Diffusion)
8. Multivariate & Exotic Diffusions (Multivariate BM, Vector OU, OU-Lévy, Quadratic Rough Heston)
"""

import sys
import time

from audits.commodities_regimes.audit_commodities_regimes import (
    run_all_commodities_regimes_audits,
)
from audits.diffusions.audit_cev_jacobi import (
    audit_cev_process,
    audit_jacobi_process,
    audit_wright_fisher,
)
from audits.diffusions.audit_cir_bessel import (
    audit_bessel_process,
    audit_cir_process,
    audit_squared_bessel,
)
from audits.diffusions.audit_kesten_logistic_quadratic import (
    audit_hyperbolic_diffusion,
    audit_kesten_process,
    audit_logistic_diffusion,
    audit_quadratic_diffusion,
)
from audits.diffusions.audit_ou_vasicek import audit_ornstein_uhlenbeck
from audits.fractional_bridges.audit_fractional_bridges import (
    run_all_fractional_bridge_audits,
)
from audits.jumps_point_processes.audit_jumps_point_processes import (
    run_all_jump_and_point_audits,
)
from audits.levy.audit_brownian import audit_brownian_motion
from audits.levy.audit_cgmy_meixner import audit_cgmy_process, audit_meixner_process
from audits.levy.audit_gamma_ig import (
    audit_gamma_process,
    audit_inverse_gaussian,
)
from audits.levy.audit_gh_stable_tempered import (
    audit_alpha_stable,
    audit_generalized_hyperbolic,
    audit_negative_binomial,
    audit_nts_process,
    audit_tempered_stable,
)
from audits.levy.audit_merton_kou import (
    audit_kou_jump_diffusion,
    audit_merton_jump_diffusion,
)
from audits.levy.audit_poisson_compound import (
    audit_compound_poisson,
    audit_poisson_process,
)
from audits.levy.audit_vg_nig import audit_nig_process, audit_variance_gamma
from audits.multivariate_exotics.audit_multivariate_exotics import (
    run_all_multivariate_exotics_audits,
)
from audits.rates_credit.audit_rates_credit import run_all_rates_credit_audits
from audits.stochastic_volatility.audit_bates_bns import (
    audit_bates_process,
    audit_bns_process,
)
from audits.stochastic_volatility.audit_cev_heston_rough_nandi import (
    audit_cev_heston_process,
    audit_heston_nandi_process,
    audit_rough_heston_process,
)
from audits.stochastic_volatility.audit_heston import audit_heston_process
from audits.stochastic_volatility.audit_sabr_three_halves import (
    audit_garch_diffusion,
    audit_sabr_process,
    audit_three_halves_process,
)
from audits.stochastic_volatility.audit_schobel_zhu_double_heston import (
    audit_double_heston_process,
    audit_schobel_zhu_process,
)


def run_master_audit() -> bool:
    start_time = time.time()
    print("\n" + "=" * 90)
    print("SPXA MATHEMATICAL AUDIT: EXHAUSTIVE VERIFICATION OF ALL STOCHASTIC PROCESSES".center(90))
    print("=" * 90 + "\n")

    suites_passed = 0
    total_suites = 8

    # 1. Lévy Processes
    print("[1/8] Executing Lévy Process Audits...")
    levy_funcs = [
        audit_brownian_motion,
        audit_poisson_process,
        audit_compound_poisson,
        audit_gamma_process,
        audit_inverse_gaussian,
        audit_variance_gamma,
        audit_nig_process,
        audit_merton_jump_diffusion,
        audit_kou_jump_diffusion,
        audit_cgmy_process,
        audit_meixner_process,
        audit_generalized_hyperbolic,
        audit_tempered_stable,
        audit_nts_process,
        audit_negative_binomial,
        audit_alpha_stable,
    ]
    levy_all_pass = True
    for fn in levy_funcs:
        res = fn()
        if not res.passed:
            print(f"FAILED: {res.process_name}")
            levy_all_pass = False
    if levy_all_pass:
        print(" -> All 16 Lévy process models PASSED.\n")
        suites_passed += 1
    else:
        print(" -> Lévy suite FAILED.\n")

    # 2. Continuous Diffusions
    print("[2/8] Executing Diffusion Audits...")
    diff_funcs = [
        audit_ornstein_uhlenbeck,
        audit_cir_process,
        audit_squared_bessel,
        audit_bessel_process,
        audit_cev_process,
        audit_jacobi_process,
        audit_wright_fisher,
        audit_kesten_process,
        audit_logistic_diffusion,
        audit_quadratic_diffusion,
        audit_hyperbolic_diffusion,
    ]
    diff_all_pass = True
    for fn in diff_funcs:
        res = fn()
        if not res.passed:
            print(f"FAILED: {res.process_name}")
            diff_all_pass = False
    if diff_all_pass:
        print(" -> All 11 Diffusion models PASSED.\n")
        suites_passed += 1
    else:
        print(" -> Diffusion suite FAILED.\n")

    # 3. Stochastic Volatility
    print("[3/8] Executing Stochastic Volatility Audits...")
    sv_funcs = [
        audit_heston_process,
        audit_sabr_process,
        audit_three_halves_process,
        audit_garch_diffusion,
        audit_bates_process,
        audit_bns_process,
        audit_schobel_zhu_process,
        audit_double_heston_process,
        audit_cev_heston_process,
        audit_heston_nandi_process,
        audit_rough_heston_process,
    ]
    sv_all_pass = True
    for fn in sv_funcs:
        res = fn()
        if not res.passed:
            print(f"FAILED: {res.process_name}")
            sv_all_pass = False
    if sv_all_pass:
        print(" -> All 11 Stochastic Volatility models PASSED.\n")
        suites_passed += 1
    else:
        print(" -> Stochastic Volatility suite FAILED.\n")

    # 4. Rates & Credit
    print("[4/8] Executing Rates, Credit & Commodity Audits...")
    rc_pass = run_all_rates_credit_audits()
    if rc_pass:
        print(" -> All 10 Rates, Credit & Commodity models PASSED.\n")
        suites_passed += 1
    else:
        print(" -> Rates & Credit suite FAILED.\n")

    # 5. Jumps & Point Processes
    print("[5/8] Executing Jumps & Point Process Audits...")
    jump_pass = run_all_jump_and_point_audits()
    if jump_pass:
        print(" -> All 9 Jump & Point Process models PASSED.\n")
        suites_passed += 1
    else:
        print(" -> Jump & Point Process suite FAILED.\n")

    # 6. Fractional & Bridge Processes
    print("[6/8] Executing Fractional & Bridge Audits...")
    fb_pass = run_all_fractional_bridge_audits()
    if fb_pass:
        print(" -> All 3 Fractional & Bridge models PASSED.\n")
        suites_passed += 1
    else:
        print(" -> Fractional & Bridge suite FAILED.\n")

    # 7. Commodities & Regime Switching
    print("[7/8] Executing Commodities & Regime Switching Audits...")
    cr_pass = run_all_commodities_regimes_audits()
    if cr_pass:
        print(" -> All 4 Commodity & Regime models PASSED.\n")
        suites_passed += 1
    else:
        print(" -> Commodity & Regime suite FAILED.\n")

    # 8. Multivariate & Exotic Diffusions
    print("[8/8] Executing Multivariate & Exotic Diffusions Audits...")
    me_pass = run_all_multivariate_exotics_audits()
    if me_pass:
        print(" -> All 4 Multivariate & Exotic models PASSED.\n")
        suites_passed += 1
    else:
        print(" -> Multivariate & Exotic suite FAILED.\n")

    elapsed = time.time() - start_time
    print("=" * 90)
    print(
        f"AUDIT SUMMARY: {suites_passed}/{total_suites} SUITES PASSED in {elapsed:.2f}s".center(90)
    )
    print("=" * 90 + "\n")

    return suites_passed == total_suites


if __name__ == "__main__":
    success = run_master_audit()
    if not success:
        sys.exit(1)
