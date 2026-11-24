r"""
spxa.compile — compile a process expression into fused numerical kernels.

A composition such as ``(BrownianMotion + CompoundPoisson) @ InverseGaussian
+ 0.5 * VarianceGamma`` is lowered to one expression DAG for its
characteristic exponent :math:`\psi(u;\theta)` over a flat parameter
vector :math:`\theta`. The DAG is simplified, differentiated symbolically and
emitted as straight-line code for NumPy, Numba or JAX.

Modules
-------
ir
    Hash-consed expression DAG with constant folding and simplification.
diff
    Symbolic derivatives in the parameters and in u, sharing nodes with the value.
evaluate
    NumPy reference interpreter and Taylor-mode interpreter (exact cumulants).
"""

from __future__ import annotations

from spxa.compile.ir import CompileError, Expr

__all__ = ["CompileError", "Expr"]
