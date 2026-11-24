r"""
spxa.compile — compile a process expression into fused numerical kernels.

A composition such as ``(BrownianMotion + CompoundPoisson) @ InverseGaussian
+ 0.5 * VarianceGamma`` is lowered to one expression DAG for its
characteristic exponent :math:`\psi(u;\theta)` over a flat parameter
vector :math:`\theta`. The DAG is simplified, differentiated symbolically and
emitted as straight-line code for NumPy, Numba or JAX.

Usage
-----
``process.compile(backend="numpy" | "numba" | "jax", free=...)`` returns a
:class:`CompiledProcess` with ``char_exponent``, ``char_exponent_and_grad``,
``char_exponent_hessian``, ``cumulants`` and ``cumulants_and_jacobian``, for
one parameter vector or a batch.

Modules
-------
ir
    Hash-consed expression DAG with constant folding and simplification.
diff
    Symbolic derivatives in the parameters and in u, sharing nodes with the value.
evaluate
    NumPy reference interpreter and Taylor-mode interpreter (exact cumulants).
rules
    Lowering rules: zoo Lévy families to IR exponents (``"<Family>.<param>"`` names).
lower
    A process expression (``+``, scaling, ``@``) to one exponent over a flat parameter vector.
codegen
    Straight-line kernels for NumPy (bit-exact with ``evaluate``), Numba and JAX.
process
    :class:`CompiledProcess`, the user-facing object.
kernel
    Compiled psi, gradient and Hessian over free parameters, batched and
    specialised on parameter branches.
"""

from __future__ import annotations

from spxa.compile.codegen import BackendUnavailable, BackendUnsupported, Program, generate
from spxa.compile.ir import CompileError, Expr
from spxa.compile.kernel import ExponentKernel
from spxa.compile.lower import Lowered, lower
from spxa.compile.process import CompiledProcess, compile_process
from spxa.compile.rules import supported_families

__all__ = [
    "CompiledProcess",
    "compile_process",
    "ExponentKernel",
    "lower",
    "Lowered",
    "generate",
    "Program",
    "supported_families",
    "CompileError",
    "BackendUnavailable",
    "BackendUnsupported",
    "Expr",
]
