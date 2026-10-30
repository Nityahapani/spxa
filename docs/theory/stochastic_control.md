# Stochastic Optimal Control

Many problems in quantitative finance are stochastic control problems: choosing a portfolio, liquidating a position, or quoting a two-sided market. A controller steers an SDE to optimise an expected criterion. `spxa.control` implements the dynamic-programming framework, four canonical problems with closed-form solutions, and a monotone finite-difference HJB solver for problems without them.

| Problem | Implementation | Aliases |
| :--- | :--- | :--- |
| Abstract problem and Monte Carlo evaluation | `ControlProblem`, `ControlledTrajectory` | |
| Linear–quadratic–Gaussian regulator | `LinearQuadraticGaussian` | `LQG`, `LinearQuadraticRegulator` |
| Merton consumption–investment | `MertonPortfolioControl` | `MertonPortfolio`, `MertonControl`, `OptimalConsumptionInvestment` |
| Almgren–Chriss optimal execution | `AlmgrenChrissExecution` | `AlmgrenChriss`, `AlmgrenChrissModel`, `OptimalExecution` |
| Avellaneda–Stoikov market making | `AvellanedaStoikovMarketMaker` | `AvellanedaStoikov`, `OptimalMarketMaking`, `HighFrequencyMarketMaker` |
| 1D HJB viscosity solver | `FiniteDifferenceHJB` | `HJBSolver`, `HamiltonJacobiBellman`, `ViscosityHJBSolver` |

---

## 1. Dynamic programming and the HJB equation

A controlled diffusion $dX_s = \mu(s,X_s,u_s)\,ds + \sigma(s,X_s,u_s)\,dW_s$ with admissible controls $u_s\in U$ has value function

$$
V(t,x) = \inf_{u}\ \mathbb{E}\Big[\int_t^T e^{-\rho(s-t)} f(s,X_s,u_s)\,ds + e^{-\rho(T-t)}g(X_T)\ \Big|\ X_t = x\Big].
$$

Bellman's principle of optimality gives the **Hamilton–Jacobi–Bellman equation**

$$
\partial_t V - \rho V + \inf_{u\in U}\Big\{\mu(t,x,u)\,\partial_x V + \tfrac12\sigma^2(t,x,u)\,\partial_{xx}V + f(t,x,u)\Big\} = 0,\qquad V(T,\cdot) = g .
$$

For maximisation problems, replace $\inf$ by $\sup$. **Verification theorem:** if a smooth solution exists and the pointwise minimiser $u^*(t,x)$ gives an admissible feedback control, then $V$ is the value function and $u^*$ is optimal. Without classical solutions, $V$ is the unique **viscosity solution** under standard comparison conditions (Crandall–Lions; Fleming & Soner 2006).

Each `ControlProblem` exposes `drift`, `diffusion`, `running_cost`, `terminal_cost`, `optimal_control` and `value_function`. `simulate(x0, num_steps, num_paths, seed)` runs an Euler–Maruyama Monte Carlo under the feedback policy $u^*(t,X_t)$ and returns a `ControlledTrajectory` holding states, controls and realised discounted payoffs. Comparing `mean_cost` with `value_function(0, x0)` checks the closed form against simulation.

---

## 2. Linear–quadratic–Gaussian control

Consider linear dynamics $dX = (AX + Bu)\,dt + C\,dW$ with quadratic cost

$$
J(u) = \mathbb{E}\Big[\int_0^T (X^\top QX + u^\top Ru)\,dt + X_T^\top M X_T\Big], \qquad Q\succeq0,\ R\succ0,\ M\succeq0 .
$$

Substituting the ansatz $V(t,x) = x^\top P(t)x + s(t)$ into the HJB equation gives the backward **matrix Riccati equation**

$$
-\dot P = PA + A^\top P - PBR^{-1}B^\top P + Q,\qquad P(T) = M,
$$

the optimal linear feedback $u^*(t,x) = -R^{-1}B^\top P(t)\,x$, and $s(t) = \int_t^T \operatorname{tr}(CC^\top P(\tau))\,d\tau$. The optimal gain does not depend on the noise $C$ (certainty equivalence). Noise only adds the constant $s(t)$. `LinearQuadraticGaussian` integrates the Riccati ODE backwards. `steady_state_riccati` solves the continuous algebraic Riccati equation $PA + A^\top P - PBR^{-1}B^\top P + Q = 0$, which is the $T\to\infty$ limit when $(A,B)$ is stabilisable and $(A,Q^{1/2})$ is detectable.

---

## 3. Merton's portfolio problem

An agent with wealth $W$ invests $\pi_t$ in a risky asset ($\mu$, $\sigma$), keeps the remainder at the risk-free rate $r$, and consumes at rate $c_t$:

$$
dW_t = \big[rW_t + \pi_t(\mu - r) - c_t\big]dt + \pi_t\sigma\,dB_t .
$$

The agent maximises CRRA utility $U(c) = c^{1-\gamma}/(1-\gamma)$ ($\gamma>0$, $\gamma\neq1$) with discount rate $\rho$ and bequest weight $\varepsilon$:

$$
\sup_{\pi,c}\ \mathbb{E}\Big[\int_t^T e^{-\rho(s-t)}\frac{c_s^{1-\gamma}}{1-\gamma}ds + \varepsilon\,e^{-\rho(T-t)}\frac{W_T^{1-\gamma}}{1-\gamma}\Big].
$$

Homotheticity reduces the HJB equation to an ODE. Its solution (Merton 1969, 1971) is

$$
\frac{\pi^*_t}{W_t} = \alpha^* = \frac{\mu - r}{\gamma\sigma^2},
\qquad
\frac{c^*_t}{W_t} = \kappa(t) = \frac{\nu}{1 + (\varepsilon^{1/\gamma}\nu - 1)\,e^{-\nu(T-t)}},
\qquad
V(t,W) = \kappa(t)^{-\gamma}\,\frac{W^{1-\gamma}}{1-\gamma},
$$

with

$$
\nu = \frac{\rho - (1-\gamma)\Big(r + \frac{(\mu-r)^2}{2\gamma\sigma^2}\Big)}{\gamma}.
$$

The Merton fraction $\alpha^*$ is constant in both time and wealth. As $T\to\infty$, the consumption rate $\kappa(t)$ tends to $\nu$ (`infinite_horizon_consumption_propensity`). Well-posedness requires $\nu > 0$. `MertonPortfolioControl.simulate` advances log-wealth (a log-Euler step), which keeps wealth strictly positive.

---

## 4. Almgren–Chriss optimal execution

The trader liquidates $Q_0$ shares over $[0,T]$ at trading rate $v_t = \dot q_t$. Permanent impact moves the price by $\gamma\,v_t\,dt$ and temporary impact makes the execution price $\tilde S_t = S_t + \eta v_t$, with $dS_t = \gamma v_t\,dt + \sigma\,dW_t$. Minimising the mean–variance criterion $\mathbb{E}[\text{cost}] + \lambda\operatorname{Var}[\text{cost}]$ over deterministic schedules leads to the Euler–Lagrange equation $\ddot q = \kappa^2 q$, where

$$
\kappa = \sqrt{\frac{\lambda\sigma^2}{\eta}}
$$

is the **urgency**. The solution is

$$
q^*(t) = Q_0\frac{\sinh(\kappa(T-t))}{\sinh(\kappa T)},\qquad
v^*(t) = -Q_0\kappa\frac{\cosh(\kappa(T-t))}{\sinh(\kappa T)}.
$$

The expected cost and its variance are

$$
\mathbb{E}[C] = \tfrac12\gamma Q_0^2 + \tfrac12\eta Q_0^2\kappa\,\frac{\sinh(\kappa T)\cosh(\kappa T) + \kappa T}{\sinh^2(\kappa T)},
\qquad
\operatorname{Var}[C] = \sigma^2 Q_0^2\,\frac{\sinh(2\kappa T) - 2\kappa T}{4\kappa\sinh^2(\kappa T)}.
$$

As $\lambda\to0$, $q^*$ becomes linear (TWAP, `is_twap`). Sweeping $\lambda$ traces the **efficient frontier** of expected cost against variance (`efficient_frontier`).

---

## 5. Avellaneda–Stoikov market making

A dealer quotes bid and ask prices $s - \delta^b$ and $s + \delta^a$ around a mid-price $dS = \sigma\,dW$. Orders arrive as Poisson processes with intensity $\lambda(\delta) = A e^{-k\delta}$. The dealer has CARA utility with risk aversion $\gamma$ over terminal wealth and inventory $q$. Avellaneda & Stoikov (2008) derive the approximate solution

$$
r(s,q,t) = s - q\,\gamma\sigma^2(T-t)
\qquad\text{(reservation price)},
$$

$$
\delta^a + \delta^b = \gamma\sigma^2(T-t) + \frac2\gamma\ln\Big(1 + \frac\gamma k\Big)
\qquad\text{(total spread)},
$$

$$
\delta^{a} = \tfrac12(\delta^a+\delta^b) - q\,\gamma\sigma^2(T-t),\qquad
\delta^{b} = \tfrac12(\delta^a+\delta^b) + q\,\gamma\sigma^2(T-t).
$$

The quotes are centred on the reservation price, not the mid-price. A long inventory therefore shifts both quotes down, encouraging sells and discouraging buys. The spread does not depend on inventory. `optimal_quotes` additionally refuses to quote on the side that would exceed `max_inventory`. Over a step $\Delta t$ the fill probabilities are $1 - e^{-\lambda(\delta)\Delta t}$ (`fill_probabilities`). `simulate` runs the joint mid-price, fill and inventory dynamics.

---

## 6. Finite-difference HJB solver

`FiniteDifferenceHJB` solves one-dimensional problems of the form in §1 on a grid $x_i = x_{\min} + i\,\Delta x$, $t_n = n\,\Delta t$, marching backwards from $V^N = g$:

$$
V^{n}_i = V^{n+1}_i + \Delta t\Big(\operatorname*{opt}_{u\in\mathcal{U}_h}\big\{\mu\,D^{\pm}V^{n+1}_i + \tfrac12\sigma^2 D^2 V^{n+1}_i + f\big\} - \rho\,V^{n+1}_i\Big),
$$

where:

* $D^{+}$ (forward difference) is used where $\mu\ge0$ and $D^{-}$ (backward difference) where $\mu<0$ (upwinding);
* $D^2$ is the central second difference;
* $\mathcal{U}_h$ is a finite set of candidate controls (`control_candidates`), searched exhaustively at every node.

The scheme is explicit and **monotone** whenever the CFL-type condition

$$
\Delta t\left(\frac{|\mu|}{\Delta x} + \frac{\sigma^2}{\Delta x^2} + \rho\right) \le 1
$$

holds over all grid points and controls. Monotonicity, stability and consistency together imply convergence to the viscosity solution (Barles & Souganidis 1991). **The solver does not enforce this condition.** Choose `n_time` large enough, or the scheme can oscillate and diverge. Boundary nodes use Neumann (zero-gradient) or Dirichlet conditions. `value(t, x)` and `optimal_control(t, x)` interpolate the solution off-grid. For a `ControlProblem` instance, `solve_problem(problem)` reads the drift, diffusion and costs directly from the problem. The closed-form problems above serve as regression tests for the numerical solver.

---

## References

* Almgren, R. & Chriss, N. (2000). Optimal execution of portfolio transactions. *Journal of Risk*, 3(2), 5–39.
* Avellaneda, M. & Stoikov, S. (2008). High-frequency trading in a limit order book. *Quantitative Finance*, 8(3), 217–224.
* Barles, G. & Souganidis, P. E. (1991). Convergence of approximation schemes for fully nonlinear second order equations. *Asymptotic Analysis*, 4(3), 271–283.
* Fleming, W. H. & Soner, H. M. (2006). *Controlled Markov Processes and Viscosity Solutions* (2nd ed.). Springer.
* Guéant, O., Lehalle, C.-A. & Fernandez-Tapia, J. (2013). Dealing with the inventory risk: a solution to the market making problem. *Mathematics and Financial Economics*, 7(4), 477–507.
* Kushner, H. J. & Dupuis, P. (2001). *Numerical Methods for Stochastic Control Problems in Continuous Time*. Springer.
* Merton, R. C. (1969). Lifetime portfolio selection under uncertainty: the continuous-time case. *Review of Economics and Statistics*, 51(3), 247–257.
* Merton, R. C. (1971). Optimum consumption and portfolio rules in a continuous-time model. *Journal of Economic Theory*, 3(4), 373–413.
* Pham, H. (2009). *Continuous-time Stochastic Control and Optimization with Financial Applications*. Springer.
