# Process discovery

`spxa.discovery` solves an inverse problem in stochastic algebra. Given data (a return series, its summary statistics, or option prices), it answers the question:

> Which composition of primitive processes explains the data, with which parameters, and how sure can we be?

It searches over *structures* (the algebraic expression built from primitives with `+`, scaling and `@`), not only over parameters. In this respect it is symbolic regression for Lévy processes.

```python
from spxa.discovery import discover

report = discover("returns.csv")          # or an array, a ReturnSeries, or summary statistics
print(report)                             # headline, ranked table, warnings
report.uncertainty().table()              # standard errors of the winner
report.compare(0, 1)                      # Vuong test, winner vs runner-up
```

The same search runs from the shell: `spxa-discover returns returns.csv --families BM,VG,NIG,Gamma`. [`examples/08_process_discovery.py`](https://github.com/Nityahapani/spxa/blob/main/examples/08_process_discovery.py) walks through all of it.

## 1. The alphabet: primitive families

Each primitive is a parametric Lévy family described by its cumulant exponent

$$
K(z) = \log E\big[e^{zX_1}\big], \qquad \operatorname{Re} z \in (z_-, z_+),
$$

on the strip of finite exponential moments. Every family supplies $K$ in closed form, its cumulants $\kappa_n = K^{(n)}(0)$ to any order, the strip $(z_-, z_+)$, and a bijection from its parameters to $\mathbb R^d$ for unconstrained optimisation.

| Name | Family | Parameters |
| :--- | :--- | :--- |
| `Drift` | deterministic drift | $\mu$ |
| `BM` | Brownian motion | $\sigma$ |
| `Gamma` | gamma subordinator | $a, b$ |
| `IG` | inverse Gaussian subordinator | $\mu, \lambda$ |
| `VG` | variance gamma (Madan, Carr & Chang 1998) | $\sigma, \nu, \theta$ |
| `NIG` | normal inverse Gaussian (Barndorff-Nielsen 1997) | $\alpha, \beta, \delta$ |
| `Meixner` | Meixner (Schoutens & Teugels 1998) | $\alpha, \beta, \delta$ |
| `Poisson` | Poisson process with fixed jump size | $\lambda, h$ |
| `NormalJumps` | compound Poisson, normal jumps (Merton 1976) | $\lambda, m, s$ |
| `DoubleExpJumps` | compound Poisson, double-exponential jumps (Kou 2002) | $\lambda, p, \eta_1, \eta_2$ |
| `CGMY` | tempered stable (Carr, Geman, Madan & Yor 2002) | $C, G, M, Y$ |

Families have no location parameter. Location is a single global `Drift` term appended to every candidate. Otherwise `VG + NIG` would have two interchangeable drifts and would not be identifiable.

## 2. The grammar: which structures are searched

A structure is a tree over three operations:

- $X + Y$, an independent sum: $K = K_X + K_Y$.
- $cX$, scaling by a fixed constant: $K(z) = K_X(cz)$.
- $X \circ T$, subordination by an independent subordinator: $K = K_T \circ K_X$ (Sato 1999, Thm 30.1).

A `Grammar` lists *atoms*: families, signed subordinators such as `-Gamma` (negative jumps only), and optional subordinated structures such as `(Drift + BM) @ (Gamma + Poisson)`. Candidates are sums of at most `max_terms` atoms.

The grammar also removes candidates whose law duplicates another candidate's:

- `BM + BM` is `BM`.
- `(Drift + BM) @ Gamma` is `VG`, `(Drift + BM) @ IG` is `NIG`, and `(Drift + BM) @ Poisson` is `NormalJumps`. Only the family form is generated.
- Sums are commutative and are put in canonical order, so each multiset of atoms is fitted once.

**Search.** `search="beam"` (the default) is forward selection with a beam. It fits every atom, then repeatedly extends the `beam_width` best structures by one atom each, warm-starting each extension from its parent's estimate. A structure reached from several parents is fitted from each parent's estimate, and the best optimum is kept, so the ranking does not depend on which parent arrives first. `search="exhaustive"` fits every structure. With a beam as wide as the atom list, both visit every structure and agree on the best BIC; this is tested.

## 3. Fitting one structure

### From a return sample: a smoothed binned likelihood

At a sampling interval of a day or a week, many pure-jump laws have unbounded densities (VG with $\Delta t/\nu < 1/2$) or atoms (compound Poisson). A pointwise density likelihood is then unreliable. Instead, the increments are binned on bins fixed once per data set, finer in the body and merged in the tails. Data and model are both smoothed by the same narrow Gaussian kernel, and the model's bin probabilities come from a COS expansion of the smoothed law (Fang & Oosterlee 2008):

$$
\ell(\theta) = \sum_j \tilde c_j \log \tilde p_j(\theta).
$$

Because the bins and kernel are shared, log-likelihoods and therefore AIC and BIC are comparable across structures. Goodness of fit is a Wald-type statistic on the smoothed counts that accounts for their covariance. The report quotes its p-value and the largest gap between the model and empirical CDFs.

**The COS interval follows the model.** A cosine series on $[a, b]$ represents the even periodic extension of the law, so mass outside the interval is not dropped but folded back inside it. If $[a, b]$ were fixed by the data, a model could gain likelihood from mass far from every observation. This happened: on 3000 daily returns of a Merton jump diffusion (jump mean $-2\%$), the maximiser on a fixed interval put the jump mean at $-85\%$. Its jumps left $[a, b] \approx [-0.44, 0.43]$, their reflections landed near zero, and it scored 3 log-likelihood units above the true parameters. With an honest expansion it scores 8700 units below them.

Each model's bin probabilities are therefore computed on an interval that contains its own range, $\kappa_1 \pm 12\sqrt{\kappa_2 + h^2 + \sqrt{\kappa_4}}$ for one step of $X_{\Delta t} + hZ$. This is Fang & Oosterlee's rule with their constant doubled.

- If the range lies within $[a, b]$, the precomputed grid is used.
- Otherwise each end moves out in whole steps of $(b-a)/4$, and the number of terms grows in proportion, which keeps the frequency cut-off at $6/h$.
- A law needing more than 12 times the data's interval gets the optimiser's penalty value. Such laws lie far from the data; generic starting points need at most 7.5.

Because the ends are quantised, the grid is piecewise constant in the parameters. Exact derivatives are those of a fixed expansion, and where the grid switches the two expansions agree to $3\times10^{-14}$. In 72 fits of `BM + NormalJumps` (24 Merton samples, 3 seeds each), the fixed interval sent the jump mean beyond $\pm 0.2$ in 43 fits, with values from $-0.85$ to $+3.2$. The adaptive interval did so in none. A binned search costs about twice as much as before.

**What the binned likelihood can see.** Information beyond the outermost bins enters only through the tail-bin counts. A component that acts mainly through a few very large jumps is therefore hard to identify from a few thousand returns. One example is a gamma subordinator with low activity $a$. The search then prefers the simpler structure, which is the correct inference from that sample. The bootstrap frequencies (§5) make this visible.

### From summary statistics: matching cumulants

`from_statistics(mean, variance, skewness, excess_kurtosis, dt, n)` gives per-unit-time cumulants $\hat\kappa_1, \dots, \hat\kappa_4$, and the fit minimises a weighted distance to the model's cumulants. Four cumulants identify at most three shape parameters, so several structures typically match exactly. The report says so explicitly and ranks the matching structures by simplicity. A return sample is needed to tell them apart.

The *moment errors* (relative errors in variance, skewness and excess kurtosis over $\Delta t$) are a diagnostic, not a ranking criterion. With heavy tails, the sample skewness and kurtosis are dominated by a handful of observations. A correct likelihood fit can miss them by tens of percent while every cumulant stays within one or two standard errors (`FitResult.cumulant_z_scores`). A regression test pins this case.

### From option prices: risk-neutral structures

`cos_prices(model, spot, strikes, maturity, rate, dividend, call)` prices European options with the model as the driver of log-returns,

$$
S_T = S_0 \exp\big((r - q)T + \omega T + X_T\big), \qquad \omega = -K_X(1),
$$

where $\omega$ is the martingale correction. It requires $1 < z_+$, i.e. $E[e^{X_1}] < \infty$, and raises an error otherwise. Prices come from the COS expansion of $\log(S_T/S_0)$, with puts priced directly and calls from put–call parity. Against the Fang–Oosterlee VG reference values (with $S_0 = 100$, $K = 90$, $r = 0.1$, $\sigma = 0.12$, $\nu = 0.2$ and $\theta = -0.14$), it gives 10.993703187 at $T = 0.1$ and 19.099354724 at $T = 1$. Both values are tested.

`discover_from_options(quotes)` fits each structure by least squares on vega-weighted price errors, which to first order are implied-volatility errors (Cont & Tankov 2004, §13.2). It reports the implied-volatility RMSE overall and per maturity. A Lévy model fixes how the smile scales with maturity, roughly as $T^{-1/2}$. When the best structure's errors depend strongly on maturity, the report warns that the quotes do not follow a single Lévy law.

### Exact gradients

All three objectives are differentiated exactly (`spxa.graph.autodiff.discovery`). Each structure is evaluated in its free coordinates with forward-mode jets, so the derivatives pass through the parameter bijections, the closed-form exponents and the composition rules:

- the binned likelihood, through the COS bin probabilities, their clip, running maximum and floor;
- the cumulant distance, through the Taylor coefficients of $K$, including the profiled drift;
- the option residuals, through the COS prices, including the dependence of the truncation interval on the cumulants.

`fit_structure(..., gradient="exact")` (L-BFGS-B) and `fit_options(..., jac="exact")` (least squares) are the defaults. `"2-point"` restores SciPy's differences. On the README example both settings recover the same model. On option quotes generated by CGMY, the exact Jacobian reaches an objective of $2\times10^{-15}$, against $3\times10^{-9}$ with differences, which stall as $Y \to 0$. Each Jacobian costs about $n + 1$ price evaluations, so fits of cheap models take roughly twice as long. [Exact derivatives](exact_derivatives.md) describes the machinery.

## 4. Ranking structures

With a likelihood, candidates are ranked by BIC (or AIC). The weights

$$
w_i \propto e^{-\Delta_i/2}, \qquad \Delta_i = \mathrm{IC}_i - \min_k \mathrm{IC}_k,
$$

give the relative support of each structure (Burnham & Anderson 2002). The headline reports the winner, its CDF gap, its goodness-of-fit p-value and its lead over the runner-up:

```
<expression> matches your data within <x>% (largest gap between model and empirical CDFs;
goodness-of-fit p = <p>); BIC lead over the runner-up <Δ>
```

Subordinators are shown as weighted unit-rate processes: $c \cdot \Gamma(a, 1)$ has the law of $\Gamma(a, 1/c)$.

**Near-equivalent structures.** Different structures can have nearly the same law on a sample. On 5000 daily returns of VG + Gamma, the bilateral gamma `-Gamma + Gamma` and `VG + Gamma` are both within reach of the top. The recovery test therefore asserts only that these two are the top pair and that every other structure trails by more than 10 BIC.

**Degenerate fits.** Some fits are limiting cases of their structure, for example `NIG(alpha=9e5, beta≈alpha)` (a one-sided limit) or `BM(sigma=1e-10) + VG` (a summand that has vanished). `FitResult.degeneracies()` flags two signs:

- an unconstrained coordinate beyond $\pm 9$, i.e. a parameter more than $e^9$ times away from 1 or within about $10^{-4}$ of an interval bound;
- a summand carrying less than $10^{-4}$ of the variance.

Tables mark these fits with `!`, a warning names the best admissible structure when a degenerate fit leads, and `exclude_degenerate=True` removes them from the ranking. The boundary check depends on units: parameters are per unit time and returns are in decimals.

## 5. How sure can we be?

**Standard errors.** The smoothed counts are not multinomial, so the inverse Fisher information is not the right covariance. `parameter_uncertainty` uses the sandwich $A^{-1} B A^{-1}$ (Huber 1967; White 1982) with per-observation scores. For cumulant fits, it uses the GMM sandwich with the sampling covariance of the cumulant estimates. The GMM standard errors are asymptotic and optimistic in small samples with heavy tails.

**Comparing two structures.** Non-nested structures are compared with Vuong's (1989) test on the pointwise log-likelihoods, with an optional AIC or BIC correction. `report.compare(i, j)` runs it on the bins the search used.

**Selection stability.** `bootstrap(report, n_boot, top)` resamples the returns and refits the leading structures from their estimates. It records how often each one wins (the bootstrap analogue of an Akaike weight; Buckland, Burnham & Augustin 1997) and percentile intervals for the parameters. `block_length > 1` gives the moving-block bootstrap (Künsch 1989) for short-range dependence.

## 6. Model evidence

BIC is a large-sample stand-in for $-2\log p(D\mid M)$. `laplace_evidence(fit, binned, prior=None)` computes the evidence

$$
p(D\mid M) = \int p(D\mid x, M)\,\pi(x\mid M)\,dx
$$

itself, for a proper prior $\pi$ on the free coordinates $x$ of the structure. It uses the Laplace approximation at the posterior mode $\hat x$:

$$
\log p(D\mid M) \approx \ell(\hat x) + \log\pi(\hat x) + \tfrac k2\log 2\pi - \tfrac12\log\det H, \qquad H = -\nabla^2[\ell+\log\pi](\hat x).
$$

The mode is found by a trust-region Newton search from the maximum-likelihood point. $H$ comes from the exact likelihood Hessian, so the only approximation is Laplace's, with relative error $O(1/n)$ (Tierney & Kadane 1986). With $H \approx n\,I_1$, $-2\log p(D\mid M) = \mathrm{BIC} + O(1)$ (Kass & Raftery 1995). `Evidence.log_occam_factor` $= \log p(D\mid M) - \ell(\hat x)$ is the price the structure pays for its parameters, measured on the actual posterior volume rather than as $\frac k2\log n$.

$\ell$ is the smoothed binned likelihood of the fit. It has no multinomial coefficient, so evidences share an additive constant per set of bins. They compare structures fitted to the same `BinnedData`, which is what `discover` produces.

**Checking Laplace.** `importance_evidence(evidence, binned, n_draws=2000)` gives an unbiased estimate of $p(D\mid M)$. Its proposal is a multivariate $t_5$ at $\hat x$ with scale $1.2^2 H^{-1}$ (Geweke 1989). It reports the delta-method standard error of $\log\hat Z$ and Kish's effective sample size.

**Sequential Monte Carlo.** Importance sampling relies on one proposal. When a posterior is far from Gaussian, only a few percent of its draws carry weight, and then the estimate and its standard error rest on those few draws. `smc_evidence(evidence, binned)` reaches the posterior gradually instead. It starts from the same normalised reference $q$ and moves $N$ particles through the geometric bridge

$$
\gamma_\beta \propto q^{1-\beta}\,(L\pi)^{\beta}, \qquad 0 = \beta_0 < \dots < \beta_T = 1
$$

(Neal 2001; Del Moral, Doucet & Jasra 2006). At each temperature:

- the particles are reweighted by $(L\pi/q)^{\beta_t-\beta_{t-1}}$;
- they are resampled (systematically) when the effective sample size falls below $N/2$;
- they are moved by random-walk Metropolis kernels that leave $\gamma_{\beta_t}$ invariant, with the particles' covariance as the proposal shape and an acceptance-tuned scale.

Because $q$ is normalised, the product of the mean incremental weights estimates $Z$. Temperatures are chosen adaptively, so that the conditional effective sample size stays at 95% of $N$ (Zhou, Johansen & Aston 2016). Particles where the likelihood is zero (laws the binned likelihood rejects) drop out at the first step. The standard error comes from independent runs, four by default; with so few runs it is itself good to only about 40%.

The engine `tempered_smc` was tested on targets with known normalising constants: a correlated Gaussian from an offset reference, the skewed $\log$-Gamma with shape 0.3, and a half-normal that is zero on half the reference. Its estimates match the exact values within their standard errors. Averaged over 60 runs per setting, the log-Gamma estimates lie within 1.2 standard errors (about 0.004) of $\log\Gamma(0.3)$. On 250 weekly VG returns, SMC and importance sampling agree to 0.02 nats, and both put Laplace 0.25 nats low. The weakly identified jump structure of §7 shows the difference between the two:

| estimate of $\log p(D\mid M)$ for `BM + NormalJumps` on 2000 Brownian returns | value |
|---|---|
| Laplace | −6512.888 |
| importance sampling, 4000 draws (ESS 2–8%), over seeds | −6512.25 to −6512.47 |
| importance sampling, 20 000 draws (ESS 2.5–4%), 3 seeds | −6512.43, −6512.38, −6512.37 |
| SMC, 4 runs × 250 particles | −6512.364 ± 0.062 |
| SMC reference, 4 runs × 2000 particles | −6512.362 ± 0.019 |

Laplace is 0.53 nats low here. The cost is about $N(1 + 3T)$ likelihood evaluations per run, with $T \approx 6$–$10$ temperatures here.

**Priors.** `GaussianPrior(mean, sd)` puts independent normals on the free coordinates. `default_prior(structure, std, scale=2)` centres them at the starting member (the data's standard deviation spread over the summands). The standard deviations are `scale` for logarithmic, logit and `arctanh` coordinates, and `scale` $\cdot\,\sigma_1$ for the three untransformed ones: the drift, the VG $\theta$ and the normal jump mean. Here $\sigma_1$ is the data's standard deviation per unit time. Bayes factors depend on the prior widths of parameters that two structures do not share (Lindley 1957). Pass your own prior when you have real information.

**Accuracy.** On Gaussian data with one or two free coordinates, the Laplace value matches adaptive quadrature of the posterior to $6\times10^{-4}$ ($n = 300$) and $8\times10^{-5}$ ($n = 2000$). Importance sampling agrees with quadrature within its standard error. On 250 weekly VG returns, the posterior is skewed and Laplace is 0.2 nats off. Importance sampling detects this (standard error 0.02), and at $n = 2000$ the gap is 0.006.

**When the mode search has converged.** The binned likelihood is piecewise smooth, with kinks of relative size about $10^{-7}$ from the CDF's clip and running maximum. At such points the gradient can stall above a fixed tolerance. Convergence is therefore judged by $\tfrac12 g^\top H^{-1} g$ (`Evidence.mode_gap`, half the squared Newton decrement). To second order, this is how far the log posterior at the reported point lies below its maximum, so it is exactly the error the point adds to $\log p(D\mid M)$. The search counts as converged below $10^{-6}$ nats. For example, BM fitted to jump-diffusion returns stalls with a gradient of $10^{-2}$, yet its gap is $10^{-8}$ nats.

## 7. Posterior over structures

`structure_posterior(report)`, or `report.posterior()`, turns the evidences of the fitted candidates into posterior probabilities,

$$
P(M_i\mid D) = \frac{\pi_i\,p(D\mid M_i)}{\sum_j \pi_j\,p(D\mid M_j)}.
$$

The prior $\pi$ over structures is uniform by default. It can also be a mapping from structure key to weight, or a callable, and it is normalised over the candidates. The probabilities are computed by shifting the log posteriors by their maximum before exponentiating. That shift is exact, whereas $z - \operatorname{logsumexp}(z)$ loses $|z|\,\varepsilon$, about $10^{-12}$ for log evidences near $-6500$. The probabilities are conditional on the candidate set: a structure that was never fitted has probability zero. Candidates without a Laplace approximation (no positive-definite Hessian at the mode) are listed as skipped.

These probabilities differ from BIC weights in two ways. They use each structure's actual posterior volume rather than $\frac k2\log n$. And they are honest about weakly identified extensions. On 2000 daily Brownian returns:

| structure | BIC weight | $P(M\mid D)$, Laplace | $P(M\mid D)$, importance |
|---|---|---|---|
| BM | 0.999 | 0.81 | 0.74 ± 0.01 |
| BM + NormalJumps | 0.000 | 0.19 | 0.26 ± 0.01 |

BIC charges the jump part $\frac32\log n \approx 11.4$ for its three parameters. The evidence charges only the prior volume the data rule out. Rare large jumps are barely constrained by 2000 Gaussian returns, so their prior volume is largely kept. On data with jumps (Merton, $n = 3000$) or VG data, the true structure gets probability above $1 - 10^{-12}$.

**Monte Carlo evidence.** `evidence="importance"` replaces each Laplace value by the importance-sampling estimate, and `evidence="smc"` by the SMC estimate (four runs of `n_draws // 4` particles). Weakly identified structures are where Laplace errs: in the table it is about 0.5 nats low for the jump structure (0.53 against the SMC reference in §6). The estimates' standard errors $s_j$ propagate to the probabilities by the delta method,

$$
\operatorname{se}(P_i)^2 = \sum_j \big(P_i(\delta_{ij} - P_j)\big)^2 s_j^2,
$$

reported as `probability_se`. Over six seeds, the spread of the estimated probability matches the reported error to within 20%.

**Model averaging.** `StructurePosterior.predictive(t, n_draws=0)` is the posterior predictive law of $X_t$. It is the mixture $\sum_i P(M_i\mid D)\,p(x\mid M_i)$ of COS laws, either at the modes or averaged over `n_draws` Laplace draws per structure. Its mean and variance are exact, by the law of total variance. `option_prices(...)` averages European prices in the same way. Each structure is made risk-neutral by its own martingale correction. The reported standard deviation combines parameter uncertainty within structures and disagreement between them. A draw whose prices are not finite is discarded and counted. Examples are an infinite $E[e^{X_1}]$, or a weakly identified jump part drawn with a jump standard deviation of 9.5 in log price, whose correction is about $-5\times10^{18}$. Averaged calls and puts satisfy put–call parity, because every component does.

## 8. Live monitoring

A fitted law is a claim about future increments too. `PageCUSUM(null, alternative, binned)` watches new increments $x_1, x_2, \dots$ for a change from the fitted law $\theta_0$ to a specified alternative $\theta_1$, for example volatility up 30% or jumps appearing. It uses Page's CUSUM (Page 1954),

$$
W_t = \max\big(0,\, W_{t-1} + \lambda(x_t)\big), \qquad
\lambda(x) = \sum_j K_j(x)\,\log\frac{\tilde p_j(\theta_1)}{\tilde p_j(\theta_0)},
$$

and raises an alarm when $W_t \ge h$. Here $\lambda$ is the smoothed binned log-likelihood ratio of §3. A new increment gets the same kernel weights $K_j(x) = \Phi((e_{j+1}-x)/h_K) - \Phi((e_j-x)/h_K)$ as the data (`BinnedData.kernel_weights`). The two laws may have different structures.

**The false-alarm guarantee is exact.** Under the null, $E_0[K_j(X)] = \tilde p_j(\theta_0)$, because the model's smoothed probabilities use the same kernel. The weights are non-negative and sum to one. By the weighted AM–GM inequality,

$$
E_0\big[e^{\lambda(X)}\big] \le E_0\Big[\sum_j K_j(X)\,\frac{\tilde p_j(\theta_1)}{\tilde p_j(\theta_0)}\Big] = \sum_j \tilde p_j(\theta_1) = 1 .
$$

So every product $\prod_{i=k+1}^{t} e^{\lambda(x_i)}$ is a non-negative supermartingale under the null. Ville's inequality bounds the chance that it ever reaches $e^h$ by $e^{-h}$. $W_t \ge h$ exactly when one of these products, started after the last zero of $W$, does. Two guarantees follow:

- the in-control average run length is at least $e^h$ (Lorden 1971), so `arl0=A` sets $h = \log A$;
- the chance of any false alarm within $n$ observations is at most $n e^{-h}$ (`false_alarm_bound(n)`).

Neither bound relies on asymptotics or on the alternative being true. The only assumptions are i.i.d. increments under the null, which is the Lévy hypothesis, and the null parameters treated as known. Parameters estimated from a short history shorten the in-control run length, so refit on enough data first.

**Detection.** After a change, the drift of $W$ is exactly $E_1[\lambda] = \mathrm{KL}(\tilde p(\theta_1)\,\|\,\tilde p(\theta_0))$ (`kl_alternative`), because $\lambda$ is linear in $K$. Before a change the drift is $-\mathrm{KL}(\tilde p(\theta_0)\,\|\,\tilde p(\theta_1))$ (`kl_null`). The expected delay is about $h/\mathrm{KL}$ (`expected_delay`). Page's CUSUM minimises the worst-case delay among detectors with the same false-alarm rate (Lorden 1971; Moustakides 1986). Each `Alarm` also reports `change_point`, the index after the last zero of $W$, where the excursion that crossed began.

On daily returns of a 20% BM, with the alternative of volatility up 30%:

| $h$ | guaranteed ARL$_0 = e^h$ | simulated ARL$_0$ | $h/\mathrm{KL}$ | simulated delay |
|---|---|---|---|---|
| 2 | 7.4 | 150 | 26 | 21 |
| 3 | 20 | 490 | 39 | 34 |
| 4 | 55 | 1420 | 52 | 48 |

The guarantee is conservative, because it holds for every alternative and every increment law with the same bin probabilities. For a target false-alarm level it is the safe choice. When volatility rises from 20% to 30% and $h = \log 10^4$, the median error of the estimated change point is 5 observations against a mean delay of 48.

**Exact run lengths.** `average_run_length(cusum, law)` computes the ARL under any law, following Brook & Evans (1972). $W$ is a Markov chain on $[0, h)$. The law of $\lambda(X)$ comes from the COS density of $X$ over one step. Discretising the state into cells and solving $(I-P)L = \mathbf 1$ gives the expected time to absorption. For the table's detector at $h = 2$, it gives 153.4. Two simulations of 2 million steps gave $153.2 \pm 1.3$ and $152.7 \pm 1.3$; at $h = 3$, the value is 503.2 against $510 \pm 8$ and $495 \pm 7$. After the change, ARL$_1$ grows by $1/\mathrm{KL}$ per unit of $h$, as Wald's identity predicts. The comparison with simulation shows how much the general bound $e^h$ gives away.

**No single alternative.** Often the change of interest is any departure, not a specific one. The Shiryaev–Roberts statistic sums the likelihood ratios over change points instead of maximising over them:

$$
R^{(j)}_t = \big(1 + R^{(j)}_{t-1}\big)\,e^{\lambda_j(x_t)}, \qquad R_t = \sum_j w_j R^{(j)}_t .
$$

It is recursive and mixes freely over alternatives $j$ with weights $w_j$. Each $E_0[e^{\lambda_j}] \le 1$, so $E_0[R_t\mid\mathcal F_{t-1}] \le 1 + R_{t-1}$. $R_t - t$ is then a supermartingale, and optional stopping at the alarm $N$ gives $E_0[N] \ge E_0[R_N] \ge e^h$ for any number of alternatives (Pollak 1987). This bound is nearly tight: with one alternative at $h = 4$, $e^h = 55$ against a simulated ARL$_0$ of $81.8 \pm 0.7$, where CUSUM at the same $h$ runs 1420. For the same guaranteed false-alarm rate, Shiryaev–Roberts therefore detects much sooner. `local_alternatives(null, binned, kl)` supplies the alternatives: $\pm$ each eigenvector of the binned Fisher information, scaled by root finding to divergence exactly `kl` from the null. A volatility rise of 30% on the 20% BM, monitored with the four local alternatives at `kl=0.02` and `arl0=200`, gives an in-control ARL of 220 and an alarm every 60 steps after the change, and 95% of alarms name the volatility-up alternative. `report.monitor()` builds the mixture from a discovery report, or a CUSUM when an alternative is given.

We do not refit the null on the stream. Re-estimating $\theta_0$ from data the detector is judging breaks $E_0[e^\lambda]\le 1$, and with it the guarantee. After an alarm, refit on the data from `change_point` onward.

## 9. When no Lévy composition fits

Lévy increments are i.i.d. `summarize` reports the autocorrelation of returns and of squared returns, with Ljung–Box tests. When volatility clusters, the search still runs, but the report warns that no composition in the grammar can explain dependence. Volatility models (`spxa.vol`) and rough models (`spxa.rough`) are the place to look then.

## 10. What is tested

The tests simulate exact increments and check:

- Brownian motion is not overfitted;
- a Merton jump diffusion is recovered as `BM + NormalJumps`;
- VG beats its neighbours with weight above 0.95;
- a VG + Gamma sum is recovered up to near-equivalence;
- the binned likelihood cannot gain from mass outside the data's range: previously aliased Merton samples recover the true jump mean, and wide-grid probabilities match Monte Carlo where the fixed interval was off by 20 standard deviations;
- posterior probabilities are Bayes' rule on the evidences to $10^{-14}$, a structure prior shifts log odds exactly, and the importance-sampling error matches the spread over seeds;
- model-averaged predictive moments and prices are exact mixtures, and averaged prices satisfy put–call parity;
- the SMC engine recovers known normalising constants (Gaussian, skewed log-Gamma, a target zero on half the reference), and SMC evidence agrees with quadrature and with importance sampling;
- monitoring: $E_0[K(X)] = \tilde p(\theta_0)$ to $10^{-10}$ and $E_0[e^{\lambda}] \le 1$ by quadrature, Ville's bound from a fixed start, run lengths above $e^h$, finite-horizon false alarms below $n e^{-h}$, delays near $h/\mathrm{KL}$, and change points located well inside the delay;
- Brook–Evans run lengths agree with simulated cycle lengths within 4 standard errors; ARL$_1$ grows by $1/\mathrm{KL}$ per unit of $h$; Shiryaev–Roberts dominates CUSUM and its in-control ARL lies between $e^h$ and $2e^h$; local alternatives sit at divergence `kl` to $10^{-8}$; and the mixture names the alternative that changed;
- sandwich standard errors agree with the sampling spread of the estimates;
- the option fitter recovers VG parameters from a 27-quote surface;
- mixed quotes trigger the maturity warning.

## References

- Barndorff-Nielsen, O. E. (1997). Normal inverse Gaussian distributions and stochastic volatility modelling. *Scand. J. Statist.* 24.
- Brook, D. & Evans, D. A. (1972). An approach to the probability distribution of CUSUM run length. *Biometrika* 59.
- Buckland, S. T., Burnham, K. P. & Augustin, N. H. (1997). Model selection: an integral part of inference. *Biometrics* 53.
- Burnham, K. P. & Anderson, D. R. (2002). *Model Selection and Multimodel Inference*, 2nd ed. Springer.
- Carr, P., Geman, H., Madan, D. & Yor, M. (2002). The fine structure of asset returns. *J. Business* 75.
- Cont, R. & Tankov, P. (2004). *Financial Modelling with Jump Processes*. Chapman & Hall.
- Del Moral, P., Doucet, A. & Jasra, A. (2006). Sequential Monte Carlo samplers. *J. R. Statist. Soc. B* 68.
- Geweke, J. (1989). Bayesian inference in econometric models using Monte Carlo integration. *Econometrica* 57.
- Fang, F. & Oosterlee, C. W. (2008). A novel pricing method for European options based on Fourier-cosine series expansions. *SIAM J. Sci. Comput.* 31.
- Hoeting, J. A., Madigan, D., Raftery, A. E. & Volinsky, C. T. (1999). Bayesian model averaging: a tutorial. *Statist. Sci.* 14.
- Huber, P. J. (1967). The behavior of maximum likelihood estimates under nonstandard conditions. *Proc. Fifth Berkeley Symp.* 1.
- Kass, R. E. & Raftery, A. E. (1995). Bayes factors. *J. Amer. Statist. Assoc.* 90.
- Kou, S. G. (2002). A jump-diffusion model for option pricing. *Management Science* 48.
- Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. *Ann. Statist.* 17.
- Lindley, D. V. (1957). A statistical paradox. *Biometrika* 44.
- Lorden, G. (1971). Procedures for reacting to a change in distribution. *Ann. Math. Statist.* 42.
- Madan, D., Carr, P. & Chang, E. (1998). The variance gamma process and option pricing. *European Finance Review* 2.
- Merton, R. C. (1976). Option pricing when underlying stock returns are discontinuous. *J. Financial Economics* 3.
- Moustakides, G. V. (1986). Optimal stopping times for detecting changes in distributions. *Ann. Statist.* 14.
- Neal, R. M. (2001). Annealed importance sampling. *Statist. Comput.* 11.
- Page, E. S. (1954). Continuous inspection schemes. *Biometrika* 41.
- Pollak, M. (1987). Average run lengths of an optimal method of detecting a change in distribution. *Ann. Statist.* 15.
- Roberts, G. O. & Rosenthal, J. S. (2001). Optimal scaling for various Metropolis–Hastings algorithms. *Statist. Sci.* 16.
- Sato, K. (1999). *Lévy Processes and Infinitely Divisible Distributions*. Cambridge.
- Schoutens, W. & Teugels, J. L. (1998). Lévy processes, polynomials and martingales. *Communications in Statistics. Stochastic Models* 14(1–2), 335–349.
- Tierney, L. & Kadane, J. B. (1986). Accurate approximations for posterior moments and marginal densities. *J. Amer. Statist. Assoc.* 81.
- Ville, J. (1939). *Étude critique de la notion de collectif*. Gauthier-Villars.
- Vuong, Q. H. (1989). Likelihood ratio tests for model selection and non-nested hypotheses. *Econometrica* 57.
- White, H. (1982). Maximum likelihood estimation of misspecified models. *Econometrica* 50.
- Zhou, Y., Johansen, A. M. & Aston, J. A. D. (2016). Toward automatic model comparison: an adaptive sequential Monte Carlo approach. *J. Comput. Graph. Statist.* 25.
