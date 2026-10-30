# Estimation and Calibration

There are two separate problems here. **Estimation** fits a model to the historical path of an observed (or partly observed) time series. **Calibration** fits a pricing model to today's cross-section of option quotes. `spxa.calibration` covers both. It has exact-likelihood estimators, moment and characteristic-function methods, nonlinear filters for latent states, smile and surface calibrators, and diagnostics that report uncertainty and goodness of fit.

| Object | Implementation |
| :--- | :--- |
| Exact transition-density MLE | `OrnsteinUhlenbeckMLE`, `VasicekMLE`, `BrownianMotionMLE`, `GeometricBrownianMotionMLE`, `CIRMaximumLikelihood`, `BESQMaximumLikelihood` |
| Heavy-tailed innovations | `StudentTInnovationsMLE`, `NormalInverseGaussianMLE` |
| Generalised method of moments | `GMMEstimator`, `OrnsteinUhlenbeckGMM`, `CIRGMM`, `newey_west_hac`, `hansen_j_test` |
| Empirical characteristic function | `ECFEstimator` |
| Latent-state filtering and QMLE | `UnscentedKalmanFilter`, `BootstrapParticleFilter`, `QuasiMaximumLikelihoodEstimator` |
| Model-specific filters | `HestonVarianceFilter`, `SchwartzTwoFactorFilter`, `DefaultIntensityFilter` |
| Smile and surface calibration | `SABRCalibrator`, `HestonCalibrator`, `BatesCalibrator`, `DoubleHestonCalibrator`, `CEVHestonCalibrator` |
| Inference and model selection | `fisher_information_matrix`, `sandwich_covariance_matrix`, `compute_parameter_inference`, `compute_profile_likelihood`, `ModelComparisonTable` |
| Forecast and fit diagnostics | `compute_pit_values`, `evaluate_pit_diagnostics`, `compute_residual_diagnostics`, `analyze_surface_residuals`, `CalibrationReport` |

---

## 1. Maximum likelihood from transition densities

Let $X_0, X_\Delta, \dots, X_{N\Delta}$ be observed on a regular grid. For a Markov diffusion the likelihood factorises into transition densities:

$$
\ell(\theta) = \sum_{k=1}^{N} \log p_\theta\big(X_{k\Delta} \mid X_{(k-1)\Delta}\big).
$$

Whenever $p_\theta$ is known in closed form, the estimators in `spxa.calibration.mle` use it. They do **not** use an Euler (Gaussian) approximation, so the MLE has no discretisation bias at any sampling frequency. Without a closed-form density one would need approximations such as the Aït-Sahalia (2002) expansion, which `spxa` does not currently implement.

| Model | Dynamics | Exact transition |
| :--- | :--- | :--- |
| Brownian motion | $dX = \mu\,dt + \sigma\,dW$ | $\mathcal N(X + \mu\Delta,\ \sigma^2\Delta)$ |
| GBM | $dS = \mu S\,dt + \sigma S\,dW$ | $\log S' \sim \mathcal N\big(\log S + (\mu-\tfrac12\sigma^2)\Delta,\ \sigma^2\Delta\big)$ |
| OU / Vasicek | $dX = \kappa(\theta - X)\,dt + \sigma\,dW$ | $\mathcal N\big(X e^{-\kappa\Delta} + \theta(1-e^{-\kappa\Delta}),\ \tfrac{\sigma^2}{2\kappa}(1-e^{-2\kappa\Delta})\big)$ |
| CIR | $dr = \kappa(\theta - r)\,dt + \sigma\sqrt r\,dW$ | scaled non-central $\chi^2$ (below) |
| BESQ$^\delta$ | $dX = \delta\,dt + 2\sqrt X\,dW$ | $X'/\Delta \sim \chi^2_\delta(X/\Delta)$ |

**CIR.** Set $c = 2\kappa / \big(\sigma^2(1 - e^{-\kappa\Delta})\big)$. Then $2c\,r_{t+\Delta}$ given $r_t$ is non-central $\chi^2$ with $4\kappa\theta/\sigma^2$ degrees of freedom and non-centrality $2c\,r_t e^{-\kappa\Delta}$. Write $u = c\,r_t e^{-\kappa\Delta}$, $v = c\,r_{t+\Delta}$ and $q = 2\kappa\theta/\sigma^2 - 1$. The density is

$$
p(r_{t+\Delta}\mid r_t) = c\, e^{-u-v}\Big(\frac{v}{u}\Big)^{q/2} I_q\big(2\sqrt{uv}\big).
$$

`CIRMaximumLikelihood` evaluates $I_q$ through the exponentially scaled Bessel function `scipy.special.ive`. This keeps $\log I_q(z) = \log \mathrm{ive}_q(z) + z$ finite for large $uv$. The log-likelihood matches `scipy.stats.ncx2.logpdf` (with Jacobian $2c$) to machine precision.

**Heavy tails.** `StudentTInnovationsMLE` fits location, scale and degrees of freedom $\nu$ to i.i.d. returns. `NormalInverseGaussianMLE` fits the four-parameter NIG law, whose density involves $K_1$. Both are i.i.d. return models. For a Lévy *process* observed at frequency $\Delta$, the increment law is the $\Delta$-convolution power, which is the setting of §2.2.

!!! note "Numerical optimisation"
    Every `TimeSeriesEstimator` maximises $\ell$ with L-BFGS-B from `default_initial_guess` within `default_bounds`. It returns an `EstimationResult` with the parameters, standard errors, log-likelihood, AIC/BIC and the number of observations. These can be fed straight into §4.

---

## 2. Moment and characteristic-function methods

### 2.1 GMM

Suppose a model implies $\mathbb E[m(X_t;\theta_0)] = 0$ for an $m$-vector of moment functions, with $m \ge k = \dim\theta$. Let $g_N(\theta) = \frac1N\sum_t m(X_t;\theta)$. `GMMEstimator` implements **two-step feasible GMM** (Hansen 1982):

$$
\hat\theta = \arg\min_\theta\ g_N(\theta)^\top W\, g_N(\theta),
\qquad W^{(1)} = I,\quad W^{(2)} = \hat\Omega^{-1}.
$$

Here $\hat\Omega$ estimates the long-run covariance $\sum_{j\in\mathbb Z}\operatorname{Cov}(m_t, m_{t-j})$ at the first-step estimate. With the efficient weight, $\sqrt N(\hat\theta - \theta_0) \Rightarrow \mathcal N\big(0, (G^\top\Omega^{-1}G)^{-1}\big)$, where $G = \mathbb E[\partial m/\partial\theta^\top]$.

`OrnsteinUhlenbeckGMM` uses the stationary mean, variance, lag-one autocovariance and third central moment:

$$
\mathbb E[X-\theta]=0,\quad
\mathbb E[(X-\theta)^2]=\tfrac{\sigma^2}{2\kappa},\quad
\mathbb E[(X_t-\theta)(X_{t-\Delta}-\theta)]=\tfrac{\sigma^2}{2\kappa}e^{-\kappa\Delta},\quad
\mathbb E[(X-\theta)^3]=0.
$$

`CIRGMM` uses the analogous conditions with stationary variance $\sigma^2\theta/(2\kappa)$.

**HAC covariance.** `newey_west_hac` returns

$$
\hat\Omega = \hat\Gamma_0 + \sum_{j\ge1} k\!\Big(\frac{j}{L+1}\Big)\big(\hat\Gamma_j + \hat\Gamma_j^\top\big),
\qquad \hat\Gamma_j = \frac1N\sum_t \tilde m_t \tilde m_{t-j}^\top,
$$

where $\tilde m$ is the demeaned moment series. The default lag is $L = \lfloor 4(N/100)^{2/9}\rfloor$. Two kernels are available: Bartlett, $k(x) = 1-x$ on $[0,1]$ (Newey & West 1987), and quadratic-spectral, $k(x) = \frac{25}{12\pi^2x^2}\big(\frac{\sin(6\pi x/5)}{6\pi x/5} - \cos(6\pi x/5)\big)$ over all lags (Andrews 1991). The result is projected onto the positive semi-definite cone by eigenvalue clipping.

**Over-identification.** `hansen_j_test` computes $J = N\,g_N(\hat\theta)^\top\hat\Omega^{-1}g_N(\hat\theta) \Rightarrow \chi^2_{m-k}$ under correct specification. With $m = k$ the model is exactly identified and $J$ is uninformative.

### 2.2 Empirical characteristic function

For Lévy and affine jump models the density is often unavailable, but the characteristic function $\varphi_\theta(u) = \mathbb E[e^{iuX_\Delta}]$ is known through Lévy–Khintchine. `ECFEstimator` minimises

$$
\hat\theta = \arg\min_\theta \int \big|\hat\varphi_N(u) - \varphi_\theta(u)\big|^2 w(u)\,du,
\qquad \hat\varphi_N(u) = \frac1N\sum_{j=1}^N e^{iuX_j}.
$$

The integral is a quadrature on a grid, by default 41 points on $[-10,10]$. The weight is Gaussian, $w(u)=e^{-u^2/2}$, or uniform. Any process in the zoo with a `char_func` can be plugged in. The estimator is consistent and asymptotically normal for i.i.d. increments (Feuerverger & McDunnough 1981; Yu 2004). Its efficiency depends on the grid and the weight.

---

## 3. Latent states: filtering and quasi-likelihood

Many models have unobserved state, such as stochastic variance, convenience yield or default intensity. Write the discrete state-space model

$$
x_k = f(x_{k-1}) + \eta_k,\ \ \eta_k\sim(0, Q(x_{k-1})),
\qquad
y_k = h(x_k) + \varepsilon_k,\ \ \varepsilon_k\sim\mathcal N(0, R).
$$

Every filter returns a `FilteredTrajectory` with the filtered means and covariances, the innovations and a log-likelihood.

### 3.1 Unscented Kalman filter

`UnscentedKalmanFilter` uses the scaled sigma-point set of van der Merwe. For state dimension $L$, $\lambda = \alpha^2(L+\kappa) - L$, and the $2L+1$ points are

$$
\mathcal X_0 = \bar x,\qquad
\mathcal X_{\pm i} = \bar x \pm \big(\sqrt{(L+\lambda)P}\big)_i,
$$

with weights

$$
W^{m}_0 = \tfrac{\lambda}{L+\lambda},\quad
W^{c}_0 = W^m_0 + 1 - \alpha^2 + \beta,\quad
W^{m}_i = W^{c}_i = \tfrac{1}{2(L+\lambda)}.
$$

The defaults are $\alpha = 10^{-3}$, $\beta = 2$ and $\kappa = 0$. Propagating the points through $f$ and $h$ gives the predicted mean and covariance, $P_{yy}$ and $P_{xy}$. These are correct to second order for any smooth nonlinearity, and **exact when $f$ and $h$ are affine**, in which case the UKF is the Kalman filter. The log-likelihood is the prediction-error decomposition

$$
\ell = -\tfrac12\sum_k\Big(\log\det(2\pi S_k) + v_k^\top S_k^{-1}v_k\Big),
\qquad v_k = y_k - \hat y_{k|k-1},\ S_k = P_{yy,k}.
$$

`QuasiMaximumLikelihoodEstimator` maximises this $\ell$ over the model parameters. When the model is not linear-Gaussian this is a *quasi*-likelihood. The estimator is still consistent under standard conditions, but standard errors should use the sandwich form of §4.

### 3.2 Bootstrap particle filter

`BootstrapParticleFilter` implements sequential importance resampling (Gordon, Salmond & Smith 1993) with $N$ particles (500 by default). Each step does the following:

1. Propagate $x_k^{(i)} \sim p(x_k\mid x_{k-1}^{(i)})$.
2. Update the weights, $w_k^{(i)} \propto w_{k-1}^{(i)}\,p(y_k\mid x_k^{(i)})$, in log-space.
3. Accumulate $\log\hat p(y_k\mid y_{1:k-1}) = \log\sum_i w_{k-1}^{(i)}p(y_k\mid x_k^{(i)})$.
4. Resample systematically if $\mathrm{ESS} = 1/\sum_i (w_k^{(i)})^2 < 0.5N$, then reset the weights to $1/N$.

The product of the incremental terms is an **unbiased** estimator of the likelihood (Del Moral 2004). This is what particle-marginal Metropolis–Hastings requires. Its logarithm is biased downwards by about half its variance. `systematic_resample`, `stratified_resample` and `effective_sample_size` are exported separately.

!!! note "Optimising a particle likelihood"
    $\hat\ell(\theta)$ is a random, non-smooth function of $\theta$. To maximise it, fix the random seed (common random numbers) and prefer derivative-free optimisers.

### 3.3 Model-specific filters

| Filter | State | Observation |
| :--- | :--- | :--- |
| `HestonVarianceFilter` | $x = \log v$, with Euler drift $\big(\kappa(\theta-v) - \tfrac12\sigma^2\big)/v$ and variance $\approx\sigma^2\Delta/v$ (Itô on $\log v$) | volatility proxy $y\approx\sqrt v = e^{x/2}$ |
| `SchwartzTwoFactorFilter` | $(\log S, \delta)$ with $d\log S = (r-\delta-\tfrac12\sigma_1^2)dt + \sigma_1dW_1$ and $d\delta = \kappa(\alpha-\delta)dt + \sigma_2dW_2$ | log-futures, affine in the state through $B(\tau) = (1-e^{-\kappa\tau})/\kappa$ |
| `DefaultIntensityFilter` | CIR intensity $\lambda$ (Duffie–Singleton) | CDS spread $\approx \mathrm{LGD}\cdot\big(\theta + (\lambda-\theta)\frac{1-e^{-\kappa T}}{\kappa T}\big)$ |

Working in $\log v$ keeps the variance filter positive. The Schwartz measurement equation is affine, so its UKF is the exact Kalman filter for the Euler-discretised state.

---

## 4. Inference, standard errors and model selection

### 4.1 Information matrices

Let $\hat\theta$ maximise $\ell(\theta) = \sum_t \ell_t(\theta)$. Let $H = \nabla^2\ell(\hat\theta)$ be the observed Hessian and $s_t = \nabla\ell_t(\hat\theta)$ the per-observation scores.

`fisher_information_matrix` returns the observed information $\mathcal I = -H$, using `compute_numerical_hessian` (central differences). Under correct specification, $\widehat{\operatorname{Var}}(\hat\theta) = \mathcal I^{-1}$.

`sandwich_covariance_matrix` returns the robust (Huber–White) estimator (White 1982):

$$
\widehat{\operatorname{Var}}(\hat\theta) = H^{-1}\Big(\sum_t s_ts_t^\top\Big)H^{-1}.
$$

This remains valid under misspecification, including QMLE (§3.1) and heavy-tailed returns fitted with a Gaussian likelihood. If the model is correct, the information-matrix equality makes the two estimators agree asymptotically.

Both use the Moore–Penrose pseudo-inverse. A near-singular $H$ signals weak identification, and its standard errors should not be trusted. [Information Geometry](information_geometry.md) interprets $\mathcal I$ as the Fisher–Rao metric.

### 4.2 Wald and profile-likelihood intervals

`compute_parameter_inference` turns a covariance matrix into standard errors $\mathrm{se}_j = \sqrt{\hat V_{jj}}$, $z$-statistics $\hat\theta_j/\mathrm{se}_j$, two-sided normal $p$-values and Wald intervals $\hat\theta_j \pm z_{1-\alpha/2}\,\mathrm{se}_j$.

Wald intervals are symmetric by construction, so they can cross parameter bounds or misstate the uncertainty when $\ell$ is skewed. This happens, for example, with a mean-reversion speed estimated from a short sample. `compute_profile_likelihood` instead evaluates

$$
\ell_p(\psi) = \max_{\lambda}\ \ell(\psi, \lambda)
$$

on a grid of the target parameter $\psi$, re-optimising the nuisance parameters $\lambda$ at each point. It reports the likelihood-ratio interval

$$
\Big\{\psi : 2\big(\ell(\hat\theta) - \ell_p(\psi)\big) \le \chi^2_{1,1-\alpha}\Big\},
$$

which is invariant to reparametrisation and can be asymmetric.

### 4.3 Information criteria

With $k$ parameters, $n$ observations and maximised log-likelihood $\hat\ell$:

| Criterion | Function | Formula |
| :--- | :--- | :--- |
| AIC | `akaike_information_criterion` | $2k - 2\hat\ell$ |
| AICc | `corrected_aic` | $\mathrm{AIC} + \dfrac{2k(k+1)}{n-k-1}$ (infinite if $n \le k+1$) |
| BIC | `bayesian_information_criterion` | $k\log n - 2\hat\ell$ |
| HQIC | `hannan_quinn_criterion` | $2k\log\log n - 2\hat\ell$ |

Smaller is better. AIC targets predictive Kullback–Leibler risk and is not consistent. BIC and HQIC are consistent for the true model when it is in the candidate set. `ModelComparisonTable` ranks fitted models by AIC and reports the differences $\Delta_i$ and the Akaike weights $e^{-\Delta_i/2}/\sum_j e^{-\Delta_j/2}$ (Burnham & Anderson 2002), with $\Delta_i = \mathrm{AIC}_i - \min_j\mathrm{AIC}_j$.

!!! warning "Comparable likelihoods only"
    Information criteria compare likelihoods of **the same data**. Do not compare a model of prices with a model of log-returns, or an exact likelihood with a filter quasi-likelihood, without correcting for the transformation.

---

## 5. Smile and surface calibration

A `MarketSurface` holds the spot, rate, dividend yield and a list of `MarketSlice`s of `MarketQuote`s (strike, maturity, quoted value and its `quote_type`, optional bid/ask, vega, forward and discount factor). A `SurfaceCalibrator` solves

$$
\hat\theta = \arg\min_{\theta\in\Theta}\ \mathcal L\big(\sigma^{\text{model}}(K_i,T_i;\theta),\ \sigma^{\text{mkt}}_i\big) + \text{penalty}(\theta).
$$

### 5.1 Models

**SABR.** `SABRCalibrator` uses the Hagan et al. (2002) expansion for the Black implied vol. At the money it reduces to

$$
\sigma_{\text{ATM}} \approx \frac{\alpha}{F^{1-\beta}}\Big[1 + \Big(\frac{(1-\beta)^2\alpha^2}{24F^{2-2\beta}} + \frac{\rho\beta\nu\alpha}{4F^{1-\beta}} + \frac{2-3\rho^2}{24}\nu^2\Big)T\Big].
$$

$\beta$ is usually fixed by convention (`fixed_beta=1` for equity and FX, $0.5$ for rates) because it is poorly identified jointly with $\rho$. `estimate_initial_alpha` gives the leading-order start $\alpha_0 = \sigma_{\text{ATM}}F^{1-\beta}$. The expansion is an asymptotic approximation and can imply a negative density far out of the money at long maturities.

**Heston family.** Calls are priced from the characteristic function of $\log S_T$ through the Gil-Pelaez / Heston (1993) form

$$
C = S_0P_1 - Ke^{-rT}P_2,\qquad
P_1 = \tfrac12 + \tfrac1\pi\int_0^\infty \operatorname{Re}\frac{e^{-iu\log K}\varphi(u-i)}{iu\,\varphi(-i)}du,\qquad
P_2 = \tfrac12 + \tfrac1\pi\int_0^\infty \operatorname{Re}\frac{e^{-iu\log K}\varphi(u)}{iu}du,
$$

where $\varphi(-i) = \mathbb E[S_T] = S_0e^{rT}$. The integral uses Gauss–Legendre quadrature on $[0, 80]$. $\varphi$ is evaluated in the Albrecher et al. (2007) "little trap" form, which avoids the complex-logarithm branch-cut discontinuity at long maturities. Model prices are then inverted to Black–Scholes implied vols by bounded root-finding.

| Calibrator | Parameters | Characteristic function |
| :--- | :--- | :--- |
| `HestonCalibrator` | $v_0,\kappa,\theta,\sigma,\rho$ | Heston (1993) |
| `BatesCalibrator` | + $\lambda, \mu_J, \sigma_J$ | Heston × Merton log-normal jumps (Bates 1996) |
| `DoubleHestonCalibrator` | two independent variance factors | product of two Heston factors (Christoffersen et al. 2009) |
| `CEVHestonCalibrator` | + CEV exponent $\gamma$ ($dS = rS\,dt + S^\gamma\sqrt v\,dW$) | heuristic: the Heston pricer with the local factor frozen at $\sqrt{S_0K}$ |

The CEV-Heston quote freezes $S^{\gamma-1}$ at $\sqrt{S_0K}$, i.e. it prices Heston with $v_0$ and $\theta$ multiplied by $(S_0K)^{\gamma-1}$. The vol-of-vol is *not* rescaled, so this is a fast heuristic rather than a convergent expansion. Validate it by Monte Carlo before relying on it away from $\gamma = 1$.

The Heston, Bates and double-Heston pricers reduce to Black–Scholes as $\sigma\to0$ with $v_0=\theta$, and the Bates pricer reduces to the Merton (1976) series. Both limits are covered by exactness tests at non-zero rates. For other zoo processes, `spxa.analytics.fourier.european_call_price` gives Carr–Madan pricing from any `char_func`.

### 5.2 Loss functions

With errors $e_i = \sigma^{\text{model}}_i - \sigma^{\text{mkt}}_i$:

| Loss | Residual |
| :--- | :--- |
| `RMSELoss` | $e_i$, optionally $\times\sqrt{w_i/\bar w}$ |
| `VegaWeightedLoss` | $e_i\sqrt{\mathcal V_i/\bar{\mathcal V}}$, with vegas floored at `min_vega` |
| `RelativeErrorLoss` | $e_i/\sigma^{\text{mkt}}_i$ |
| `ArbitragePenalizedLoss` | base loss plus penalties for violating the Feller condition $2\kappa\theta\ge\sigma^2$ and for $\lvert\rho\rvert\to1$, plus an optional user penalty |

Vega weighting approximates a price-space fit, because $\Delta C \approx \mathcal V\,\Delta\sigma$, while staying well scaled across strikes. The Feller penalty is a *preference*, not a no-arbitrage requirement. Equity-index Heston fits routinely violate it, and the CIR variance then touches zero but stays non-negative.

### 5.3 Optimisers

Box constraints are handled by `ParameterTransform`, the logit map $u = \log\frac{x-a}{b-x}$ and its logistic inverse. This lets unconstrained optimisers work in $\mathbb R^k$.

* `LeastSquaresOptimizer`: bounded trust-region reflective least squares (`scipy.optimize.least_squares`, method `trf`). Fast but only local.
* `DifferentialEvolutionOptimizer`: global and derivative-free. Robust to the flat valleys typical of Heston ($\kappa$–$\theta$ and $\sigma$–$\rho$ trade-offs).
* `HybridOptimizer`: differential evolution followed by a TRF polish. This is the recommended default for stochastic-volatility models.

### 5.4 Fit diagnostics

`analyze_surface_residuals` reports RMSE, MAE, maximum error, mean relative error and per-tenor RMSE. It also counts static-arbitrage violations in the fitted surface:

* **calendar**: total variance $w = \sigma^2T$ must be non-decreasing in $T$, checked at strikes common to adjacent maturities. This is exact for zero carry. With carry, the condition holds at fixed forward moneyness.
* **butterfly**: call prices must be convex in strike.

---

## 6. Forecast diagnostics

A fitted model is a sequence of one-step-ahead predictive distributions $F_t$. If the model is correct, the **probability integral transforms** $u_t = F_t(y_t)$ are i.i.d. $\mathcal U(0,1)$ (Rosenblatt 1952; Diebold, Gunther & Tay 1998). `compute_pit_values` computes $u_t$, clipped to $[10^{-6}, 1-10^{-6}]$. `evaluate_pit_diagnostics` then runs two tests:

* **Kolmogorov–Smirnov** test of uniformity.
* **Berkowitz (2001)** likelihood-ratio test, described below.

For the Berkowitz test, map $z_t = \Phi^{-1}(u_t)$ and fit the Gaussian AR(1) model $z_t - \mu = \rho(z_{t-1}-\mu) + \sigma\varepsilon_t$. Then test $H_0: \mu=0,\ \rho=0,\ \sigma^2=1$ with

$$
\mathrm{LR} = 2\big(\ell(\hat\mu,\hat\rho,\hat\sigma^2) - \ell(0,0,1)\big) \Rightarrow \chi^2_3.
$$

Berkowitz has much more power than KS in small samples and also detects serial dependence, which KS cannot.

`compute_residual_diagnostics` applies Jarque–Bera (normality) and Ljung–Box (serial correlation, 10 lags by default) to standardised residuals or filter innovations. `CalibrationReport` collects parameters, standard errors, criteria and diagnostics into one summary, and `plot_smile_overlay`, `plot_surface_error_heatmap` and `plot_filtered_trajectory` visualise them.

---

## References

* Aït-Sahalia, Y. (2002). Maximum likelihood estimation of discretely sampled diffusions: a closed-form approximation approach. *Econometrica*, 70(1), 223–262.
* Albrecher, H., Mayer, P., Schoutens, W. & Tistaert, J. (2007). The little Heston trap. *Wilmott Magazine*, January, 83–92.
* Andrews, D. W. K. (1991). Heteroskedasticity and autocorrelation consistent covariance matrix estimation. *Econometrica*, 59(3), 817–858.
* Bates, D. S. (1996). Jumps and stochastic volatility: exchange rate processes implicit in Deutsche Mark options. *Review of Financial Studies*, 9(1), 69–107.
* Berkowitz, J. (2001). Testing density forecasts, with applications to risk management. *Journal of Business & Economic Statistics*, 19(4), 465–474.
* Burnham, K. P. & Anderson, D. R. (2002). *Model Selection and Multimodel Inference* (2nd ed.). Springer.
* Carr, P. & Madan, D. (1999). Option valuation using the fast Fourier transform. *Journal of Computational Finance*, 2(4), 61–73.
* Christoffersen, P., Heston, S. & Jacobs, K. (2009). The shape and term structure of the index option smirk: why multifactor stochastic volatility models work so well. *Management Science*, 55(12), 1914–1932.
* Del Moral, P. (2004). *Feynman–Kac Formulae: Genealogical and Interacting Particle Systems with Applications*. Springer.
* Diebold, F. X., Gunther, T. A. & Tay, A. S. (1998). Evaluating density forecasts with applications to financial risk management. *International Economic Review*, 39(4), 863–883.
* Feuerverger, A. & McDunnough, P. (1981). On the efficiency of empirical characteristic function procedures. *Journal of the Royal Statistical Society B*, 43(1), 20–27.
* Gordon, N. J., Salmond, D. J. & Smith, A. F. M. (1993). Novel approach to nonlinear/non-Gaussian Bayesian state estimation. *IEE Proceedings F*, 140(2), 107–113.
* Hagan, P. S., Kumar, D., Lesniewski, A. S. & Woodward, D. E. (2002). Managing smile risk. *Wilmott Magazine*, September, 84–108.
* Hansen, L. P. (1982). Large sample properties of generalized method of moments estimators. *Econometrica*, 50(4), 1029–1054.
* Heston, S. L. (1993). A closed-form solution for options with stochastic volatility with applications to bond and currency options. *Review of Financial Studies*, 6(2), 327–343.
* Merton, R. C. (1976). Option pricing when underlying stock returns are discontinuous. *Journal of Financial Economics*, 3(1–2), 125–144.
* Newey, W. K. & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703–708.
* Schwartz, E. S. (1997). The stochastic behavior of commodity prices: implications for valuation and hedging. *Journal of Finance*, 52(3), 923–973.
* Rosenblatt, M. (1952). Remarks on a multivariate transformation. *Annals of Mathematical Statistics*, 23(3), 470–472.
* Wan, E. A. & van der Merwe, R. (2000). The unscented Kalman filter for nonlinear estimation. *Proc. IEEE Adaptive Systems for Signal Processing, Communications, and Control Symposium*, 153–158.
* White, H. (1982). Maximum likelihood estimation of misspecified models. *Econometrica*, 50(1), 1–25.
* Yu, J. (2004). Empirical characteristic function estimation and its applications. *Econometric Reviews*, 23(2), 93–123.
