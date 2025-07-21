# Statistical Test Discovery & Formal Verification Framework

## Table of Contents

1. [Motivation & First‑Principles Vision](#motivation)
2. [Problem Statement](#problem)
3. [End‑to‑End Workflow](#workflow)
4. [Module Specifications](#modules)

   * 4.1 [Search‑Space API](#search)
   * 4.2 [Simulation Engine](#sim)
   * 4.3 [RL Optimisation Loop](#rl)
   * 4.4 [Calibration & Critical‑Value Estimation](#calib)
   * 4.5 [Formal Proof Layer](#proof)
   * 4.6 [Continuous Integration & Reproducibility](#ci)
5. [Prototype: Rediscovering the *z*‑Test](#proto‑z)
6. [Path to a Unit‑Root (ADF‑like) Test](#proto‑adf)
7. [Tech Stack](#tech)
8. [Repository Layout](#repo)
9. [Contribution Guide](#contrib)
10. [Road‑Map & Open Questions](#roadmap)
11. [Glossary](#glossary)
12. [Key References](#refs)

---

<a name="motivation"></a>

## 1. Motivation & First‑Principles Vision

> **Goal:** Build an open, reproducible pipeline that *discovers* powerful statistical tests from raw simulated data and then *certifies* them with machine‑checked proofs.
>
> **First Principle:**  A statistical test is valuable when it simultaneously (i) controls the Type‑I error for **all** distributions in a null model and (ii) achieves high power for realistic alternatives.  Nothing in classical theory prevents novel tests—*if* we can search the infinite design space and give iron‑clad guarantees.
>
> **Thought Experiment:**  Imagine the Augmented Dickey–Fuller (ADF) or the classical *z*‑test had never been written down.  Our framework should be capable of rediscovering an equivalent (or better) procedure—*purely* by simulation‑driven search—**and** output a Lean‑verified theorem that its size ≤ α.

---

<a name="problem"></a>

## 2. Problem Statement

Given:

* A **null family** 𝒫₀ (e.g. all I(1) processes, or all i.i.d. samples with mean 0).
* A set of **alternative generators** 𝒫₁ (e.g. stationary AR processes, or mean‑shifted normals).
* A target significance level α (e.g. 0.05).

We seek an algorithm that outputs a pair *(T̂, ĉ\_α)* such that

1. *(Validity)* ∀ P ∈ 𝒫₀: P\[T̂(X) > ĉ\_α] ≤ α.
2. *(High Power)* ∀ fixed P ∈ 𝒫₁: P\[T̂(X) > ĉ\_α] → 1 as n → ∞.

---

<a name="workflow"></a>

## 3. End‑to‑End Workflow

```
┌──────────┐   samples   ┌────────────┐   reward   ┌─────────┐
│ Simulator├───────────►│   RL/ES    ├───────────►│  θ_t    │
└──────────┘            └────────────┘            └──┬──────┘
      ▲                   ▲   ▲                      │
      │ null / alt        │   │                      │ T_θ(X)
      │                   │   │                      ▼
      │            size‑penalty │            ┌─────────────────┐
      │                           └──────────┤Calibration cv_α│
      │                                      └─────────────────┘
      │                                               │
      │                                 Lean proof of │ size ≤ α
      └───────────────────────────────────────────────┘
```

1. **Simulation** draws batches from null & alternative models.
2. **RL / Evolution Search** adjusts parameters θ of a candidate statistic family *T\_θ*.
3. **Calibration** computes a critical value *c\_α(θ)* to keep empirical size near α.
4. **Formal Verification** (Lean) proves size control (and ideally consistency) *uniformly* over the whole search‑space Θ.
5. **Evaluation** reports finite‑sample power curves & publishes the certified test.

---

<a name="modules"></a>

## 4. Module Specifications

### <a name="search"></a>4.1 Search‑Space API

* **Definition:** A Python class `StatFamily` exposing

  * `T(theta, sample) -> float`
  * `critical_value(theta, alpha, sample_or_n) -> float`
* **Assumptions for Proof:**  For every θ ∈ Θ

  * `T` is measurable; moments up to order 2 exist.
  * Calibration routine targets α via bootstrap/permutation/analytic quantile.
* **Extensibility:**  New statistic families register by subclassing `StatFamily`.

### <a name="sim"></a>4.2 Simulation Engine

* Vectorised NumPy/JAX samplers for null & alt.
* Pluggable noise, lag‑structure, and dependence models.
* CI test‑set ensures deterministic seeds for reproducibility.

### <a name="rl"></a>4.3 RL Optimisation Loop

* **Algorithm:** default PPO (SB3) or CMA‑ES (OpenAI Evolution).
* **Reward Function:**

  ```
     r = 1{reject under alt} − λ·max(0, reject_under_null − α)
  ```

  Dual‑gradient step on λ enforces the size constraint.
* **Parallelism:** gym‑vector > 32 envs; Ray for scale‑out.

### <a name="calib"></a>4.4 Calibration & Critical‑Value Estimation

* Default: percentile bootstrap with B = 2 000.
* Block bootstrap helper for dependent data.
* Guarantees empirical size ≤ α *by definition*; formal proof extends this to uniform size via consistency of the bootstrap quantile.

### <a name="proof"></a>4.5 Formal Proof Layer

* **Assistant:** Lean 4 + mathlib4 probability.
* **Core Theorem (template):**

  ```lean
  theorem size_control (θ : Θ) : ∀ n, P₀ (T θ n > cv θ α n) ≤ α
  ```
* **Automation:** APOLLO / DeepSeek‑Prover attempts proof; CI fails if kernel rejects.
* Optional: consistency proof via CLT or martingale SLLN.

### <a name="ci"></a>4.6 Continuous Integration

* GitHub Actions steps:

  1. `python train.py --fast` (short smoke run)
  2. Cache or download pre‑trained θ̂.
  3. `lean --make proof/` (must pass).
  4. Run unit tests + power benchmarks; upload plots as artefacts.

---

<a name="proto‑z"></a>

## 5. Prototype 1 – Rediscovering the *z*‑Test

* **Search space:** linear combination of sample mean, variance, skew.
* **n = 30**, σ = 1 known, α = 0.05.
* **Outcome:** RL converges to θ≈(√n,≈0,≈0). Lean proof trivial (bootstrap quantile).  End‑to‑end runtime < 10 min on CPU.

---

<a name="proto‑adf"></a>

## 6. Prototype 2 – Toward an ADF‑like Unit‑Root Test

* **Search space:** studentised OLS t‑ratios on lagged level + k lagged differences.
* **Simulation:** ARIMA(0,1,0) for null; AR(p) ρ∈{0.6,0.8} for alt.
* **Calibration:** block bootstrap under unit‑root.
* **Proof tasks:** functional CLT for partial‑sum process; validity of block bootstrap (existing theorems in econ library).

---

<a name="tech"></a>

## 7. Tech Stack

| Layer      | Libs / Tools                                        |
| ---------- | --------------------------------------------------- |
| Simulation | NumPy, JAX, SciPy, NetworkX                         |
| RL / Optim | Stable‑Baselines3, Ray RLlib, CMA‑ES, Optuna        |
| Proof      | Lean 4, mathlib4, APOLLO / DeepSeek‑Prover          |
| DevOps     | GitHub Actions, Docker, pre‑commit, `black`, `ruff` |
| Viz / Eval | Matplotlib, seaborn, pandas, Weights & Biases       |

---

<a name="repo"></a>

## 8. Repository Layout

```
stat‑test‑discovery/
├── search/           # RL & ES entry‑points
│   ├── train.py
│   └── envs.py
├── statfamily/       # pluggable statistic modules
│   ├── base.py
│   └── linear_mean_var.py  # prototype 1
├── proof/            # Lean files
│   ├── Size.lean
│   └── Consistency.lean
├── data/             # cached θ̂ & CV tables
├── experiments/      # notebooks & plots
├── ci/               # GitHub Action workflows
└── README.md         # quick‑start & badge
```

---

<a name="contrib"></a>

## 9. Contribution Guide

1. **Fork & clone**; install via `poetry install --with dev`.
2. Add a new statistic: subclass `StatFamily`, write a docstring describing assumptions.
3. Provide a minimal Lean lemma file or extend existing theorems.
4. Run `pytest` and `lean --make` locally; open a PR.
5. CI must stay green (training, proofs, lint).

---

<a name="roadmap"></a>

## 10. Road‑Map & Open Questions

| Milestone                | ETA     | Owner                |
| ------------------------ | ------- | -------------------- |
| Full ADF‑like demo       | Q3 2025 | core team            |
| Graph two‑sample test    | Q4 2025 | community            |
| Privacy‑preserving tests | 2026    | TBD                  |
| Isabelle backend         | 2026    | external contributor |

*Open Issues*: heavy‑tail robustness, sequential tests, formalising bootstrap for mixing processes.

---

<a name="glossary"></a>

## 11. Glossary

* **Size (α):** maximal Type‑I error probability.
* **Power:** probability to reject under a specific alternative.
* **Consistency:** power → 1 as n → ∞.
* **Search space Θ:** parameter set for candidate statistics.
* **Lean:** theorem prover / proof assistant based on dependent type theory.

---

<a name="refs"></a>

## 12. Key References

1. Dickey, D. & Fuller, W. (1979) *Distribution of the Estimators for Autoregressive Time Series*.
2. Efron, B. (1979) *Bootstrap Methods*.
3. Kirchler, Lin et al. (2020) *Learning Two‑Sample Tests via Deep Kernels*.
4. DeepSeek‑Prover Team (2024), *Large‑Scale Automated Theorem Proving*.
5. Stable‑Baselines3 docs, RLlib docs, Lean `mathlib4`.

---
