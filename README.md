# DIF‑PI — Decision Intelligence Framework for Predicting Purchase Intentions in e‑Commerce

DIF‑PI operationalizes a Decision Intelligence pipeline that moves from *prediction* (when a purchase is likely) to *prescription* (what pricing action to take, when to apply it, and whether the action should be screened before execution).

The framework combines the following modules:

- *Purchase intention / timing (NPD)* - estimate when customers are likely to purchase next
- *Synthetic data generation (SDG)* - improve robustness for sparse or intermittent demand histories
- *What‑if pricing (TBWISA)* - simulate counterfactual price interventions
- *Demand forecasting* - roll out each scenario with a global Transformer forecaster
- *Window optimization* - identify the revenue-maximizing execution window
- *Explainability screening (X‑TBWISA)* - flag unstable or implausible recommendations before export


## Quickstart

Recommended path for the executive DIF‑PI demo: EDA → training → `dif-pi.ipynb`

1. Run `eda-complete-journey.ipynb` to build the processed DIF‑PI inputs in *datasets/processed/*.
2. Run `train-forecaster.ipynb` to save the global scenario forecaster in *artifacts/models/scenario_gen_transformer_global/*.
3. Run `train-npd.ipynb` to save the NPD artifact bundle in *artifacts/models/npd_transformer_bundle/*.
4. Run `dif-pi.ipynb` to generate the end-to-end decision outputs in *artifacts/difpi_exec_demo/*.

Optional validation path for thesis reporting:

5. Run `npd-validation.ipynb` for purchase-intention validation.
6. Run `sdg-validation.ipynb` for synthetic-demand validation.
7. Run `tbwisa-validation.ipynb` for standalone scenario-generation validation.
8. Run `x-tbwisa-validation.ipynb` for explainability and screening validation.
9. Run `dif-pi-ablation.ipynb` for module-level and mechanism-level ablation analysis.


## Repository structure

```text
DIF-PI/
├── artifacts/
│   ├── difpi_exec_demo/                         # exported DIF-PI decisions
│   ├── npd_validation/                          # NPD validation exports
│   ├── sdg_validation/                          # SDG validation exports
│   ├── tbwisa_validation_outputs/               # TBWISA validation exports
│   ├── x_tbwisa_validation_outputs/             # X-TBWISA validation exports
│   └── models/
│       ├── npd_transformer_bundle/              # global + clustered NPD models
│       ├── scenario_gen_transformer_global/     # global forecaster checkpoint
│       └── sdg_chronos_t5_small_dunnhumby/      # SDG checkpoint
├── datasets/
│   ├── raw/
│   └── processed/
├── src/
│   ├── autots.py
│   ├── dgan.py
│   ├── gan.py
│   ├── loglinear_scenarios.py
│   ├── lstm.py
│   ├── npd.py
│   ├── npd_baselines.py
│   ├── sdg.py
│   ├── sdg_compare_utils.py
│   ├── sdg_integration.py
│   ├── tbwisa.py
│   ├── timegan.py
│   ├── transformer_forecaster.py
│   └── xgboost_scenarios.py
├── dif-pi.ipynb
├── dif-pi-ablation.ipynb
├── eda-complete-journey.ipynb
├── npd-validation.ipynb
├── sdg-validation.ipynb
├── tbwisa-validation.ipynb
├── x-tbwisa-validation.ipynb
├── train-forecaster.ipynb
├── train-npd.ipynb
├── train-sdg.ipynb
├── LICENSE
└── README.md
```


## DIF‑PI framework

DIF‑PI follows a *predict → simulate → forecast → optimize → screen → export* workflow:

1. **Data unification**  
   Build a continuous daily SKU panel with aligned timestamp, price, and demand.

2. **Purchase intention modeling**  
   Use customer inter-purchase-gap histories as a timing signal for downstream decision support.

3. **What-if scenario generation**  
   Generate counterfactual demand trajectories under price interventions using TBWISA, plus log-linear and XGBoost baselines.

4. **Demand forecasting**  
   Apply the global Transformer forecaster to roll each scenario forward across the decision horizon.

5. **Revenue window optimization**  
   Search for the best execution interval under each intervention.

6. **Explainability screening**  
   Apply X‑TBWISA checks before recommending an action.

7. **Executive export**  
   Save the final outputs as CSV/JSON artifacts for analysis, reporting, or dashboards.

### Decision formulation

For a selected SKU $s$ and decision date $t_0$, DIF‑PI returns one decision object that combines the selected intervention, execution window, score, uplift, screening status, and audit trail:

```math
z_s = (\delta^{*}, \tau^{*}, \ell^{*}, U_s^{*}, \mathrm{uplift}_s, \chi_s, \mathrm{audit}_s)
```

where $\delta^{*}$ is the recommended price intervention, $\tau^{*}$ is the start offset, $\ell^{*}$ is the execution-window length, and $\chi_s$ is the X‑TBWISA screening status:

```math
\chi_s(\delta^{*}, \tau^{*}, \ell^{*}) \in \{\mathrm{Accept},\ \mathrm{Accept\_Caution},\ \mathrm{Flag}\}
```

### Mapping to repo assets

| DIF‑PI stage | Notebook / Module |
|---|---|
| Data unification and exports | `eda-complete-journey.ipynb` |
| Purchase intention (NPD) | `src/npd.py`, `train-npd.ipynb`, `npd-validation.ipynb` |
| SDG model training and validation | `src/sdg.py`, `src/sdg_integration.py`, `train-sdg.ipynb`, `sdg-validation.ipynb` |
| Global forecasting training | `src/transformer_forecaster.py`, `train-forecaster.ipynb` |
| What-if scenario generation | `src/tbwisa.py`, `src/loglinear_scenarios.py`, `src/xgboost_scenarios.py`, `tbwisa-validation.ipynb` |
| End-to-end executive decision run | `dif-pi.ipynb` |
| Explainability screening | `dif-pi.ipynb`, `x-tbwisa-validation.ipynb` |
| Module and mechanism ablation | `dif-pi-ablation.ipynb` |


## Core components

### 1) Data unification
`eda-complete-journey.ipynb` converts raw transactions into a daily SKU panel with aligned timestamp, price, and demand, and exports the processed files used by the rest of the repository, including:

- *datasets/processed/difpi_transactions.csv*
- *datasets/processed/difpi_pricing_demand.csv*
- *datasets/processed/difpi_pricing_demand_panel.csv*
- *datasets/processed/top_skus.csv*
- *datasets/processed/difpi_metadata.json*

**Daily aggregation (per SKU)**  
Let $\mathcal{T}(s,t)$ be the set of transactions for SKU $s$ on day $t$, with quantity $q_i$ and unit price $p_i$. Daily demand is the total sold quantity:

```math
Q_{s,t} = \sum_{i\in \mathcal{T}(s,t)} q_i
```

Daily price is computed as a quantity-weighted unit price when the sold quantity is positive, and as the simple average observed price otherwise:

```math
P_{s,t} =
\begin{cases}
\frac{\sum_{i\in \mathcal{T}(s,t)} p_i q_i}{\sum_{i\in \mathcal{T}(s,t)} q_i}, & \sum q_i > 0 \\
\frac{1}{|\mathcal{T}(s,t)|}\sum_{i\in \mathcal{T}(s,t)} p_i, & \text{otherwise}
\end{cases}
```

The resulting $(P_{s,t}, Q_{s,t})$ series is reindexed to daily continuity over the active span of each SKU. Missing demand values are treated as zero sales and missing prices are filled forward to preserve a valid intervention reference for scenario generation and revenue evaluation.

### 2) Purchase intention: Next Purchase Day (NPD)
DIF‑PI uses NPD as a timing signal that complements the pricing decision stage. For each customer $c$, ordered purchase dates are denoted by $\tau_{c,1}, \tau_{c,2}, \dots, \tau_{c,n_c}$. The inter-purchase gaps are:

```math
g_{c,j} = \tau_{c,j} - \tau_{c,j-1}, \quad j = 2,\dots,n_c
```

For a fixed history length $K$, supervised NPD samples are constructed as:

```math
\mathbf{x}_{c,i} = [g_{c,i-K}, g_{c,i-K+1}, \dots, g_{c,i-1}], \qquad y_{c,i} = g_{c,i}
```

Inside DIF‑PI, the NPD output is summarized as a horizon-level purchase-intention profile and a reliability weight:

```math
\mathbf{i}_{t_0+1:t_0+H} = \mathrm{NPD}(\{g_c\}_c), \qquad \alpha = \Gamma(\mathrm{MAE}_{NPD})
```

The reliability gate prevents the executive layer from overusing NPD when timing predictions are weak. In the current implementation, low MAE gives stronger timing influence, acceptable MAE gives mild influence, and high MAE disables the timing contribution.

### 3) Synthetic data generation (SDG)
SDG is used when demand histories are sparse, short, intermittent, or need robustness support. The current DIF‑PI implementation keeps the token-based foundation of the published SDG work, while using `amazon/chronos-t5-small` as the encoder-decoder backbone and adding practical safeguards for retail demand generation.

For a context window $y_1, y_2, \dots, y_L$, the module first computes a mean-absolute scale:

```math
a = \max\left(\varepsilon,\frac{1}{L}\sum_{i=1}^{L}|y_i|\right)
```

The demand values are then scaled as:

```math
\tilde{y}_i = \frac{y_i}{a}
```

The scaled values are quantized into a fixed vocabulary of time-series tokens. In the default configuration, the implementation uses 4094 bins and clips the scaled value range to $[-5,5]$. The source sequence follows the thesis prompt structure:

```math
\text{task prefix}\ |\ \text{horizon}=H\ |\ \text{metadata}\ |\ \text{context: token sequence}
```

The default context length is 140 observations and the default prediction length is 30 observations. The SDG module supports symbolic time-series tokens, optional calendar and sequence-state metadata tokens, low-rank adaptation with QLoRA loading when available, fallback to LoRA or plain sequence-to-sequence loading, privacy-aware candidate filtering, training-time jitter, seasonality-aware calibration, sparse-series handling, and a moving-block bootstrap fallback for conservative robustness checks.

Synthetic data are treated as augmentation and repair support. They do not replace real held-out targets in validation.

### 4) What-if scenario generation: TBWISA
`src/tbwisa.py` implements TBWISA as an SCM-inspired scenario generator with controlled structural elasticity estimation, counterfactual price interventions, non-linear elasticity adjustment, residual reconstruction, and controlled stochasticity.

The controlled elasticity estimator first keeps only rows where log-space fitting is valid and where lagged demand can be used safely:

```math
M = \{t \in \{2,\dots,T\}: p_t > 0,\ d_t > 0,\ d_{t-1} > 0\}
```

When event-based fitting is enabled, the estimator can focus on rows where the price movement is large enough:

```math
E_{\tau} = \{t \in M: |\log p_t - \log p_{t-1}| \geq \log(1+\tau)\}
```

If enough event rows exist, $E_{\tau}$ is used; otherwise the estimator falls back to $M$. On the retained rows, TBWISA fits a robust Huber regression in log space:

```math
\log d_t =
a + \beta \log p_t + \gamma \log d_{t-1} + \eta z_t
+ \sum_{j=1}^{J}\left[
u_j \sin\left(\frac{2\pi t}{s_j}\right)
+ v_j \cos\left(\frac{2\pi t}{s_j}\right)
\right]
+ \varepsilon_t
```

where $a$ is the intercept, $\beta$ is the own-price elasticity coefficient, $\log d_{t-1}$ is the lagged demand control, $z_t$ is the z-scored trend term, and the Fourier terms capture seasonality. The implementation uses weekly and annual seasonal periods, Huber regression, a conservative negative prior when the raw coefficient is non-negative, and optional clipping of extreme elasticity magnitudes.

For an intervention $\delta$ (%) and $p_δ=\frac{\delta}{100}$, the adjusted price path is:

```math
p_t^{(\delta)} = p_t\left(1+\frac{\delta}{100}\right)
```

The implemented non-linear elasticity adjustment is:

```math
\beta^{(\delta)} = \beta(1+0.5p_\delta^2)
```

The counterfactual demand anchor is reconstructed in log space as:

```math
\tilde{d}_t^{(\delta)} =
\exp\left(a + \beta^{(\delta)}\log p_t^{(\delta)} + e_t\right)
```

where $e_t$ is the residual series retained from the elasticity stage. A controlled stochastic perturbation is then added:

```math
\xi_t^{(\delta)} \sim \mathcal{N}(0,\sigma^2\Delta^2), \qquad
\xi_t^{(\delta)} \in [-c,c]
```

and the final scenario demand is:

```math
d_t^{(\delta)} = \tilde{d}_t^{(\delta)}(1+\xi_t^{(\delta)})
```

The repository also includes log-linear and XGBoost scenario baselines for comparison. These baselines are useful references, but they do not use the full controlled elasticity logic, lagged-demand control, Fourier seasonality controls, sign-constrained elasticity, or stochastic counterfactual perturbation used by TBWISA.

### 5) Global demand forecasting
`train-forecaster.ipynb` trains the global Transformer used to roll out each candidate scenario across the future horizon. The forecaster is a global univariate encoder-only Transformer trained on pooled SKU demand windows.

Let $y_{s,t}$ denote the observed daily demand for SKU $s$ at time $t$. For sequence length $L$, the supervised training pair is:

```math
\mathbf{x}_{s,t} = [y_{s,t-L}, y_{s,t-L+1}, \dots, y_{s,t-1}], \qquad y_{s,t}
```

The model learns:

```math
\hat{y}_{s,t} = f_{\theta}(\mathbf{x}_{s,t})
```

Each SKU is scaled before pooling its windows into the global training set:

```math
\tilde{y}_{t}^{(s)} =
\frac{y_t^{(s)} - \min(y^{(s)})}
{\max(y^{(s)}) - \min(y^{(s)})}
```

At inference time, the saved model is reused for each scenario history without retraining. The one-step forecast from a scenario history $h_{1:T}^{(\delta)}$ is:

```math
\hat{y}_{T+1}^{(\delta)}
= f_{\theta}(h_{T-L+1}^{(\delta)}, \dots, h_T^{(\delta)})
```

The multi-step rollout is recursive:

```math
\hat{y}_{T+h}^{(\delta)}
= f_{\theta}(\hat{y}_{T+h-L}^{(\delta)}, \dots, \hat{y}_{T+h-1}^{(\delta)}),
\qquad h = 1,\dots,H
```

Scaling is fitted only on the available scenario history during inference, which keeps the rollout leakage-safe.

### 6) Revenue window optimization
For each intervention $\delta$, DIF‑PI computes expected daily revenue over the forecast horizon:

```math
R_t^{(\delta)} = P_t^{(\delta)} \cdot \hat{Q}_t^{(\delta)}
```

For a start offset $\tau$ and window length $\ell$, the average revenue inside the candidate execution window is:

```math
\bar{R}_{\tau,\ell}^{(\delta)}
= \frac{1}{\ell}\sum_{h=\tau}^{\tau+\ell-1} R_{t_0+h}^{(\delta)}
```

The implementation-level window score is:

```math
\mathrm{score}(\tau,\ell) = \bar{R}_{\tau,\ell}^{(\delta)} - \lambda \ell
```

where $\lambda \geq 0$ controls the length penalty. The selected window is:

```math
(\tau^{*},\ell^{*}) =
\arg\max_{\tau,\ell}\ \mathrm{score}(\tau,\ell)
```

The thesis also defines the more general decision objective with feasibility and instability penalties:

```math
S(\delta,a,b)
= \sum_{\tau=a}^{b} P_{\tau}^{(\delta)} \cdot \hat{Q}_{\tau}^{(\delta)}
- \lambda \cdot \mathrm{Penalty}(b-a+1)
- \gamma \cdot \mathrm{Volatility}(\hat{Q}_{a:b}^{(\delta)})
```

The optimizer searches valid windows, ranks candidate decisions, and then passes the top alternatives to X‑TBWISA screening and executive guardrails.

### 7) Explainability screening: X‑TBWISA
Before export, DIF‑PI applies X‑TBWISA to distinguish stable decisions from risky ones. The teacher is the global forecaster used in the TBWISA workflow. X‑TBWISA builds a surrogate dataset from intervention-level features and teacher outputs:

```math
\mathcal{S} =
\{(\mathbf{x}_{\delta,t}, \hat{y}_{\delta,t}^{teacher})\}_{\delta\in\Delta,\ t=1,\dots,H}
```

An XGBoost surrogate is fitted on this dataset. If $y_i$ is the teacher output and $\hat{y}_i$ is the surrogate prediction, surrogate fidelity is reported as:

```math
\mathrm{MAE} =
\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|,
\qquad
\mathrm{RMSE} =
\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
```

X‑TBWISA then computes SHAP-based diagnostics. Adjacent-delta explanation drift is measured as a normalized $L_1$ distance:

```math
d_{SHAP}(\delta,\delta+\Delta)
=
\frac{\|\phi_{\delta}-\phi_{\delta+\Delta}\|_1}
{\|\phi_{\delta}\|_1+\varepsilon}
```

The price-drift ratio checks whether drift is concentrated in intervention-related features:

```math
r_{price}(\delta)
=
\frac{|\phi_{\delta,\mathrm{delta\_pct}}| + |\phi_{\delta,\mathrm{last\_price}}|}
{\sum_j |\phi_{\delta,j}|+\varepsilon}
```

These diagnostics are combined with local monotonicity, price-alignment, economic plausibility, and surrogate-fidelity checks. The final screening labels are:

- *Accept*
- *Accept_Caution*
- *Flag*

The explanation layer is diagnostic rather than causal: SHAP values are used for auditing and screening the scenario behavior, while the intervention semantics remain defined by the TBWISA structural scenario engine.


## Key defaults

- *Eligibility mode:* adaptive, strict, relaxed
- *Delta grid (δ):* [-15, -12, -10, -7, -5, -2, 0, 2, 5, 7, 10, 12, 15]
- *Forecast horizon:* 30 days
- *Global model location:* artifacts/models/scenario_gen_transformer_global/
- *NPD model location:* artifacts/models/npd_transformer_bundle/
- *Exports:* artifacts/difpi_exec_demo/


## Notebook guide

| Notebook | Purpose | Main outputs |
|---|---|---|
| `eda-complete-journey.ipynb` | Preprocess raw retail transactions and export DIF‑PI-ready files | *datasets/processed/* panel, transactions, metadata, top SKUs |
| `train-forecaster.ipynb` | Train the global Transformer scenario forecaster | *artifacts/models/scenario_gen_transformer_global/* |
| `dif-pi.ipynb` | Run the end-to-end decision pipeline for a case SKU | *artifacts/difpi_exec_demo/* |
| `train-npd.ipynb` | Train global and clustered Transformer NPD models | *artifacts/models/npd_transformer_bundle/* |
| `npd-validation.ipynb` | Validate the NPD module and export benchmark tables/plots | *artifacts/npd_validation/* |
| `train-sdg.ipynb` | Train the LLM-based SDG module | *artifacts/models/sdg_chronos_t5_small_dunnhumby/* |
| `sdg-validation.ipynb` | Validate the SDG module and compare against benchmark generators | *artifacts/sdg_validation/* |
| `tbwisa-validation.ipynb` | Validate TBWISA against log-linear and XGBoost scenario generators | *artifacts/tbwisa_validation_outputs/* |
| `x-tbwisa-validation.ipynb` | Validate X‑TBWISA surrogate fidelity, SHAP diagnostics, drift, and screening labels | *artifacts/x_tbwisa_validation_outputs/* |
| `dif-pi-ablation.ipynb` | Run module-level and mechanism-level ablations for the integrated DIF‑PI workflow | *artifacts/difpi_exec_demo/* |


## Core source modules

### Main DIF‑PI modules
- `src/npd.py` - NPD utilities and purchase-intention signal construction
- `src/tbwisa.py` - SCM-inspired what-if scenario generation and window search helpers
- `src/transformer_forecaster.py` - global Transformer forecaster for scenario roll-out
- `src/sdg.py` - LLM-based synthetic demand generation
- `src/sdg_integration.py` - SDG loading, compatibility checks, and pipeline integration

### Scenario-generation baselines
- `src/loglinear_scenarios.py`
- `src/xgboost_scenarios.py`

### NPD baselines
- `src/autots.py`
- `src/npd_baselines.py`

### SDG benchmark and evaluation utilities
- `src/gan.py`
- `src/dgan.py`
- `src/timegan.py`
- `src/lstm.py`
- `src/sdg_compare_utils.py`


## Experiments

### DIF‑PI executive experiment
The main `dif-pi.ipynb` notebook keeps the current DIF‑PI case-study flow:
- build a representative SKU series;
- generate price-intervention scenarios;
- forecast demand under each scenario;
- search for the best revenue window;
- screen the recommendation with X‑TBWISA;
- export a compact executive decision summary.

### NPD validation
The NPD validation workflow adds a dedicated training and evaluation path:
- `train-npd.ipynb` trains the Global Transformer and Clustered Transformer NPD models;
- `npd-validation.ipynb` evaluates the NPD module on the next 5 purchase gaps and prepares thesis-ready tables and figures;
- the validation stage compares the Transformer-based NPD models against classical and neural baselines and exports RMSE/MAE summaries together with case-customer results.

### SDG validation
The SDG validation workflow adds a separate reproducible benchmark path:
- `train-sdg.ipynb` trains and saves the proposed SDG checkpoint;
- `sdg-validation.ipynb` evaluates the proposed SDG on held-out SKU futures and case-SKU reconstruction;
- the validation stage reports *fidelity*, *utility*, and *privacy* diagnostics and compares the proposed generator against benchmark sequence generators.

### TBWISA validation
The TBWISA validation workflow separates the scenario-generation evidence from the integrated executive run:
- `tbwisa-validation.ipynb` loads the processed SKU panel and the saved global Transformer forecaster;
- the notebook evaluates TBWISA against log-linear and XGBoost scenario generators;
- the validation stage reports baseline forecast accuracy, implied elasticity behavior, monotonicity consistency, economic plausibility, elasticity-noise sensitivity, and optimal-window stability;
- the outputs are saved under *artifacts/tbwisa_validation_outputs/*.

### X‑TBWISA validation
The X‑TBWISA validation workflow evaluates the explainability and screening layer as a standalone component:
- `x-tbwisa-validation.ipynb` builds a surrogate dataset from TBWISA intervention scenarios and teacher-forecaster outputs;
- an XGBoost surrogate is used for TreeSHAP analysis and surrogate-fidelity reporting;
- the validation stage measures adjacent-delta SHAP drift, price-feature drift, monotonicity, and price-alignment;
- each scenario receives an *Accept*, *Accept_Caution*, or *Flag* status under the screening policy;
- the outputs are saved under *artifacts/x_tbwisa_validation_outputs/*.

### DIF‑PI ablation analysis
The ablation notebook evaluates which modules and mechanisms affect the final executive recommendation:
- `dif-pi-ablation.ipynb` runs the full reference workflow and compares it with ablated variants;
- tested variants include `sdg_off`, `clustering_off`, `npd_timing_weight_off`, `controlled_elasticity_to_simple_loglog`, `nonlinear_elasticity_off`, `stochastic_simulation_off`, `x_tbwisa_screening_off`, and `forecaster_replacement_arima`;
- the analysis summarizes decision-score changes, uplift changes, screening-status shifts, recommended-delta shifts, and execution-window shifts;
- the outputs are saved under *artifacts/difpi_exec_demo/*.

### Experiment gallery

> **Note:** The figures below correspond to the repository experiments and can be generated from the notebook workflow.

#### E1 — Purchase intention distribution (NPD signal)
This histogram summarizes purchase days per customer. It motivates the NPD layer: customers exhibit strong heterogeneity and a long tail, so DIF‑PI treats NPD as a timing signal and
down‑weights it when error is high, as a reliability gate.

![Purchase days per customer (NPD context)](assets/npd.png)

**How to interpret and use**
- The long-tail distribution indicates strong heterogeneity in shopping frequency, many infrequent buyers.
- Use this as justification for timing-aware decisions: DIF‑PI can align actions to periods with higher predicted purchase activity, but will down-weight timing when NPD is unreliable.
- Operationally, this plot supports whether to treat the upcoming horizon as “high intent” vs “low intent” for planning the intervention window.


#### E2 — Synthetic data generation (SDG)
This plot compares real demand vs SDG synthetic demand on the same training segment.
The purpose is not perfect matching, but distributional plausibility (spikes, volatility, seasonality proxy) to support robustness experiments (real‑only vs real+synthetic training).

![Demand — real vs SDG synthetic (train segment)](assets/synthetic_data.png)

**How to interpret and use**
- If the synthetic series matches key stylized facts (volatility, spikes, rough seasonality), it is suitable for robustness testing.
- Use SDG to compare real-only vs real+synthetic forecaster training, then report both forecast error changes and decision stability (recommended $\delta$ consistency).


#### E3 — Price/demand visualization for a representative SKU (SKU 1070820)
This is a lightweight check that the SKU panel is aligned and economically plausible.

![Price time series](assets/price.png)
![Demand time series](assets/demand.png)

**What to report**
- date range, number of observations, missingness rate
- a short qualitative observation (e.g., “promotional price drops coincide with demand spikes”)

#### E4 — TBWISA what‑if scenarios + selected window
This figure shows forecasted revenue trajectories under a grid of price interventions $\delta$,
with the selected optimal window highlighted. It visualizes the prescriptive logic:
generate scenarios → forecast demand → compute revenue → select best execution window.

![TBWISA — revenue scenarios + selected window](assets/what_if_scenarios.png)

**How to interpret and use**
- Each curve is the forecasted revenue trajectory under a candidate intervention $\delta$.
- The highlighted segment is the selected optimal execution window $(s^{*},\ell^{*})$ for the chosen $\delta$.
- Implementation: apply the chosen $\delta$ for $\ell^{*}$ days starting at offset $s^{*}$ (relative to the decision date), and use X‑TBWISA screening to decide EXECUTE vs PILOT vs REVIEW.


#### E5 — Executive decision view
This figure summarizes the final decision logic: window score vs delta.
- Green points: deltas that pass screening (Accept)  
- Red points: deltas flagged by X‑TBWISA (Flag)  
- Black star: raw optimizer pick (best score before screening and guardrails)  
- Black diamond: final executive pick (after screening and guardrails; aligns with the exported decision card)

![Executive decision view (TBWISA): Score vs delta with X‑TBWISA screening](assets/executive_decision.png)

**Interpretation**
- Screening can downgrade the raw optimum when the response is unstable or implausible.
- The final executive recommendation is exported in `executive_summary_case_sku_<SKU>.json`.


## Executive decision outcome

For deployment-oriented use, DIF‑PI exports a single decision card:

- `artifacts/difpi_exec_demo/executive_summary_case_sku_<SKU>.json`

This file contains the final recommendation after screening, guardrails, and optional NPD-based timing support.

### Example executive decision card

```json
{
  "case_sku": "1070820",
  "forecast_backend": "scenario_gen_transformer_global",
  "npd_mae_days": 1.078113866446046,
  "horizon_days": 30,
  "recommendation": {
    "model": "tbwisa",
    "forecast_backend": "scenario_gen_transformer_global",
    "recommended_delta_pct": -15,
    "window_start_offset_days": 23,
    "window_length_days": 7,
    "avg_rev_forecast": 23.14267851303652,
    "baseline_avg_rev": 21.462066568588508,
    "uplift_ratio": 0.07830615653897083,
    "window_score": 27.80628383928755,
    "x_tbwisa_status": "Accept",
    "x_tbwisa_reason": "ok"
  }
}
```

**How to read it**
- *recommended_delta_pct* and *window_*\* define what to do and when.
- *avg_rev_forecast*, *baseline_avg_rev*, and *uplift_ratio* quantify the projected gain.
- *x_tbwisa_status* is the screening result used to distinguish direct execution from caution or review.


## Outputs

### DIF‑PI decision outputs
Running `dif-pi.ipynb` writes the main decision artifacts under *artifacts/difpi_exec_demo/*, including:

- `decision_table_case_sku_<SKU>.csv`
- `decision_table_screened_case_sku_<SKU>.csv`
- `metrics_case_sku_<SKU>.csv`
- `windows_case_sku_<SKU>.csv`
- `shap_global_importance_case_sku_<SKU>.csv`
- `executive_summary_case_sku_<SKU>.json`

### NPD validation outputs
`npd-validation.ipynb` exports benchmark and summary files under *artifacts/npd_validation/*, including customer-level results, summary tables, and reusable text/plot outputs for the thesis.

### SDG validation outputs
`sdg-validation.ipynb` exports evaluation tables under *artifacts/sdg_validation/*, including fidelity summaries, utility scores, privacy diagnostics, and benchmark comparison files.

### TBWISA validation outputs
`tbwisa-validation.ipynb` exports scenario-generation validation files under *artifacts/tbwisa_validation_outputs/*, including diagnostic CSV files and validation figures for accuracy, plausibility, elasticity, noise sensitivity, and window stability.

### X‑TBWISA validation outputs
`x-tbwisa-validation.ipynb` exports explainability and screening validation files under *artifacts/x_tbwisa_validation_outputs/*, including validation figures for surrogate fidelity, global SHAP, local waterfall, adjacent-delta drift, and screening outcomes.

### DIF‑PI ablation outputs
`dif-pi-ablation.ipynb` exports ablation results under *artifacts/difpi_exec_demo/*, including ablation figures for score change, uplift change, screening-status distribution, recommended-delta shift, and window shift.


## What is required before running the validation notebooks

The validation notebooks are designed to load saved artifacts, not to retrain the models inside validation.

To run the full workflow you still need:

- the raw dunnhumby-Complete Journey export under `datasets/raw/complete_journey/`;
- the processed DIF‑PI files generated by `eda-complete-journey.ipynb`;
- the saved global forecaster checkpoint from `train-forecaster.ipynb` for `dif-pi.ipynb`, `tbwisa-validation.ipynb`, `x-tbwisa-validation.ipynb`, and `dif-pi-ablation.ipynb`;
- the saved NPD artifact bundle from `train-npd.ipynb` for `npd-validation.ipynb` and the NPD-enabled DIF‑PI run;
- the saved SDG checkpoint from `train-sdg.ipynb` for `sdg-validation.ipynb` and optional SDG-enabled experiments;
- the saved train/test SKU split when available, so TBWISA, X‑TBWISA, integrated DIF‑PI, and ablation experiments use the same holdout protocol.


## Setup

### Python
Python **3.10+** is recommended.

### Suggested dependencies
Base notebooks use: *numpy, pandas, scipy, scikit-learn, xgboost, statsmodels, tensorflow, matplotlib, shap*

Additional dependencies used by the NPD and SDG training and validation notebooks: *torch, transformers, datasets, sentencepiece, peft, accelerate, autots, huggingface-hub*

Optional acceleration packages such as `bitsandbytes` can be installed separately when using quantized QLoRA loading on compatible hardware, but they are not required for the default repository workflow.

Install:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```


## Data

Raw datasets are **not committed** because of licensing and size constraints.

### dunnhumby “The Complete Journey”

Download the dataset from [dunnhumby.com](https://www.dunnhumby.com/source-files/) and place the export under: *datasets/raw/complete_journey/*

Then run `eda-complete-journey.ipynb` to generate the processed DIF‑PI inputs in `datasets/processed/`.


## Reproducibility notes

- Fixed seeds are used where possible.
- `train-forecaster.ipynb` writes train/test SKU lists next to the saved model.
- DIF‑PI includes decision guardrails to downgrade risky actions, especially when elasticity or uplift patterns are unstable.
- The validation notebooks are evaluation notebooks, so they assume the corresponding training notebooks have already produced the required artifacts.


## Resources

The framework integrates and extends the following research components:

1. [**Synthetic Time Series Generation for Decision Intelligence Using Large Language Models**](https://www.mdpi.com/2227-7390/12/16/2494)  
2. [**Transformer‑Based Model for Predicting Customers’ Next Purchase Day in e‑Commerce**](https://www.mdpi.com/2079-3197/11/11/210)  
3. [**Generating and Optimizing What‑If Scenarios Using a Transformers‑Based Forecasting Model**](https://ieeexplore.ieee.org/document/11240464)  
4. **X‑TBWISA: Explainable What‑If Scenario Generation Using Transformers and SHAP Guidance**


## License
Apache License 2.0 (`Apache-2.0`) — see `LICENSE`.
