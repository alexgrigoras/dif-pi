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


## Repository structure

```text
DIF-PI/
├── artifacts/
│   ├── difpi_exec_demo/                         # exported DIF-PI decisions
│   ├── npd_validation/                          # NPD validation exports
│   ├── sdg_validation/                          # SDG validation exports
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
├── eda-complete-journey.ipynb
├── npd-validation.ipynb
├── sdg-validation.ipynb
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

### Mapping to repo assets

| DIF‑PI stage | Notebook / Module |
|---|---|
| Data unification and exports | `eda-complete-journey.ipynb` |
| Purchase intention (NPD) | `src/npd.py`, `train-npd.ipynb`, `npd-validation.ipynb` |
| SDG model training and validation | `src/sdg.py`, `src/sdg_integration.py`, `train-sdg.ipynb`, `sdg-validation.ipynb` |
| Global forecasting training | `src/transformer_forecaster.py`, `train-forecaster.ipynb` |
| What-if scenario generation | `src/tbwisa.py`, `src/loglinear_scenarios.py`, `src/xgboost_scenarios.py` |
| End-to-end executive decision run | `dif-pi.ipynb` |
| Explainability screening | `dif-pi.ipynb` |


## Core components

### 1) Data unification
`eda-complete-journey.ipynb` converts raw transactions into a daily SKU panel with aligned timestamp, price, and demand, and exports the processed files used by the rest of the repository, including:

- *datasets/processed/difpi_transactions.csv*
- *datasets/processed/difpi_pricing_demand.csv*
- *datasets/processed/difpi_pricing_demand_panel.csv*
- *datasets/processed/top_skus.csv*
- *datasets/processed/difpi_metadata.json*

**Daily aggregation (per SKU)**  
Let $\mathcal{T}(s,t)$ be the set of transactions for SKU $s$  on day $t$, with quantity $q_i$ and unit price $p_i$:

- **Demand** (units)
```math
Q_{s,t} = \sum_{i\in \mathcal{T}(s,t)} q_i
```

- **Price** (quantity-weighted unit price)
```math
P_{s,t} =
\begin{cases}
\frac{\sum_{i\in \mathcal{T}(s,t)} p_i q_i}{\sum_{i\in \mathcal{T}(s,t)} q_i}, & \sum q_i > 0 \\
\frac{1}{|\mathcal{T}(s,t)|}\sum_{i\in \mathcal{T}(s,t)} p_i, & \text{otherwise}
\end{cases}
```

The panel is reindexed to daily continuity, missing demand is set to 0, and missing price is forward-filled to preserve a usable intervention reference.

### 2) Purchase intention: Next Purchase Day (NPD)
DIF‑PI uses NPD as a timing signal that complements the pricing decision stage.

For each customer $c$ with purchase dates $(d_1, d_2, \dots, d_n)$, inter-purchase gaps are defined as:

```math
g_i = d_{i+1} - d_i,\quad i=1,\dots,n-1
```

Given a history length $k$, supervised samples are constructed as:

```math
\mathbf{x}_i = [g_{i-k}, \dots, g_{i-1}],\qquad y_i = g_i
```

The predicted next gap $hat g_c$ is mapped to a next purchase day:

```math
\hat d_c = d_{\max} + \mathrm{round}(\hat g_c)
```

In the framework, NPD error is summarized as MAE in days and converted into a timing-reliability weight. This prevents the executive layer from overusing NPD when timing predictions are weak.

### 3) What-if scenario generation: TBWISA
`src/tbwisa.py` implements TBWISA as an SCM-inspired elasticity and residual model with controlled stochasticity.

#### Elasticity model
```math
\log Q_t = \beta_0 + \varepsilon \log P_t + u_t
```

where $\varepsilon# is price elasticity and $u_t$ are residuals.

#### Counterfactual under a price intervention
For an intervention $\delta$ (%), the counterfactual price is:

```math
P_t^{(\delta)} = P_t\left(1+\frac{\delta}{100}\right)
```

and the demand trajectory is reconstructed by applying the elasticity response plus residual structure. The repository also includes log-linear and XGBoost scenario baselines for comparison.

### 4) Demand forecasting
`train-forecaster.ipynb` trains the global Transformer used to roll out each candidate scenario across the future horizon.

**Multi-step forecasting**
```math
\hat y_{t+1}=f_\theta([y_{t-L+1},\dots,y_t]),\quad
\hat y_{t+h}=f_\theta([\hat y_{t+h-L},\dots,\hat y_{t+h-1}])
```

In practice, the forecaster receives the scenario-adjusted history and produces the demand path needed for revenue evaluation. Scaling is applied per SKU at inference time to reduce leakage.

### 5) Revenue window optimization
For each price intervention $\delta$, DIF‑PI computes revenue over the forecast horizon:

```math
R_t^{(\delta)} = P_t^{(\delta)} \cdot \hat Q_t^{(\delta)}
```

For a start offset $s$ and window length $\ell$, the average revenue inside the candidate window is:

```math
\bar R_{s,\ell}^{(\delta)} = \frac{1}{\ell}\sum_{t=s}^{s+\ell-1} R_t^{(\delta)}
```

A simple score used in the repository is:

```math
\text{score}(s,\ell)=\bar R_{s,\ell}^{(\delta)} - \lambda \ell
```

where $\lambda$ is an optional length penalty. The selected window is:

```math
(s^{*},\ell^{*})=\arg\max_{s,\ell}\ \text{score}(s,\ell)
```

The optimizer searches over valid windows and keeps the configuration that maximizes the selected score while respecting the practical demand and stability checks implemented in the notebook pipeline.

### 6) Explainability screening: X‑TBWISA
Before export, DIF‑PI applies a screening layer to distinguish stable decisions from risky ones.

A surrogate can be fit to approximate the teacher forecaster over intervention outputs, and the approximation quality can be summarized by:

```math
\mathrm{MAE}=\frac{1}{N}\sum_{i=1}^{N}\left|\hat y_i^{(\text{teacher})}-\hat y_i^{(\text{surrogate})}\right|,
\quad
\mathrm{RMSE}=\sqrt{\frac{1}{N}\sum_{i=1}^{N}\left(\hat y_i^{(\text{teacher})}-\hat y_i^{(\text{surrogate})}\right)^2}
```

This is combined with SHAP-based diagnostics and economic plausibility checks to assign one of three labels:
- *Accept*
- *Accept_Caution*
- *Flag*

### 7) Synthetic data generation (SDG)
SDG is used when demand histories are sparse or intermittent. The repository includes the proposed LLM-based SDG module together with benchmark generators. In practice, SDG supports robustness experiments such as real-only versus real+synthetic training, and the effect is assessed through fidelity, utility, privacy, and downstream decision stability.


## Key defaults

- *Eligibility mode:* adaptive, strict, relaxed
- *Delta grid (δ):* [-15, -10, -5, 0, 5, 10, 15]
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
- Use SDG to compare real-only vs real + synthetic forecaster training, then report both forecast error changes and decision stability (recommended $\delta$ consistency).


#### E3 — Price/demand visualization for a representative SKU (SKU 1070820)
This is a lightweight , paper‑grade check that the SKU panel is aligned and economically plausible.

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
- Screening can override the raw optimum when the response is unstable or implausible.
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


## What is required before running the validation notebooks

The validation notebooks are designed to load saved artifacts, not to retrain the models inside validation.

To run the full workflow you still need:

- the raw dunnhumby-Complete Journey export under `datasets/raw/complete_journey/`;
- the processed DIF‑PI files generated by `eda-complete-journey.ipynb`;
- the saved global forecaster checkpoint from `train-forecaster.ipynb`;
- the saved NPD artifact bundle from `train-npd.ipynb` for `npd-validation.ipynb` and the NPD-enabled DIF‑PI run;
- the saved SDG checkpoint from `train-sdg.ipynb` for `sdg-validation.ipynb` and optional SDG-enabled experiments.


## Setup

### Python
Python **3.10+** is recommended.

### Suggested dependencies
Base notebooks use: *numpy, pandas, scipy, scikit-learn, xgboost, statsmodels, tensorflow, matplotlib, shap, joblib

Additional dependencies used by the NPD and SDG training and validation notebooks: torch, transformers, datasets, sentencepiece, peft, accelerate, autots, bitsandbytes

Install:

```bash
python -m venv .venv
python -m pip install --upgrade pip
source .venv/bin/activate
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
