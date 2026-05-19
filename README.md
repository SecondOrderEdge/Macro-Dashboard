# Macro Risk Cockpit

> A unified view of U.S. recession risk through three independent macro lenses.

![status](https://img.shields.io/badge/status-live-5ba3a3)
![python](https://img.shields.io/badge/python-3.11+-d4a574)
![license](https://img.shields.io/badge/license-MIT-6b7280)

[Methodology notebook](notebooks/methodology.ipynb)

## What this is

A single-indicator recession model gives you a precise number and the false comfort of a precise number. The 10-year/3-month yield curve has the best track record of any single signal in the post-war U.S. sample, and it has nonetheless run hot for stretches when the rest of the economy was demonstrably fine. Street estimates routinely disagree by twenty or thirty percentage points without disclosing what is driving the spread.

The cockpit responds with three independent lenses that can disagree readably. A 5-submodel probit ensemble covers 41 FRED series across yield curve, labor, credit, housing, and sentiment. A 10-indicator labor composite (LAME) summarises the regime in a single inverse-volatility-weighted z-score. A yield-curve module exposes the term structure and inversion statistics directly. The headline is a 0–100 composite built from a 50/25/25 weighted blend.

What's novel — for a public dashboard — is the transparent decomposition. Every cell of the headline can be opened: each submodel reports its probability, each retained feature reports its marginal contribution in percentage points, and the LAME breakdown shows the z-score, weight, and contribution of each of its ten indicators. When this model disagrees with the Street, the disagreement is auditable.

## The three modules

**Recession ensemble.** Five thematic probit submodels (yield curve, labor, credit, housing, sentiment), each estimated with sign-constrained BIC stepwise selection on an expanding window from 1976. The ensemble probability is the arithmetic mean. Calibration is measured by Brier score and AUC on the in-sample history, with a reliability diagram for visual inspection.

**LAME — Labor Aggregate Market Engine.** Ten labor indicators (`UNRATE`, `ICSA`, `CCSA`, `JTSJOL`, `JTSQUR`, `AWHAETP`, `TEMPHELPS`, `PAYEMS`, `U6RATE`, `CIVPART`) are transformed per the registry, signed so that positive means expansionary, z-scored against their own expanding-window history, and combined with inverse-volatility weights computed from a rolling 5-year window.

**Yield curve.** Daily term structure with 11 maturities, three benchmark spreads (10Y−3M, 10Y−2Y, 5Y−2Y), and inversion statistics that condition on inversions lasting more than three months to avoid noise.

## Methodology

Each probit submodel is built from a thematic subset of features. Coefficients are first fit on the full feature set; any feature whose coefficient sign disagrees with the economic prior is dropped. The remaining features are then put through BIC-stepwise: at each step we drop the feature with the smallest absolute t-statistic, refit, and accept the drop if BIC strictly improves and the sign constraints still hold. The procedure stops when no drop improves BIC.

The dependent variable is the standard 12-month-forward NBER recession indicator. Estimation uses an expanding sample from 1976 onward — the earliest date at which most of the series in the panel are available. Each submodel produces a probability; the ensemble is the simple arithmetic mean of the five submodel probabilities. Driver contributions at the current observation are computed as `coef × (value − mean) × φ(z)`, expressed in percentage points.

LAME's expanding-window z-scoring requires a minimum of 60 monthly observations. Inverse-volatility weights are computed from the rolling 5-year volatility of each signed z-score; weights are normalised to sum to 1 at every date, with indicators with missing readings dropping out of the basket for that month.

See [`notebooks/methodology.ipynb`](notebooks/methodology.ipynb) for the full walkthrough, including a side-by-side comparison against a naive base-rate baseline.

## Calibration

Brier score and AUC are reported live in the *Under the Hood* tab of the recession view, with a reliability diagram binned into deciles. On synthetic noise the ensemble's Brier sits below the unconditional base-rate Brier; on real FRED data the in-sample fit is materially better still.

## Quick start

```bash
git clone https://github.com/SecondOrderEdge/macro-cockpit.git
cd macro-cockpit
pip install -r requirements.txt
cp .env.example .env
# Add your FRED_API_KEY to .env — free key at https://fred.stlouisfed.org/docs/api/api_key.html
streamlit run app.py
```

The first load fetches ~30 series from FRED and fits the ensemble; subsequent loads come from Streamlit's cache for six hours. Tests run with `pytest` and do not require network access.

## How it differs from Street estimates

Pure yield-curve models (NY Fed) tend to print higher than us when the curve is inverted but labor and credit are calm — we discount the curve signal by ensembling against four other lenses. Pure equity-vol or growth-momentum estimates (Goldman) tend to print lower when the curve is inverted but everything else looks fine — we lift their reading by including the curve. The composite's job is not to be the best single indicator; it is to be the indicator that disagrees least with the rest of the dashboard when you click through.

The Street comparison values in *vs. Street* live in `data/street_estimates.csv` and are maintained manually. They are a sanity-check, not a benchmark.

## Data notes

- `BAMLH0A0HYM2` (ICE BofA High Yield OAS) replaces older retail-credit series and starts in 1996; the credit submodel's effective sample begins there.
- `JTSJOL` and `JTSQUR` (JOLTS openings and quits rate) begin in 2000 — the labor submodel uses them when available and otherwise leans on the four older series.
- All daily series are resampled to month-end before being aligned into the monthly panel used by the probit submodels.

## Disclaimer

This is a research project. Not investment advice. No warranty.

## License

MIT — see [LICENSE](LICENSE).
