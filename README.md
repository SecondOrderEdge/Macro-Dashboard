# U.S. Macro Dashboard

> A unified view of U.S. recession risk through three independent macro lenses.

![status](https://img.shields.io/badge/status-live-5ba3a3)
![python](https://img.shields.io/badge/python-3.11+-d4a574)
![license](https://img.shields.io/badge/license-MIT-6b7280)

[Methodology notebook](notebooks/methodology.ipynb)

## What this is

A single-indicator recession model gives you a precise number and the false comfort of a precise number. The 10-year/3-month yield curve has the best track record of any single signal in the post-war U.S. sample, and it has nonetheless run hot for stretches when the rest of the economy was demonstrably fine. Street estimates routinely disagree by twenty or thirty percentage points without disclosing what is driving the spread.

The dashboard responds with three independent lenses that can disagree readably. A four-model probit ensemble (NY Fed, Wright, BIC-selected, Estrella–Mishkin) estimates 12-month recession probability live from a 37-series FRED universe, with Chauvet–Piger shown alongside as a coincident benchmark. A 10-indicator labor composite (LAME) summarises the regime in a single inverse-volatility-weighted z-score. A yield-curve module exposes the term structure and inversion statistics directly. The headline is a 0–100 composite built from a 50/25/25 weighted blend.

What's novel — for a public dashboard — is the transparent decomposition. Every cell of the headline can be opened: each model reports its probability, the watchlist reports the exact indicator value that would trip a higher reading, and the LAME breakdown shows the z-score, weight, and contribution of each of its ten indicators. When the models disagree, the disagreement is auditable.

## The three modules

**Recession ensemble.** Four methodologically distinct 12-month-ahead probit specifications — NY Fed (term spread), Wright (spread + fed funds), BIC-selected (sign-constrained multivariate), and Estrella–Mishkin (closed form) — estimated over a shared 37-series FRED universe on an expanding window from 1967; the ensemble probability is their arithmetic mean. Chauvet–Piger (FRED's smoothed Markov-switching series) is reported alongside as a *coincident* benchmark and excluded from the average so forecast horizons aren't blended. Calibration is measured by Brier score and AUC both in-sample and via a walk-forward backtest, each with a reliability diagram.

**LAME — Labor Aggregate Market Engine.** Ten labor indicators (`UNRATE`, `ICSA`, `CCSA`, `JTSJOL`, `JTSQUR`, `AWHAETP`, `TEMPHELPS`, `PAYEMS`, `U6RATE`, `CIVPART`) are transformed per the registry, signed so that positive means expansionary, z-scored against their own expanding-window history, and combined with inverse-volatility weights computed from a rolling 5-year window.

**Yield curve.** Daily term structure with 11 maturities, three benchmark spreads (10Y−3M, 10Y−2Y, 5Y−2Y), and inversion statistics that condition on inversions lasting more than three months to avoid noise.

## Methodology

The BIC-selected model is built by forward-stepwise selection over the full FRED universe: a candidate feature is accepted only if it improves BIC, does not induce quasi-complete separation, and keeps every coefficient on the economically correct side (lower spread → higher risk; rising unemployment → higher risk; weaker sentiment and contracting credit → higher risk). The NY Fed and Wright models are re-estimated probits on fixed feature sets; Estrella–Mishkin uses frozen 2006 coefficients; Chauvet–Piger is FRED's published `RECPROUSM156N`.

The dependent variable is the within-12-months NBER target (`y_t = 1` if `USREC = 1` in any month from `t+1` to `t+12`) — matching how the headline "12-month recession probability" is read. Estimation uses an expanding sample from 1967 onward. The ensemble is the simple arithmetic mean of the four forward-model probabilities (Estrella–Mishkin keeps its frozen point-in-time coefficients). The walk-forward backtest only trains on observations whose 12-month-ahead label was already observable at each refit date, so future labels can't leak in; BIC feature selection is done once on the full sample (an in-sample-selection caveat noted on the methodology page).

LAME's expanding-window z-scoring requires a minimum of 60 monthly observations. Inverse-volatility weights are computed from the rolling 5-year volatility of each signed z-score; weights are normalised to sum to 1 at every date, with indicators with missing readings dropping out of the basket for that month.

A full **Methodology** tab is built into the dashboard itself — it auto-generates the data sources table from the registry and shows the feature set retained by the live fit, so it cannot drift from the code. See also [`notebooks/methodology.ipynb`](notebooks/methodology.ipynb) for a step-by-step walkthrough including a side-by-side comparison against a naive base-rate baseline.

## Calibration

Brier score and AUC are reported on the Methodology page, both in-sample (section 9) and out-of-sample via the walk-forward backtest (section 10), each with a decile reliability diagram and a base-rate skill score. The Recession page's *Under the Hood* tab shows the per-model comparison and indicator percentiles.

## Quick start

```bash
git clone https://github.com/SecondOrderEdge/Macro-Dashboard.git
cd Macro-Dashboard
pip install -r requirements.txt
cp .env.example .env
# Add your FRED_API_KEY to .env — free key at https://fred.stlouisfed.org/docs/api/api_key.html
streamlit run app.py
```

The first load fetches the FRED panel, fits the four-model ensemble, and runs the walk-forward backtest (tens of seconds); subsequent loads come from Streamlit's cache for six hours. Tests run with `pytest` and do not require network access.

## Recession page — four-model probit ensemble + benchmark

The Recession page averages four methodologically distinct, academically grounded 12-month-ahead models over a shared FRED universe: **NY Fed** (term-spread probit), **Wright** (spread + fed funds), **BIC-selected** (sign-constrained multivariate probit), and **Estrella–Mishkin** (closed form). **Chauvet–Piger** (FRED's smoothed Markov-switching series `RECPROUSM156N`) is shown beside them as a coincident benchmark — it nowcasts whether we're in recession now, a different horizon — and is excluded from the ensemble average. The page also reports a bootstrap 90% CI, per-indicator watchlist trigger levels, a 24-month trend attribution, and an interactive scenario tool. Every model probability is computed live from FRED — there are no hand-entered comparison values.

## Data notes

- Recession dates are sourced live from FRED's `USREC` (NBER-based recession indicator), falling back to the bundled `data/nber_recessions.csv` if the fetch is unavailable.
- The probit model drops any candidate feature covering less than 80% of the target window, so short-history series (e.g. JOLTS from 2000) don't shrink the estimation sample.
- Non-monthly series are resampled to month-start (weekly→mean, daily→last, quarterly→forward-fill) before being aligned into the monthly panel the probit ensemble consumes.
- The **Policy Path** tab reads a bundled CSV (`data/market_probability_tracker.csv`) built from the [Atlanta Fed Market Probability Tracker](https://www.atlantafed.org/cenfis/market-probability-tracker)'s *MPT Historical Data* (`.xlsx`) export — the market-implied distribution of the FOMC policy rate from CME SOFR options. A scheduled GitHub Action (`.github/workflows/refresh-market-probability.yml`) refreshes it daily: it downloads the `.xlsx`, validates it through the parser, and commits a new CSV only when the data changes (which redeploys the app). If the source blocks automated access, the last committed snapshot is served and can be refreshed manually by replacing the file; the tab and dashboard card show the snapshot **as-of date** so staleness is visible. Enabling it requires *Settings → Actions → General → Workflow permissions → Read and write*.

## Disclaimer

This is a research project. Not investment advice. No warranty.

## License

MIT — see [LICENSE](LICENSE).
