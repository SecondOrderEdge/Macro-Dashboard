"""Recession Probability Ensemble.

Five probit submodels — yield curve, labor, credit, housing, sentiment —
fit on FRED data with NBER 12-month-forward recession as the dependent
variable. The ensemble is the simple arithmetic mean of the five submodel
probabilities.

Within each submodel we apply:

* Sign-constrained selection: features whose coefficient sign disagrees
  with economic priors are dropped before the BIC sweep.
* BIC-based selection: starting from the full feature set, drop the worst
  feature (by individual t-stat) until BIC stops improving.
* Expanding-window estimation from 1976 onward.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import norm

try:
    from statsmodels.discrete.discrete_model import Probit
except ImportError:  # pragma: no cover - dependency guard
    Probit = None  # type: ignore[assignment]

from src.data.series_registry import SERIES_REGISTRY, transform_series


# ---------------------------------------------------------------------- specs


@dataclass(frozen=True)
class FeatureSpec:
    """A single regressor: how to build it from the FRED panel, plus a sign prior."""

    name: str
    sign: int  # +1 expected positive in probit, -1 expected negative
    builder: str  # one of: "fred", "yoy", "diff_12m", "term_premium_proxy"
    source: str = ""  # registry key or FRED ID
    transform: str | None = None  # extra transform on top of fetch (e.g. "ma_3m")


@dataclass(frozen=True)
class SubmodelSpec:
    name: str
    label: str
    features: list[FeatureSpec] = field(default_factory=list)


# Submodel definitions. Sign priors here are the *expected* sign of the
# coefficient in a probit predicting recession-in-next-12m.

_YIELD_CURVE = SubmodelSpec(
    name="yield_curve",
    label="Yield Curve",
    features=[
        FeatureSpec("t10y3m", sign=-1, builder="fred", source="T10Y3M"),
        FeatureSpec("t10y2y", sign=-1, builder="fred", source="T10Y2Y"),
        FeatureSpec("term_premium_proxy", sign=-1, builder="term_premium_proxy"),
    ],
)

_LABOR = SubmodelSpec(
    name="labor",
    label="Labor",
    features=[
        FeatureSpec("unrate_d12", sign=+1, builder="diff_12m", source="UNRATE"),
        FeatureSpec("icsa_ma4", sign=+1, builder="fred", source="ICSA", transform="ma4"),
        FeatureSpec("jtsjol", sign=-1, builder="fred", source="JTSJOL"),
        FeatureSpec("jtsqur", sign=-1, builder="fred", source="JTSQUR"),
    ],
)

_CREDIT = SubmodelSpec(
    name="credit",
    label="Credit",
    features=[
        FeatureSpec("baa10y", sign=+1, builder="fred", source="BAA10Y"),
        FeatureSpec("drtscilm", sign=+1, builder="fred", source="DRTSCILM"),
        FeatureSpec("hy_oas", sign=+1, builder="fred", source="BAMLH0A0HYM2"),
    ],
)

_HOUSING = SubmodelSpec(
    name="housing",
    label="Housing",
    features=[
        FeatureSpec("permit_yoy", sign=-1, builder="yoy", source="PERMIT"),
        FeatureSpec("houst", sign=-1, builder="fred", source="HOUST"),
        FeatureSpec("pcec96_yoy", sign=-1, builder="yoy", source="PCEC96"),
    ],
)

_SENTIMENT = SubmodelSpec(
    name="sentiment",
    label="Sentiment",
    features=[
        FeatureSpec("vix_3m", sign=+1, builder="fred", source="VIXCLS", transform="ma_3m"),
        FeatureSpec("usslind", sign=-1, builder="fred", source="USSLIND"),
        FeatureSpec("sp500_6m", sign=-1, builder="fred", source="SP500", transform="ret_6m"),
    ],
)

SUBMODELS: list[SubmodelSpec] = [_YIELD_CURVE, _LABOR, _CREDIT, _HOUSING, _SENTIMENT]


# ----------------------------------------------------------- feature building


def _to_monthly(series: pd.Series) -> pd.Series:
    """Convert any cadence to month-end last-of-period."""
    return series.dropna().resample("ME").last()


def _build_feature(spec: FeatureSpec, panel: pd.DataFrame) -> pd.Series | None:
    """Build a single feature column from the raw FRED panel."""
    if spec.builder == "term_premium_proxy":
        for col in ("DGS10", "DGS2", "DGS5"):
            if col not in panel.columns:
                return None
        dgs10 = _to_monthly(panel["DGS10"])
        dgs2 = _to_monthly(panel["DGS2"])
        dgs5 = _to_monthly(panel["DGS5"])
        s = dgs10 - 0.5 * (dgs2 + dgs5)
        return s.rename(spec.name)

    if spec.source not in panel.columns:
        return None
    raw = _to_monthly(panel[spec.source])
    if raw.empty:
        return None

    if spec.builder == "fred":
        s = raw
    elif spec.builder == "yoy":
        s = transform_series(raw, "yoy")
    elif spec.builder == "diff_12m":
        s = raw.diff(12)
    else:  # pragma: no cover
        return None

    if spec.transform:
        s = transform_series(s, spec.transform)
    return s.rename(spec.name)


def _build_design_matrix(spec: SubmodelSpec, panel: pd.DataFrame) -> pd.DataFrame:
    cols = []
    for feat in spec.features:
        built = _build_feature(feat, panel)
        if built is not None:
            cols.append(built)
    if not cols:
        return pd.DataFrame()
    df = pd.concat(cols, axis=1)
    df.index = pd.DatetimeIndex(df.index).to_period("M").to_timestamp()
    return df


# ---------------------------------------------------------------- submodel fit


@dataclass
class _FittedSubmodel:
    spec: SubmodelSpec
    feature_names: list[str]
    coefs: pd.Series  # includes 'const'
    feature_means: pd.Series
    last_X: pd.Series  # last observation (without const)

    def predict_one(self, x: pd.Series) -> float:
        """Predict probability for a single feature row (no constant)."""
        x_aligned = x.reindex(self.feature_names).fillna(self.feature_means)
        const = self.coefs.get("const", 0.0)
        z = const + float(np.dot(self.coefs.drop("const").values, x_aligned.values))
        return float(norm.cdf(z))

    def driver_contributions(self, x: pd.Series) -> list[tuple[str, float]]:
        """Marginal contribution (pp) of each retained feature at observation x."""
        x_aligned = x.reindex(self.feature_names).fillna(self.feature_means)
        const = self.coefs.get("const", 0.0)
        z = const + float(np.dot(self.coefs.drop("const").values, x_aligned.values))
        pdf_z = float(norm.pdf(z))
        out: list[tuple[str, float]] = []
        for name in self.feature_names:
            coef = float(self.coefs[name])
            delta = float(x_aligned[name] - self.feature_means[name])
            contrib_pp = coef * delta * pdf_z * 100.0  # in percentage points
            out.append((name, contrib_pp))
        out.sort(key=lambda t: abs(t[1]), reverse=True)
        return out

    def predict_path(self, X: pd.DataFrame) -> pd.Series:
        X_aligned = X.reindex(columns=self.feature_names).fillna(self.feature_means)
        const = self.coefs.get("const", 0.0)
        lin = const + X_aligned.values @ self.coefs.drop("const").values
        return pd.Series(norm.cdf(lin), index=X_aligned.index)


def _fit_probit(X: pd.DataFrame, y: pd.Series):
    """Wrap statsmodels Probit fit; return result object or None on singular matrix."""
    if Probit is None:
        raise RuntimeError("statsmodels is required to fit the recession ensemble.")
    X_with_const = X.copy()
    X_with_const.insert(0, "const", 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = Probit(y.astype(int).values, X_with_const.values)
            return model.fit(disp=False, maxiter=200, method="newton")
        except Exception:
            try:
                model = Probit(y.astype(int).values, X_with_const.values)
                return model.fit(disp=False, maxiter=500, method="bfgs")
            except Exception:
                return None


def _select_submodel(spec: SubmodelSpec, X: pd.DataFrame, y: pd.Series) -> _FittedSubmodel | None:
    """Sign-constrained + BIC-based stepwise selection within one submodel."""
    if X.empty or y.empty:
        return None

    full = pd.concat([X, y.rename("y")], axis=1).dropna()
    if len(full) < 60 or full["y"].sum() < 5 or full["y"].sum() > len(full) - 5:
        return None

    X_full = full.drop(columns=["y"])
    y_full = full["y"]
    feature_names = list(X_full.columns)
    sign_priors = {f.name: f.sign for f in spec.features}

    # Round 1: drop features whose coefficient sign disagrees with prior.
    res = _fit_probit(X_full[feature_names], y_full)
    if res is None:
        return None
    params = pd.Series(res.params, index=["const"] + feature_names)
    keep = [n for n in feature_names if np.sign(params[n]) == sign_priors[n]]
    if not keep:
        # Fallback: keep the single feature with the strongest |t-stat|.
        tvals = pd.Series(res.tvalues, index=["const"] + feature_names).drop("const")
        tvals_abs = tvals.abs().replace([np.inf, -np.inf], np.nan).dropna()
        if tvals_abs.empty:
            return None
        keep = [str(tvals_abs.idxmax())]

    # Round 2: BIC-stepwise — drop the feature with the smallest |t| until BIC worsens.
    current = list(keep)
    res = _fit_probit(X_full[current], y_full)
    if res is None:
        return None
    best_bic = res.bic
    best_params = pd.Series(res.params, index=["const"] + current)

    while len(current) > 1:
        tvals = pd.Series(res.tvalues, index=["const"] + current).drop("const")
        tvals_abs = tvals.abs().replace([np.inf, -np.inf], np.nan).dropna()
        if tvals_abs.empty:
            break
        candidate = str(tvals_abs.idxmin())
        trial = [n for n in current if n != candidate]
        trial_res = _fit_probit(X_full[trial], y_full)
        if trial_res is None:
            break
        # Enforce signs after refit too.
        trial_params = pd.Series(trial_res.params, index=["const"] + trial)
        signs_ok = all(np.sign(trial_params[n]) == sign_priors[n] for n in trial)
        if not signs_ok:
            break
        if trial_res.bic < best_bic:
            best_bic = trial_res.bic
            best_params = trial_params
            current = trial
            res = trial_res
        else:
            break

    feature_means = X_full[current].mean()
    last_X = X_full[current].iloc[-1]

    return _FittedSubmodel(
        spec=spec,
        feature_names=current,
        coefs=best_params,
        feature_means=feature_means,
        last_X=last_X,
    )


# ------------------------------------------------------------------- ensemble


class RecessionEnsemble:
    """Five-submodel probit ensemble. See module docstring for the algorithm."""

    SUBMODELS: list[SubmodelSpec] = SUBMODELS
    START_DATE: pd.Timestamp = pd.Timestamp("1976-01-01")

    def __init__(self):
        self._fitted: dict[str, _FittedSubmodel] = {}
        self._designs: dict[str, pd.DataFrame] = {}
        self._target: pd.Series | None = None
        self._panel: pd.DataFrame | None = None

    # ------------------------------------------------------------------ fit

    def fit(self, data: pd.DataFrame, nber: pd.Series) -> None:
        """Fit all five submodels.

        ``data`` is the raw FRED panel (columns = FRED IDs); ``nber`` is the
        ``recession_in_next_12m`` target indexed by month.
        """
        self._panel = data.copy()
        target = nber.copy()
        target.index = pd.DatetimeIndex(target.index).to_period("M").to_timestamp()
        target = target.loc[target.index >= self.START_DATE]
        self._target = target.astype(int)

        self._fitted.clear()
        self._designs.clear()

        for spec in self.SUBMODELS:
            X = _build_design_matrix(spec, data)
            if X.empty:
                continue
            X = X.loc[X.index >= self.START_DATE]
            common = X.index.intersection(self._target.index)
            X = X.loc[common]
            y = self._target.loc[common]
            fitted = _select_submodel(spec, X, y)
            if fitted is None:
                continue
            self._fitted[spec.name] = fitted
            self._designs[spec.name] = X

    # -------------------------------------------------------- introspection

    def predict_current(self) -> dict:
        if not self._fitted:
            raise RuntimeError("Call fit(...) before predict_current().")

        submodel_probs: dict[str, float] = {}
        drivers: dict[str, list[tuple[str, float]]] = {}
        latest_ts = pd.Timestamp.min

        for name, fitted in self._fitted.items():
            X = self._designs[name]
            X_clean = X.dropna(how="any")
            if X_clean.empty:
                continue
            ts = X_clean.index.max()
            x = X_clean.loc[ts]
            prob = fitted.predict_one(x) * 100.0
            submodel_probs[name] = float(prob)
            drivers[name] = fitted.driver_contributions(x)
            latest_ts = max(latest_ts, ts)

        ensemble = float(np.mean(list(submodel_probs.values()))) if submodel_probs else float("nan")
        return {
            "ensemble": ensemble,
            "submodels": submodel_probs,
            "drivers": drivers,
            "as_of": latest_ts if submodel_probs else pd.Timestamp.now(),
        }

    def predict_history(self) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call fit(...) before predict_history().")

        cols: dict[str, pd.Series] = {}
        for name, fitted in self._fitted.items():
            X = self._designs[name].dropna(how="any")
            if X.empty:
                continue
            cols[name] = fitted.predict_path(X) * 100.0

        if not cols:
            return pd.DataFrame()

        df = pd.concat(cols.values(), axis=1, keys=cols.keys())
        df["ensemble"] = df.mean(axis=1, skipna=True)
        # Reorder so 'ensemble' is first.
        order = ["ensemble"] + [c for c in df.columns if c != "ensemble"]
        return df[order]

    def calibration_stats(self) -> dict:
        if not self._fitted or self._target is None:
            raise RuntimeError("Call fit(...) before calibration_stats().")
        history = self.predict_history()
        if history.empty or "ensemble" not in history.columns:
            return {"brier": float("nan"), "auc": float("nan"), "reliability_curve": pd.DataFrame()}

        pred = history["ensemble"] / 100.0
        truth = self._target.reindex(pred.index).astype(float)
        df = pd.concat([pred.rename("p"), truth.rename("y")], axis=1).dropna()

        brier = float(((df["p"] - df["y"]) ** 2).mean())
        auc = _auc(df["y"].values, df["p"].values)

        bins = np.linspace(0, 1, 11)
        df["bin"] = pd.cut(df["p"], bins=bins, include_lowest=True)
        rel = (
            df.groupby("bin", observed=True)
            .agg(predicted=("p", "mean"), actual=("y", "mean"), n=("y", "size"))
            .reset_index(drop=True)
            .dropna()
        )
        return {"brier": brier, "auc": auc, "reliability_curve": rel}


def _auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Mann-Whitney AUC. Returns NaN if only one class is present."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Rank-based AUC.
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    pos_ranks = ranks[: len(pos)]
    auc = (pos_ranks.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return float(auc)
