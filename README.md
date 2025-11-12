#!/usr/bin/env python3
"""
Trading pipeline for SMM248 - prediction and trading coursework.

Core idea and timing
--------------------
The coursework describes a logit of the form

    Y(i, t) = 1{ P(i,t) > P(i,t-1) }

with one-month lagged predictors X_j(i, t-1), Z_j(t-1). The trading rule then
uses the estimated model with information at time t to form positions over the
next month.

To both (i) mirror the brief literally and (ii) keep economic timing transparent,
this script constructs two consistent specifications from the same panel:

1. Backward + lagged specification (literal brief)
   ------------------------------------------------
   For each stock i and month t:

       ret_1m_arith(i, t) = P(i,t) / P(i,t-1) - 1
       ret_1m_log(i, t)   = log(P(i,t) / P(i,t-1))

       Y_back(i, t)       = 1{ ret_1m_arith(i, t) > 0 }

   Predictors are 1-month lagged:

       X_lag1(i, t) = X(i, t-1),   Z_lag1(t) = Z(t-1).

   This corresponds exactly to the PDF notation:
       Y(i,t) with predictors X(i,t-1), Z(t-1).

   We use this version for:
       - statsmodels Logit diagnostics (coef, p-values, pseudo R²)
       - a baseline per-stock classification backtest strictly aligned with the brief

2. Forward + contemporaneous specification (economically clean)
   ------------------------------------------------------------
   We re-index time so each row is the start of a 1-month holding period.
   For month t:

       ret_fwd_arith(i, t) = P(i, t+1) / P(i, t) - 1
       ret_fwd_log(i, t)   = log(P(i, t+1) / P(i, t))

       Y_fwd(i, t)         = 1{ ret_fwd_arith(i, t) > 0 }

   Predictors are measured at the start of the period:

       X(i, t), Z(t).

   We then estimate, for each stock i:

       P{ Y_fwd(i, t) = 1 | X(i, t), Z(t) },

   and use the predicted probability to generate trading positions and evaluate
   performance over the same horizon (t -> t+1):

       information at t  -> prediction about t->t+1 -> trade over t->t+1.

   Algebraically, this is equivalent to the brief notation after a one-period
   shift of the time index t' = t+1. Economically, it keeps the timing clear.

Main features
-------------
- compute_predictors_from_raw (optional):
    - if Data.xlsx contains raw variables instead of pre-engineered predictors,
      this helper constructs:
        * momentum_12m_excl1m from prices
        * gross_profitability from gross_profit / total_assets
        * sentiment_change from analyst_rating
        * yc_slope from long and short yields
        * m2_yoy from money_supply
      It is safe to skip if Data.xlsx already contains these columns.

- generate_summary_tables:
    - produces summary statistics and correlation matrix for returns and predictors

- prepare_data:
    - computes backward 1-month returns R_1m(i, t-1 -> t)
    - defines Y_back(i, t) = 1{R_1m(i, t-1 -> t) > 0}
    - constructs 1-month lagged predictors X_lag1(i, t) = X(i, t-1), Z_lag1(t) = Z(t-1)
    - computes forward returns R_fwd(i, t -> t+1)
    - defines Y_fwd(i, t) = 1{R_fwd(i, t -> t+1) > 0}
    - attaches both predictor sets as metadata:
        * predictor_cols_forward
        * predictor_cols_lagged

- fit_logistic:
    - scikit-learn pipeline (StandardScaler + LogisticRegression)

- fit_random_forest:
    - RandomForestClassifier as an AI alternative to logit

- compute_logit_diagnostics:
    - per-stock statsmodels Logit on any target (Y_back or Y_fwd) and any predictors
    - outputs coef, p-values, pseudo R²
    - builds cross-sectional summary (mean coef, mean p-value, share significant)

- expanding_window_backtest / rolling_window_backtest:
    - per-stock out-of-sample predictions for any target (Y_back or Y_fwd)
    - expanding vs rolling windows (no look-ahead)
    - configurable target and predictors

- expanding_window_backtest_rf:
    - per-stock out-of-sample predictions for Random Forest (expanding, forward spec)

- calculate_hit_rate:
    - per-stock hit rates
    - equal-weight overall hit rate (as in assignment formula)
    - pooled overall hit rate across all predictions

- compute_classification_metrics:
    - pooled precision, recall and F1-score for the positive class

- backtest_trading_strategy:
    - mapping probabilities to positions using tau_lower and tau_upper
    - equal-weight portfolio and benchmark from log or arithmetic returns
    - optional transaction costs based on turnover

- optimize_thresholds:
    - small grid search over (tau_lower, tau_upper) pairs
    - picks the pair that maximises Sharpe ratio of the strategy

- save_outputs:
    - saves predictions, performance and summary statistics to CSV
    - computes Sharpe ratios with explicit risk-free rate and frequency
    - computes skewness and max drawdown
    - creates performance plots and saves them as PNG

- run_full_pipeline:
    - end to end driver that takes a DataFrame and writes artefacts
    - calls generate_summary_tables
    - optional feature engineering via compute_predictors_from_raw
    - uses forward spec (Y_fwd, X(t)) for main predictive model and trading
    - runs statsmodels diagnostics for both backward+lagged and forward specs
    - compares expanding vs rolling and logit vs Random Forest (classification)
    - compares logit based vs RF based trading strategies
    - includes a baseline out-of-sample classification evaluation using
      Y_back with lagged predictors to demonstrate strict alignment with the brief
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
import statsmodels.api as sm

# ----------------------------------------------------------------------------
# Configuration / Logging
# ----------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=ConvergenceWarning)
np.random.seed(42)
logger.info("Random seed set to 42.")

# Columns expected in the engineered input data
REQUIRED_COLS = {
    "ticker",
    "date",
    "price",
    "momentum_12m_excl1m",
    "gross_profitability",
    "sentiment_change",
    "yc_slope",
    "m2_yoy",
}

# Predictor columns as supplied in the dataset (level variables at time t)
PREDICTOR_COLS = [
    "momentum_12m_excl1m",
    "gross_profitability",
    "sentiment_change",
    "yc_slope",
    "m2_yoy",
]

# ----------------------------------------------------------------------------
# STEP 0: OPTIONAL FEATURE ENGINEERING AND SUMMARY STATISTICS
# ----------------------------------------------------------------------------


def compute_predictors_from_raw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optional feature engineering step.

    If Data.xlsx already contains the five predictor columns:

        momentum_12m_excl1m
        gross_profitability
        sentiment_change
        yc_slope
        m2_yoy

    this function does nothing.

    If those columns are missing but raw variables are present, this function
    constructs them. This allows you to demonstrate in the report that you know
    how to build the predictors from raw data, as suggested by the coursework.

    Expected raw columns (if available):

        - price: for momentum
        - gross_profit, total_assets: for gross_profitability
        - analyst_rating: for sentiment_change (e.g. 1 = strong sell to 5 = strong buy)
        - yield_10y, yield_3m: for yc_slope
        - money_supply_m2: for m2_yoy

    All computations are done at monthly frequency per stock or at aggregate level
    where appropriate.
    """
    df = df.copy()

    # Momentum: 12-month trailing return excluding last month
    if "momentum_12m_excl1m" not in df.columns and "price" in df.columns:
        logger.info("Computing momentum_12m_excl1m from price.")
        df = df.sort_values(["ticker", "date"])
        # Monthly simple returns
        df["ret_1m_from_price"] = (
            df.groupby("ticker")["price"].pct_change().astype(float)
        )
        # 12-month trailing return excluding last month: product of months t-12 to t-2
        def _mom_12m_excl1m(x: pd.Series) -> pd.Series:
            # shift by 1 to drop last month, then rolling 11 (months t-12 to t-2)
            r = x.shift(1)
            return (1.0 + r).rolling(window=11, min_periods=11).apply(
                lambda arr: float(np.prod(arr) - 1.0), raw=False
            )

        df["momentum_12m_excl1m"] = df.groupby("ticker")["ret_1m_from_price"].apply(
            _mom_12m_excl1m
        )

    # Gross profitability: gross_profit / total_assets
    if (
        "gross_profitability" not in df.columns
        and "gross_profit" in df.columns
        and "total_assets" in df.columns
    ):
        logger.info("Computing gross_profitability from gross_profit and total_assets.")
        df["gross_profitability"] = df["gross_profit"] / df["total_assets"]

    # Sentiment change: first difference of analyst_rating
    if "sentiment_change" not in df.columns and "analyst_rating" in df.columns:
        logger.info("Computing sentiment_change from analyst_rating.")
        df = df.sort_values(["ticker", "date"])
        df["sentiment_change"] = df.groupby("ticker")["analyst_rating"].diff()

    # Yield curve slope: 10-year minus 3-month
    if (
        "yc_slope" not in df.columns
        and "yield_10y" in df.columns
        and "yield_3m" in df.columns
    ):
        logger.info("Computing yc_slope from yield_10y and yield_3m.")
        df["yc_slope"] = df["yield_10y"] - df["yield_3m"]

    # M2 YoY growth: 12 month percentage change in money_supply_m2
    if "m2_yoy" not in df.columns and "money_supply_m2" in df.columns:
        logger.info("Computing m2_yoy from money_supply_m2.")
        df = df.sort_values("date")
        df["m2_yoy"] = df["money_supply_m2"].pct_change(periods=12)

    return df


def generate_summary_tables(
    df: pd.DataFrame,
    out_dir: str = "outputs",
    prefix: str = "summary",
) -> Dict[str, str]:
    """
    Produce basic summary statistics and correlation matrix for the key variables.

    This addresses the coursework request for code that produces summary statistics.

    Outputs:
        - {prefix}_descriptive_{timestamp}.csv: mean, std, min, max, percentiles
        - {prefix}_corr_{timestamp}.csv: correlation matrix of predictors and returns
    """
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Variables for summary statistics
    cols_for_stats = []
    for col in [
        "ret_1m_arith",
        "ret_1m_log",
        "ret_fwd_arith",
        "ret_fwd_log",
    ] + PREDICTOR_COLS:
        if col in df.columns:
            cols_for_stats.append(col)

    if not cols_for_stats:
        logger.info("No variables available for summary statistics.")
        return {}

    stats_df = df[cols_for_stats].describe(
        percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
    ).T
    stats_path = os.path.join(out_dir, f"{prefix}_descriptive_{stamp}.csv")
    stats_df.to_csv(stats_path)

    corr_df = df[cols_for_stats].corr()
    corr_path = os.path.join(out_dir, f"{prefix}_corr_{stamp}.csv")
    corr_df.to_csv(corr_path)

    logger.info("Summary statistics saved: %s", stats_path)
    logger.info("Correlation matrix saved: %s", corr_path)

    return {"descriptive": stats_path, "correlation": corr_path}


# ----------------------------------------------------------------------------
# STEP 1: DATA PREPARATION
# ----------------------------------------------------------------------------


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare dataset for logistic regression and trading evaluation.

    Builds both the backward+lagged spec (literal brief) and the
    forward+contemporaneous spec (economically clean) from the same panel.

    Constructs
    ----------
    Backward returns and Y_back (brief spec):
        price_prev(i, t)    = P(i, t-1)
        ret_1m_arith(i, t)  = P(i, t) / P(i, t-1) - 1
        ret_1m_log(i, t)    = log(P(i, t) / P(i, t-1))
        Y_back(i, t)        = 1{ ret_1m_arith(i, t) > 0 }

        predictors_lag1(i, t) = predictors(i, t-1)

    Forward returns and Y_fwd (trading spec):
        price_next(i, t)    = P(i, t+1)
        ret_fwd_arith(i, t) = P(i, t+1) / P(i, t) - 1
        ret_fwd_log(i, t)   = log(P(i, t+1) / P(i, t))
        Y_fwd(i, t)         = 1{ ret_fwd_arith(i, t) > 0 }

        predictors_forward(i, t) = predictors(i, t)

    To have all these defined, we need at least t-1, t, t+1, so the first and
    last rows per stock are dropped automatically by the NaN filtering.
    """
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in input DataFrame: {missing}")

    df = df.copy()

    # Ensure date is datetime and sorted by ticker, date
    if not np.issubdtype(df["date"].dtype, np.datetime64):
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Backward-looking 1-month return: from t-1 to t
    df["price_prev"] = df.groupby("ticker")["price"].shift(1)
    df["ret_1m_log"] = np.log(df["price"] / df["price_prev"])
    df["ret_1m_arith"] = df["price"] / df["price_prev"] - 1
    df["Y_back"] = (df["ret_1m_arith"] > 0).astype(int)

    # Forward-looking 1-month return: from t to t+1
    df["price_next"] = df.groupby("ticker")["price"].shift(-1)
    df["ret_fwd_log"] = np.log(df["price_next"] / df["price"])
    df["ret_fwd_arith"] = df["price_next"] / df["price"] - 1
    df["Y_fwd"] = (df["ret_fwd_arith"] > 0).astype(int)

    # 1-month lagged predictors: X(i,t-1), Z(t-1)
    predictor_cols_lagged: List[str] = []
    for col in PREDICTOR_COLS:
        lag_col = f"{col}_lag1"
        df[lag_col] = df.groupby("ticker")[col].shift(1)
        predictor_cols_lagged.append(lag_col)

    # Require:
    #   - backward returns (for Y_back)
    #   - forward returns (for Y_fwd and trading)
    #   - predictors at time t (forward spec)
    #   - lagged predictors at time t (backward spec)
    df = df.dropna(
        subset=[
            "ret_1m_arith",
            "ret_1m_log",
            "ret_fwd_arith",
            "ret_fwd_log",
        ]
        + PREDICTOR_COLS
        + predictor_cols_lagged
    )

    # Attach predictor column names as metadata for downstream functions
    df.attrs["predictor_cols_forward"] = PREDICTOR_COLS
    df.attrs["predictor_cols_lagged"] = predictor_cols_lagged

    logger.info("Data prepared. Final shape: %s rows, %s columns", *df.shape)
    return df


# ----------------------------------------------------------------------------
# STEP 2: MODEL FITTING (sklearn)
# ----------------------------------------------------------------------------


def fit_logistic(
    X: pd.DataFrame,
    y: pd.Series,
    penalty: str = "l2",
    C: float = 1.0,
    solver: str = "liblinear",
    max_iter: int = 200,
):
    """Fit a scikit-learn LogisticRegression inside a scaling pipeline."""
    if y.isna().any():
        mask = ~y.isna()
        X = X.loc[mask]
        y = y.loc[mask]

    if len(y) == 0:
        logger.warning("Empty training target provided to fit_logistic")
        return None

    try:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                penalty=penalty,
                C=C,
                solver=solver,
                max_iter=max_iter,
            ),
        )
        model.fit(X, y)
        return model
    except Exception as exc:
        logger.warning("Logistic fit failed: %s", exc)
        return None


def fit_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int = 200,
    max_depth: Optional[int] = None,
    min_samples_leaf: int = 5,
    random_state: int = 42,
):
    """Fit a RandomForestClassifier as an AI alternative to logit."""
    if y.isna().any():
        mask = ~y.isna()
        X = X.loc[mask]
        y = y.loc[mask]

    if len(y) == 0:
        logger.warning("Empty training target provided to fit_random_forest")
        return None

    try:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
        model.fit(X, y)
        return model
    except Exception as exc:
        logger.warning("Random Forest fit failed: %s", exc)
        return None


# ----------------------------------------------------------------------------
# STEP 2b: MODEL DIAGNOSTICS (statsmodels Logit)
# ----------------------------------------------------------------------------


def compute_logit_diagnostics(
    df_clean: pd.DataFrame,
    predictor_cols: List[str],
    target_col: str,
    min_train_samples: int = 50,
    pvalue_threshold: float = 0.05,
    out_dir: str = "outputs",
    label: str = "",
) -> Dict[str, Optional[str]]:
    """Fit statsmodels Logit per stock and collect coef, p-values, pseudo R².

    Parameters
    ----------
    df_clean : DataFrame
        Prepared data.
    predictor_cols : list of str
        Columns to use as regressors.
    target_col : str
        Name of the binary dependent variable (e.g. 'Y_back' or 'Y_fwd').
    label : str
        Short label included in output filenames to distinguish specs
        (e.g. 'backward' vs 'forward').

    Returns
    -------
    dict with keys:
        - "detail": path to CSV with coef, pvalue, pseudo R² per ticker-variable
        - "summary": path to CSV with per-variable average coef and share significant
    (values can be None if no diagnostics were produced).
    """
    rows: List[Dict[str, object]] = []

    for ticker in df_clean["ticker"].unique():
        df_stock = df_clean[df_clean["ticker"] == ticker].copy()
        if len(df_stock) < min_train_samples or df_stock[target_col].nunique() < 2:
            logger.info(
                "%s: skipped diagnostics for '%s' (too few obs or no variation in %s)",
                ticker,
                label or target_col,
                target_col,
            )
            continue

        try:
            X = sm.add_constant(df_stock[predictor_cols])
            y = df_stock[target_col]
            model = sm.Logit(y, X).fit(disp=0)

            logger.info(
                "%s: statsmodels Logit (%s) pseudo R^2 = %.4f",
                ticker,
                label or target_col,
                model.prsquared,
            )

            for var in model.params.index:
                rows.append(
                    {
                        "ticker": ticker,
                        "spec": label or target_col,
                        "variable": var,
                        "coef": model.params[var],
                        "pvalue": model.pvalues[var],
                        "pseudo_r2": model.prsquared,
                    }
                )
        except Exception as exc:
            logger.warning(
                "%s: statsmodels Logit (%s) failed: %s",
                ticker,
                label or target_col,
                exc,
            )

    if not rows:
        logger.info("No logit diagnostics produced for spec '%s'.", label or target_col)
        return {"detail": None, "summary": None}

    diag_df = pd.DataFrame(rows)

    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    label_suffix = f"_{label}" if label else ""
    detail_path = os.path.join(
        out_dir, f"logit_diagnostics_detail{label_suffix}_{stamp}.csv"
    )
    diag_df.to_csv(detail_path, index=False)
    logger.info(
        "Statsmodels logit diagnostics (detail, %s) saved: %s",
        label or target_col,
        detail_path,
    )

    # Cross-sectional summary by predictor: average coef and share significant
    summary = (
        diag_df.assign(
            significant=lambda d: d["pvalue"] < pvalue_threshold,
        )
        .groupby(["spec", "variable"])
        .agg(
            mean_coef=("coef", "mean"),
            mean_pvalue=("pvalue", "mean"),
            share_significant=("significant", "mean"),
            mean_pseudo_r2=("pseudo_r2", "mean"),
        )
        .reset_index()
    )

    summary_path = os.path.join(
        out_dir, f"logit_diagnostics_summary{label_suffix}_{stamp}.csv"
    )
    summary.to_csv(summary_path, index=False)
    logger.info(
        "Statsmodels logit diagnostics (summary, %s) saved: %s",
        label or target_col,
        summary_path,
    )

    return {"detail": detail_path, "summary": summary_path}


# ----------------------------------------------------------------------------
# STEP 3: OUT-OF-SAMPLE - EXPANDING & ROLLING WINDOWS
# ----------------------------------------------------------------------------


def expanding_window_backtest(
    df: pd.DataFrame,
    ticker: str,
    predictor_cols: List[str],
    target_col: str = "Y_fwd",
    train_pct: float = 0.6,
    min_train_samples: int = 50,
    penalty: str = "l2",
    C: float = 1.0,
    solver: str = "liblinear",
    max_iter: int = 200,
) -> pd.DataFrame:
    """Generate expanding window out-of-sample predictions for one ticker (logit).

    Uses the fraction `train_pct` of that ticker's sample as the initial
    training size, then expands the window one observation at a time.
    By default uses the forward spec target 'Y_fwd'.
    """
    df_stock = df[df["ticker"] == ticker].copy().reset_index(drop=True)
    n_total = len(df_stock)
    n_train_init = int(n_total * train_pct)

    records: List[Dict[str, object]] = []
    if n_total <= n_train_init:
        return pd.DataFrame(records)

    for i in range(n_train_init, n_total):
        df_train = df_stock.iloc[:i].copy()
        df_test = df_stock.iloc[i : i + 1].copy()

        record: Dict[str, object] = {
            "ticker": ticker,
            "date": df_test["date"].values[0],
            "Y_actual": int(df_test[target_col].values[0]),
            # forward returns from t to t+1 for trading
            "ret_fwd_arith": df_test["ret_fwd_arith"].values[0],
            "ret_fwd_log": df_test["ret_fwd_log"].values[0],
        }

        if len(df_train) < min_train_samples or df_train[target_col].nunique() < 2:
            record.update({"Y_pred_prob": np.nan, "Y_pred_binary": np.nan})
            records.append(record)
            continue

        X_train = df_train[predictor_cols]
        y_train = df_train[target_col]
        X_test = df_test[predictor_cols]

        model = fit_logistic(
            X_train, y_train, penalty=penalty, C=C, solver=solver, max_iter=max_iter
        )

        if model is None:
            record.update({"Y_pred_prob": np.nan, "Y_pred_binary": np.nan})
        else:
            try:
                prob = float(model.predict_proba(X_test)[0, 1])
                record.update(
                    {
                        "Y_pred_prob": prob,
                        "Y_pred_binary": int(prob > 0.5),
                    }
                )
            except Exception as exc:
                logger.warning("Prediction error for %s at i=%s: %s", ticker, i, exc)
                record.update({"Y_pred_prob": np.nan, "Y_pred_binary": np.nan})

        records.append(record)

    return pd.DataFrame(records)


def rolling_window_backtest(
    df: pd.DataFrame,
    ticker: str,
    predictor_cols: List[str],
    target_col: str = "Y_fwd",
    train_pct: float = 0.6,
    min_train_samples: int = 50,
    penalty: str = "l2",
    C: float = 1.0,
    solver: str = "liblinear",
    max_iter: int = 200,
) -> pd.DataFrame:
    """Generate rolling window out-of-sample predictions for one ticker (logit).

    Training window has constant length:
        window = max(min_train_samples, int(train_pct * n_total))
    and rolls forward one observation at a time.
    By default uses the forward spec target 'Y_fwd'.
    """
    df_stock = df[df["ticker"] == ticker].copy().reset_index(drop=True)
    n_total = len(df_stock)
    window = max(min_train_samples, int(n_total * train_pct))

    records: List[Dict[str, object]] = []
    if n_total <= window:
        return pd.DataFrame(records)

    for i in range(window, n_total):
        df_train = df_stock.iloc[i - window : i].copy()
        df_test = df_stock.iloc[i : i + 1].copy()

        record: Dict[str, object] = {
            "ticker": ticker,
            "date": df_test["date"].values[0],
            "Y_actual": int(df_test[target_col].values[0]),
            "ret_fwd_arith": df_test["ret_fwd_arith"].values[0],
            "ret_fwd_log": df_test["ret_fwd_log"].values[0],
        }

        if len(df_train) < min_train_samples or df_train[target_col].nunique() < 2:
            record.update({"Y_pred_prob": np.nan, "Y_pred_binary": np.nan})
            records.append(record)
            continue

        X_train = df_train[predictor_cols]
        y_train = df_train[target_col]
        X_test = df_test[predictor_cols]

        model = fit_logistic(
            X_train, y_train, penalty=penalty, C=C, solver=solver, max_iter=max_iter
        )

        if model is None:
            record.update({"Y_pred_prob": np.nan, "Y_pred_binary": np.nan})
        else:
            try:
                prob = float(model.predict_proba(X_test)[0, 1])
                record.update(
                    {
                        "Y_pred_prob": prob,
                        "Y_pred_binary": int(prob > 0.5),
                    }
                )
            except Exception as exc:
                logger.warning("Prediction error for %s at i=%s: %s", ticker, i, exc)
                record.update({"Y_pred_prob": np.nan, "Y_pred_binary": np.nan})

        records.append(record)

    return pd.DataFrame(records)


def expanding_window_backtest_rf(
    df: pd.DataFrame,
    ticker: str,
    predictor_cols: List[str],
    target_col: str = "Y_fwd",
    train_pct: float = 0.6,
    min_train_samples: int = 50,
) -> pd.DataFrame:
    """Generate expanding window out-of-sample predictions for one ticker (Random Forest).

    Used for AI accuracy comparison vs logit. Defaults to forward spec (Y_fwd).
    """
    df_stock = df[df["ticker"] == ticker].copy().reset_index(drop=True)
    n_total = len(df_stock)
    n_train_init = int(n_total * train_pct)

    records: List[Dict[str, object]] = []
    if n_total <= n_train_init:
        return pd.DataFrame(records)

    for i in range(n_train_init, n_total):
        df_train = df_stock.iloc[:i].copy()
        df_test = df_stock.iloc[i : i + 1].copy()

        record: Dict[str, object] = {
            "ticker": ticker,
            "date": df_test["date"].values[0],
            "Y_actual": int(df_test[target_col].values[0]),
        }

        if len(df_train) < min_train_samples or df_train[target_col].nunique() < 2:
            record.update({"Y_pred_prob": np.nan, "Y_pred_binary": np.nan})
            records.append(record)
            continue

        X_train = df_train[predictor_cols]
        y_train = df_train[target_col]
        X_test = df_test[predictor_cols]

        model = fit_random_forest(X_train, y_train)

        if model is None:
            record.update({"Y_pred_prob": np.nan, "Y_pred_binary": np.nan})
        else:
            try:
                prob = float(model.predict_proba(X_test)[0, 1])
                record.update(
                    {
                        "Y_pred_prob": prob,
                        "Y_pred_binary": int(prob > 0.5),
                    }
                )
            except Exception as exc:
                logger.warning(
                    "Random Forest prediction error for %s at i=%s: %s", ticker, i, exc
                )
                record.update({"Y_pred_prob": np.nan, "Y_pred_binary": np.nan})

        records.append(record)

    return pd.DataFrame(records)


# ----------------------------------------------------------------------------
# STEP 4: HIT RATE AND METRICS
# ----------------------------------------------------------------------------


def calculate_hit_rate(
    df_pred: pd.DataFrame,
    actual_col: str = "Y_actual",
    pred_col: str = "Y_pred_binary",
) -> Dict[str, float]:
    """Calculate hit rates.

    Returns a dict with:
      - 'overall': equal-weighted average of per-ticker hit rates
      - 'overall_pooled': pooled hit rate across all predictions
      - one entry per ticker with its own hit rate

    NaN predictions are ignored in all calculations.
    """
    results: Dict[str, float] = {}

    # Pooled hit rate across all predictions
    df_valid = df_pred.dropna(subset=[actual_col, pred_col])
    if len(df_valid) > 0:
        results["overall_pooled"] = (
            df_valid[actual_col] == df_valid[pred_col]
        ).mean()
    else:
        results["overall_pooled"] = np.nan

    ticker_hits: List[float] = []
    for ticker in df_pred["ticker"].unique():
        tdf = df_pred[df_pred["ticker"] == ticker].dropna(
            subset=[actual_col, pred_col]
        )
        if len(tdf) > 0:
            hit = (tdf[actual_col] == tdf[pred_col]).mean()
            results[ticker] = hit
            ticker_hits.append(hit)
        else:
            results[ticker] = np.nan

    results["overall"] = float(np.mean(ticker_hits)) if ticker_hits else np.nan

    return results


def compute_classification_metrics(
    df_pred: pd.DataFrame,
    actual_col: str = "Y_actual",
    pred_col: str = "Y_pred_binary",
) -> Dict[str, float]:
    """Compute pooled precision, recall and F1-score for the positive class.

    Addresses the note that hit rate alone does not distinguish false positives
    vs false negatives.
    """
    df_valid = df_pred.dropna(subset=[actual_col, pred_col])
    if len(df_valid) == 0:
        return {
            "precision_pos": np.nan,
            "recall_pos": np.nan,
            "f1_pos": np.nan,
        }

    y_true = df_valid[actual_col].astype(int)
    y_pred = df_valid[pred_col].astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        "precision_pos": float(precision),
        "recall_pos": float(recall),
        "f1_pos": float(f1),
    }


# ----------------------------------------------------------------------------
# STEP 5: TRADING STRATEGY BACKTEST
# ----------------------------------------------------------------------------


def backtest_trading_strategy(
    df_pred: pd.DataFrame,
    tau_lower: float = 0.4,
    tau_upper: float = 0.6,
    return_type: str = "log",
    transaction_cost: float = 0.0,
) -> pd.DataFrame:
    """Generate portfolio and benchmark returns from probabilistic signals.

    Positions:
        +1 if Y_pred_prob > tau_upper
        -1 if Y_pred_prob < tau_lower
         0 otherwise.

    Returns are aggregated cross-sectionally by simple average.

    transaction_cost:
        cost per unit change in position, applied to
        absolute change |w(i,t) - w(i,t-1)|. Set to 0.0 for no costs.
    """
    strat = df_pred.copy().sort_values(["date", "ticker"])

    strat["position"] = 0
    strat.loc[strat["Y_pred_prob"] > tau_upper, "position"] = 1
    strat.loc[strat["Y_pred_prob"] < tau_lower, "position"] = -1

    # Transaction costs: per-stock position changes
    strat["position_prev"] = strat.groupby("ticker")["position"].shift(1).fillna(0)
    strat["turnover"] = (strat["position"] - strat["position_prev"]).abs()
    strat["trading_cost"] = transaction_cost * strat["turnover"]

    ret_col = "ret_fwd_arith" if return_type == "arith" else "ret_fwd_log"
    strat["gross_strategy_return"] = strat["position"] * strat[ret_col]
    # Net strategy return after transaction costs
    strat["strategy_return"] = strat["gross_strategy_return"] - strat["trading_cost"]

    portfolio = (
        strat.groupby("date")
        .agg(
            portfolio_return=("strategy_return", "mean"),
            benchmark_return=(ret_col, "mean"),
        )
        .reset_index()
    )

    return portfolio


def optimize_thresholds(
    df_oos: pd.DataFrame,
    taus: List[Tuple[float, float]],
    return_type: str = "log",
    transaction_cost: float = 0.0,
    risk_free_annual: float = 0.0,
    freq: int = 12,
) -> Tuple[Tuple[float, float], float, Optional[pd.DataFrame]]:
    """Grid search over threshold pairs to find the one maximising Sharpe ratio."""
    if df_oos.empty:
        logger.warning("optimize_thresholds: df_oos is empty; cannot optimise.")
        return (0.4, 0.6), np.nan, None

    rf_period = (1.0 + risk_free_annual) ** (1.0 / freq) - 1.0
    best_sharpe = -np.inf
    best_tau = (0.4, 0.6)  # default
    best_perf: Optional[pd.DataFrame] = None

    for tau_l, tau_u in taus:
        perf = backtest_trading_strategy(
            df_oos,
            tau_lower=tau_l,
            tau_upper=tau_u,
            return_type=return_type,
            transaction_cost=transaction_cost,
        )
        mean_r = perf["portfolio_return"].mean()
        std_r = perf["portfolio_return"].std()
        sharpe = (mean_r - rf_period) / std_r if std_r > 0 else np.nan
        logger.info(
            "Thresholds (%.2f, %.2f): Sharpe = %s",
            tau_l,
            tau_u,
            f"{sharpe:.4f}" if not np.isnan(sharpe) else "nan",
        )
        if not np.isnan(sharpe) and sharpe > best_sharpe:
            best_sharpe = sharpe
            best_tau = (tau_l, tau_u)
            best_perf = perf

    logger.info(
        "Best thresholds: (%.2f, %.2f) with Sharpe = %s",
        best_tau[0],
        best_tau[1],
        f"{best_sharpe:.4f}" if not np.isnan(best_sharpe) else "nan",
    )
    return best_tau, best_sharpe, best_perf


# ----------------------------------------------------------------------------
# STEP 6: SAVE RESULTS AND PLOTS
# ----------------------------------------------------------------------------


def save_outputs(
    df_oos_predictions: pd.DataFrame,
    portfolio_performance: pd.DataFrame,
    hit_rates_oos: Dict[str, float],
    out_dir: str = "outputs",
    return_type: str = "log",
    tickers: Optional[List[str]] = None,
    skip_plots: bool = False,
    risk_free_annual: float = 0.0,
    freq: int = 12,
    prefix: str = "",
) -> Dict[str, str]:
    """Persist artefacts (CSV and plots) and return the generated file paths.

    `prefix` is a short label (e.g. 'logit' or 'rf') to distinguish filenames
    when saving multiple strategies.
    """
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix_str = f"{prefix}_" if prefix else ""

    oos_file = os.path.join(out_dir, f"{prefix_str}predictions_oos_{stamp}.csv")
    perf_file = os.path.join(out_dir, f"{prefix_str}portfolio_performance_{stamp}.csv")
    summary_file = os.path.join(out_dir, f"{prefix_str}summary_statistics_{stamp}.csv")
    plot_file = os.path.join(out_dir, f"{prefix_str}performance_analysis_{stamp}.png")

    df_oos_predictions.to_csv(oos_file, index=False)
    portfolio_performance.to_csv(perf_file, index=False)

    mean_col = "portfolio_return"
    mean_return = portfolio_performance[mean_col].mean()
    std_return = portfolio_performance[mean_col].std()

    bench_col = "benchmark_return"
    mean_benchmark = portfolio_performance[bench_col].mean()
    std_benchmark = portfolio_performance[bench_col].std()

    # Convert annual risk free to per period
    rf_period = (1.0 + risk_free_annual) ** (1.0 / freq) - 1.0

    sharpe_ratio = (
        (mean_return - rf_period) / std_return if std_return > 0 else np.nan
    )
    sharpe_benchmark = (
        (mean_benchmark - rf_period) / std_benchmark
        if std_benchmark > 0
        else np.nan
    )

    # Additional stats: skewness and max drawdown of strategy
    strat_returns = portfolio_performance["portfolio_return"].dropna()
    skewness = strat_returns.skew() if len(strat_returns) > 1 else np.nan

    # Max drawdown based on cumulative returns
    portfolio = portfolio_performance.copy()
    if return_type == "arith":
        portfolio["cum_return_strategy"] = (1 + portfolio["portfolio_return"]).cumprod()
        portfolio["cum_return_benchmark"] = (
            (1 + portfolio["benchmark_return"]).cumprod()
        )
    else:
        portfolio["cum_return_strategy"] = np.exp(
            portfolio["portfolio_return"].cumsum()
        )
        portfolio["cum_return_benchmark"] = np.exp(
            portfolio["benchmark_return"].cumsum()
        )

    cum = portfolio["cum_return_strategy"]
    running_max = cum.cummax()
    drawdown = cum / running_max - 1.0
    max_drawdown = drawdown.min() if len(drawdown) > 0 else np.nan

    summary_stats = pd.DataFrame(
        {
            "Metric": [
                "Mean Return",
                "Std Deviation",
                "Sharpe Ratio",
                "Hit Rate (OOS equal weight)",
                "Hit Rate (OOS pooled)",
                "Annualized Return (approx.)",
                "Annualized Volatility",
                "Skewness (strategy)",
                "Max Drawdown (strategy)",
            ],
            "Strategy": [
                mean_return,
                std_return,
                sharpe_ratio,
                hit_rates_oos.get("overall", np.nan),
                hit_rates_oos.get("overall_pooled", np.nan),
                mean_return * freq,          # linear approx for annualisation
                std_return * np.sqrt(freq),
                skewness,
                max_drawdown,
            ],
            "Benchmark": [
                mean_benchmark,
                std_benchmark,
                sharpe_benchmark,
                0.5,  # random classifier benchmark
                0.5,
                mean_benchmark * freq,
                std_benchmark * np.sqrt(freq),
                np.nan,
                np.nan,
            ],
        }
    )
    summary_stats.to_csv(summary_file, index=False)

    artefacts = {
        "predictions": oos_file,
        "performance": perf_file,
        "summary": summary_file,
    }

    logger.info("Files saved (%s): %s", prefix or "strategy", ", ".join(artefacts.values()))

    if skip_plots:
        logger.info("Skipping plots as requested.")
        return artefacts

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        f"Trading Strategy Performance Analysis ({prefix or 'strategy'})",
        fontsize=16,
        fontweight="bold",
    )

    # Panel 1: cumulative returns
    axes[0, 0].plot(
        portfolio["date"],
        portfolio["cum_return_strategy"],
        label="Strategy",
        lw=2,
    )
    axes[0, 0].plot(
        portfolio["date"],
        portfolio["cum_return_benchmark"],
        label="Benchmark",
        lw=2,
        ls="--",
    )
    axes[0, 0].set_title("Cumulative Returns")
    axes[0, 0].set_ylabel("Cumulative Return")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Panel 2: distribution of period returns
    axes[0, 1].hist(
        portfolio["portfolio_return"].dropna(),
        bins=30,
        alpha=0.7,
        edgecolor="black",
    )
    axes[0, 1].axvline(mean_return, ls="--", lw=2, label=f"Mean: {mean_return:.3%}")
    axes[0, 1].set_title("Distribution of Period Returns")
    axes[0, 1].set_xlabel("Return")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Panel 3: hit rates by stock
    tickers_for_plot = tickers or []
    tickers_for_plot = [t for t in tickers_for_plot if t in hit_rates_oos]
    if tickers_for_plot:
        hit_rate_data = pd.DataFrame(
            {
                "Ticker": tickers_for_plot,
                "Hit_Rate": [hit_rates_oos[t] for t in tickers_for_plot],
            }
        )
        axes[1, 0].bar(
            hit_rate_data["Ticker"],
            hit_rate_data["Hit_Rate"],
            edgecolor="black",
        )
        axes[1, 0].axhline(0.5, linestyle="--", linewidth=2, label="Random (50 percent)")
        axes[1, 0].set_title("Out-of-Sample Hit Rate by Stock")
        axes[1, 0].set_ylabel("Hit Rate")
        axes[1, 0].grid(True, axis="y", alpha=0.3)
        plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45, ha="right")
        axes[1, 0].legend()
    else:
        axes[1, 0].set_visible(False)

    # Panel 4: rolling Sharpe ratio
    rolling_mean = portfolio["portfolio_return"].rolling(freq).mean()
    rolling_std = portfolio["portfolio_return"].rolling(freq).std()
    rolling_sharpe = (rolling_mean - rf_period) / rolling_std
    axes[1, 1].plot(portfolio["date"], rolling_sharpe, lw=2)
    axes[1, 1].axhline(0, ls="--", lw=1)
    axes[1, 1].set_title(f"Rolling {freq}-Period Sharpe Ratio")
    axes[1, 1].set_ylabel("Sharpe Ratio")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Plot saved (%s): %s", prefix or "strategy", plot_file)

    artefacts["plot"] = plot_file
    return artefacts


# ----------------------------------------------------------------------------
# STEP 7: FULL PIPELINE
# ----------------------------------------------------------------------------


def run_full_pipeline(
    df: pd.DataFrame,
    train_pct: float = 0.6,
    tau_lower: float = 0.4,
    tau_upper: float = 0.6,
    return_type: str = "log",
    out_dir: str = "outputs",
    min_train_samples: int = 50,
    penalty: str = "l2",
    C: float = 1.0,
    solver: str = "liblinear",
    max_iter: int = 200,
    skip_plots: bool = False,
    risk_free_annual: float = 0.0,
    freq: int = 12,
    oos_mode: str = "expanding",  # "expanding" or "rolling"
    transaction_cost: float = 0.0,
    optimize_taus: bool = True,
    assume_engineered_predictors: bool = True,
) -> Dict[str, str]:
    """
    Run the complete analysis pipeline with configurable options.

    Steps:
        0. Optional feature engineering and summary statistics
        1. Data preparation and construction of Y_back and Y_fwd
        1b. Logit diagnostics with statsmodels for backward and forward specs
        2. In-sample logit estimation (forward spec)
        3. Out-of-sample logit (forward spec, rolling or expanding)
        3b. Out-of-sample Random Forest (forward spec)
        3c. Baseline out-of-sample classification for Y_back with lagged predictors
            to demonstrate strict alignment with the coursework notation
        4. Trading strategy backtest using logit and RF signals (forward spec)
        5. Saving all artefacts
    """
    os.makedirs(out_dir, exist_ok=True)

    logger.info("STEP 0: OPTIONAL FEATURE ENGINEERING")
    if not assume_engineered_predictors:
        df = compute_predictors_from_raw(df)

    logger.info("STEP 1: DATA PREPARATION")
    df_clean = prepare_data(df)
    predictor_cols_forward: List[str] = df_clean.attrs.get(
        "predictor_cols_forward", PREDICTOR_COLS
    )
    predictor_cols_lagged: List[str] = df_clean.attrs.get(
        "predictor_cols_lagged", [f"{c}_lag1" for c in predictor_cols_forward]
    )

    tickers = df_clean["ticker"].unique()
    logger.info("Number of stocks: %s", len(tickers))
    logger.info(
        "Date range: %s to %s",
        df_clean["date"].min().date(),
        df_clean["date"].max().date(),
    )
    logger.info("Total observations after cleaning: %s", len(df_clean))

    # Summary statistics based on prepared data (includes returns and predictors)
    logger.info("STEP 1a: SUMMARY STATISTICS")
    summary_paths = generate_summary_tables(df_clean, out_dir=out_dir, prefix="data")

    # Statsmodels diagnostics for both specs
    logger.info("STEP 1b: LOGIT DIAGNOSTICS WITH STATSMODELS (backward + lagged)")
    diag_back = compute_logit_diagnostics(
        df_clean,
        predictor_cols=predictor_cols_lagged,
        target_col="Y_back",
        min_train_samples=min_train_samples,
        out_dir=out_dir,
        label="backward_lagged",
    )

    logger.info("STEP 1c: LOGIT DIAGNOSTICS WITH STATSMODELS (forward + contemporaneous)")
    diag_fwd = compute_logit_diagnostics(
        df_clean,
        predictor_cols=predictor_cols_forward,
        target_col="Y_fwd",
        min_train_samples=min_train_samples,
        out_dir=out_dir,
        label="forward_contemporaneous",
    )

    # In-sample logit fit (forward spec for trading)
    logger.info("STEP 2: IN-SAMPLE MODEL ESTIMATION (SKLEARN - LOGIT, forward spec)")
    insample_list: List[pd.DataFrame] = []
    for ticker in tickers:
        df_stock = df_clean[df_clean["ticker"] == ticker].copy()
        n_stock = len(df_stock)

        probs = np.full(n_stock, np.nan, dtype=float)
        yhat_binary = np.full(n_stock, np.nan, dtype=float)
        model = None

        y_series = df_stock["Y_fwd"]
        if n_stock >= min_train_samples and y_series.nunique() >= 2:
            model = fit_logistic(
                df_stock[predictor_cols_forward],
                y_series,
                penalty=penalty,
                C=C,
                solver=solver,
                max_iter=max_iter,
            )
            if model is not None:
                probs = model.predict_proba(df_stock[predictor_cols_forward])[:, 1]
                yhat_binary = (probs > 0.5).astype(float)
                dump(model, os.path.join(out_dir, f"model_logit_forward_{ticker}.joblib"))

        df_stock["Y_fwd"] = df_stock["Y_fwd"].astype(int)
        df_stock["Y_pred_prob_insample"] = probs
        df_stock["Y_pred_binary_insample"] = yhat_binary

        insample_list.append(
            df_stock[
                [
                    "ticker",
                    "date",
                    "Y_fwd",
                    "Y_pred_prob_insample",
                    "Y_pred_binary_insample",
                ]
            ]
        )

        logger.info(
            "%s: In-sample forward-spec logit model %s",
            ticker,
            "fitted" if model is not None else "skipped",
        )

    df_insample = (
        pd.concat(insample_list, ignore_index=True) if insample_list else pd.DataFrame()
    )
    df_insample_for_hr = df_insample.rename(
        columns={"Y_fwd": "Y_actual", "Y_pred_binary_insample": "Y_pred_binary"}
    )
    hit_rates_insample = calculate_hit_rate(
        df_insample_for_hr,
        actual_col="Y_actual",
        pred_col="Y_pred_binary",
    )
    metrics_insample = compute_classification_metrics(
        df_insample_for_hr,
        actual_col="Y_actual",
        pred_col="Y_pred_binary",
    )
    logger.info(
        "In-sample overall hit rate (equal weight across stocks): %s",
        f"{hit_rates_insample.get('overall', np.nan):.2%}"
        if hit_rates_insample
        else "nan",
    )
    logger.info(
        "In-sample overall pooled hit rate: %s",
        f"{hit_rates_insample.get('overall_pooled', np.nan):.2%}"
        if hit_rates_insample
        else "nan",
    )
    logger.info(
        "In-sample precision/recall/F1 (positive): %.3f / %.3f / %.3f",
        metrics_insample["precision_pos"],
        metrics_insample["recall_pos"],
        metrics_insample["f1_pos"],
    )

    # Out-of-sample logistic (forward spec, rolling or expanding)
    logger.info("STEP 3: OUT-OF-SAMPLE (LOGIT, %s WINDOW, forward spec) EVALUATION", oos_mode.upper())
    oos_frames: List[pd.DataFrame] = []
    for ticker in tickers:
        if oos_mode.lower() == "rolling":
            df_pred = rolling_window_backtest(
                df_clean,
                ticker,
                predictor_cols=predictor_cols_forward,
                target_col="Y_fwd",
                train_pct=train_pct,
                min_train_samples=min_train_samples,
                penalty=penalty,
                C=C,
                solver=solver,
                max_iter=max_iter,
            )
        else:
            df_pred = expanding_window_backtest(
                df_clean,
                ticker,
                predictor_cols=predictor_cols_forward,
                target_col="Y_fwd",
                train_pct=train_pct,
                min_train_samples=min_train_samples,
                penalty=penalty,
                C=C,
                solver=solver,
                max_iter=max_iter,
            )
        oos_frames.append(df_pred)
        logger.info("%s: %s out-of-sample rows (logit, forward)", ticker, len(df_pred))

    df_oos = pd.concat(oos_frames, ignore_index=True) if oos_frames else pd.DataFrame()
    hit_rates_oos = calculate_hit_rate(
        df_oos,
        actual_col="Y_actual",
        pred_col="Y_pred_binary",
    )
    metrics_oos_logit = compute_classification_metrics(
        df_oos,
        actual_col="Y_actual",
        pred_col="Y_pred_binary",
    )
    logger.info(
        "Out-of-sample (logit, forward) overall hit rate (equal weight): %s",
        f"{hit_rates_oos.get('overall', np.nan):.2%}" if hit_rates_oos else "nan",
    )
    logger.info(
        "Out-of-sample (logit, forward) overall pooled hit rate: %s",
        f"{hit_rates_oos.get('overall_pooled', np.nan):.2%}"
        if hit_rates_oos
        else "nan",
    )
    logger.info(
        "Out-of-sample (logit, forward) precision/recall/F1 (positive): %.3f / %.3f / %.3f",
        metrics_oos_logit["precision_pos"],
        metrics_oos_logit["recall_pos"],
        metrics_oos_logit["f1_pos"],
    )

    # Out-of-sample Random Forest (expanding window, forward spec), AI comparison
    logger.info("STEP 3b: OUT-OF-SAMPLE (RANDOM FOREST, EXPANDING, forward spec) EVALUATION")
    oos_frames_rf: List[pd.DataFrame] = []
    for ticker in tickers:
        df_pred_rf = expanding_window_backtest_rf(
            df_clean,
            ticker,
            predictor_cols=predictor_cols_forward,
            target_col="Y_fwd",
            train_pct=train_pct,
            min_train_samples=min_train_samples,
        )
        oos_frames_rf.append(df_pred_rf)
        logger.info("%s: %s out-of-sample rows (Random Forest, forward)", ticker, len(df_pred_rf))

    df_oos_rf = (
        pd.concat(oos_frames_rf, ignore_index=True) if oos_frames_rf else pd.DataFrame()
    )
    hit_rates_oos_rf = calculate_hit_rate(
        df_oos_rf,
        actual_col="Y_actual",
        pred_col="Y_pred_binary",
    )
    metrics_oos_rf = compute_classification_metrics(
        df_oos_rf,
        actual_col="Y_actual",
        pred_col="Y_pred_binary",
    )
    logger.info(
        "Out-of-sample (RF, forward) overall hit rate (equal weight): %s",
        f"{hit_rates_oos_rf.get('overall', np.nan):.2%}" if hit_rates_oos_rf else "nan",
    )
    logger.info(
        "Out-of-sample (RF, forward) overall pooled hit rate: %s",
        f"{hit_rates_oos_rf.get('overall_pooled', np.nan):.2%}"
        if hit_rates_oos_rf
        else "nan",
    )
    logger.info(
        "Out-of-sample (RF, forward) precision/recall/F1 (positive): %.3f / %.3f / %.3f",
        metrics_oos_rf["precision_pos"],
        metrics_oos_rf["recall_pos"],
        metrics_oos_rf["f1_pos"],
    )

    # Baseline: out-of-sample classification for backward spec (Y_back, lagged predictors)
    logger.info("STEP 3c: BASELINE OUT-OF-SAMPLE CLASSIFICATION (LOGIT, BACKWARD + LAGGED)")
    oos_frames_back: List[pd.DataFrame] = []
    for ticker in tickers:
        df_pred_back = expanding_window_backtest(
            df_clean,
            ticker,
            predictor_cols=predictor_cols_lagged,
            target_col="Y_back",
            train_pct=train_pct,
            min_train_samples=min_train_samples,
            penalty=penalty,
            C=C,
            solver=solver,
            max_iter=max_iter,
        )
        oos_frames_back.append(df_pred_back)
        logger.info(
            "%s: %s out-of-sample rows (logit, backward spec)", ticker, len(df_pred_back)
        )

    df_oos_back = (
        pd.concat(oos_frames_back, ignore_index=True) if oos_frames_back else pd.DataFrame()
    )
    hit_rates_oos_back = calculate_hit_rate(
        df_oos_back,
        actual_col="Y_actual",
        pred_col="Y_pred_binary",
    )
    metrics_oos_back = compute_classification_metrics(
        df_oos_back,
        actual_col="Y_actual",
        pred_col="Y_pred_binary",
    )
    logger.info(
        "Out-of-sample (logit, backward spec) overall hit rate (equal weight): %s",
        f"{hit_rates_oos_back.get('overall', np.nan):.2%}" if hit_rates_oos_back else "nan",
    )
    logger.info(
        "Out-of-sample (logit, backward spec) overall pooled hit rate: %s",
        f"{hit_rates_oos_back.get('overall_pooled', np.nan):.2%}"
        if hit_rates_oos_back
        else "nan",
    )
    logger.info(
        "Out-of-sample (logit, backward spec) precision/recall/F1 (positive): %.3f / %.3f / %.3f",
        metrics_oos_back["precision_pos"],
        metrics_oos_back["recall_pos"],
        metrics_oos_back["f1_pos"],
    )

    # Trading strategy backtest using logit OOS signals (forward spec)
    logger.info("STEP 4: TRADING STRATEGY BACKTEST (LOGIT SIGNALS, forward spec)")
    if optimize_taus:
        tau_grid = [
            (0.2, 0.8),
            (0.25, 0.75),
            (0.3, 0.7),
            (0.35, 0.65),
            (0.4, 0.6),
            (0.45, 0.55),
        ]
        best_tau, best_sharpe, portfolio_performance = optimize_thresholds(
            df_oos,
            taus=tau_grid,
            return_type=return_type,
            transaction_cost=transaction_cost,
            risk_free_annual=risk_free_annual,
            freq=freq,
        )
        logger.info(
            "Using optimised thresholds for logit strategy: tau_lower=%.2f, tau_upper=%.2f, Sharpe=%.4f",
            best_tau[0],
            best_tau[1],
            best_sharpe if not np.isnan(best_sharpe) else float("nan"),
        )
        tau_lower_used, tau_upper_used = best_tau
    else:
        portfolio_performance = backtest_trading_strategy(
            df_oos,
            tau_lower=tau_lower,
            tau_upper=tau_upper,
            return_type=return_type,
            transaction_cost=transaction_cost,
        )
        tau_lower_used, tau_upper_used = tau_lower, tau_upper

    # Save artefacts and plots for logit strategy
    logger.info("STEP 5: SAVING RESULTS AND PLOTS (LOGIT STRATEGY)")
    artefacts = save_outputs(
        df_oos,
        portfolio_performance,
        hit_rates_oos,
        out_dir=out_dir,
        return_type=return_type,
        tickers=list(tickers),
        skip_plots=skip_plots,
        risk_free_annual=risk_free_annual,
        freq=freq,
        prefix="logit",
    )

    # Trading strategy backtest using RF OOS signals (forward spec)
    logger.info("STEP 4b: TRADING STRATEGY BACKTEST (RANDOM FOREST SIGNALS, forward spec)")
    if not df_oos_rf.empty:
        portfolio_performance_rf = backtest_trading_strategy(
            df_oos_rf,
            tau_lower=tau_lower_used,
            tau_upper=tau_upper_used,
            return_type=return_type,
            transaction_cost=transaction_cost,
        )
        logger.info(
            "RF strategy evaluated with tau_lower=%.2f, tau_upper=%.2f.",
            tau_lower_used,
            tau_upper_used,
        )
        artefacts_rf = save_outputs(
            df_oos_rf,
            portfolio_performance_rf,
            hit_rates_oos_rf,
            out_dir=out_dir,
            return_type=return_type,
            tickers=list(tickers),
            skip_plots=skip_plots,
            risk_free_annual=risk_free_annual,
            freq=freq,
            prefix="rf",
        )
        # Merge RF artefacts into main dict (with prefixes)
        artefacts.update(
            {
                "rf_predictions": artefacts_rf["predictions"],
                "rf_performance": artefacts_rf["performance"],
                "rf_summary": artefacts_rf["summary"],
                "rf_plot": artefacts_rf.get("plot", ""),
            }
        )
    else:
        logger.info("RF OOS dataframe is empty; skipping RF trading backtest.")

    # Add diagnostic paths and summary stats paths, if any
    if diag_back["detail"] is not None:
        artefacts["logit_diag_backward_detail"] = diag_back["detail"]
    if diag_back["summary"] is not None:
        artefacts["logit_diag_backward_summary"] = diag_back["summary"]
    if diag_fwd["detail"] is not None:
        artefacts["logit_diag_forward_detail"] = diag_fwd["detail"]
    if diag_fwd["summary"] is not None:
        artefacts["logit_diag_forward_summary"] = diag_fwd["summary"]
    artefacts.update({f"data_{k}": v for k, v in summary_paths.items()})

    # Save classification metrics comparison (logit vs RF, in-sample vs OOS)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cls_metrics_path = os.path.join(out_dir, f"classification_metrics_{stamp}.csv")
    metrics_rows = [
        {
            "model": "logit_forward",
            "sample": "in_sample",
            "overall_hit_equal_weight": hit_rates_insample.get("overall", np.nan),
            "overall_hit_pooled": hit_rates_insample.get("overall_pooled", np.nan),
            **metrics_insample,
        },
        {
            "model": "logit_forward",
            "sample": "oos_" + oos_mode.lower(),
            "overall_hit_equal_weight": hit_rates_oos.get("overall", np.nan),
            "overall_hit_pooled": hit_rates_oos.get("overall_pooled", np.nan),
            **metrics_oos_logit,
        },
        {
            "model": "random_forest_forward",
            "sample": "oos_expanding",
            "overall_hit_equal_weight": hit_rates_oos_rf.get("overall", np.nan),
            "overall_hit_pooled": hit_rates_oos_rf.get("overall_pooled", np.nan),
            **metrics_oos_rf,
        },
        {
            "model": "logit_backward_lagged",
            "sample": "oos_expanding",
            "overall_hit_equal_weight": hit_rates_oos_back.get("overall", np.nan),
            "overall_hit_pooled": hit_rates_oos_back.get("overall_pooled", np.nan),
            **metrics_oos_back,
        },
    ]
    pd.DataFrame(metrics_rows).to_csv(cls_metrics_path, index=False)
    artefacts["classification_metrics"] = cls_metrics_path
    logger.info("Classification metrics saved: %s", cls_metrics_path)

    logger.info("Pipeline complete.")
    return artefacts


__all__ = [
    "compute_predictors_from_raw",
    "generate_summary_tables",
    "prepare_data",
    "fit_logistic",
    "fit_random_forest",
    "compute_logit_diagnostics",
    "expanding_window_backtest",
    "rolling_window_backtest",
    "expanding_window_backtest_rf",
    "calculate_hit_rate",
    "compute_classification_metrics",
    "backtest_trading_strategy",
    "optimize_thresholds",
    "save_outputs",
    "run_full_pipeline",
]


# ----------------------------------------------------------------------------
# Command line entry point
# ----------------------------------------------------------------------------


if __name__ == "__main__":
    default_data_path_xlsx = "Data.xlsx"
    default_data_path_csv = "Data.csv"

    if os.path.exists(default_data_path_xlsx):
        logger.info("Loading data from %s", default_data_path_xlsx)
        data = pd.read_excel(default_data_path_xlsx)
    elif os.path.exists(default_data_path_csv):
        logger.info("Loading data from %s", default_data_path_csv)
        data = pd.read_csv(default_data_path_csv)
    else:
        raise FileNotFoundError(
            "No data file found. Place your data as 'Data.xlsx' or 'Data.csv' "
            "in the current directory and re-run the script."
        )

    # Default run: expanding OOS as in coursework. Set oos_mode="rolling"
    # for rolling-window evaluation. Adjust tau_* and transaction_cost as desired.
    run_full_pipeline(
        data,
        oos_mode="expanding",
        transaction_cost=0.0,
        optimize_taus=True,
        assume_engineered_predictors=True,
    )

