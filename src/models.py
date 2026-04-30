import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit

from .metrics import evaluate, print_metrics

SEED = 42
MODELS_DIR = Path(__file__).parents[1] / "outputs"

LAG_LIST    = [7, 14, 30, 90, 180, 365, 730]
WINDOW_LIST = [7, 14, 30, 90, 180]
EWM_SPANS   = [7, 30]


def forward_walk_cv(df: pd.DataFrame, feature_cols: list[str],
                    target: str = "Revenue", n_splits: int = 5,
                    lgbm_params: dict | None = None) -> list[dict]:
    df = df.dropna(subset=feature_cols + [target]).reset_index(drop=True)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    params = lgbm_params or _default_lgbm_params()
    results = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(df)):
        X_tr, y_tr = df.loc[train_idx, feature_cols], df.loc[train_idx, target]
        X_val, y_val = df.loc[val_idx, feature_cols], df.loc[val_idx, target]

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
        preds = model.predict(X_val)
        m = evaluate(y_val, preds)
        m["fold"] = fold + 1
        print_metrics(m, label=f"Fold {fold + 1}")
        results.append(m)

    return results


def train_lgbm(df: pd.DataFrame, feature_cols: list[str],
               target: str = "Revenue",
               lgbm_params: dict | None = None,
               weight_decay: float | None = None) -> lgb.LGBMRegressor:
    """
    weight_decay: if set, apply exponential sample weighting exp((year-2012)*decay).
    Upweights recent years without zeroing out COVID patterns (unlike downweighting approach).
    """
    params = lgbm_params or _default_lgbm_params()
    df_clean = df.dropna(subset=feature_cols + [target]).reset_index(drop=True)
    n_val = max(60, int(len(df_clean) * 0.10))
    df_tr  = df_clean.iloc[:-n_val]
    df_val = df_clean.iloc[-n_val:]

    sw_tr = None
    if weight_decay is not None and "year" in df_tr.columns:
        sw_tr = np.exp((df_tr["year"].values - 2012) * weight_decay)

    model = lgb.LGBMRegressor(**params)
    model.fit(
        df_tr[feature_cols], df_tr[target],
        sample_weight=sw_tr,
        eval_set=[(df_val[feature_cols], df_val[target])],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )
    return model


def train_prophet(df: pd.DataFrame, target: str = "Revenue"):
    from prophet import Prophet
    pdf = df[["Date", target]].rename(columns={"Date": "ds", target: "y"})
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.1,
        seasonality_prior_scale=15,
    )
    m.add_country_holidays(country_name="VN")
    m.fit(pdf)
    return m


def predict_lgbm(model: lgb.LGBMRegressor, df: pd.DataFrame,
                 feature_cols: list[str]) -> np.ndarray:
    return model.predict(df[feature_cols])


def predict_prophet(model, future_dates: pd.Series) -> np.ndarray:
    future   = pd.DataFrame({"ds": future_dates})
    forecast = model.predict(future)
    return forecast["yhat"].values


def get_prophet_components(model, dates: pd.Series) -> pd.DataFrame:
    """Extract yhat, trend, and seasonal components from a fitted Prophet model.

    Returns a DataFrame with Date + prophet_* columns that can be merged as
    LGBM features, letting the tree model learn *when* to trust Prophet rather
    than hard-coding a fixed blend weight.
    """
    future   = pd.DataFrame({"ds": pd.to_datetime(dates).values})
    forecast = model.predict(future)
    result   = pd.DataFrame({"Date": pd.to_datetime(dates).values})
    result["prophet_yhat"]     = forecast["yhat"].values
    result["prophet_trend"]    = forecast["trend"].values
    result["prophet_seasonal"] = (forecast["yhat"] - forecast["trend"]).values
    if "weekly" in forecast.columns:
        result["prophet_weekly"] = forecast["weekly"].values
    if "yearly" in forecast.columns:
        result["prophet_yearly"] = forecast["yearly"].values
    return result


def predict_autoregressive(lgbm_model: lgb.LGBMRegressor,
                            prophet_model,
                            df_train: pd.DataFrame,
                            df_test: pd.DataFrame,
                            feature_cols: list[str],
                            w_lgbm: float = 0.7) -> np.ndarray:
    """
    Predict test Revenue day-by-day, feeding each day's prediction
    back as lag/rolling/EWM inputs for subsequent days.
    df_test must have all non-lag features pre-computed (calendar, promo, web, inv).
    """
    # Initialise running history from training data
    history: dict[pd.Timestamp, float] = dict(
        zip(df_train["Date"], df_train["Revenue"].values)
    )

    # Pre-compute Prophet for all test dates at once (faster)
    prophet_preds_all = predict_prophet(prophet_model, df_test.sort_values("Date")["Date"])
    prophet_map = dict(zip(df_test.sort_values("Date")["Date"], prophet_preds_all))

    test_sorted = df_test.sort_values("Date").reset_index(drop=True)
    pred_map: dict[pd.Timestamp, float] = {}

    # EWM state (incremental, span=7 and span=30)
    hist_series = df_train.sort_values("Date").set_index("Date")["Revenue"]
    ewm7_state  = float(hist_series.ewm(span=7,  adjust=False).mean().iloc[-1])
    ewm30_state = float(hist_series.ewm(span=30, adjust=False).mean().iloc[-1])
    alpha7  = 2 / (7  + 1)
    alpha30 = 2 / (30 + 1)

    for i in range(len(test_sorted)):
        row  = test_sorted.iloc[i].copy()
        date = row["Date"]

        # ── Lag features ─────────────────────────────────────────
        for lag in LAG_LIST:
            lag_date = date - pd.Timedelta(days=lag)
            row[f"Revenue_lag_{lag}"] = history.get(lag_date, np.nan)

        # ── Rolling features ──────────────────────────────────────
        for w in WINDOW_LIST:
            vals = [
                history[d]
                for d in pd.date_range(date - pd.Timedelta(days=w),
                                        date - pd.Timedelta(days=1), freq="D")
                if d in history
            ]
            if vals:
                arr = np.array(vals, dtype=float)
                row[f"Revenue_roll_mean_{w}"] = arr.mean()
                row[f"Revenue_roll_std_{w}"]  = arr.std(ddof=1) if len(arr) > 1 else 0.0
                row[f"Revenue_roll_max_{w}"]  = arr.max()
            else:
                row[f"Revenue_roll_mean_{w}"] = np.nan
                row[f"Revenue_roll_std_{w}"]  = np.nan
                row[f"Revenue_roll_max_{w}"]  = np.nan

        # ── EWM features (incremental) ────────────────────────────
        row["Revenue_ewm_7"]  = ewm7_state
        row["Revenue_ewm_30"] = ewm30_state

        # ── Predict ───────────────────────────────────────────────
        X          = pd.DataFrame([row])[feature_cols]
        lgbm_pred  = max(float(lgbm_model.predict(X)[0]), 0.0)
        proph_pred = max(float(prophet_map[date]), 0.0)
        pred       = max(w_lgbm * lgbm_pred + (1 - w_lgbm) * proph_pred, 0.0)

        pred_map[date] = pred
        history[date]  = pred

        # Update EWM state with this prediction
        ewm7_state  = pred * alpha7  + ewm7_state  * (1 - alpha7)
        ewm30_state = pred * alpha30 + ewm30_state * (1 - alpha30)

    # Return in original df_test row order
    return np.array([pred_map[d] for d in df_test["Date"]])


def ensemble_predict(lgbm_pred: np.ndarray, prophet_pred: np.ndarray,
                     w_lgbm: float = 0.7) -> np.ndarray:
    return w_lgbm * lgbm_pred + (1 - w_lgbm) * prophet_pred


def detect_drift(df_train: pd.DataFrame, df_test: pd.DataFrame,
                 feature_cols: list[str], alpha: float = 0.05,
                 lookback_days: int = 365) -> pd.DataFrame:
    from scipy.stats import ks_2samp
    train_window = df_train.sort_values("Date").tail(lookback_days)
    test_window  = df_test.sort_values("Date").head(lookback_days)
    rows = []
    for col in feature_cols:
        a = train_window[col].dropna().values
        b = test_window[col].dropna().values
        if len(a) == 0 or len(b) == 0:
            continue
        stat, p_val = ks_2samp(a, b)
        rows.append({"feature": col, "ks_stat": stat, "p_value": p_val,
                     "drifted": p_val < alpha})
    report = pd.DataFrame(rows).sort_values("ks_stat", ascending=False)
    n_drifted = report["drifted"].sum()
    print(f"Drift check: {n_drifted}/{len(report)} features drifted (alpha={alpha})")
    return report


def save_model(model, name: str) -> Path:
    path = MODELS_DIR / f"{name}.pkl"
    joblib.dump(model, path)
    return path


def load_model(name: str):
    path = MODELS_DIR / f"{name}.pkl"
    return joblib.load(path)


def _default_lgbm_params() -> dict:
    return {
        "n_estimators":      5000,
        "learning_rate":     0.01,
        "num_leaves":        31,
        "max_depth":         6,
        "min_child_samples": 15,
        "subsample":         0.75,
        "subsample_freq":    5,
        "colsample_bytree":  0.75,
        "reg_alpha":         0.02,
        "reg_lambda":        0.5,
        "random_state":      SEED,
        "n_jobs":            -1,
        "verbose":           -1,
    }
