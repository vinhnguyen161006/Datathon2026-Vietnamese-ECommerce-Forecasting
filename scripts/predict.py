"""Generate submission.csv from saved models. Run: python scripts/predict.py

Ý tưởng B: Prophet components are injected as LGBM input features.
LGBM makes the final prediction directly — no explicit blend weight needed.
Recovery scaling is still applied to compensate for post-COVID trend shift.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd
import random
import joblib

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

from src.data_loader import (
    load_all_tables, load_submission_template,
    daily_promo_features, monthly_inventory_features,
)
from src.features import build_features
from src.models import load_model, get_prophet_components, LAG_LIST, WINDOW_LIST

OUTPUT = Path(__file__).parents[1] / "outputs"

# Monthly COGS/Revenue margin computed from training data
MONTHLY_MARGIN = {
    1: 0.8118, 2: 0.8114, 3: 0.8584, 4: 0.8458,
    5: 0.8005, 6: 0.8388, 7: 0.9007, 8: 0.9791,
    9: 0.8894, 10: 0.8038, 11: 0.8628, 12: 0.9663,
}

# Smooth linear recovery scale (replaces per-year step function).
# Scale interpolates continuously from SCALE_START (Jan 1, 2023) to SCALE_END (Jul 1, 2024).
# Calibrated so that 2023 and 2024 yearly means match the same targets as the
# step-function baseline (4,141,935 and 5,324,607 respectively).
# Key benefit: eliminates the artificial jump at Jan 1, 2024 (was 1.453→1.549=+6.6%).
SCALE_START = 1.4643   # Jan 1 2023  (v28 reg_lambda=0.3 — BEST 808,731)
SCALE_END   = 1.5057   # Jul 1 2024
_SCALE_T0   = pd.Timestamp("2023-01-01")
_SCALE_DAYS = (pd.Timestamp("2024-07-01") - _SCALE_T0).days  # 547


def _smooth_scale(dates: pd.Series) -> np.ndarray:
    t = (dates - _SCALE_T0).dt.days.values / _SCALE_DAYS
    t = np.clip(t, 0, 1)
    return SCALE_START + t * (SCALE_END - SCALE_START)


def _make_synthetic_promotions() -> pd.DataFrame:
    """Generate 2023-2024 promotions from deterministic alternating pattern.

    Odd years (2013,2015,...,2021,2023): 6 promos, avg 20.8% discount.
    Even years (2014,2016,...,2022,2024): 4 promos, avg 15.0% discount.
    Exact same dates and discount values repeat each year of the same parity.
    """
    rows = [
        # 2023 — odd year, 6 promos (identical dates/values to 2021/2019/...)
        {"start_date": "2023-01-30", "end_date": "2023-03-01",  "discount_value": 15.0, "stackable_flag": 1},
        {"start_date": "2023-03-18", "end_date": "2023-04-17",  "discount_value": 12.0, "stackable_flag": 1},
        {"start_date": "2023-06-23", "end_date": "2023-07-22",  "discount_value": 18.0, "stackable_flag": 1},
        {"start_date": "2023-07-30", "end_date": "2023-09-02",  "discount_value": 50.0, "stackable_flag": 0},
        {"start_date": "2023-08-30", "end_date": "2023-10-01",  "discount_value": 10.0, "stackable_flag": 0},
        {"start_date": "2023-11-18", "end_date": "2024-01-02",  "discount_value": 20.0, "stackable_flag": 1},
        # 2024 — even year, 4 promos (test ends 2024-07-01, only spring/summer matter)
        {"start_date": "2024-03-18", "end_date": "2024-04-17",  "discount_value": 12.0, "stackable_flag": 1},
        {"start_date": "2024-06-23", "end_date": "2024-07-22",  "discount_value": 18.0, "stackable_flag": 0},
    ]
    df = pd.DataFrame(rows)
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"]   = pd.to_datetime(df["end_date"])
    return df


def _project_web_traffic(web: pd.DataFrame, target_dates: pd.Series) -> pd.DataFrame:
    """Project 2023-2024 web traffic using lag-365 from real data.

    Sessions plateaued at ~30,300/day in 2021-2022 (+0.7% YoY).
    We shift each day's value forward by 365 days (preserves seasonal pattern).
    Iterates in date order so 2024 dates can chain through projected 2023 values.
    """
    web = web.sort_values("date").copy()
    web_lookup: dict = {row["date"]: row.to_dict() for _, row in web.iterrows()}

    projected = []
    for d in sorted(pd.to_datetime(target_dates)):
        d_ref = d - pd.Timedelta(days=365)
        if d_ref not in web_lookup:
            d_ref = d - pd.Timedelta(days=364)
        if d_ref in web_lookup:
            base = web_lookup[d_ref]
        else:
            m = d.month
            candidates = [v for k, v in web_lookup.items() if pd.Timestamp(k).month == m]
            base = {col: float(np.mean([c[col] for c in candidates if isinstance(c.get(col), (int, float))]))
                    for col in ["sessions", "unique_visitors", "page_views", "bounce_rate", "avg_session_duration_sec"]}
        row = {
            "date":                    d,
            "sessions":                float(base.get("sessions",                30000)),
            "unique_visitors":         float(base.get("unique_visitors",         23000)),
            "page_views":              float(base.get("page_views",              130000)),
            "bounce_rate":             float(base.get("bounce_rate",             0.0045)),
            "avg_session_duration_sec":float(base.get("avg_session_duration_sec", 300)),
            "traffic_source":          "projected",
        }
        projected.append(row)
        web_lookup[d] = row

    synth = pd.DataFrame(projected)
    synth["date"] = pd.to_datetime(synth["date"])
    return pd.concat([web, synth], ignore_index=True).sort_values("date").reset_index(drop=True)


def _project_inventory_monthly(inv_monthly: pd.DataFrame, test_dates: pd.Series) -> pd.DataFrame:
    """Project 2023-2024 monthly inventory by repeating 2022 monthly patterns."""
    inv_2022 = inv_monthly[inv_monthly["year"] == 2022].copy()
    test_periods = pd.to_datetime(test_dates).dt.to_period("M").unique()

    rows = []
    for period in test_periods:
        year, month = period.year, period.month
        already = inv_monthly[(inv_monthly["year"] == year) & (inv_monthly["month"] == month)]
        if already.empty:
            base = inv_2022[inv_2022["month"] == month]
            if len(base) > 0:
                new_row = base.iloc[0].copy()
                new_row["year"] = year
                rows.append(new_row.to_dict())

    if not rows:
        return inv_monthly
    return pd.concat([inv_monthly, pd.DataFrame(rows)], ignore_index=True)


def _add_extra_features(df: pd.DataFrame, hist_monthly_avg: dict) -> pd.DataFrame:
    df["revenue_hist_monthly_mean"] = df["month"].map(hist_monthly_avg)
    df["days_since_2020"] = np.maximum(
        (df["Date"] - pd.Timestamp("2020-01-01")).dt.days, 0
    ).astype(float)
    for lag in [1, 7, 30, 365]:
        col = f"Revenue_lag_{lag}"
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))
    for w in [7, 30]:
        col = f"Revenue_roll_mean_{w}"
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))
    return df


def predict_autoregressive(lgbm_model, df_train, df_test, feature_cols, hist_monthly_avg,
                           ratio_mode: bool = False,
                           prophet_ratio_mode: bool = False,
                           prophet_yhat_map: dict = None):
    """Autoregressive prediction with prophet components pre-baked into df_test rows.

    If ratio_mode=True: model predicts Revenue/Revenue_lag_365 (YoY growth ratio).
    Final prediction = ratio_pred * lag_365_actual. For 2023 dates, lag_365 is from
    actual training data (no autoregressive error on the base). For 2024, lag_365 is
    from predicted 2023 values.
    """
    history = dict(zip(df_train["Date"], df_train["Revenue"].values))

    test_sorted = df_test.sort_values("Date").reset_index(drop=True)
    pred_map: dict = {}

    hist_series = df_train.sort_values("Date").set_index("Date")["Revenue"]
    ewm7_state  = float(hist_series.ewm(span=7,  adjust=False).mean().iloc[-1])
    ewm30_state = float(hist_series.ewm(span=30, adjust=False).mean().iloc[-1])
    alpha7  = 2 / (7  + 1)
    alpha30 = 2 / (30 + 1)

    for i in range(len(test_sorted)):
        row  = test_sorted.iloc[i].copy()
        date = row["Date"]

        # ── Lag features (updated from history) ─────────────────
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

        # ── EWM ───────────────────────────────────────────────────
        row["Revenue_ewm_7"]  = ewm7_state
        row["Revenue_ewm_30"] = ewm30_state

        # ── Extra features (log-lags, calendar) ──────────────────
        for lag in [1, 7, 30, 365]:
            col = f"Revenue_lag_{lag}"
            row[f"log_{col}"] = np.log1p(max(row.get(col, 0) or 0, 0))
        for w in [7, 30]:
            col = f"Revenue_roll_mean_{w}"
            row[f"log_{col}"] = np.log1p(max(row.get(col, 0) or 0, 0))
        row["revenue_hist_monthly_mean"] = hist_monthly_avg.get(int(date.month), 0)
        row["days_since_2020"] = max((date - pd.Timestamp("2020-01-01")).days, 0)

        X = pd.DataFrame([row])[feature_cols]

        if prophet_ratio_mode and prophet_yhat_map:
            # Model predicts Revenue/prophet_yhat; multiply by prophet_yhat_test to get Revenue.
            # prophet_yhat_test is the extrapolated Prophet baseline for 2023-2024.
            prophet_yhat_t = prophet_yhat_map.get(date, np.nan)
            ratio_pred = max(float(lgbm_model.predict(X)[0]), 0.1)
            if np.isnan(prophet_yhat_t) or prophet_yhat_t <= 0:
                pred = max(float(lgbm_model.predict(X)[0]), 0.0)
            else:
                pred = max(ratio_pred * prophet_yhat_t, 0.0)
        elif ratio_mode:
            # Model predicts YoY ratio; multiply by lag_365 to get absolute Revenue
            lag365_val = history.get(date - pd.Timedelta(days=365), np.nan)
            ratio_pred = float(lgbm_model.predict(X)[0])
            ratio_pred = max(ratio_pred, 0.1)
            if np.isnan(lag365_val) or lag365_val <= 0:
                pred = 0.0
            else:
                pred = max(ratio_pred * lag365_val, 0.0)
        else:
            pred = max(float(lgbm_model.predict(X)[0]), 0.0)

        pred_map[date] = pred
        history[date]  = pred

        ewm7_state  = pred * alpha7  + ewm7_state  * (1 - alpha7)
        ewm30_state = pred * alpha30 + ewm30_state * (1 - alpha30)

    return np.array([pred_map[d] for d in df_test["Date"]])


def main():
    print("Loading models and config...")
    lgbm_revenue    = load_model("lgbm_revenue")
    prophet_revenue = load_model("prophet_revenue")
    config          = joblib.load(OUTPUT / "train_config.pkl")
    hist_monthly_avg  = config["hist_monthly_avg"]
    feature_cols      = config["feature_cols"]
    prophet_feat_cols = config.get("prophet_feat_cols", [])
    ratio_mode        = config.get("ratio_mode", False)
    print(f"  Feature count: {len(feature_cols)} | Prophet features: {prophet_feat_cols} | ratio_mode={ratio_mode}")

    print("Loading data...")
    tables  = load_all_tables()
    sales   = tables["sales"]
    web     = tables["web_traffic"]
    promos  = tables["promotions"]
    inv     = tables["inventory"]
    sub     = load_submission_template()

    # ── Augment with synthetic 2023-2024 data (derived from training patterns) ──
    print("Augmenting with synthetic 2023-2024 estimates...")

    synth_promos = _make_synthetic_promotions()
    promos_aug   = pd.concat([promos, synth_promos], ignore_index=True)
    print(f"  Promos: {len(promos)} real + {len(synth_promos)} synthetic")

    web_aug = _project_web_traffic(web, sub["Date"])
    print(f"  Web traffic: {len(web)} real + {len(sub)} projected rows")

    inv_monthly = monthly_inventory_features(inv)
    inv_monthly_aug = _project_inventory_monthly(inv_monthly, sub["Date"])
    print(f"  Inventory months: {len(inv_monthly)} real + {len(inv_monthly_aug)-len(inv_monthly)} projected")

    # ── Build features for all dates ────────────────────────────
    print("Building feature matrix...")
    all_dates = pd.concat(
        [
            sales[["Date", "Revenue", "COGS"]],
            sub[["Date"]].assign(Revenue=0.0, COGS=0.0),
        ],
        ignore_index=True,
    ).sort_values("Date").reset_index(drop=True)

    all_promo = daily_promo_features(
        promos_aug,
        pd.concat([sales["Date"], sub["Date"]]).drop_duplicates().sort_values(),
    )

    df_all = build_features(all_dates, web=web_aug, promo_daily=all_promo, inv_monthly=inv_monthly_aug)
    df_all = _add_extra_features(df_all, hist_monthly_avg)

    # ── Inject Prophet components for ALL dates (Ý tưởng B) ─────
    if prophet_feat_cols:
        print(f"Injecting Prophet component features: {prophet_feat_cols}...")
        prophet_all_feats = get_prophet_components(prophet_revenue, df_all["Date"])
        df_all = df_all.merge(prophet_all_feats, on="Date", how="left")

    df_train = df_all[df_all["Date"].isin(sales["Date"])].copy().reset_index(drop=True)
    df_test  = df_all[df_all["Date"].isin(sub["Date"])].copy().reset_index(drop=True)

    # Safety-net fill for any remaining NaNs in web/inv features
    web_inv_cols = [c for c in feature_cols
                    if c in ("session_lag_1", "bounce_rate_7d_avg",
                             "unique_visitors_lag_1", "avg_fill_rate",
                             "avg_stockout_flag", "avg_overstock_flag",
                             "total_units_sold")]
    trailing = df_train[web_inv_cols].iloc[-30:].mean()
    for col in web_inv_cols:
        df_test[col] = df_test[col].fillna(trailing[col])

    # Diagnostic: promo coverage in test
    promo_days = (df_test["n_active_promos"] > 0).sum()
    print(f"  Test days with active promos: {promo_days}/{len(df_test)} ({100*promo_days/len(df_test):.0f}%)")

    print(f"Train: {len(df_train)} rows | Test: {len(df_test)} rows | Features: {len(feature_cols)}")

    # ── Autoregressive Revenue prediction (LGBM with Prophet features) ──
    mode_label = "ratio×lag365" if ratio_mode else "absolute"
    print(f"Predicting Revenue (autoregressive, mode={mode_label})...")
    revenue_pred = predict_autoregressive(
        lgbm_model=lgbm_revenue,
        df_train=df_train,
        df_test=df_test,
        feature_cols=feature_cols,
        hist_monthly_avg=hist_monthly_avg,
        ratio_mode=ratio_mode,
    )
    revenue_pred = np.maximum(revenue_pred, 0)

    # ── Diagnostic: raw prediction means (before scale) ─────────────────
    for yr in [2023, 2024]:
        mask = df_test["Date"].dt.year == yr
        print(f"  Raw mean {yr}: {revenue_pred[mask].mean():,.0f}")

    # ── Post-COVID recovery scaling (smooth linear, no year-boundary jump) ──
    rec_scale    = _smooth_scale(df_test["Date"])
    revenue_pred = revenue_pred * rec_scale
    print(f"Smooth recovery scale: {rec_scale.min():.4f} (Jan 2023) to {rec_scale.max():.4f} (Jul 2024)")

    # ── COGS via monthly margin ratio ────────────────────────────
    print("Predicting COGS (monthly margin ratio)...")
    test_months = sub["Date"].dt.month.values
    cogs_pred   = revenue_pred * np.array([MONTHLY_MARGIN[m] for m in test_months])
    cogs_pred   = np.maximum(cogs_pred, 0)

    # ── Write submission ─────────────────────────────────────────
    print("Writing submission.csv...")
    submission = sub[["Date"]].copy()
    submission["Revenue"] = revenue_pred
    submission["COGS"]    = cogs_pred
    submission["Date"]    = submission["Date"].dt.strftime("%Y-%m-%d")

    out_path = OUTPUT / "submission.csv"
    submission.to_csv(out_path, index=False)
    print(f"Saved: {out_path}  ({len(submission)} rows)")
    print(submission.head(10).to_string(index=False))

    yr_mean = {}
    for yr in [2023, 2024]:
        mask = pd.to_datetime(submission["Date"]).dt.year == yr
        yr_mean[yr] = submission.loc[mask, "Revenue"].mean()
    print(f"\nYearly prediction means:")
    for yr, mean in yr_mean.items():
        print(f"  {yr}: {mean:,.0f}")
    print(f"\nRevenue range: {revenue_pred.min():,.0f} – {revenue_pred.max():,.0f}")


if __name__ == "__main__":
    main()
