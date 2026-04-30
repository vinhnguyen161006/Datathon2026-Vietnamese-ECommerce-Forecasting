"""End-to-end training pipeline. Run from project root: python scripts/train_final.py

Prophet Components as LGBM Features (current baseline, v13/v16+)
  - Train Prophet first on Revenue
  - Extract yhat / trend / seasonal / weekly / yearly components as features
  - LGBM learns *when* to trust Prophet adaptively (vs fixed 80/20 blend)
  - Post-hoc smooth linear recovery scale applied in predict.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

(Path(__file__).parents[1] / "outputs").mkdir(exist_ok=True)

import numpy as np
import random
import pandas as pd

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

from src.data_loader import (
    load_all_tables, daily_promo_features, monthly_inventory_features,
)
from src.features import build_features, get_feature_cols
from src.models import (
    forward_walk_cv, train_lgbm, train_prophet, get_prophet_components,
    save_model,
)
from src.metrics import evaluate, print_metrics


def _add_extra_features(df: pd.DataFrame, train_monthly_avg: dict) -> pd.DataFrame:
    df["revenue_hist_monthly_mean"] = df["month"].map(train_monthly_avg)
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


def main():
    print("=" * 60)
    print("Loading data...")
    tables = load_all_tables()
    sales  = tables["sales"]
    web    = tables["web_traffic"]
    promos = tables["promotions"]
    inv    = tables["inventory"]

    # ── Historical monthly averages ───────────────────────────────
    hist_monthly_avg = (
        sales.assign(_m=sales["Date"].dt.month)
        .groupby("_m")["Revenue"].mean()
        .to_dict()
    )
    print("Historical monthly Revenue avg:", {k: f"{v:,.0f}" for k, v in hist_monthly_avg.items()})

    # ── Build base feature matrix (train) ────────────────────────
    print("\nBuilding feature matrix (train)...")
    promo_daily = daily_promo_features(promos, sales["Date"])
    inv_monthly = monthly_inventory_features(inv)
    df_train = build_features(sales, web=web, promo_daily=promo_daily, inv_monthly=inv_monthly)
    df_train = _add_extra_features(df_train, hist_monthly_avg)

    # ── [1/3] Train Prophet first (Ý tưởng B) ────────────────────
    print("\n[1/3] Training Prophet (Revenue)...")
    prophet_rev = train_prophet(df_train, target="Revenue")
    save_model(prophet_rev, "prophet_revenue")
    print("  Saved: prophet_revenue.pkl")

    # Extract in-sample components → add as LGBM features
    prophet_train_feats = get_prophet_components(prophet_rev, df_train["Date"])
    df_train = df_train.merge(prophet_train_feats, on="Date", how="left")
    PROPHET_FEAT_COLS = [c for c in prophet_train_feats.columns if c != "Date"]
    print(f"  Prophet component features: {PROPHET_FEAT_COLS}")

    feature_cols = get_feature_cols(df_train)
    print(f"Train rows: {len(df_train)} | Features: {len(feature_cols)}")

    # ── [2/3] Forward-walk CV (5 splits) ─────────────────────────
    print("\n[2/3] Forward-walk CV (LightGBM + Prophet features, 5 splits)...")
    cv_results = forward_walk_cv(df_train, feature_cols, target="Revenue", n_splits=5)
    avg_mae  = np.mean([r["MAE"]  for r in cv_results])
    avg_rmse = np.mean([r["RMSE"] for r in cv_results])
    avg_r2   = np.mean([r["R2"]   for r in cv_results])
    print(f"\nCV avg — MAE={avg_mae:,.0f}  RMSE={avg_rmse:,.0f}  R2={avg_r2:.4f}")

    # ── [3/3] Train final LightGBM ────────────────────────────────
    print("\n[3/3] Training final LightGBM (all train data)...")
    lgbm_rev = train_lgbm(df_train, feature_cols, target="Revenue")
    save_model(lgbm_rev, "lgbm_revenue")

    lgbm_cogs = train_lgbm(df_train, feature_cols, target="COGS")
    save_model(lgbm_cogs, "lgbm_cogs")
    print("  Saved: lgbm_revenue.pkl, lgbm_cogs.pkl")

    # ── Save config ───────────────────────────────────────────────
    import joblib
    config = {
        "hist_monthly_avg":    hist_monthly_avg,
        "feature_cols":        feature_cols,
        "prophet_feat_cols":   PROPHET_FEAT_COLS,
        "ratio_mode":          False,
    }
    joblib.dump(config, Path(__file__).parents[1] / "outputs" / "train_config.pkl")
    print("  Saved: train_config.pkl")

    # ── Holdout evaluation (last 180 days) ───────────────────────
    df_clean  = df_train.dropna(subset=feature_cols + ["Revenue"]).reset_index(drop=True)
    val_df    = df_clean.iloc[-180:]
    val_preds = lgbm_rev.predict(val_df[feature_cols])
    m = evaluate(val_df["Revenue"].values, val_preds)
    print_metrics(m, label="LightGBM holdout (last 180d, with Prophet features)")

    # Feature importance — top 15
    fi = pd.Series(lgbm_rev.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nTop 15 features:")
    print(fi.head(15).to_string())

    print("\n" + "=" * 60)
    print("Training complete. Run scripts/predict.py to generate submission.csv")


if __name__ == "__main__":
    main()
