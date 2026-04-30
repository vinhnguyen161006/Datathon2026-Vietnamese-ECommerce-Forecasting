"""Generate SHAP summary plot from saved LightGBM model. Run: python scripts/generate_shap.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import shap

SEED = 42
np.random.seed(SEED)
OUTPUT = Path(__file__).parents[1] / "outputs"
REPORTS = Path(__file__).parents[1] / "reports"
REPORTS.mkdir(exist_ok=True)

from src.data_loader import load_all_tables, daily_promo_features, monthly_inventory_features
from src.features import build_features, get_feature_cols
from src.models import get_prophet_components
import random
random.seed(SEED)


def _add_extra_features(df, hist_monthly_avg):
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


def main():
    print("Loading models and config...")
    lgbm_revenue    = joblib.load(OUTPUT / "lgbm_revenue.pkl")
    prophet_revenue = joblib.load(OUTPUT / "prophet_revenue.pkl")
    config = joblib.load(OUTPUT / "train_config.pkl")
    hist_monthly_avg  = config["hist_monthly_avg"]
    feature_cols      = config["feature_cols"]
    prophet_feat_cols = config.get("prophet_feat_cols", [])

    print("Loading data...")
    tables = load_all_tables()
    sales  = tables["sales"]
    web    = tables["web_traffic"]
    promos = tables["promotions"]
    inv    = tables["inventory"]

    promo_daily = daily_promo_features(promos, sales["Date"])
    inv_monthly = monthly_inventory_features(inv)
    df = build_features(sales, web=web, promo_daily=promo_daily, inv_monthly=inv_monthly)
    df = _add_extra_features(df, hist_monthly_avg)

    if prophet_feat_cols:
        print(f"Injecting Prophet features: {prophet_feat_cols}...")
        prophet_feats = get_prophet_components(prophet_revenue, df["Date"])
        df = df.merge(prophet_feats, on="Date", how="left")

    df = df.dropna(subset=feature_cols + ["Revenue"]).reset_index(drop=True)

    # Sample up to 500 rows for SHAP (TreeExplainer is fast but we cap for speed)
    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(df), size=min(500, len(df)), replace=False)
    X_sample = df.iloc[idx][feature_cols]

    print(f"Computing SHAP values for {len(X_sample)} samples...")
    explainer = shap.TreeExplainer(lgbm_revenue)
    shap_values = explainer.shap_values(X_sample)

    # ── Summary plot (beeswarm) ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_sample,
        max_display=20,
        show=False,
        plot_size=None,
    )
    plt.title("SHAP Feature Importance — LightGBM Revenue Model\n(Top 20 features, 500-sample subset)", fontsize=12)
    plt.tight_layout()
    out_path = REPORTS / "shap_summary.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

    # ── Bar plot (mean |SHAP|) ──────────────────────────────────────
    mean_abs = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_cols)
    top20 = mean_abs.sort_values(ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(9, 7))
    top20[::-1].plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Mean |SHAP| — Top 20 Features (LightGBM Revenue)", fontsize=12)
    ax.set_xlabel("Mean |SHAP value| (VND impact on prediction)")
    plt.tight_layout()
    bar_path = REPORTS / "shap_bar.png"
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {bar_path}")

    print("\nTop 10 features by mean |SHAP|:")
    for feat, val in top20.head(10).items():
        print(f"  {feat:<35} {val:>10,.0f}")


if __name__ == "__main__":
    main()
