import numpy as np
import pandas as pd

SEED = 42

LAG_LIST    = [7, 14, 30, 90, 180, 365, 730]
WINDOW_LIST = [7, 14, 30, 90, 180]
EWM_SPANS   = [7, 30]


def add_calendar_features(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    d = df[date_col]
    df["day_of_week"]  = d.dt.dayofweek
    df["day_of_month"] = d.dt.day
    df["day_of_year"]  = d.dt.dayofyear
    df["week_of_year"] = d.dt.isocalendar().week.astype(int)
    df["month"]        = d.dt.month
    df["quarter"]      = d.dt.quarter
    df["year"]         = d.dt.year
    df["is_weekend"]   = (d.dt.dayofweek >= 5).astype(int)
    df["is_month_end"] = d.dt.is_month_end.astype(int)
    # Cyclical encoding for day-of-year (helps model see annual cycle)
    df["sin_doy"] = np.sin(2 * np.pi * d.dt.dayofyear / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * d.dt.dayofyear / 365.25)
    df["sin_month"] = np.sin(2 * np.pi * d.dt.month / 12)
    df["cos_month"] = np.cos(2 * np.pi * d.dt.month / 12)
    return df


def add_business_flags(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    d = df[date_col]
    df["is_sale_season"] = ((d.dt.month == 11) | (d.dt.month == 12)).astype(int)
    df["is_spring_peak"] = ((d.dt.month >= 3) & (d.dt.month <= 6)).astype(int)
    df["is_mid_month"]   = d.dt.day.between(13, 17).astype(int)
    df["is_month_start"] = (d.dt.day <= 5).astype(int)
    df["is_tet"]         = ((d.dt.month == 1) | (d.dt.month == 2)).astype(int)
    return df


def add_lag_features(df: pd.DataFrame, target: str = "Revenue",
                     lags: list[int] | None = None) -> pd.DataFrame:
    if lags is None:
        lags = LAG_LIST
    for lag in lags:
        df[f"{target}_lag_{lag}"] = df[target].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, target: str = "Revenue",
                         windows: list[int] | None = None) -> pd.DataFrame:
    if windows is None:
        windows = WINDOW_LIST
    for w in windows:
        s = df[target].shift(1)
        df[f"{target}_roll_mean_{w}"] = s.rolling(w).mean()
        df[f"{target}_roll_std_{w}"]  = s.rolling(w).std()
        df[f"{target}_roll_max_{w}"]  = s.rolling(w).max()
    return df


def add_ewm_features(df: pd.DataFrame, target: str = "Revenue",
                     spans: list[int] | None = None) -> pd.DataFrame:
    if spans is None:
        spans = EWM_SPANS
    for span in spans:
        df[f"{target}_ewm_{span}"] = df[target].shift(1).ewm(span=span, adjust=False).mean()
    return df


def add_yoy_features(df: pd.DataFrame, target: str = "Revenue") -> pd.DataFrame:
    roll7    = df[target].shift(1).rolling(7).mean()
    roll7_ly = df[target].shift(365).rolling(7).mean()
    df["yoy_7d_ratio"]  = roll7 / (roll7_ly + 1e-6)

    roll30    = df[target].shift(1).rolling(30).mean()
    roll30_ly = df[target].shift(365).rolling(30).mean()
    df["yoy_30d_ratio"] = roll30 / (roll30_ly + 1e-6)
    return df


def add_web_traffic_features(df: pd.DataFrame, web: pd.DataFrame) -> pd.DataFrame:
    daily_web = (
        web.groupby("date")
        .agg(
            sessions          = ("sessions", "sum"),
            unique_visitors   = ("unique_visitors", "sum"),
            bounce_rate       = ("bounce_rate", "mean"),
            avg_session_dur   = ("avg_session_duration_sec", "mean"),
        )
        .reset_index()
        .rename(columns={"date": "Date"})
    )
    daily_web["session_lag_1"]          = daily_web["sessions"].shift(1)
    daily_web["bounce_rate_7d_avg"]     = daily_web["bounce_rate"].rolling(7).mean()
    daily_web["unique_visitors_lag_1"]  = daily_web["unique_visitors"].shift(1)

    df = df.merge(
        daily_web[["Date", "session_lag_1", "bounce_rate_7d_avg", "unique_visitors_lag_1"]],
        on="Date", how="left",
    )
    return df


def build_features(sales: pd.DataFrame, web: pd.DataFrame | None = None,
                   promo_daily: pd.DataFrame | None = None,
                   inv_monthly: pd.DataFrame | None = None,
                   target: str = "Revenue") -> pd.DataFrame:
    df = sales.copy().sort_values("Date").reset_index(drop=True)

    df = add_calendar_features(df)
    df = add_business_flags(df)
    df = add_lag_features(df, target=target)
    df = add_rolling_features(df, target=target)
    df = add_ewm_features(df, target=target)
    df = add_yoy_features(df, target=target)

    if web is not None:
        df = add_web_traffic_features(df, web)

    if promo_daily is not None:
        df = df.merge(promo_daily, on="Date", how="left")
        df["n_active_promos"]    = df["n_active_promos"].fillna(0)
        df["has_stackable_promo"] = df["has_stackable_promo"].fillna(0)
        df["promo_discount_avg"] = df["promo_discount_avg"].fillna(0)

    if inv_monthly is not None:
        df = df.merge(inv_monthly, on=["year", "month"], how="left")

    return df


def get_feature_cols(df: pd.DataFrame, exclude: list[str] | None = None) -> list[str]:
    if exclude is None:
        exclude = ["Date", "Revenue", "COGS"]
    return [c for c in df.columns if c not in exclude and df[c].dtype != object]
