import pandas as pd
from pathlib import Path

RAW = Path(__file__).parents[1] / "data" / "raw"
PROCESSED = Path(__file__).parents[1] / "data" / "processed"

_PARSE_DATES = {
    "sales":             ["Date"],
    "sample_submission": ["Date"],
    "shipments":         ["ship_date", "delivery_date"],
    "web_traffic":       ["date"],
    "customers":         ["signup_date"],
    "inventory":         ["snapshot_date"],
    "orders":            ["order_date"],
    "promotions":        ["start_date", "end_date"],
    "returns":           ["return_date"],
    "reviews":           ["review_date"],
}

_DTYPES = {
    "products":  {"category": "category", "segment": "category", "color": "category", "size": "category"},
    "orders":    {"order_status": "category", "payment_method": "category",
                  "device_type": "category", "order_source": "category"},
    "customers": {"gender": "category", "age_group": "category", "acquisition_channel": "category"},
    "order_items": {"promo_id": "str", "promo_id_2": "str"},
}

_ALL_TABLES = [
    "sales", "sample_submission", "shipments", "web_traffic",
    "customers", "geography", "inventory", "order_items",
    "orders", "payments", "products", "promotions", "returns", "reviews",
]


def _load_table(name: str) -> pd.DataFrame:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    pq = PROCESSED / f"{name}.parquet"
    if pq.exists():
        try:
            return pd.read_parquet(pq)
        except Exception:
            pq.unlink(missing_ok=True)
    kwargs = {}
    if name in _PARSE_DATES:
        kwargs["parse_dates"] = _PARSE_DATES[name]
    if name in _DTYPES:
        kwargs["dtype"] = _DTYPES[name]
    df = pd.read_csv(RAW / f"{name}.csv", **kwargs)
    try:
        df.to_parquet(pq, index=False)
    except Exception:
        pass
    return df


def load_all_tables() -> dict[str, pd.DataFrame]:
    return {name: _load_table(name) for name in _ALL_TABLES}


def load_sales_train() -> pd.DataFrame:
    df = _load_table("sales")
    return df.sort_values("Date").reset_index(drop=True)


def load_submission_template() -> pd.DataFrame:
    return _load_table("sample_submission")


def daily_promo_features(promotions: pd.DataFrame, date_range: pd.DatetimeIndex) -> pd.DataFrame:
    """Vectorized: explode promo intervals then groupby date."""
    promo = promotions[["start_date", "end_date", "stackable_flag", "discount_value"]].copy()
    promo["Date"] = [
        pd.date_range(s, e, freq="D")
        for s, e in zip(promo["start_date"], promo["end_date"])
    ]
    promo = promo.explode("Date")

    date_set = set(date_range)
    promo = promo[promo["Date"].isin(date_set)]

    agg = (
        promo.groupby("Date")
        .agg(
            n_active_promos=("discount_value", "count"),
            has_stackable_promo=("stackable_flag", lambda x: int(x.sum() > 0)),
            promo_discount_avg=("discount_value", "mean"),
        )
        .reset_index()
    )

    result = pd.DataFrame({"Date": date_range}).merge(agg, on="Date", how="left")
    result["n_active_promos"] = result["n_active_promos"].fillna(0).astype(int)
    result["has_stackable_promo"] = result["has_stackable_promo"].fillna(0).astype(int)
    result["promo_discount_avg"] = result["promo_discount_avg"].fillna(0.0)
    return result


def monthly_inventory_features(inventory: pd.DataFrame) -> pd.DataFrame:
    """Aggregate inventory flags to month level (for join with daily sales)"""
    inv = inventory.copy()
    inv["year"] = inv["snapshot_date"].dt.year
    inv["month"] = inv["snapshot_date"].dt.month
    agg = (
        inv.groupby(["year", "month"])
        .agg(
            avg_fill_rate=("fill_rate", "mean"),
            avg_stockout_flag=("stockout_flag", "mean"),
            avg_overstock_flag=("overstock_flag", "mean"),
            total_units_sold=("units_sold", "sum"),
        )
        .reset_index()
    )
    return agg
