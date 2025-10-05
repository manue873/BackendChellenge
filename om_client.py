    # om_client.py
import requests
import pandas as pd

OM_URL = "https://archive-api.open-meteo.com/v1/archive"

def fetch_daily(lat: float, lon: float, year: int) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": f"{year}-01-01",
        "end_date":   f"{year}-12-31",
        "daily": "precipitation_sum,wind_speed_10m_mean",
        "wind_speed_unit": "ms",
        "timezone": "auto"
    }
    r = requests.get(OM_URL, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()
    if "daily" not in js or not js["daily"]:
        raise RuntimeError(f"Open-Meteo sin datos para {lat},{lon} {year}")
    d = js["daily"]
    df = pd.DataFrame({
        "date": d["time"],
        "precipitation_sum_mm": d.get("precipitation_sum", []),
        "wind_speed_10m_mean_ms": d.get("wind_speed_10m_mean", [])
    })
    df["date"] = pd.to_datetime(df["date"])
    return df

def compute_stagnation_share(daily_df: pd.DataFrame,
                             wind_thresh_ms: float = 3.2,
                             precip_thresh_mm: float = 1.0) -> float:
    ok = (
        (daily_df["wind_speed_10m_mean_ms"] < wind_thresh_ms) &
        (daily_df["precipitation_sum_mm"]   < precip_thresh_mm)
    )
    return float(ok.mean()) if len(daily_df) else 0.0
