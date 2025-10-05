"""
Entrena un RandomForest que predice la fracción anual de días con estancamiento
(stagnation_share) usando tus gases (PM10, PM25, NO2) + (Year, lat, lon, Country).

- Etiquetas (target) se construyen con Open-Meteo Historical (diario):
  estancado = wind_speed_10m_mean < 3.2 m/s AND precipitation_sum < 1 mm
  stagnation_share = #dias_estancados / #dias

Incluye:
- Caché HTTP (requests-cache)
- Concurrencia (ThreadPoolExecutor) + reintentos/backoff
- Filtros por países y años (rango o lista)
- Geocodificación opcional City+Country -> lat/lon con geocode.maps.co
- Muestreo aleatorio para --limit (opción de balancear PM25)
- Checkpoints CSV
- Validación GroupKFold por ciudad (CV dinámico)
- Baseline opcional si no hay muestras (DummyRegressor)

Uso típico:
  python train_stagnation_rf.py --data-path data/who_aap_clean.json \
      --years 2015-2018 --countries Peru,Italy --workers 4 --cache \
      --maps-api-key TU_API_KEY_MAPS --limit 200 --balance-pm25

Recomendación: ejecutar con -u para ver logs en tiempo real.
"""

import argparse
import json
import time
from typing import Optional, Tuple, List

import joblib
import numpy as np
import pandas as pd
import requests

# ---- Caché opcional ----
try:
    import requests_cache
except Exception:
    requests_cache = None

from concurrent.futures import ThreadPoolExecutor, as_completed

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_val_score


# ------------------ Open-Meteo ------------------
OM_URL = "https://archive-api.open-meteo.com/v1/archive"

def fetch_daily(lat: float, lon: float, year: int, retries: int = 3, backoff: float = 1.5) -> pd.DataFrame:
    """
    Descarga clima diario para un año con Open-Meteo, con reintentos/backoff.
    """
    for attempt in range(retries):
        try:
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
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep((attempt + 1) * backoff)


def compute_stagnation_share(daily_df: pd.DataFrame,
                             wind_thresh_ms: float = 3.2,
                             precip_thresh_mm: float = 1.0) -> float:
    """
    Fracción de días con viento medio < 3.2 m/s y precipitación < 1 mm.
    """
    if daily_df is None or len(daily_df) == 0:
        return np.nan
    ok = (
        (daily_df["wind_speed_10m_mean_ms"] < wind_thresh_ms) &
        (daily_df["precipitation_sum_mm"]   < precip_thresh_mm)
    )
    return float(ok.mean()) if len(daily_df) else np.nan


# ------------------ Geocodificación opcional (maps.co) ------------------
GEOCODE_URL = "https://geocode.maps.co/search"

def geocode_city_country(city: str, country: str, api_key: str) -> Optional[Tuple[float,float]]:
    """
    Devuelve (lat, lon) para City, Country usando maps.co si se pasa api_key.
    """
    if not api_key:
        return None
    q = f"{city}, {country}"
    params = {"q": q, "api_key": api_key}
    try:
        r = requests.get(GEOCODE_URL, params=params, timeout=20)
        r.raise_for_status()
        arr = r.json()
        if not arr: 
            return None
        lat = float(arr[0]["lat"]); lon = float(arr[0]["lon"])
        return lat, lon
    except Exception:
        return None


# ------------------ Utilidades ------------------
def parse_years(spec: str) -> List[int]:
    """
    '2015-2018' -> [2015,2016,2017,2018]
    '2015,2017,2019' -> [2015,2017,2019]
    """
    spec = spec.strip()
    if not spec:
        return []
    if "-" in spec:
        a, b = spec.split("-", 1)
        a, b = int(a), int(b)
        if a > b: a, b = b, a
        return list(range(a, b + 1))
    years = []
    for tok in spec.split(","):
        tok = tok.strip()
        if tok:
            years.append(int(tok))
    return years


def print_nonnull_summary(df: pd.DataFrame, cols: List[str], header: str):
    print(header, flush=True)
    for col in cols:
        nn = df[col].notna().sum()
        print(f"  {col}: {nn} no-nulos de {len(df)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", default="data/who_aap_clean.json", help="Ruta del JSON (lista de objetos)")
    ap.add_argument("--out-model", default="stagnation_rf.joblib", help="Ruta de salida del modelo")
    ap.add_argument("--maps-api-key", default="", help="API key de maps.co para geocodificar City,Country si faltan coords")
    ap.add_argument("--countries", default="", help="Lista separada por coma (e.g., Peru,Italy). Vacío = todos.")
    ap.add_argument("--years", default="", help="Rango '2015-2018' o lista '2015,2017'. Vacío = todos.")
    ap.add_argument("--limit", type=int, default=0, help="Limitar # de (City,Year) para pruebas (0 = sin límite)")
    ap.add_argument("--workers", type=int, default=4, help="Hilos para etiquetado concurrente")
    ap.add_argument("--cache", action="store_true", help="Activar caché HTTP (requests-cache) 30 días")
    ap.add_argument("--checkpoint", default="labels_checkpoint.csv", help="CSV de checkpoint (opcional)")
    ap.add_argument("--bench", type=int, default=0, help="Benchmark: #muestras para estimar tiempo por descarga (0=off)")
    ap.add_argument("--balance-pm25", action="store_true", help="Al limitar, balancear ~50% con PM25 presente")
    ap.add_argument("--allow-baseline", action="store_true", help="Si no hay muestras, guardar un modelo baseline")
    args = ap.parse_args()

    # Caché HTTP opcional
    if args.cache and requests_cache is not None:
        requests_cache.install_cache("om_cache", expire_after=60*60*24*30)  # 30 días
        print("Caché HTTP habilitada: om_cache.sqlite (30 días)", flush=True)

    # 1) Cargar JSON
    with open(args.data_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    df = pd.DataFrame(raw)

    # 2) Columnas mínimas
    must = ["Country","City","Year","PM10","PM25","NO2"]
    for k in ["Latitude","Longitude"]:
        if k not in df.columns:
            df[k] = np.nan
    df = df[must + ["Latitude","Longitude"]].copy()
    df = df.dropna(subset=["Country","City","Year"])
    df["Year"] = df["Year"].astype(int)

    # Filtros opcionales
    if args.countries:
        keep_c = [c.strip().lower() for c in args.countries.split(",") if c.strip()]
        df = df[df["Country"].str.lower().isin(keep_c)]
    if args.years:
        keep_y = set(parse_years(args.years))
        if keep_y:
            df = df[df["Year"].isin(keep_y)]

    # Deduplicar (Country,City,Year)
    df = df.drop_duplicates(subset=["Country","City","Year"]).reset_index(drop=True)

    # Geocodificar si faltan coords
    if df["Latitude"].isna().any() or df["Longitude"].isna().any():
        print("Geocodificando ciudades sin coords...", flush=True)
        missing_coords = df["Latitude"].isna() | df["Longitude"].isna()
        for i in df[missing_coords].index:
            row = df.loc[i]
            coords = geocode_city_country(str(row["City"]), str(row["Country"]), args.maps_api_key)
            if coords:
                df.at[i,"Latitude"], df.at[i,"Longitude"] = coords
                time.sleep(0.35)  # rate limit de cortesía
        n0 = len(df)
        df = df.dropna(subset=["Latitude","Longitude"]).reset_index(drop=True)
        print(f"Descartadas {n0 - len(df)} filas sin coords.", flush=True)

    # Resumen de no-nulos post-filtros (antes de limit)
    print_nonnull_summary(df, ["PM10","PM25","NO2","Year","Latitude","Longitude"],
                          "No nulos por columna (post-filtros, antes del limit):")

    # Limitar (muestreo aleatorio o balanceado por PM25)
    if args.limit and args.limit > 0 and len(df) > args.limit:
        need = args.limit
        if args.balance_pm25:
            with_pm25 = df[df["PM25"].notna()]
            without_pm25 = df[df["PM25"].isna()]
            n1 = min(len(with_pm25), need // 2)   # ~50% con PM25
            n2 = need - n1
            df = pd.concat([
                with_pm25.sample(n=n1, random_state=42),
                without_pm25.sample(n=n2, random_state=42)
            ]).sample(frac=1.0, random_state=42)  # mezclar
            print(f"Limit balanceado: {len(df)} filas (PM25 presentes: {n1})", flush=True)
        else:
            df = df.sample(n=need, random_state=42).copy()
            print(f"Limit aleatorio: {len(df)} filas", flush=True)

    # Benchmark opcional
    if args.bench and len(df) > 0:
        print("Benchmarking llamadas a Open-Meteo...", flush=True)
        sample_idx = np.random.choice(df.index, size=min(args.bench, len(df)), replace=False)
        times = []
        for i in sample_idx:
            r = df.loc[i]
            t0 = time.time()
            _ = fetch_daily(float(r["Latitude"]), float(r["Longitude"]), int(r["Year"]))
            times.append(time.time() - t0)
        avg = float(np.mean(times))
        print(f"Promedio por llamada: {avg:.2f} s; Estimación total (~N={len(df)}): {avg*len(df):.1f} s", flush=True)

    # 4) Etiquetado concurrente con progreso y checkpoints
    print(f"Descargando meteo y calculando etiquetas para N={len(df)} combos City-Year...", flush=True)
    labels = [np.nan] * len(df)
    errors = 0

    def label_one(i, lat, lon, year):
        daily = fetch_daily(lat, lon, year)
        return i, compute_stagnation_share(daily)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = []
        for i, row in df.iterrows():
            futures.append(ex.submit(label_one, i, float(row["Latitude"]), float(row["Longitude"]), int(row["Year"])))
        done = 0
        total = len(futures)
        for fut in as_completed(futures):
            try:
                i, y = fut.result()
                labels[i] = y
            except Exception as e:
                errors += 1
                print("ERROR etiquetando una fila:", e, flush=True)
            done += 1
            if done % 10 == 0 or done == total:
                print(f"Progreso: {done}/{total} completados; errores: {errors}", flush=True)
            if args.checkpoint and (done % 25 == 0 or done == total):
                df_cp = df.copy()
                df_cp["stagnation_share"] = labels
                df_cp.to_csv(args.checkpoint, index=False)
                print(f"Checkpoint {done}/{total} → {args.checkpoint}", flush=True)

    df["stagnation_share"] = labels
    df = df.dropna(subset=["stagnation_share"]).reset_index(drop=True)
    print(f"Filas con etiqueta: {len(df)} (errores durante etiquetado: {errors})", flush=True)
    print(f"Post-filtros: samples={len(df)}, ciudades={df['City'].nunique()}, años={df['Year'].nunique()}", flush=True)

    # 5) Entrenamiento (o baseline)
    if len(df) == 0:
        if args.allow_baseline:
            print("Sin muestras etiquetadas. Creando modelo baseline de emergencia (DummyRegressor).", flush=True)
            from sklearn.dummy import DummyRegressor
            # Pipeline compatible con tus features:
            X_dummy = pd.DataFrame([{
                "PM10": 0, "PM25": 0, "NO2": 0, "Year": 2000,
                "Latitude": 0.0, "Longitude": 0.0, "Country": "NA"
            }])
            y_dummy = [0.2]  # valor constante (20% días estancados)
            num_cols = ["PM10","PM25","NO2","Year","Latitude","Longitude"]
            cat_cols = ["Country"]
            num_tf = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
            cat_tf = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                               ("onehot", OneHotEncoder(handle_unknown="ignore"))])
            pre = ColumnTransformer([("num", num_tf, num_cols), ("cat", cat_tf, cat_cols)])
            dummy = DummyRegressor(strategy="constant", constant=y_dummy[0])
            pipe = Pipeline([("pre", pre), ("rf", dummy)])
            pipe.fit(X_dummy, y_dummy)
            joblib.dump(pipe, args.out_model)
            print(f"Modelo baseline guardado en: {args.out_model}", flush=True)
            return
        else:
            raise SystemExit("Sin muestras etiquetadas. Revisa filtros/coords/API key/red.")

    # Selección dinámica de columnas numéricas (PM25 solo si hay datos)
    num_cols = ["PM10","NO2","Year","Latitude","Longitude"]
    if df["PM25"].notna().sum() > 0:
        num_cols.insert(1, "PM25")  # ["PM10","PM25","NO2","Year","Latitude","Longitude"]

    features = num_cols + ["Country"]
    X = df[features].copy()
    y = df["stagnation_share"].astype(float)

    num_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    pre = ColumnTransformer([
        ("num", num_tf, num_cols),
        ("cat", cat_tf, ["Country"])
    ])

    rf = RandomForestRegressor(
        n_estimators=500,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )

    pipe = Pipeline([("pre", pre), ("rf", rf)])

    # CV dinámico por ciudad
    n_samples = len(df)
    n_cities  = df["City"].nunique()

    if n_samples < 2 or n_cities < 2:
        print(f"Datos insuficientes para CV (samples={n_samples}, cities={n_cities}). Entrenando sin CV.", flush=True)
        pipe.fit(X, y)
    else:
        cv_splits = min(5, max(2, n_cities))
        gkf = GroupKFold(n_splits=cv_splits)
        scores = cross_val_score(
            pipe, X, y,
            cv=gkf.split(X, y, groups=df["City"]),
            scoring="r2"
        )
        print("R2 (city holdout):", round(scores.mean(), 3), "+/-", round(scores.std(), 3), flush=True)
        pipe.fit(X, y)

    joblib.dump(pipe, args.out_model)
    print(f"Modelo guardado en: {args.out_model}", flush=True)


if __name__ == "__main__":
    main()
