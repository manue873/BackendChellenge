from __future__ import annotations
import json, os
from typing import Any, Dict, List, Optional
from collections import defaultdict

import requests
from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import joblib
import pandas as pd

# =========================
# CONFIG (sin env vars)
# =========================
API_URL = "https://geocode.maps.co/reverse"
API_KEY = "68e18d6a56c8d645937370ykja04fb1"   # para resolver país desde lat/lon
DATA_PATH = os.path.join("data", "who_aap_clean.json")
MODEL_PATH = "stagnation_rf.joblib"

# =========================
# DATASET + ÍNDICES
# =========================
def load_dataset(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("El archivo debe contener una lista JSON de objetos.")
    return data

DATASET: List[Dict[str, Any]] = load_dataset(DATA_PATH)

INDEX_BY_COUNTRY: Dict[str, List[int]] = defaultdict(list)
for i, row in enumerate(DATASET):
    c = str(row.get("Country", "")).strip()
    if c:
        INDEX_BY_COUNTRY[c].append(i)

# Stats para scores heurísticos
def _col_stats(key: str):
    vals = [row.get(key) for row in DATASET if row.get(key) is not None]
    vals = np.array(vals, dtype=float) if vals else np.array([0.0])
    return {"mean": float(vals.mean()), "std": float(vals.std() if vals.std() > 1e-9 else 1.0)}

PM10_STATS = _col_stats("PM10")
NO2_STATS  = _col_stats("NO2")

# =========================
# UTILIDADES
# =========================
def get_country_from_coords(lat: float, lon: float) -> str:
    params = {"lat": lat, "lon": lon, "api_key": API_KEY}
    try:
        resp = requests.get(API_URL, params=params, timeout=12)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Error de conexión geocoder: {e}")
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    country = (resp.json().get("address") or {}).get("country")
    if not country:
        raise HTTPException(status_code=404, detail="No se pudo determinar el país.")
    return country

def filter_by_country(country: str) -> List[Dict[str, Any]]:
    idxs = INDEX_BY_COUNTRY.get(country, [])
    return [DATASET[i] for i in idxs]

def find_row_for_prediction(country: str, city: Optional[str], year: int) -> Optional[dict]:
    # 1) país+ciudad+year
    if city:
        m = [r for r in DATASET if str(r.get("Country","")).lower()==country.lower()
             and str(r.get("City","")).lower()==city.lower()
             and int(r.get("Year",-1))==year]
        if m: return m[0]
    # 2) país+year
    m = [r for r in DATASET if str(r.get("Country","")).lower()==country.lower()
         and int(r.get("Year",-1))==year]
    if m: return m[0]
    # 3) año más cercano dentro del país (y ciudad si se especificó)
    pool = [r for r in DATASET if str(r.get("Country","")).lower()==country.lower()
            and "Year" in r]
    if city:
        pool_city = [r for r in pool if str(r.get("City","")).lower()==city.lower()]
        if pool_city:
            return min(pool_city, key=lambda r: abs(int(r["Year"]) - year))
    if pool:
        return min(pool, key=lambda r: abs(int(r["Year"]) - year))
    return None

def compute_scores(row: Dict[str, Any]) -> Dict[str, Any]:
    pm10 = row.get("PM10"); no2 = row.get("NO2")
    pm10_z = (pm10 - PM10_STATS["mean"]) / (PM10_STATS["std"]) if pm10 is not None else 0.0
    no2_z  = (no2  - NO2_STATS["mean"])  / (NO2_STATS["std"])  if no2  is not None else 0.0
    stagnation_score = float(np.clip(50 + 25*pm10_z + 25*no2_z, 0, 100))
    haze = "bajo"
    if pm10 is not None:
        if pm10 >= 50: haze = "alto"
        elif pm10 >= 30: haze = "medio"
    return {"StagnationScore": stagnation_score, "HazeRisk": haze}

# =========================
# APP
# =========================
app = FastAPI(title="Reverse Country + Stagnation API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # abre todo; restringe si usarás un front específico
    allow_methods=["*"],
    allow_headers=["*"],
)

class CoordIn(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)

# Modelo (si existe)
try:
    RF_MODEL = joblib.load(MODEL_PATH)
except Exception:
    RF_MODEL = None

@app.get("/health")
def health():
    return {
        "status": "ok",
        "dataset_records": len(DATASET),
        "countries_indexed": len(INDEX_BY_COUNTRY),
        "model_loaded": RF_MODEL is not None
    }

@app.get("/country")
def country_from_coords(lat: float = Query(...), lon: float = Query(...)):
    c = get_country_from_coords(lat, lon)
    return {"lat": lat, "lon": lon, "country": c, "records_count": len(INDEX_BY_COUNTRY.get(c, []))}

@app.get("/data/by-country")
def data_by_country(lat: float = Query(...), lon: float = Query(...), limit: int = 0, offset: int = 0):
    c = get_country_from_coords(lat, lon)
    items = filter_by_country(c)
    total = len(items)
    if limit > 0:
        items = items[offset: offset + limit]
    return {"country": c, "total": total, "count": len(items), "items": items}

@app.get("/scores")
def scores(lat: float = Query(...), lon: float = Query(...), year: int = Query(...), city: Optional[str] = None):
    c = get_country_from_coords(lat, lon)
    row = find_row_for_prediction(c, city, year)
    if not row:
        raise HTTPException(404, f"No hay gases para {c} ({city or '—'}) en {year}")
    s = compute_scores(row)
    return {"query": {"lat": lat, "lon": lon, "year": year, "country": c, "city": city},
            "scores": s, "row": row}

@app.get("/stagnation/predict")
def stagnation_predict(lat: float = Query(...), lon: float = Query(...), year: int = Query(...), city: Optional[str] = None):
    if RF_MODEL is None:
        raise HTTPException(500, "Modelo no cargado. Ejecuta entrenamiento (stagnation_rf.joblib).")
    c = get_country_from_coords(lat, lon)
    row = find_row_for_prediction(c, city, year)
    if not row:
        raise HTTPException(404, f"No hay gases para {c} ({city or '—'}) en {year}")

    feat = {
        "PM10": row.get("PM10"),
        "PM25": row.get("PM25"),
        "NO2":  row.get("NO2"),
        "Year": int(row.get("Year", year)),
        "Latitude": float(lat),
        "Longitude": float(lon),
        "Country": row.get("Country")
    }
    X = pd.DataFrame([feat])
    y_hat = float(RF_MODEL.predict(X)[0])
    return {"query": {"lat": lat, "lon": lon, "year": year, "country": c, "city": city},
            "prediction": {"stagnation_share": y_hat}}

class BatchItem(BaseModel):
    lat: float
    lon: float
    year: int
    city: Optional[str] = None

@app.post("/stagnation/predict-batch")
def stagnation_predict_batch(items: List[BatchItem] = Body(...)):
    if RF_MODEL is None:
        raise HTTPException(500, "Modelo no cargado. Ejecuta entrenamiento.")
    out = []
    for it in items:
        try:
            c = get_country_from_coords(it.lat, it.lon)
            row = find_row_for_prediction(c, it.city, it.year)
            if not row:
                out.append({"ok": False, "error": f"Sin gases para {c} {it.year}",
                            "input": it.dict()})
                continue
            feat = {
                "PM10": row.get("PM10"),
                "PM25": row.get("PM25"),
                "NO2":  row.get("NO2"),
                "Year": int(row.get("Year", it.year)),
                "Latitude": float(it.lat),
                "Longitude": float(it.lon),
                "Country": row.get("Country")
            }
            X = pd.DataFrame([feat])
            y_hat = float(RF_MODEL.predict(X)[0])
            out.append({"ok": True, "input": it.dict(),
                        "country": c, "prediction": {"stagnation_share": y_hat}})
        except Exception as e:
            out.append({"ok": False, "input": it.dict(), "error": str(e)})
    return {"results": out}
