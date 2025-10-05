from __future__ import annotations
import json
import os
from typing import Any, Dict, List
from collections import defaultdict

import requests
from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# =========================
# CONFIG FIJA (sin env vars)
# =========================
API_URL = "https://geocode.maps.co/reverse"
API_KEY = "68e18d6a56c8d645937370ykja04fb1"  # ⚠️ Para despliegue público, NO es recomendable hardcodear
DATA_PATH = os.path.join("data", "who_aap_clean.json")

# Normalización opcional por si el nombre del API no coincide con tu JSON
COUNTRY_NORMALIZATION_MAP: Dict[str, str] = {
    # "United States": "United States of America",
    # "Russia": "Russian Federation",
}

# =========================
# CARGA E ÍNDICE
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

# =========================
# FASTAPI
# =========================
app = FastAPI(title="Reverse Country Filter API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class CoordIn(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)

def get_country_from_coords(lat: float, lon: float) -> str:
    params = {"lat": lat, "lon": lon, "api_key": API_KEY}
    try:
        resp = requests.get(API_URL, params=params, timeout=12)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Error de conexión: {e}")
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    payload = resp.json()
    country = (payload.get("address") or {}).get("country")
    if not country:
        raise HTTPException(status_code=404, detail="No se pudo determinar el país.")
    return COUNTRY_NORMALIZATION_MAP.get(country, country)

def filter_by_country(country: str) -> List[Dict[str, Any]]:
    idxs = INDEX_BY_COUNTRY.get(country, [])
    return [DATASET[i] for i in idxs]

@app.get("/health")
def health():
    return {
        "status": "ok",
        "dataset_records": len(DATASET),
        "countries_indexed": len(INDEX_BY_COUNTRY),
        "data_path": DATA_PATH
    }

@app.get("/country")
def country_from_coords(lat: float = Query(...), lon: float = Query(...)):
    country = get_country_from_coords(lat, lon)
    return {"lat": lat, "lon": lon, "country": country, "records_count": len(INDEX_BY_COUNTRY.get(country, []))}

@app.get("/data/by-country")
def data_by_country(lat: float = Query(...), lon: float = Query(...), limit: int = 0, offset: int = 0):
    country = get_country_from_coords(lat, lon)
    items = filter_by_country(country)
    total = len(items)
    if limit > 0:
        items = items[offset: offset + limit]
    return {"country": country, "total": total, "count": len(items), "items": items}

@app.post("/data/by-country")
def data_by_country_post(body: CoordIn = Body(...), limit: int = 0, offset: int = 0):
    return data_by_country(body.lat, body.lon, limit, offset)

@app.get("/data/by-name")
def data_by_name(country: str = Query(...), limit: int = 0, offset: int = 0):
    items = filter_by_country(country.strip())
    total = len(items)
    if limit > 0:
        items = items[offset: offset + limit]
    return {"country": country, "total": total, "count": len(items), "items": items}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
