# Reverse Country API

API en FastAPI que, dado `lat` y `lon`, detecta el **país** con geocode.maps.co y devuelve
los registros de un archivo JSON cuyo campo `Country` coincide con ese país.

## Endpoints
- `GET /health`
- `GET /country?lat=..&lon=..`
- `GET /data/by-country?lat=..&lon=..&limit=&offset=`
- `POST /data/by-country` (body: `{ "lat": .., "lon": .. }`)
- `GET /data/by-name?country=...&limit=&offset=`

## Desarrollo local
```bash
pip install -r requirements.txt
uvicorn app:app --reload
