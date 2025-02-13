import openmeteo_requests
import pandas as pd
import requests_cache
import pandas as pd
from retry_requests import retry
from fastapi import FastAPI, HTTPException
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import joblib
import numpy as np
import requests
from pydantic import BaseModel



model = joblib.load("random_forest_model.joblib")
scaler = joblib.load("scaler.joblib")

app = FastAPI()

cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

locations = {
    "حوزه دریائی": {"lat": 33.11972, "lon": 69.62407},
    "شمل پکتیکا": {"lat": 32.99863, "lon": 68.77225}
}

class LocationInput(BaseModel):
    place: str  


@app.get("/locations")
def get_locations():
    return {"available_locations": locations}

@app.post("/predict")
def predict(data: LocationInput):
    # Step 1: Get Latitude & Longitude from Dictionary
    place_name = data.place

    if place_name not in locations:
        raise HTTPException(status_code=404, detail="Place not found in predefined list")
    
    lat = locations[place_name]["lat"]
    lon = locations[place_name]["lon"]

    # Step 2: Get Data from OpenMeteo API
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "soil_temperature_6cm",
    }
    responses = openmeteo.weather_api(url, params=params)
    
    print(responses)
    
