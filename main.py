from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import requests

# Load the pre-trained model and scaler
model = joblib.load("random_forest_model.joblib")
scaler = joblib.load("scaler.joblib")

# Initialize FastAPI app
app = FastAPI()

# Predefined dictionary mapping place names to latitude & longitude
locations = {
    "حوزه دریائی": {"lat": 33.11972, "lon": 69.62407},
    "شمل پکتیکا": {"lat": 32.99863, "lon": 68.77225}
}

# Define input data structure
class LocationInput(BaseModel):
    place: str  

@app.get("/locations")
def get_locations():
    return {"available_locations": locations}

# Function to fetch data from Open-Meteo API
def fetch_weather_data(url: str):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for non-200 status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch data: {str(e)}")

@app.post("/predict")
def predict(data: LocationInput):
    # Step 1: Get Latitude & Longitude
    place_name = data.place
    if place_name not in locations:
        raise HTTPException(status_code=404, detail="Place not found in predefined list")
    
    lat, lon = locations[place_name]["lat"], locations[place_name]["lon"]

    # Step 2: Fetch Weather Data from Open-Meteo API
    api_urls = {
        "precipitation": f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=precipitation&timezone=auto",
        "soil_moisture": f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=soil_moisture_0_to_1cm&timezone=auto",
        "river_discharge": f"https://flood-api.open-meteo.com/v1/flood?latitude={lat}&longitude={lon}&daily=river_discharge&forecast_days=1&timezone=auto"
    }

    # Fetch and extract data from API responses
    try:
        precipitation = fetch_weather_data(api_urls["precipitation"])["current"]["precipitation"]
        soil_moisture = fetch_weather_data(api_urls["soil_moisture"])["hourly"]["soil_moisture_0_to_1cm"][0]
        river_discharge = fetch_weather_data(api_urls["river_discharge"])["daily"]["river_discharge"][0]
    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"Missing data in Open-Meteo response: {str(e)}")

    # Calculate water height using the given formula
    water_height = 0.1 * (river_discharge ** 0.5) + 0.3 * soil_moisture + 0.2 * precipitation

    # Step 3: Prepare Input for ML Model
    input_features = np.array([precipitation, soil_moisture, river_discharge, water_height]).reshape(1, -1)

    # Step 4: Scale Input and Predict
    scaled_features = scaler.transform(input_features)
    prediction = model.predict(scaled_features)

    return {
        "location": place_name,
        "lat": lat,
        "lon": lon,
        "weather_data": {
            "precipitation": precipitation,
            "soil_moisture": soil_moisture,
            "river_discharge": river_discharge,
            "water_height": water_height
        },
        "prediction": prediction.tolist()
    }
