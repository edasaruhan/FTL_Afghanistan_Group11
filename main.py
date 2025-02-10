from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import requests
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.requests import Request
from fastapi import FastAPI, HTTPException, Form
import json


# Mount the templates and static files
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# Load the pre-trained model and scaler
model = joblib.load("flood_prediction_model.pkl")
scaler = joblib.load("flood_prediction_scaler.pkl")

# Initialize FastAPI app
app = FastAPI()

# Predefined dictionary mapping place names to latitude & longitude
locations = {
    "On the water": {"lat": 35.01361, "lon": 69.17139},
    "Under the bed": {"lat": 34.5, "lon": 69.15},
    "Deh Sabz": {"lat": 34.66555, "lon": 69.308},
    "Gul Gondi": {"lat": 35.01914, "lon": 69.15114},
    "Shatoot": {"lat": 34.5, "lon": 69.15},
    "Jalga Station": {"lat": 32.41667, "lon": 68.75},
    "Black Stone": {"lat": 34.77472, "lon": 111.18139},
    "Pirakoti": {"lat": 32.89864, "lon": 69.27371},
    "Dasht Palto": {"lat": 33.10647, "lon": 69.05464},
    "Dasht Park": {"lat": 32.99863, "lon": 68.77225},
    "Ambar Khail Lake": {"lat": 33.42906, "lon": 68.30439},
    "Wheel Station": {"lat": 33.75764, "lon": 68.91321},
    "Above Bandsardi": {"lat": 33.00754, "lon": 68.11957},
    "Ergon": {"lat": 43.29289, "lon": 45.86691},
    "Ghazni Bridge": {"lat": 33.18896, "lon": 67.84132},
    "Daryaee Jaghori": {"lat": 33.14341, "lon": 67.46384},
    "Matun Station": {"lat": 33.33951, "lon": 69.92041},
    "Zakla Station": {"lat": 33.33951, "lon": 69.92041},
    "Sepira Station": {"lat": 33.20204, "lon": 69.5152},
    "Tora Tike Station": {"lat": 33.11972, "lon": 69.62407},
    "Shaki": {"lat": 35.3234, "lon": 71.5585},
    "Khorram": {"lat": 32.41667, "lon": 68.75},
    "Red Bridge": {"lat": 34.3241, "lon": 68.84932},
    "Deewal Neck": {"lat": 34.46456, "lon": 67.58532},
    "Sheikh Abad": {"lat": 34.08697, "lon": 68.75635},
    "Khawat": {"lat": 34.07963, "lon": 68.12614},
    "Above Band Sultan": {"lat": 33.76204, "lon": 68.37837},
    "Tangi Seydan": {"lat": 34.40563, "lon": 69.09699},
    "Market": {"lat": 34, "lon": 69.25},
    "Ahmad Ziyoklah": {"lat": 34, "lon": 69.25},
    "Hawza Muhammad Agha": {"lat": 31.19104, "lon": 61.83774},
    "Kamal Khil": {"lat": 34, "lon": 69.25},
    "Dahno": {"lat": 34, "lon": 69.25},
    "Inscription": {"lat": 34.52813, "lon": 69.17233},
    "Mughal Khel": {"lat": 34.15607, "lon": 69.05976},
    "Sorkh Ab": {"lat": 35.60671, "lon": 68.67885},
    "Tangy Gharu": {"lat": 34.55, "lon": 69.5},
    "Badr Ab": {"lat": 35.40248, "lon": 69.98291},
    "Tangi Gol Bahar": {"lat": 35.40248, "lon": 69.98291},
    "Estalf Station": {"lat": 34.84847, "lon": 69.02353},
    "Shakardara Station": {"lat": 34.68616, "lon": 69.00805},
    "Shaka Village": {"lat": 34.33164, "lon": 67.39182},
    "Korek": {"lat": 34.42647, "lon": 70.45153},
    "Sayedabad": {"lat": 34.42647, "lon": 70.45153},
    "Samarkhel": {"lat": 34.42647, "lon": 70.45153},
    "Pirakoti Station": {"lat": 32.89864, "lon": 69.27371},
    "Dasht Park Station": {"lat": 32.99863, "lon": 68.77225},
    "Argon Station": {"lat": 32.87036, "lon": 69.1301},
    "Bande Sardi Upper Station": {"lat": 33.00754, "lon": 68.11957},
    "Dasht Palto Station": {"lat": 33.10647, "lon": 69.05464},
    "Sheikh Abad Station": {"lat": 34.08697, "lon": 68.75635},
    "Spin Khore": {"lat": 34.42647, "lon": 70.45153},
    "Dhan Rashqa": {"lat": 34.3665, "lon": 67.48142},
    "Mia Khur": {"lat": 34.42647, "lon": 70.45153},
    "Pul Islamabad": {"lat": 34.72816, "lon": 70.11407},
    "Qarghai Bridge": {"lat": 34.51253, "lon": 70.19568},
    "Bandar": {"lat": 34.42647, "lon": 70.45153},
    "Khawat": {"lat": 34.07963, "lon": 68.12614},
    "Shakardara Station": {"lat": 34.68616, "lon": 69.00805},
    "Above Band Sultan": {"lat": 33.32114, "lon": 68.63682},
    "Chaparhar Khwar": {"lat": 34.27737, "lon": 70.3618},
    "Shaka Loya Lari Khwar": {"lat": 34.42647, "lon": 70.45153},
    "Dhan Rashqa": {"lat": 34.3665, "lon": 67.48142},
    "Mia Khur": {"lat": 34.42647, "lon": 70.45153},
    "Pul Islamabad": {"lat": 34.72816, "lon": 70.11407},
    "Nelyar Bridge": {"lat": 34.72816, "lon": 70.11407},
    "Qarghai Bridge": {"lat": 34.51253, "lon": 70.19568},
    "Khorsaracha": {"lat": 34.42647, "lon": 70.45153},
    "Kabul Camp": {"lat": 34.42647, "lon": 70.45153},
    "Thamarkhel Khor": {"lat": 34.42647, "lon": 70.45153},
    "Bikhiy Khor": {"lat": 34.42647, "lon": 70.45153},
    "Patar": {"lat": 34.42647, "lon": 70.45153},
    "Maroof Chena": {"lat": 34.42647, "lon": 70.45153},
    "Station near Matun": {"lat": 33.33951, "lon": 69.92041},
    "Pirakoti Station": {"lat": 32.89864, "lon": 69.27371},
    "Dasht Park Station": {"lat": 32.99863, "lon": 68.77225},
    "Argon Station": {"lat": 32.87036, "lon": 69.1301},
    "Bande Sardi Upper Station": {"lat": 33.00754, "lon": 68.11957},
    "Dasht Palto Station": {"lat": 33.10647, "lon": 69.05464},
    "Second Domandeh Upper Station": {"lat": 32.41667, "lon": 68.75},
    "Second Domandeh Lower Station": {"lat": 32.41667, "lon": 68.75},
    "Seydan Narrow Station": {"lat": 34.40563, "lon": 69.09699},
    "Red Bridge Station": {"lat": 34.3241, "lon": 68.84932},
    "Stone Inscription Station": {"lat": 34.52813, "lon": 69.17233},
    "Tangi Gharo Station": {"lat": 34.55, "lon": 69.5},
    "Ghazni Station": {"lat": 33.55391, "lon": 68.42096},
    "Qala-e-Malik Paghman Station": {"lat": 34.58787, "lon": 68.95091},
    "Canal Entrance Station of Band-e-Qargah": {"lat": 34.58787, "lon": 68.95091},
    "Khawat Station": {"lat": 34.06609, "lon": 67.96216},
    "Upper Band Sultan Station": {"lat": 33.76204, "lon": 68.37837},
    "Jalkeh Station": {"lat": 32.41667, "lon": 68.75},
    "Sheikh Abad Station": {"lat": 34.08697, "lon": 68.75635}
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

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "locations": locations})

@app.post("/predict")
def predict(location: str = Form(...)):
    if location not in locations:
        raise HTTPException(status_code=404, detail="Location not found")
    
    lat, lon = locations[location]["lat"], locations[location]["lon"]
    api_urls = {
        "precipitation": f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=precipitation&timezone=auto",
        "soil_moisture": f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=soil_moisture_0_to_1cm&timezone=auto",
        "river_discharge": f"https://flood-api.open-meteo.com/v1/flood?latitude={lat}&longitude={lon}&daily=river_discharge&forecast_days=1&timezone=auto"
    }
   def fetch_data(url):
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=500, detail=f"Failed to fetch data: {str(e)}")
    
    try:
        precipitation = fetch_data(api_urls["precipitation"])['current']['precipitation']
        soil_moisture = fetch_data(api_urls["soil_moisture"])['hourly']['soil_moisture_0_to_1cm'][0]
        river_discharge = fetch_data(api_urls["river_discharge"])['daily']['river_discharge'][0]
    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"Missing data in response: {str(e)}")
    
    water_height = 0.1 * (river_discharge ** 0.5) + 0.3 * soil_moisture + 0.2 * precipitation
    
    return {
        "location": location,
        "lat": lat,
        "lon": lon,
        "weather_data": {
            "precipitation": precipitation,
            "soil_moisture": soil_moisture,
            "river_discharge": river_discharge,
            "water_height": water_height
        },
        "prediction": "Flood risk calculated"
    }
