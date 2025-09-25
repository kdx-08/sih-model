import os

import requests
from dotenv import load_dotenv
from fastapi import FastAPI
from llm import llmchat

load_dotenv()

app = FastAPI()


@app.get("/check")
async def checker(lat: float, lon: float):
    API_KEY = os.getenv("weather_api_key")
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"

    response = requests.get(url)
    data = response.json()
    result = llmchat(lat, lon, weather_data=data["current_weather"])
    return result
