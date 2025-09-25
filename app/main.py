import getpass
import os

import requests
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")


prompt_template = ChatPromptTemplate(
    [
        (
            "system",
            "You are a tourism safety analyst assistant. You must always respond with valid JSON only, no additional text.",
        ),
        (
            "user",
            """
You are given weather data, nearby tourism safety news, and a location (latitude and longitude).
Analyze the data and determine the severity of risk specifically for tourists and travelers in that area.

Consider factors like:
- Temperature extremes (too hot/cold for tourists)
- Wind speed (dangerous winds that affect sightseeing)
- Weather conditions (storms, fog, heavy rain/snow affecting tourism)
- Visibility issues (affecting tourist activities)
- Day/night conditions (tourist safety considerations)

PRIORITIZE NEWS-BASED RISKS for tourists:
- Tourist-targeted crimes (robbery, theft, scams, attacks)
- Travel advisories and safety warnings for visitors
- Incidents affecting popular tourist areas or attractions
- Transportation disruptions affecting travelers
- Security issues in tourist zones
- Health emergencies affecting visitors
- Civil unrest or protests in tourist areas
- Accidents at tourist attractions or hotels

Weight news-based risks HIGHER than weather-based risks when assessing tourist safety.

Output ONLY valid JSON in exactly this format (no markdown, no explanations):

{{
  "Severity status": "<low/medium/high> risk",
  "Risk factor": <number between 1-10>,
  "Reasons": ["reason 1", "reason 2", "reason 3"]
}}

Weather data: {weather_data}
Tourism Safety News: {news_data}
Location: latitude {lat}, longitude {lon}
""",
        ),
    ]
)

import json
from typing import Literal

from pydantic import BaseModel

# Weather code mapping
weather_codes = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}

# News fetching function using Tavily Search
from datetime import datetime, timedelta


def get_nearby_news(lat: float, lon: float, radius_km: int = 50):
    """
    Fetch nearby news using Tavily Search API.
    Tavily provides better location-based search and real-time results.
    Falls back to NewsAPI if Tavily is unavailable.
    """
    try:
        # Try Tavily first (recommended)
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if tavily_api_key and tavily_api_key != "your_tavily_api_key_here":
            return _fetch_tavily_news(lat, lon, tavily_api_key)

        # Fallback to NewsAPI
        news_api_key = os.getenv("NEWS_API_KEY")
        if news_api_key and news_api_key != "your_news_api_key_here":
            return _fetch_news_api(lat, lon, news_api_key)

        # No API keys available
        return {
            "articles": [],
            "status": "no_api_key",
            "message": "Neither Tavily nor NewsAPI key configured. Add API keys to .env file for news-based risk analysis.",
        }

    except Exception as e:
        return {"error": f"News fetch error: {str(e)}", "articles": []}


def _fetch_tavily_news(lat: float, lon: float, api_key: str):
    """Fetch tourism safety and crime-related news using Tavily Search API"""
    try:
        # Import Tavily here to avoid import errors if not installed
        try:
            from tavily import TavilyClient
        except ImportError:
            return {
                "error": "Tavily package not installed. Run: pip install tavily-python",
                "articles": [],
            }

        client = TavilyClient(api_key=api_key)

        # Get location name for better search results
        location_name = _get_location_name(lat, lon)

        # Create tourism safety and crime specific query
        query = f"tourist safety crime incidents travel advisory {location_name} recent"

        try:
            response = client.search(
                query=query,
                search_depth="basic",
                max_results=5,
                include_domains=[
                    "timesofindia.com",
                    "indianexpress.com",
                    "hindustantimes.com",
                    "ndtv.com",
                    "thehindu.com",
                    "deccanherald.com",
                    "news18.com",
                    "reuters.com",
                    "bbc.com",
                    "cnn.com",
                    "apnews.com",
                    "traveladvisory.state.gov",
                ],
                exclude_domains=[
                    "twitter.com",
                    "facebook.com",
                    "instagram.com",
                    "reddit.com",
                ],
                include_raw_content=False,
            )

            # Filter and format results focusing on tourism safety
            articles = []
            for result in response.get("results", []):
                title = result.get("title", "")
                content = result.get("content", "")
                url = result.get("url", "")

                # Filter for tourism safety, crime, and security-related content
                safety_keywords = [
                    "tourist",
                    "tourism",
                    "travel",
                    "visitor",
                    "crime",
                    "safety",
                    "security",
                    "robbery",
                    "theft",
                    "scam",
                    "attack",
                    "warning",
                    "advisory",
                    "caution",
                    "incident",
                    "assault",
                    "harassment",
                    "fraud",
                    "danger",
                    "risk",
                ]

                if any(
                    keyword.lower() in (title + content).lower()
                    for keyword in safety_keywords
                ):
                    articles.append(
                        {
                            "title": title,
                            "content": content[:300] + "..."
                            if len(content) > 300
                            else content,
                            "url": url,
                            "relevance": "tourism_safety",
                        }
                    )

            return {
                "articles": articles,
                "status": "success",
                "source": "tavily",
                "query_used": query,
                "location": location_name,
            }

        except Exception as e:
            return {"error": f"Tavily API error: {str(e)}", "articles": []}

    except Exception as e:
        return {"error": f"Tavily integration error: {str(e)}", "articles": []}


def _get_location_name(lat: float, lon: float) -> str:
    """Get location name from coordinates using a simple geocoding approach"""
    try:
        # You can integrate with a proper geocoding service here
        # For now, return a simple coordinate-based identifier
        return f"lat_{lat:.2f}_lon_{lon:.2f}"
    except:
        return f"location_{lat:.1f}_{lon:.1f}"


def _fetch_news_api(lat: float, lon: float, api_key: str):
    """Fallback: Fetch news using NewsAPI"""
    try:
        # Calculate date range (last 7 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)

        # Search for location-based safety keywords
        keywords = "disaster OR emergency OR protest OR accident OR crime OR safety OR warning OR alert"

        # NewsAPI parameters
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": keywords,
            "from": start_date.strftime("%Y-%m-%d"),
            "to": end_date.strftime("%Y-%m-%d"),
            "sortBy": "publishedAt",
            "language": "en",
            "pageSize": 5,
            "apiKey": api_key,
        }

        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        if response.status_code == 200 and data.get("status") == "ok":
            # Format articles to match Tavily format
            articles = []
            for article in data.get("articles", []):
                articles.append(
                    {
                        "title": article.get("title", ""),
                        "content": article.get("description", ""),
                        "url": article.get("url", ""),
                        "published_date": article.get("publishedAt", ""),
                        "source": article.get("source", {}).get("name", ""),
                    }
                )
            return {"articles": articles, "status": "success", "source": "newsapi"}
        else:
            return {
                "error": f"NewsAPI error: {data.get('message', 'Unknown error')}",
                "articles": [],
            }

    except requests.RequestException as e:
        return {"error": f"NewsAPI network error: {str(e)}", "articles": []}


def _get_location_name(lat: float, lon: float):
    """
    Convert coordinates to location name using reverse geocoding.
    Falls back to coordinate string if geocoding fails.
    """
    try:
        # You can use a free geocoding service like OpenStreetMap Nominatim
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {"lat": lat, "lon": lon, "format": "json", "accept-language": "en"}
        headers = {"User-Agent": "GuardianEye-SafetyApp/1.0"}

        response = requests.get(url, params=params, headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            # Extract city, state, country information
            address = data.get("address", {})
            location_parts = []

            for key in ["city", "town", "village", "state", "country"]:
                if key in address:
                    location_parts.append(address[key])

            if location_parts:
                return " ".join(location_parts[:3])  # Limit to 3 parts

    except Exception:
        pass

    # Fallback to coordinates
    return f"location {lat} {lon}"


# Pydantic models
class CurrentWeather(BaseModel):
    time: str
    interval: int
    temperature: float
    windspeed: float
    winddirection: int
    is_day: Literal[0, 1]
    weathercode: int


class WeatherInput(BaseModel):
    current_weather: CurrentWeather


# LLM wrapper function
def llmchat(lat: float, lon: float, weather_data: dict):
    """
    Calls the LLM with weather data, news data and coordinates, returning structured risk info.
    Converts numeric weather code into a string description.
    """
    # Wrap the inner dict to match WeatherInput model
    weather_data_wrapped = {"current_weather": weather_data}
    weather = WeatherInput(**weather_data_wrapped)
    weather_dict = weather.dict()

    # Add human-readable weather description
    code = weather_dict["current_weather"]["weathercode"]
    weather_dict["current_weather"]["weather_description"] = weather_codes.get(
        code, "Unknown"
    )

    # Fetch nearby news
    print(f"Fetching news for location: {lat}, {lon}")
    news_data = get_nearby_news(lat, lon)
    print(f"News data: {news_data}")

    # Create a chain and invoke the LLM
    chain = prompt_template | model
    response = chain.invoke(
        {"weather_data": weather_dict, "news_data": news_data, "lat": lat, "lon": lon}
    )
    print(f"LLM Response: {response}")

    # Extract string content from AI message
    response_text = response.content.strip()

    # Clean up potential markdown formatting
    if response_text.startswith("```json"):
        response_text = response_text.replace("```json", "").replace("```", "").strip()
    elif response_text.startswith("```"):
        response_text = response_text.replace("```", "").strip()

    # Parse JSON safely
    try:
        risk_info = json.loads(response_text)

        # Validate required fields
        if not all(
            key in risk_info for key in ["Severity status", "Risk factor", "Reasons"]
        ):
            raise ValueError("Missing required fields in response")

    except (json.JSONDecodeError, ValueError) as e:
        print(f"JSON parsing/validation error: {e}")
        print(f"Raw response: {response_text}")

        # Fallback response based on weather conditions
        code = weather_dict["current_weather"]["weathercode"]
        temp = weather_dict["current_weather"]["temperature"]
        wind = weather_dict["current_weather"]["windspeed"]

        # Simple fallback risk assessment
        if code in [95, 96, 99] or wind > 20:  # Thunderstorms or high winds
            severity = "high risk"
            risk_factor = 8
        elif (
            code in [61, 63, 65, 82] or temp < 0 or temp > 40
        ):  # Heavy rain or extreme temps
            severity = "medium risk"
            risk_factor = 5
        else:
            severity = "low risk"
            risk_factor = 2

        risk_info = {
            "Severity status": severity,
            "Risk factor": risk_factor,
            "Reasons": [
                "Fallback assessment due to LLM parsing error",
                f"Weather: {weather_dict['current_weather']['weather_description']}",
            ],
            "error": f"LLM response parsing failed: {str(e)}",
            "raw_output": response_text,
        }

    return risk_info


load_dotenv()

app = FastAPI()
app.cors_origins = ["*"]


@app.get("/check")
async def checker(lat: float, lon: float):
    API_KEY = os.getenv("weather_api_key")
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"

    response = requests.get(url)
    data = response.json()
    result = llmchat(lat, lon, weather_data=data["current_weather"])
    return result
