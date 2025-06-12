import requests
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("API key niet gevonden. Voeg GOOGLE_API_KEY toe.")


def get_address_from_coordinates(lat: float,lng: float) -> str:
    url = "https://maps.googleapis.com/maps/api/geocode/json"

    params = {"latlng": f"{lat},{lng}", "key": API_KEY}
    response = requests.get(url, params=params)
    data = response.json()
    if data["status"] == "OK" and data["results"]:
        return data["results"][0]["formatted_address"]
    else:
        raise Exception(f"Geocoding mislukt: {data.get('status')} - {data.get('error_message', 'Geen extra info')}")
