import requests
import os
from dotenv import load_dotenv

# Load in the environment secrets
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("API key niet gevonden. Voeg GOOGLE_API_KEY toe.")


def get_address_from_coordinates(lat: float,lng: float) -> str:
    """
    Get a formatted address from geographic coordinates using Google's Geocoding API.

    This function sends a request to the Google Maps Geocoding API with the given
    latitude and longitude as input parameters. It parses the response JSON to extract
    the formatted address if the request is successful. If the request fails or the
    response does not contain an address, an exception is raised.

    :param lat: Latitude of the location for which the address is being searched.
    :type lat: float
    :param lng: Longitude of the location for which the address is being searched.
    :type lng: float
    :return: The formatted address corresponding to the provided latitude and longitude.
    :rtype: str
    :raises Exception: If the Geocoding API operation fails or an address cannot be
        retrieved from the response.
    """
    url = "https://maps.googleapis.com/maps/api/geocode/json"

    params = {"latlng": f"{lat},{lng}", "key": API_KEY}
    response = requests.get(url, params=params)
    data = response.json()
    if data["status"] == "OK" and data["results"]:
        return data["results"][0]["formatted_address"]
    else:
        raise Exception(f"Geocoding mislukt: {data.get('status')} - {data.get('error_message', 'Geen extra info')}")
