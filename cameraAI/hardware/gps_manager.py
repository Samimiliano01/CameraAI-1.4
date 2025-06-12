import zoneinfo
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2
from serial import Serial
from pyubx2 import UBXReader, NMEA_PROTOCOL, UBX_PROTOCOL
import timezonefinder
import threading
import time

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












class GPSData:
  def __init__(self):
    self._lock = threading.Lock()
    self.coords: tuple[float, float] | None = None

  def unset(self):
    with self._lock:
      self.coords = None

  def set(self, lat: float, lon: float):
    with self._lock:
      self.coords = (lat, lon)

  def get(self) -> tuple[float, float] | None:
    with self._lock:
      return self.coords

def get_gps_coordinates():
  print("Running...")
  while True:
    try:
      with Serial('/dev/ttyACM0', 9600, timeout=3) as stream:
        ubr = UBXReader(stream, protfilter=NMEA_PROTOCOL | UBX_PROTOCOL)
        raw_data, parsed_data = ubr.read()
        if parsed_data is not None:
          payload = parsed_data.payload
          print(payload)
          if len(payload) < 6:
            print("Couldn't get location!")
            gps_data.unset()
            continue
          # if on the other side of UTC meridian or southern hemisphere multiply latitude/longitude by -1.
          latitude = payload[2] if payload[3] == 'N' else payload[2] * -1
          longitude = payload[4] if payload[5] == 'E' else payload[4] * -1
          if latitude is not None and longitude is not None and latitude != "" and longitude != "":
            gps_data.set(float(latitude) / 100, float(longitude) / 100)
            continue
        continue
    except Exception as e:
      print(f"Error reading GPS: {e}")
    time.sleep(5)

def get_distance_between(coord1, coord2) -> float:
  earth_radius = 6371000
  lat1, lon1 = coord1
  lat2, lon2 = coord2

  dlat = radians(lat2 - lat1)
  dlon = radians(lon2 - lon1)
  lat1 = radians(lat1)
  lat2 = radians(lat2)

  a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
  c = 2 * atan2(sqrt(a), sqrt(1 - a))

  return earth_radius * c

def get_local_time(coord) -> datetime:
  if coord is None:
    return datetime.now()
  tzf = timezonefinder.TimezoneFinder()
  tz_name = tzf.timezone_at(lat=coord[0], lng=coord[1])
  tz = zoneinfo.ZoneInfo(tz_name)
  now = datetime.now(tz)
  return now

gps_data: GPSData

def main():
  global gps_data
  gps_data = GPSData()
  threading.Thread(target=get_gps_coordinates, daemon=True).start()

  try:
    while True:
      time.sleep(1)
  except KeyboardInterrupt:
    print("Exiting gracefully.")

main()






