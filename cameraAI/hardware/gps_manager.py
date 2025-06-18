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

# Load in the environment variables
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("API key niet gevonden. Voeg GOOGLE_API_KEY toe.")


def get_address_from_coordinates(lat: float,lng: float) -> str:
    """
    Fetches a formatted address as a string from coordinates (latitude and longitude)
    by making a request to the Google Maps Geocoding API. It converts the given coordinate
    pair into a readable address. The function expects valid float values for the
    latitude and longitude within their acceptable ranges.

    :param lat: The latitude of the coordinate pair.
    :type lat: float
    :param lng: The longitude of the coordinate pair.
    :type lng: float
    :return: The formatted address corresponding to the given coordinates.
    :rtype: str
    :raises Exception: If the geocoding request fails or does not return expected results.
    """
    url = "https://maps.googleapis.com/maps/api/geocode/json"

    params = {"latlng": f"{lat},{lng}", "key": API_KEY}
    response = requests.get(url, params=params)
    data = response.json()
    if data["status"] == "OK" and data["results"]:
        return data["results"][0]["formatted_address"]
    else:
        raise Exception(f"Geocoding mislukt: {data.get('status')} - {data.get('error_message', 'Geen extra info')}")

class GPSData:
  """
  Represents a GPS data handler for storing and retrieving coordinates.

  This class provides a thread-safe mechanism to store GPS coordinates
  (latitude and longitude). It allows setting, unsetting, and retrieving
  the coordinates. The coordinates are stored as a tuple of latitude and
  longitude values or set to None when not available.

  :ivar coords: Holds the current GPS coordinates as a tuple of latitude
      and longitude. It is None if no coordinates are set.
  :type coords: Tuple[float, float] | None
  """
  def __init__(self):
    self._lock = threading.Lock()
    self.coords: tuple[float, float] | None = None

  def unset(self):
    """
    Unset the `coords` attribute safely within a thread-safe context using
    a lock. This method ensures that the `coords` attribute is set to None
    in a thread-safe manner by acquiring a lock before modification.

    :return: None
    """
    with self._lock:
      self.coords = None

  def set(self, lat: float, lon: float):
    """
    Sets the coordinates using the provided latitude and longitude values.

    This method updates the internal coordinates attribute by acquiring
    a lock to ensure thread safety during the modification. The new
    coordinates are specified by the parameters `lat` and `lon`.

    :param lat: The latitude value to set.
    :param lon: The longitude value to set.
    :return: None
    """
    with self._lock:
      self.coords = (lat, lon)

  def get(self) -> tuple[float, float] | None:
    """
    Retrieves the current coordinates stored in the object.

    If the coordinates are available, they are returned as a tuple
    containing two float values. If the coordinates are not available,
    None is returned. This method ensures thread-safe access to the
    coordinate data by using a lock.

    :returns: A tuple containing two float values representing the
              coordinates in the format: [Latitude, Longitude] if available, otherwise None.
    :rtype: tuple[float, float] | None
    """
    with self._lock:
      return self.coords

def get_gps_coordinates():
  """
  Retrieve GPS coordinates using the Serial interface and parse relevant data
  using the UBXReader. Continuously attempts to read data from the GPS device
  and update the `gps_data` with latitude and longitude if available. Handles
  parsing and validation to ensure proper GPS data retrieval.

  The function operates indefinitely, retrying upon failure and waiting
  between retries. Errors encountered during reading or parsing are logged,
  allowing the process to continue.

  :raises Exception: Captures and logs all exceptions encountered during
      device reading or parsing.

  :return: This function does not return any value; it updates shared data
      structures as its primary operation.
  """
  print("Running...")
  while True:
    try:
      # Open a data stream on the gps usb.
      with Serial('/dev/ttyACM0', 9600, timeout=3) as stream:
        ubr = UBXReader(stream, protfilter=NMEA_PROTOCOL | UBX_PROTOCOL)
        raw_data, parsed_data = ubr.read()
        if parsed_data is not None:
          payload = parsed_data.payload
          # If the length is shorter than 6 something has gone wrong with the GPS retrieval (probably because there is no signal).
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
  """
  Calculate the great-circle distance between two points on the Earth's surface.

  This function computes the shortest distance over the Earth's surface using the haversine
  formula, accounting for the curvature of the Earth. It takes in two coordinates, each provided
  as latitude and longitude in decimal degrees, and returns the great-circle distance between
  them in meters. The calculation assumes a spherical Earth with a fixed radius.

  :param coord1: A tuple representing the latitude and longitude (in decimal degrees) of the
      first location.
  :param coord2: A tuple representing the latitude and longitude (in decimal degrees) of the
      second location.
  :return: The great-circle distance between the two locations in meters as a float.
  """
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
  """
  Determine the local time at a specified geographical coordinate.

  This function calculates the local time for a given geographical coordinate using
  the associated timezone. If no coordinate is supplied, it defaults to returning
  the current local time in the system's timezone.

  :param coord: A tuple containing latitude and longitude in decimal degrees.
  :type coord: tuple[float, float] | None
  :return: The current local time for the specified location or the system's
      local time if no location is provided.
  :rtype: datetime
  """
  if coord is None:
    return datetime.now()
  tzf = timezonefinder.TimezoneFinder()
  tz_name = tzf.timezone_at(lat=coord[0], lng=coord[1])
  tz = zoneinfo.ZoneInfo(tz_name)
  now = datetime.now(tz)
  return now

gps_data: GPSData

def main():
  """
  Initializes and runs the main execution loop for the program.

  This function starts a separate thread to continuously fetch GPS coordinates
  using the `get_gps_coordinates` function. It sets up a global `gps_data` object,
  which is an instance of the `GPSData` class. The main thread then enters an
  infinite loop, which can be interrupted by a keyboard interrupt to terminate
  the program gracefully.

  :global gps_data: A global instance of the `GPSData` class used to store GPS
                    data collected by the program.
  :raises KeyboardInterrupt: Raised when the user manually interrupts the
                             program's execution, e.g., pressing Ctrl+C.
  :return: None
  """
  global gps_data
  gps_data = GPSData()
  # Start a thread
  threading.Thread(target=get_gps_coordinates, daemon=True).start()

main()






