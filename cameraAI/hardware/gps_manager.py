import zoneinfo
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2
from serial import Serial
from pyubx2 import UBXReader, NMEA_PROTOCOL, UBX_PROTOCOL
import timezonefinder

def get_gps_coordinates() -> tuple[float, float] | None:
  with Serial('/dev/ttyACM0', 9600, timeout=3) as stream:
    ubr = UBXReader(stream, protfilter=NMEA_PROTOCOL | UBX_PROTOCOL)
    raw_data, parsed_data = ubr.read()
    if parsed_data is not None:
      payload = parsed_data.payload
      # if on the other side of UTC meridian or southern hemisphere multiply latitude/longitude by -1.
      latitude = payload[2] if payload[3] is 'N' else payload[2] * -1
      longitude = payload[4] if payload[5] is 'E' else payload[4] * -1
      return latitude, longitude
    return None

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
  tzf = timezonefinder.TimezoneFinder()
  tz_name = tzf.timezone_at(lat=coord[0], lng=coord[1])
  tz = zoneinfo.ZoneInfo(tz_name)
  now = datetime.now(tz)
  return now