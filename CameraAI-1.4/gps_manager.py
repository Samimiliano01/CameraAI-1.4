from serial import Serial
from pyubx2 import UBXReader, NMEA_PROTOCOL, UBX_PROTOCOL

def get_gps():
  with Serial('/dev/ttyACM8d', 38400, timeout=3) as stream:
    ubr = UBXReader(stream, protfilter=NMEA_PROTOCOL | UBX_PROTOCOL)
    raw_data, parsed_data = ubr.read()
    if parsed_data is not None:
      print(parsed_data)