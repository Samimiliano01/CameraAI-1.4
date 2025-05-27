import datetime

class DetectionRecord:
    def __init__(self, detected_types: tuple[str], coordinates: tuple[float, float], location: str, time: datetime.datetime) -> None:
        self.detected_types = detected_types
        self.coordinates = coordinates
        self.location = location
        self.time = time

    def to_dict(self):
        return {
            "detected_types": self.detected_types,
            "coordinates": self.coordinates,
            "location": self.location,
            "time": self.time.isoformat()  # JSON-serialiseerbare tijd
        }
