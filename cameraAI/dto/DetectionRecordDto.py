import datetime

class DetectionRecordDto:
    def __init__(self, category: str, coordinates: tuple[float, float], location: str, time: datetime.datetime) -> None:
        self.category = category
        self.coordinates = coordinates
        self.location = location
        self.time = time

    def to_dict(self):
        return {
            "category": self.category,
            "coordinates": self.coordinates,
            "location": self.location,
            "time": self.time.isoformat()  # JSON-serialiseerbare tijd
        }
