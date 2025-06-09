import datetime

class DetectionRecordDto:
    def __init__(self, category: str, coordinates: tuple[float, float], location: str, time: datetime.datetime) -> None:
        self.category = category
        self.coordinates = coordinates
        self.location = location
        self.time = time

    def to_dict(self):
        return {
            "Category": self.category,
            "Coordinates": self.coordinates,
            "Location": self.location,
            "Time": self.time.isoformat()  # JSON-serialiseerbare tijd
        }
