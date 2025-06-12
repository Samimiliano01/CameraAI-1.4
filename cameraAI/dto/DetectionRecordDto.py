import datetime

class DetectionRecordDto:
    def __init__(self, type_of_trash: str, coordinates: tuple[float, float], location: str, time: datetime.datetime) -> None:
        self.typeOfTrash = type_of_trash
        self.coordinates = coordinates
        self.location = location
        self.time = time

    def to_dict(self):
        return {
            "typeOfTrash": self.typeOfTrash,
            "Coordinates": self.coordinates,
            "Location": self.location,
            "Time": self.time.isoformat()  # JSON-serialiseerbare tijd
        }
