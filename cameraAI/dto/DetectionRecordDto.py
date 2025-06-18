import datetime

class DetectionRecordDto:
    """
    Represents a record of a detected trash item including its type, location,
    and time of detection. This class is designed to encapsulate necessary
    information about trash detection and provide a method to convert the
    information into a dictionary format suitable for external processing,
    such as JSON serialization.

    :ivar typeOfTrash: Specifies the type of trash detected.
    :ivar coordinates: A tuple (latitude, longitude) representing
        geographic coordinates of the detection.
    :ivar location: A textual description of the location of detection.
    :ivar time: Datetime object indicating when the detection occurred.
    """
    def __init__(self, type_of_trash: str, coordinates: tuple[float, float], location: str, time: datetime.datetime) -> None:
        self.typeOfTrash = type_of_trash
        self.coordinates = coordinates
        self.location = location
        self.time = time

    def to_dict(self):
        """
        Converts the object's data to a dictionary representation.

        This method creates and returns a dictionary containing specific attributes
        of the class, such as 'typeOfTrash', 'coordinates', 'location', and 'time'.
        The 'time' attribute will be serialized into an ISO 8601 string format to
        ensure JSON-compatible data output.

        :return: A dictionary representing the object's data with attributes
            properly serialized, including 'typeOfTrash', 'coordinates',
            'location', and 'time'.
        :rtype: dict
        """
        return {
            "typeOfTrash": self.typeOfTrash,
            "Coordinates": self.coordinates,
            "Location": self.location,
            "Time": self.time.isoformat()  # JSON-serialiseerbare tijd
        }
