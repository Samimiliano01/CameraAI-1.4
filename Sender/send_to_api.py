import requests
import json
from DetectionRecord import DetectionRecord

API_ENDPOINT = "lokaal ip adres | de port"
API_KEY = "XXXXXXXXXXXXXXXXX"


def Post_Detection_Record(detection_record: DetectionRecord)->DetectionRecord:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    data = json.dumps(detection_record.to_dict())   # of detection_record.to_dict() als je die hebt

    response = requests.post(url=API_ENDPOINT, data=data, headers=headers)

    if response.status_code == 200:
        print("Success:", response.json())
    else:
        print("Error:", response.status_code, response.text)

    return response