import requests
import json
from cameraAI.dto.DetectionRecordDto import DetectionRecordDto

API_ENDPOINT = "lokaal ip adres : de port"
API_KEY = "XXXXXXXXXXXXXXXXX"


def post_detection_record(detection_record: DetectionRecordDto)->DetectionRecordDto:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    data = json.dumps(detection_record.to_dict())

    response = requests.post(url=API_ENDPOINT, data=data, headers=headers)

    if response.status_code == 200:
        print("Success:", response.json())
    else:
        print("Error:", response.status_code, response.text)

    return response