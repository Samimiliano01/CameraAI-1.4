import requests
import os
from cameraAI.dto.DetectionRecordDto import DetectionRecordDto
from dotenv import load_dotenv
import queue
from cameraAI.hardware import gps_manager
from cameraAI.external_api import external_api
import threading

load_dotenv()
API_ENDPOINT = os.getenv("API_URL")
API_KEY = os.getenv("API_KEY")
detection_queue = queue.Queue()

def detection_worker():
    while True:
        data = detection_queue.get()
        if data is None:
            break  # Allows clean shutdown

        result, coords = data
        try:
            address = external_api.get_address_from_coordinates(*coords) if coords else "No GPS"
            post_detection_record(
                DetectionRecordDto(
                    result,
                    coords if coords else (0, 0),
                    address,
                    gps_manager.get_local_time(coords)
                )
            )
        except Exception as e:
            print("[Worker error]", e)
        detection_queue.task_done()

def post_detection_record(detection_record: DetectionRecordDto):
    headers = {
        "Content-Type": "application/json",
        "x-api-key": API_KEY
    }

    data = detection_record.to_dict()

    response = requests.post(url=API_ENDPOINT + "/litters", json=data, headers=headers)

    if response.status_code == 200:
        print("Success:", response.json())
    else:
        print("Error:", response.status_code, response.text)

threading.Thread(target=detection_worker, daemon=True).start()