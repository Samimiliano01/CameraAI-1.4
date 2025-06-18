import requests
import os
from cameraAI.dto.DetectionRecordDto import DetectionRecordDto
from dotenv import load_dotenv
import queue
from cameraAI.hardware import gps_manager
from cameraAI.external_api import external_api
import threading

# Load in secrets
load_dotenv()
API_ENDPOINT = os.getenv("API_URL")
API_KEY = os.getenv("API_KEY")

detection_queue = queue.Queue()

def detection_worker():
    """
    Processes detection tasks from a queue in a loop and creates detection records.

    This function runs indefinitely until a `None` value is retrieved from the
    queue, which signals a shutdown. For each item processed, it extracts the
    detection result and corresponding coordinates, retrieves the address
    associated with the coordinates (if available), and creates a detection
    record by calling `post_detection_record`.

    If any errors occur during processing, they are logged to the standard output.

    :raises Exception: Logs exceptions encountered during detection record
        creation or address retrieval, but does not stop the worker thread.
    """
    while True:
        data = detection_queue.get()
        if data is None:
            # Allows clean shutdown
            break

        result, coords = data
        try:
            # Retrieve the address based on coördinates.
            address = external_api.get_address_from_coordinates(*coords) if coords else "No GPS"
            # Post the record to the API.
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
    """
    Posts a detection record to a specified API endpoint.

    This function sends a POST request to the configured API endpoint with a
    detection record payload. The payload is in JSON format, and the request
    includes an API key in the header for authentication.

    The function attempts to log the server's response, printing either a success
    message with the response data, or an error message with the status code and
    response body.

    :param detection_record: An instance of DetectionRecordDto containing the
        detection record data to be sent.
    :return: None
    """
    headers = {
        "Content-Type": "application/json",
        "x-api-key": API_KEY
    }

    data = detection_record.to_dict()

    response = requests.post(url=API_ENDPOINT + "/litters", json=data, headers=headers)

    # Print a result depending on if the request was succesfull
    if response.status_code == 200:
        print("Success:", response.json())
    else:
        print("Error:", response.status_code, response.text)

# Start the detection thread which sends the detections over to our API.
threading.Thread(target=detection_worker, daemon=True).start()