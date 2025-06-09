import requests
import os
import json
from threading import Timer
from cameraAI.dto.DetectionRecordDto import DetectionRecordDto
from dotenv import load_dotenv

class LoginInfo:

    def __init__(self, access_token: str="", expires_in: float=0, refresh_token: str=""):
        self.access_token = access_token
        self.expires_in = expires_in
        self.refresh_token = refresh_token

    def refresh(self):
        Timer(self.expires_in, lambda: login(self))


load_dotenv()
API_ENDPOINT = os.getenv("API_URL")
login_info: LoginInfo | None = None

def login(info: LoginInfo):
    headers = {
        "Content-Type": "application/json",
    }

    data = {"email": os.getenv("API_USERNAME"), "password": os.getenv("API_PASSWORD")}

    response = requests.post(API_ENDPOINT + "/account/login", headers=headers, data=data)

    if response.status_code == 200:
        json_response = response.json()
        info.access_token = json_response["access_token"]
        info.expires_in = json_response["expires_in"]
        info.refresh_token = json_response["refresh_token"]
        info.refresh()
    else:
        print("Error:", response.status_code, response.text)


def post_detection_record(detection_record: DetectionRecordDto):
    if login_info is None:
        login(LoginInfo())

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {login_info.access_token}"
    }

    data = json.dumps(detection_record.to_dict())

    response = requests.post(url=API_ENDPOINT + "Litter", data=data, headers=headers)

    if response.status_code == 200:
        print("Success:", response.json())
    else:
        print("Error:", response.status_code, response.text)