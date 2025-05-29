import datetime

import cv2
import depthai as dai
from cameraAI.hardware import gps_manager
from cameraAI.detection import detect
from ultralytics import YOLO
from cameraAI.sender import send_to_api
from cameraAI.dto import DetectionRecordDto


def main():
    previous_coords = (0, 0)

    print("starting camera")

    pipeline = dai.Pipeline()

    color_cam = pipeline.createColorCamera()
    color_cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)  # New recommended socket
    color_cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    color_cam.setInterleaved(False)
    color_cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    color_cam.setFps(1)  # Reduce FPS to lower USB load

    xout = pipeline.createXLinkOut()
    xout.setStreamName("color")
    color_cam.preview.link(xout.input)
    color_cam.setPreviewSize(1280, 720)

    with dai.Device(pipeline) as device:
        queue = device.getOutputQueue(name="color", maxSize=4, blocking=False)
        model = YOLO("../AImodel/datasets/solidwaste_project/yolov8n_v1_results/weights/best.pt")


        while True:
            frame = queue.tryGet()
            if frame is not None:
                color_frame = frame.getCvFrame()
                cv2.imshow("Color Camera", color_frame)
                detected = detect.detect_litter(color_frame, model, 0.7)
                if len(detected) > 0:
                    for cat in detected:
                        coords = gps_manager.get_gps_coordinates()
                        if gps_manager.get_distance_between(coords, previous_coords) < 2:
                            break
                        previous_coords = coords
                        send_to_api.post_detection_record(DetectionRecordDto.DetectionRecordDto(cat, coords, "Home", gps_manager.get_local_time(coords)))

            if cv2.waitKey(1) == ord('q'):
                break

    cv2.destroyAllWindows()
