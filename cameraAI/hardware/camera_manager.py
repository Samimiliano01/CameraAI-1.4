import depthai as dai
from cameraAI.hardware import gps_manager
from cameraAI.detection.detect import postprocess_blob_output  # You must create this
from cameraAI.sender import send_to_api
from cameraAI.dto import DetectionRecordDto
from cameraAI.external_api import external_api

def main():
    previous_coords = (0, 0)
    blob_path = "AImodel/datasets/solidwaste_project/yolov8n_v1_results/weights/best.blob"

    print("Starting OAK-D with blob model...")

    # Create pipeline
    pipeline = dai.Pipeline()

    cam = pipeline.create(dai.node.ColorCamera)
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setPreviewSize(640, 640)  # match YOLO input
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.setFps(30)

    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setBlobPath(blob_path)
    cam.preview.link(nn.input)

    nn_out = pipeline.create(dai.node.XLinkOut)
    nn_out.setStreamName("nn")
    nn.out.link(nn_out.input)

    with dai.Device(pipeline, usb2Mode=True) as device:
        output_queue = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

        while True:
            in_nn = output_queue.tryGet()
            if in_nn is None:
                continue

            raw_output = in_nn.getFirstLayerFp16()

            detections = postprocess_blob_output(raw_output, conf_threshold=0.6)
            print(detections)
            if len(detections) > 0:
                for cat in detections:
                    coords = gps_manager.get_gps_coordinates()
                    if gps_manager.get_distance_between(coords, previous_coords) < 2:
                        continue
                    previous_coords = coords
                    print("sending result!")
                    send_to_api.post_detection_record(
                        DetectionRecordDto.DetectionRecordDto(
                            cat,
                            coords,
                            external_api.get_address_from_coordinates(coords[0], coords[1]),
                            gps_manager.get_local_time(coords)
                        )
                    )
