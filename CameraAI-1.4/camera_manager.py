import cv2
import depthai as dai

def main():
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

        while True:
            frame = queue.tryGet()
            if frame is not None:
                color_frame = frame.getCvFrame()
                cv2.imshow("Color Camera", color_frame)

            if cv2.waitKey(1) == ord('q'):
                break

    cv2.destroyAllWindows()
