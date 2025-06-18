from cameraAI.hardware import gps_manager
from cameraAI.sender import send_to_api
from cameraAI.detection import config
from cameraAI.detection import utils
import cv2
import depthai as dai
from imutils.video import FPS
import time

def main():
    """
    Main function to initialize and run DepthAI camera pipeline for real-time
    inference using YOLOv8. Processes video frames and detections, annotates
    frames with results, calculates FPS, and optionally sends detection data to
    an API. The function uses a video writer to output the processed video and
    allows exiting via the 'q' key.

    :raises RuntimeError: If there are issues initializing or starting the DepthAI
        device.
    :raises ValueError: When the detections or GPS data do not conform to expected
        conditions.

    :return: None
    """
    # initialize a depthai camera pipeline
    print("[INFO] initializing a depthai images pipeline...")


    pipeline = utils.create_camera_pipeline(config_path=config.YOLOV8N_CONFIG,
                                            model_path=config.YOLOV8N_MODEL)

    output_video = config.OUTPUT_VIDEO_YOLOv8n
    previous_coords = (0,0)
    # set the video codec to use with video writer
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')

    out = cv2.VideoWriter(
       output_video,
       fourcc,
       20.0,
       config.CAMERA_PREV_DIM
    )

    # pipeline defined, now the device is assigned and pipeline is started
    with dai.Device(pipeline, usb2Mode=True) as device:
        # output queues will be used to get the rgb frames
        # and nn data from the outputs defined above
        qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
        # initialize variables like frame, start time for NN FPS
        # also start the FPS module timer, define color pattern for FPS text
        # frame = None
        startTime = time.monotonic()
        fps = FPS().start()
        counter = 0
        color2 = (255, 255, 255)

        print("[INFO] starting inference with OAK camera...")
        while True:
            # fetch the RGB frames and YOLO detections for the frame
            inRgb = qRgb.get()
            inDet = qDet.get()
            if inRgb is not None:
                pass
                #used to show video

                # convert inRgb output to a format OpenCV library can work
                frame = inRgb.getCvFrame()
                # annotate the frame with FPS information
                cv2.putText(frame, "NN fps: {:.2f}".format(counter / (time.monotonic() - startTime)),
                            (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.8, color2)
                # update the FPS counter
                fps.update()
            if inDet is not None:
                # if inDet is not none, fetch all the detections for a frame
                detections = inDet.detections

                coords = gps_manager.gps_data.get()

                if len(detections) >= 0 and (
                        previous_coords is None or coords is None or gps_manager.get_distance_between(coords, previous_coords) >= 2):
                    for detection in detections:
                        result = config.LABELS[detection.label]
                        print("type: " + result)
                        print("confidence: " + str(detection.confidence))
                        if detection.confidence < config.CONFIDENCE:
                            continue

                        previous_coords = coords

                        send_to_api.detection_queue.put((result, coords))
                        time.sleep(5)

                counter += 1


            #This is used to show the video

            if frame is not None:
                # annotate frame with detection results
                frame = utils.annotateFrame(frame, detections, "video")
                # display the frame with gesture output on the screen
                cv2.imshow("video", frame)
            # write the annotated frame to the file
            out.write(frame)
            # break out of the while loop if `q` key is pressed
            if cv2.waitKey(1) == ord('q'):
                break

    #stop the timer and display FPS information
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    # do a bit of cleanup
    out.release()
    cv2.destroyAllWindows()
