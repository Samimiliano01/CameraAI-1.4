# import the necessary packages
import config
import utils
from imutils.video import FPS
import time
import cv2
import depthai as dai

# initialize a depthai camera pipeline
print("[INFO] initializing a depthai camera pipeline...")


pipeline = utils.create_camera_pipeline(config_path=config.YOLOV8N_CONFIG,
                                   model_path=config.YOLOV8N_MODEL)
output_video = config.OUTPUT_VIDEO_YOLOv8n

# set the video codec to use with video writer
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# create video writer object with parameters: output video path,
# video codec, frame rate of output video, and dimensions of video frame
out = cv2.VideoWriter(
   output_video,
   fourcc,
   20.0,
   config.CAMERA_PREV_DIM
)

# pipeline defined, now the device is assigned and pipeline is started
with dai.Device(pipeline) as device:
    # output queues will be used to get the rgb frames
    # and nn data from the outputs defined above
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
    # initialize variables like frame, start time for NN FPS
    # also start the FPS module timer, define color pattern for FPS text
    frame = None
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
            counter += 1
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

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# do a bit of cleanup
out.release()
cv2.destroyAllWindows()
