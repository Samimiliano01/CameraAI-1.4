# import the necessary packages
from cameraAI.detection import config
import json
import numpy as np
import cv2
from pathlib import Path
import depthai as dai
def create_image_pipeline(config_path, model_path):
   # initialize a depthai pipeline
   pipeline = dai.Pipeline()
   # load model config file and fetch nn_config parameters
   print("[INFO] loading model config...")
   configPath = Path(config_path)
   model_config = load_config(configPath)
   nnConfig = model_config.get("nn_config", {})

   print("[INFO] extracting metadata from model config...")
   # using nnConfig extract metadata like classes,
   # iou and confidence threshold, number of coordinates
   metadata = nnConfig.get("NN_specific_metadata", {})
   classes = metadata.get("classes", {})
   coordinates = metadata.get("coordinates", {})
   anchors = metadata.get("anchors", {})
   anchorMasks = metadata.get("anchor_masks", {})
   iouThreshold = metadata.get("iou_threshold", {})
   confidenceThreshold = metadata.get("confidence_threshold", {})

   print("[INFO] configuring inputs and output...")
   # configure inputs for depthai pipeline
   # since this pipeline is dealing with images an XLinkIn node is created
   detectionIN = pipeline.createXLinkIn()
   # create a Yolo detection node
   detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
   # create a XLinkOut node for fetching the neural network outputs to host
   nnOut = pipeline.create(dai.node.XLinkOut)
   print("[INFO] setting stream names for queues...")
   # set stream names used in queue to fetch data when the pipeline is started
   nnOut.setStreamName("nn")
   detectionIN.setStreamName("detection_in")

   print("[INFO] setting YOLO network properties...")
   # network specific settings - parameters read from config file
   # confidence and iou threshold, classes, coordinates are set
   # most important the model .blob file is used to load weights
   detectionNetwork.setConfidenceThreshold(confidenceThreshold)
   detectionNetwork.setNumClasses(classes)
   detectionNetwork.setCoordinateSize(coordinates)
   detectionNetwork.setAnchors(anchors)
   detectionNetwork.setAnchorMasks(anchorMasks)
   detectionNetwork.setIouThreshold(iouThreshold)
   detectionNetwork.setBlobPath(model_path)
   detectionNetwork.setNumInferenceThreads(2)
   detectionNetwork.input.setBlocking(False)

   print("[INFO] creating links...")
   # linking the nodes - image node output is linked to detection node
   # detection network node output is linked to XLinkOut input
   detectionIN.out.link(detectionNetwork.input)
   detectionNetwork.out.link(nnOut.input)
   # return the pipeline to the calling function
   return pipeline

def create_camera_pipeline(config_path, model_path):
   # initialize a depthai pipeline
   pipeline = dai.Pipeline()
   # load model config file and fetch nn_config parameters
   print("[INFO] loading model config...")
   configPath = Path(config_path)
   model_config = load_config(configPath)
   nnConfig = model_config.get("nn_config", {})
   print("[INFO] extracting metadata from model config...")
   # using nnConfig extract metadata like classes,
   # iou and confidence threshold, number of coordinates
   metadata = nnConfig.get("NN_specific_metadata", {})
   classes = metadata.get("classes", {})
   coordinates = metadata.get("coordinates", {})
   anchors = metadata.get("anchors", {})
   anchorMasks = metadata.get("anchor_masks", {})
   iouThreshold = metadata.get("iou_threshold", {})
   confidenceThreshold = metadata.get("confidence_threshold", {})
   # output of metadata - feel free to tweak the threshold parameters
   #{'classes': 5, 'coordinates': 4, 'anchors': [], 'anchor_masks': {},
   # 'iou_threshold': 0.5, 'confidence_threshold': 0.5}
   print(metadata)

   print("[INFO] configuring source and outputs...")
   # define sources and outputs
   # since OAK's camera is used in this pipeline
   # a color camera node is defined
   camRgb = pipeline.create(dai.node.ColorCamera)
   # create a Yolo detection node
   detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
   xoutRgb = pipeline.create(dai.node.XLinkOut)
   # create a XLinkOut node for getting the detection results to host
   nnOut = pipeline.create(dai.node.XLinkOut)
   print("[INFO] setting stream names for queues...")
   # set stream names used in queue to fetch data when the pipeline is started
   xoutRgb.setStreamName("rgb")
   nnOut.setStreamName("nn")

   print("[INFO] setting camera properties...")
   # setting camera properties like the output preview size,
   # camera resolution, color channel ordering and FPS
   camRgb.setPreviewSize(config.CAMERA_PREV_DIM)
   camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
   camRgb.setInterleaved(False)
   camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
   camRgb.setFps(40)

   print("[INFO] setting YOLO network properties...")
   # network specific settings - parameters read from config file
   # confidence and iou threshold, classes, coordinates are set
   # most important the model .blob file is used to load weights
   detectionNetwork.setConfidenceThreshold(confidenceThreshold)
   detectionNetwork.setNumClasses(classes)
   detectionNetwork.setCoordinateSize(coordinates)
   detectionNetwork.setAnchors(anchors)
   detectionNetwork.setAnchorMasks(anchorMasks)
   detectionNetwork.setIouThreshold(iouThreshold)
   detectionNetwork.setBlobPath(model_path)
   detectionNetwork.setNumInferenceThreads(2)
   detectionNetwork.input.setBlocking(False)
   print("[INFO] creating links...")
   # linking the nodes - camera stream output is linked to detection node
   # RGB frame is passed through detection node linked with XLinkOut
   # used for annotating the frame with detection output
   # detection network node output is linked to XLinkOut input
   camRgb.preview.link(detectionNetwork.input)

   detectionNetwork.passthrough.link(xoutRgb.input)
   detectionNetwork.out.link(nnOut.input)
   # return the pipeline to the calling function
   return pipeline

def load_config(config_path):
   # open the config file and load using json module
   with config_path.open() as f:
       config = json.load(f)
       return config
def annotateFrame(frame, detections, model_name):
    # loops over all detections in a given frame
    # annotates the frame with model name, class label,
    # confidence score, and draw bounding box on the object
    color = (0, 0, 255)
    for detection in detections:
        bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
        cv2.putText(frame, model_name, (20, 40), cv2.FONT_HERSHEY_TRIPLEX, 1,
                    color)
        cv2.putText(frame, config.LABELS[detection.label], (bbox[0] + 10, bbox[1] + 25), cv2.FONT_HERSHEY_TRIPLEX, 1,
                    color)
        cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 60),
                    cv2.FONT_HERSHEY_TRIPLEX, 1, color)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
    return frame

def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
   # resize the image array and modify the channel dimensions
   resized = cv2.resize(arr, shape)
   return resized.transpose(2, 0, 1)
def frameNorm(frame, bbox):
   # nn data, being the bounding box locations, are in <0..1> range
   # normalized them with frame width/height
   normVals = np.full(len(bbox), frame.shape[0])
   normVals[::2] = frame.shape[1]
   return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)