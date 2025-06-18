# import the necessary packages
from cameraAI.detection import config
import json
import numpy as np
import cv2
from pathlib import Path
import depthai as dai


def create_image_pipeline(config_path, model_path):
   """
   Creates and configures a DepthAI pipeline for YOLO-based image detection.

   This function initializes a DepthAI pipeline and sets up the required nodes,
   including input, detection network, and output nodes. It parses the model
   configuration file to extract necessary metadata required for configuring
   the YOLO detection network, such as classes, coordinates, anchors, IOU
   threshold, and confidence threshold. The function links the nodes to complete
   the pipeline and prepares it for execution.

   :param config_path:
       Path to the configuration file containing the YOLO model configurations,
       including neural network configuration details like classes, anchors,
       confidence threshold, and other metadata.
   :param model_path:
       Path to the model blob file containing the weights and trained parameters
       for the YOLO neural network.
   :return:
       A configured DepthAI pipeline ready for execution with nodes set up for
       YOLO image detection.
   :rtype: dai.Pipeline
   """
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
   """
   Creates and configures a DepthAI pipeline utilizing an OAK camera for object
   detection. The function initializes a depthai pipeline, sets up sources,
   outputs, and defines a YOLO detection network for object detection based
   on the provided configuration and model files.

   :param config_path: The file path to the JSON configuration file containing
       the settings for the neural network model, such as classes, IOU and
       confidence thresholds, anchors, etc.
   :type config_path: str

   :param model_path: The file path to the compiled model in a .blob format
       which contains model weights and inference-related data compatible with
       DepthAI.
   :type model_path: str

   :return: A depthai.Pipeline object that is ready to be used with a DepthAI
       device. The pipeline includes a camera source, a YOLO object detection
       network, and the required data outputs for frames and detections.
   :rtype: dai.Pipeline
   """
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
   """
   Loads a configuration from a JSON file.

   The function reads the specified configuration file and parses it
   using the `json` module to return the loaded configuration data.
   This function assumes that the file exists and contains valid JSON data.

   :param config_path: A Path object representing the path to the
       configuration file.
   :type config_path: pathlib.Path

   :return: A dictionary containing the parsed configuration data
       from the JSON file.
   :rtype: dict
   """
   # open the config file and load using json module
   with config_path.open() as f:
       config = json.load(f)
       return config

def annotateFrame(frame, detections, model_name):
   """
   Annotates a given video or image frame with detection results including the model name,
   class labels, confidence scores, and bounding boxes around detected objects.

   This function processes each detection in the provided frame by normalizing bounding
   box coordinates, adding text annotations for model name, detection class, confidence
   percentage, and drawing a bounding box around objects. The annotations are styled
   with red color for better visibility.

   :param frame: The video or image frame to annotate.
   :type frame: numpy.ndarray
   :param detections: A list of detection objects. Each detection contains details
                      about the bounding box, confidence score, and class label.
   :type detections: list[detection]
   :param model_name: The name of the model used for predictions, which will be displayed
                      on the annotated frame.
   :type model_name: str
   :return: The annotated frame with all detection results visualized.
   :rtype: numpy.ndarray
   """
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
   """
   Resize the given NumPy array to the specified shape and modify its channel
   dimensions to a planar format. The function performs a resizing operation
   and then transposes the channel dimensions.

   :param arr: Input NumPy array representing the image data.
   :type arr: numpy.ndarray
   :param shape: Target shape as a tuple (width, height).
   :type shape: tuple
   :return: Transposed NumPy array in planar format.
   :rtype: numpy.ndarray
   """
   # resize the image array and modify the channel dimensions
   resized = cv2.resize(arr, shape)
   return resized.transpose(2, 0, 1)

def frameNorm(frame, bbox):
   """
   Normalizes bounding box coordinates to the frame's dimensions and ensures the
   values are clipped to fall within the specified range. It scales the bounding box
   coordinates (bbox) using the width and height of the frame and returns the result
   as integer values.

   :param frame: The input frame, typically as a NumPy array representing an image.
   :param bbox: List or array representing the bounding box coordinates normalized
       within the range <0..1>.
   :return: The bounding box coordinates adjusted to the dimensions of the frame as
       integers.
   """
   # nn data, being the bounding box locations, are in <0..1> range
   # normalized them with frame width/height
   normVals = np.full(len(bbox), frame.shape[0])
   normVals[::2] = frame.shape[1]
   return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)