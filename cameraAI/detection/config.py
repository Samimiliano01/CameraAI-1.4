# import the necessary packages
import os
import glob
# define path to the model, test data directory and results
#AImodel/datasets/solidwaste_project/yolov8n_v1_results/weights/best.blob
YOLOV8N_MODEL = os.path.join(
    "AImodel", "datasets", "solidwaste_project","yolov8n_v1_results", "weights","best_openvino_2022.1_6shave.blob"
    #"..","..","AImodel", "datasets", "solidwaste_project","yolov8n_v1_results", "weights","best_openvino_2022.1_6shave.blob"
)
YOLOV8N_CONFIG = os.path.join(
    "AImodel", "datasets", "solidwaste_project","yolov8n_v1_results", "weights","best.json"
    #"..","..","AImodel", "datasets", "solidwaste_project","yolov8n_v1_results", "weights","best.json"
)

CONFIDENCE = 0.8

TEST_DATA = glob.glob("AImodel/datasets/litter_dataset-1/test/images/*.jpg")
#TEST_DATA = glob.glob("../../AImodel/datasets/litter_dataset-1/test/images/*.jpg")
OUTPUT_IMAGES_YOLOv8n = os.path.join("results", "gesture_pred_images_v8n")
OUTPUT_IMAGES_YOLOv8s = os.path.join("results", "gesture_pred_images_v8s")
OUTPUT_VIDEO_YOLOv8n = os.path.join("results", "gesture_camera_v8n.mp4")
OUTPUT_VIDEO_YOLOv8s = os.path.join("results", "gesture_camera_v8s.mp4")
# define camera preview dimensions same as YOLOv8 model input size
CAMERA_PREV_DIM = (416, 416)
# define the class label names list
LABELS = [
    "cans", "cardboard", "colored glass bottles", "face mask", "glass bottle",
    "HDPE", "LDPE", "PET", "PVC", "paper bag", "paper cup",
    "paperboard", "peel", "pile of leaves", "rags", "styrofoam", "tetra pak"
]