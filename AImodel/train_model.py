import os
from roboflow import Roboflow
from ultralytics import YOLO

def train_model():
    """
    Train a YOLOv8 model for a specific dataset by downloading the latest dataset version,
    training the model using specified settings, and exporting the trained model.

    The function performs the following steps:
    - Sets the working directory for the dataset.
    - Retrieves the Roboflow API key from environment variables.
    - Downloads the latest version of the dataset from Roboflow.
    - Trains a YOLOv8 model using the downloaded dataset.
    - Validates the trained model.
    - Exports the trained model in ONNX format upon successful training.

    :raises ValueError: If the Roboflow API key is not set as an environment variable.

    :return: None
    """
    # Zet werkdirectory
    HOME = os.getcwd()
    DATASET_DIR = os.path.join(HOME, "datasets")
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.chdir(DATASET_DIR)

    API_KEY = os.getenv("ROBOFLOW_API_KEY")
    if not API_KEY:
        raise ValueError("Zet je ROBOFLOW_API_KEY als omgevingsvariabele!")

    rf = Roboflow(api_key=API_KEY)
    workspace = rf.workspace("test-mncux")
    project = workspace.project("litter_dataset-2vtbq")

    print("Beschikbare datasetversies:")
    for v in project.versions():
        print(f"- Versie {v.version} (ID: {v.id})")

    dataset_version = project.versions()[0]  # Nieuwste versie
    dataset = dataset_version.download("yolov8")
    print(f"✅ Gedownloade versie {dataset_version.version} naar {dataset.location}")

    model = YOLO("yolov8n.pt")
    results = model.train(
        data=os.path.join(dataset.location, "data.yaml"),
        epochs=50,
        imgsz=640,
        project="solidwaste_project",
        name=f"yolov8n_v{dataset_version.version}_results",
        exist_ok=True
    )

    val_results = model.val()

    success = model.export(format='onnx')
    print("Training klaar en model succesvol geëxporteerd:", success)

if __name__ == "__main__":
    train_model()
