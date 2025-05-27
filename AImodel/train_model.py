import os
from roboflow import Roboflow
from ultralytics import YOLO

# Zet werkdirectory
HOME = os.getcwd()
DATASET_DIR = os.path.join(HOME, "datasets")
os.makedirs(DATASET_DIR, exist_ok=True)
os.chdir(DATASET_DIR)

API_KEY = os.getenv("ROBOFLOW_API_KEY")
rf = Roboflow(api_key=API_KEY)

# Haal project op uit juiste workspace
workspace = rf.workspace("vape-0gytc")
project = workspace.project("solidwaste-detection")

# debug
print("Beschikbare datasetversies:")
for v in project.versions():
    print(f"- Versie {v.version} (ID: {v.id})")

dataset_version = project.versions()[0]  # Nieuwste versie bovenaan
dataset = dataset_version.download("yolov8")
print(f"✅ Gedownloade versie {dataset_version.version} naar {dataset.location}")


model = YOLO("yolov8n.pt")

# Train het model
results = model.train(
    data=os.path.join(dataset.location, "data.yaml"),
    epochs=50,
    imgsz=640,
    project="solidwaste_project",
    name=f"yolov8n_v{dataset_version.version}_results",
    exist_ok=True
)

val_results = model.val()

# Exporteer naar ONNX-formaat
success = model.export(format='onnx')

print("🎉 Training klaar en model succesvol geëxporteerd:", success)
