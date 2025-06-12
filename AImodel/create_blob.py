import pathlib

# Define paths
onnx_model_path = pathlib.Path("datasets/solidwaste_project/yolov8n_v1_results/weights/best.onnx")
xml_path = pathlib.Path("datasets/solidwaste_project/yolov8n_v1_results/weights/best.xml")
bin_path = pathlib.Path("datasets/solidwaste_project/yolov8n_v1_results/weights/best.bin")
blob_output_path = pathlib.Path("datasets/solidwaste_project/yolov8n_v1_results/weights/best.blob")

import blobconverter

blob_path = blobconverter.from_onnx(
    model=str(onnx_model_path),
    data_type="FP16",      # Use FP16 for OAK-D
    shaves=6,
    use_cache=False
)
print(f"Compatible blob saved at: {blob_path}")
