import numpy as np
from PIL import Image
import torch
import onnxruntime as ort
from openvino.runtime import Core
import depthai as dai
import cv2
from ultralytics import YOLO

# --- Shared Preprocessing ---
def preprocess_image(image_path, img_size=640):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((img_size, img_size))
    img_np = np.array(img).astype(np.float32) / 255.0
    img_np = np.transpose(img_np, (2, 0, 1))  # HWC -> CHW
    img_np = np.expand_dims(img_np, 0)  # Add batch
    return img_np

image_path = "img.png"
img = preprocess_image(image_path, 640)
img_torch = torch.from_numpy(img)

# --- PT Model ---
print("\nRunning PT model...")
pt_model = YOLO("best.pt")
print("Number of classes in PT model:", pt_model.model.nc)  # nc = number of classes

# Use model.model() to get raw output tensor (before NMS)
pt_output_tensor = pt_model.model(img_torch)[0].detach().cpu()
pt_output = pt_output_tensor.numpy()
print("Output channels:", pt_output.shape[1])
print("Sample bbox coords (first 4 channels):", pt_output[0, :4, 0])
print("Sample next channels (next 17 classes?):", pt_output[0, 4:21, 0])

print("PT output shape:", pt_output.shape)  # should be (1, 22, 8400) for 17 classes
print("PT output min/max:", pt_output.min(), pt_output.max())

# --- ONNX Model ---
print("\nRunning ONNX model...")
onnx_sess = ort.InferenceSession("best.onnx")
onnx_input_name = onnx_sess.get_inputs()[0].name
onnx_output = onnx_sess.run(None, {onnx_input_name: img.astype(np.float32)})[0]
print("ONNX output shape:", onnx_output.shape)
print("ONNX output min/max:", onnx_output.min(), onnx_output.max())

# --- OpenVINO Model ---
print("\nRunning OpenVINO model...")
ie = Core()
ov_model = ie.read_model(model="best_openvino_model/best.xml")
compiled_model = ie.compile_model(ov_model, "CPU")
input_layer = compiled_model.input(0)
ov_output = compiled_model([img])[compiled_model.output(0)]
print("OpenVINO output shape:", ov_output.shape)
print("OpenVINO output min/max:", ov_output.min(), ov_output.max())

# --- OAK-D Model ---
print("\nRunning OAK-D blob model...")
pipeline = dai.Pipeline()
nn = pipeline.create(dai.node.NeuralNetwork)
nn.setBlobPath("best.blob")

manip = pipeline.create(dai.node.ImageManip)
manip.initialConfig.setResize(640, 640)
manip.setMaxOutputFrameSize(640 * 640 * 3)

xlinkIn = pipeline.create(dai.node.XLinkIn)
xlinkIn.setStreamName("input")
xlinkIn.out.link(manip.inputImage)
manip.out.link(nn.input)

xlinkOut = pipeline.create(dai.node.XLinkOut)
xlinkOut.setStreamName("output")
nn.out.link(xlinkOut.input)

with dai.Device(pipeline) as device:
    input_queue = device.getInputQueue("input")
    output_queue = device.getOutputQueue("output", maxSize=1, blocking=True)

    # Prepare image frame for OAK-D
    frame = cv2.imread(image_path)
    frame = cv2.resize(frame, (640, 640))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Send frame
    nn_data = dai.ImgFrame()
    nn_data.setData(frame.transpose(2, 0, 1).flatten())
    nn_data.setTimestamp(dai.Clock.now())
    nn_data.setWidth(640)
    nn_data.setHeight(640)
    input_queue.send(nn_data)

    # Receive output
    out = output_queue.get().getFirstLayerFp16()
    blob_output = np.array(out).reshape((1, 22, 8400))  # 22 channels (4 bbox + 1 obj + 17 classes)
    print("Blob output shape:", blob_output.shape)
    print("Blob output min/max:", blob_output.min(), blob_output.max())

# --- Comparison ---
print("\n--- Mean Absolute Differences ---")
print("PT vs ONNX:", np.mean(np.abs(pt_output - onnx_output)))
print("PT vs OpenVINO:", np.mean(np.abs(pt_output - ov_output)))
print("PT vs Blob:", np.mean(np.abs(pt_output - blob_output)))
