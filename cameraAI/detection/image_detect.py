# import the necessary packages
import config
import utils
import argparse
import cv2
import depthai as dai


# initialize a depthai camera pipeline
print("[INFO] initializing a depthai images pipeline...")


pipeline = utils.create_image_pipeline(config_path=config.YOLOV8N_CONFIG,
                                        model_path=config.YOLOV8N_MODEL)
output_image_path = config.OUTPUT_IMAGES_YOLOv8n

# pipeline defined, now the device is assigned and pipeline is started
with dai.Device(pipeline) as device:
   # define the queues that will be used in order to communicate with
   # depthai and then send our input image for predictions
   detectionIN = device.getInputQueue("detection_in")
   detectionNN = device.getOutputQueue("nn")
   print("[INFO] loading image from disk...")
   for img_path in config.TEST_DATA:
       # load the input image and then resize it
       image = cv2.imread(img_path)
       image_res = cv2.resize(image, config.CAMERA_PREV_DIM)
       # create a copy of image for inference
       image_copy = image.copy()
       # initialize depthai NNData() class which is fed with the
       # image data resized and transposed to model input shape
       nn_data = dai.NNData()
       nn_data.setLayer(
           "input",
           utils.to_planar(image_copy, config.CAMERA_PREV_DIM)
       )
       # send the image to detectionIN queue further passed
       # to the detection network for inference as defined in pipeline
       detectionIN.send(nn_data)

       print("[INFO] fetching neural network output for {}".
           format(img_path.split('/')[2]))
       # fetch the neural network output
       inDet = detectionNN.get()
       # if detection is available for given image, fetch the detections
       if inDet is not None:
           detections = inDet.detections
           # if object detected, annotate the image
           image_res = utils.annotateFrame(image_res, detections, "image")
       # finally write the image to the output path
       print(output_image_path +"\\"+img_path.split('/')[2])
       cv2.imwrite(
           "C:\\Users\\lvgra\\Documents\\school\\projecten\\CameraAI-1.4\\AImodel\\result.png",
           image_res
       )