from ultralytics import YOLO
import glob
import random
import matplotlib.pyplot as plt
import cv2

# https://pyimagesearch.com/2023/05/01/training-the-yolov8-object-detector-for-oak-d/
class Predictor:
   def __init__(self, model_path, test_folder_path):
       self.model = YOLO(model_path)
       self.test_folder = glob.glob(test_folder_path)
   def classify_random_images(self, num_images=10):
       # Generate num_images random numbers between
       # 0 and length of test folder
       random_list = random.sample(range(0, len(self.test_folder)), num_images)
       plt.figure(figsize=(20, 20))
       for i, idx in enumerate(random_list):
           plt.subplot(5,5,i+1)
           plt.xticks([])
           plt.yticks([])
           plt.grid(True)
           img = cv2.imread(self.test_folder[idx])
           results = self.model.predict(source=img)
           res_plotted = results[0].plot()
           # cv2_imshow(res_plotted)
           # convert the image frame BGR to RGB and display it
           image = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
           plt.imshow(image)
       plt.show()

classifier = Predictor("datasets/solidwaste_project/yolov8n_v1_results/weights/best.pt", "datasets/litter_dataset-1/test/images/*.jpg")
classifier.classify_random_images(num_images=10)