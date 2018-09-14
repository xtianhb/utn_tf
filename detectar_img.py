######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/20/18
# Description: 
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier uses it to perform object detection on a webcam feed.
# It draws boxes and scores around the objects of interest in each frame from
# the webcam.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.


# python detectar_img.py training/inf_graph/frozen_inference_graph.pb samples/label_map.txt test

# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import glob
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.insert(0, "../models/research/object_detection")

# Import utilites
from object_detection.utils import ops as utils_ops
from utils import label_map_util
from utils import visualization_utils as vis_util

if len(sys.argv) <2:
	print("Falta path a frozen_inference_graph.pb")
	print("python detectar_img.py training/inf_graph/frozen_inference_graph.pb samples/label_map.txt test")
	exit(-1)

if len(sys.argv) <3:
	print("Falta path a label_map.txt")
	print("python detectar_img.py training/inf_graph/frozen_inference_graph.pb samples/label_map.txt test")
	exit(-1)

if len(sys.argv) <4:
	print("Falta carpeta de test imagenes")
	print("python detectar_img.py training/inf_graph/frozen_inference_graph.pb samples/label_map.txt test")
	exit(-1)
	
# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = sys.argv[1]

# Path to label map file
PATH_TO_LABELS = sys.argv[2]

# Number of classes the object detector can identify
#NUM_CLASSES = int(sys.argv[3])
NUM_CLASSES = 1

#IMAGENES INPUT
IMGS_INPUT=sys.argv[3]

## Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)


# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Inicializa imagenes
print("Buscando imagenes jpg disponible en input_dir")
ImgsPaths=glob.glob(IMGS_INPUT+"/*.jpg")

if len(ImgsPaths)==0:
	print("no hay imagenes .jpg en input dir")
	exit(-1)

for ImgPath in ImgsPaths:

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    Img = Image.open(ImgPath)
    
    (W, H) = Img.size
    Image_np = np.array(Img)
    Img_expanded = np.expand_dims(Image_np, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: Img_expanded})
	
    # Draw the results of the detection (aka 'visulaize the results')
    vis_util.visualize_boxes_and_labels_on_image_array(
        Image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=2,
        min_score_thresh=0.4)
	
    img_cvt=cv2.cvtColor(Image_np, cv2.COLOR_BGR2RGB)
    plt.imshow(img_cvt)
    plt.show()


print("End")
