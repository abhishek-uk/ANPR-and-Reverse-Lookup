import os

import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util

import cv2
import numpy as np
from matplotlib import pyplot as plt

import easyocr

import csv
import uuid

import time

CUSTOM_MODEL_NAME = 'my_ssd_mobnet'
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'

paths = {
    'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
    'SCRIPTS_PATH': os.path.join('Tensorflow','scripts'),
    'APIMODEL_PATH': os.path.join('Tensorflow','models'),
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
    'IMAGE_PATH': os.path.join('Tensorflow', 'workspace','images'),
    'MODEL_PATH': os.path.join('Tensorflow', 'workspace','models'),
    'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace','pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME),
    'OUTPUT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'export'),
    'TFJS_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'),
    'TFLITE_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'),
    'PROTOC_PATH':os.path.join('Tensorflow','protoc')
 }

files = {
    'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME),
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}


# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-21')).expect_partial()  # change checkpoint

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])



# Plate Detection Function
def detect_plate(img, detection_threshold=0.5, img_from_path=False):
    if img_from_path: 
        img_array = cv2.imread(img) # only for detect from direct img, if np array this dont need 
    else:
        img_array = img
    
    img_np = np.array(img_array)

    input_tensor = tf.convert_to_tensor(np.expand_dims(img_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    # return img_np, detections
    scores = list(filter(lambda x: x > detection_threshold, detections['detection_scores']))
    # print(scores)
    boxes = detections['detection_boxes'][:len(scores)]
    classes = detections['detection_classes'][:len(scores)]

    height = img_np.shape[0]
    width = img_np.shape[1]
    plates = []
    
    for box in boxes:
        roi = box * [height, width, height, width]
        region =  img_np[int(roi[0]) : int(roi[2]), int(roi[1]) : int(roi[3])]    
        plates.append(region)

    return plates



# Function to Temp-Save data on csv file
def save_results_as_csv(plate, text):
    img_name = '{}.jpg'.format(uuid.uuid1())
    cv2.imwrite(os.path.join('Detection_Images', img_name), plate)
    with open('detection_results.csv', mode='a', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow([img_name, text])



# Plate Recognition Function
reader = easyocr.Reader(['en'], gpu=False)
def rec_text(plate):
    text = ''
    ocr_result = reader.readtext(plate)
    for part in ocr_result:
        text += part[1]
    return text



print('started ')


def save_plates_from_video(vid_path):
    cap = cv2.VideoCapture(vid_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            plates = detect_plate(frame)  
            for plate in plates:
                plate_number = rec_text(plate)
                #  code to store data in to database 
                if not os.path.exists(f'Temp/{plate_number}.png'):
                    cv2.imwrite(f'Temp/{plate_number}.png', plate)
        else:
            break
    cap.release()



print('done')