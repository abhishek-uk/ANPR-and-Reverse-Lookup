import os

import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util

import cv2
import numpy as np
from matplotlib import pyplot as plt

# import csv
import uuid
import time

import re
import pytesseract

import mysql.connector

import datetime


CUSTOM_MODEL_NAME = 'my_ssd_mobnet'
LABEL_MAP_NAME = 'label_map.pbtxt'
paths = {
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME),
}
files = {
    'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
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
def detect_plate(img, detection_threshold=0.3, img_from_path=False):
    if img_from_path: 
        img_array = cv2.imread(img) # only for detect from direct img, if np array this dont need 
    else:
        img_array = img
    
    img_np = np.array(img_array)

    input_tensor = tf.convert_to_tensor(np.expand_dims(img_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    # detections['num_detections'] = num_detections       # some times dont need, need to focus here
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



# Adding custom options
custom_config = r'--oem 3 --psm 6'
def rec_plate(plate):
    text = pytesseract.image_to_string(plate, config=custom_config)
    text = re.sub(r'\W+', '', text)
    # print(text)
    return text




mydb = mysql.connector.connect(
    host='localhost',
    user='root',
    password='P@ssw0rd',
    database="mydatabase"
)
mycursor = mydb.cursor()

# mycursor.execute('''CREATE TABLE IF NOT EXISTS recognized_vehicle (
#                                         id INT AUTO_INCREMENT PRIMARY KEY, 
#                                         plate_number VARCHAR(255),
#                                         plate_img_name VARCHAR(255),
#                                         video_file_name VARCHAR(255),
#                                         frame_number INT,
#                                         frame_time INT,
#                                         loaction VARCHAR(255),
#                                         date VARCHAR(255), 
#                                         time VARCHAR(255)
#                                         )  ''')

# mycursor.execute('show tables')
# for i in mycursor:
#     print(i)



sql = 'INSERT INTO recognized_vehicle (plate_number, plate_img_name, video_file_name, frame_number, frame_time, loaction, date, time) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)'
def save_db(plate, plate_number, video_name, frame_nr, frame_time):
    img_name = '{}.jpg'.format(uuid.uuid1())
    cv2.imwrite(os.path.join('instance', 'detected_plates', img_name), plate)
    location, date, time = video_name.split('.')[0].split('_')
    # date_time = '{}-{}-{} {}:{}:{}'.format(date[4:], date[2:4], date[:2], time[:2], time[2:], '00')
    date = '{}-{}-{}'.format(date[:2], date[2:4], date[4:])

    time = '{}:{}:{}'.format(time[:2], time[2:], '00')
    time_obj = datetime.datetime.strptime(time, '%H:%M:%S')
    time_obj += datetime.timedelta(seconds=frame_time)
    time = time_obj.strftime('%H:%M:%S')

    val = (plate_number, img_name, video_name, frame_nr, frame_time, location, date, time)
    mycursor.execute(sql, val)
    mydb.commit()



def save_from_video(vid_path, new_fps = 20):
    cap = cv2.VideoCapture(vid_path)
    print(cap.get(cv2.CAP_PROP_FRAME_COUNT), end=', ')
    print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT), end=', ')
    print(cap.get(cv2.CAP_PROP_FRAME_WIDTH), end=', ')
    print(cap.get(cv2.CAP_PROP_FPS))

    fps = int(round(cap.get(cv2.CAP_PROP_FPS) / 10) * 10)
    print(fps)
    readed_plates = set()
    current_frame = 0
    saves = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if( ( new_fps * current_frame ) % fps == 0):
                plates = detect_plate(frame, detection_threshold= 0.2)
                for plate in plates:
                    plate_number = rec_plate(plate) 
                    if plate_number not in readed_plates:
                        if 6 <= len(plate_number) <= 12:    
                            save_db(
                                plate=plate,
                                plate_number=plate_number,
                                video_name=os.path.basename(vid_path).split('/')[-1],
                                frame_nr=current_frame,
                                frame_time = int(current_frame / cap.get(cv2.CAP_PROP_FPS))
                            )
                            readed_plates.add(plate_number)
                            saves += 1
            current_frame += 1
        else:
            break

    cap.release()
    print(f'saves: {saves}')
    print(f'current_frame: {current_frame}\n')


print('done')