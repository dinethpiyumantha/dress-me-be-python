"""
This program utilizes PoseNet and MoveNet models to obtain keypoints 
from the human body, as well as a waist detection model. The purpose 
of this program is to analyze and track human body by detecting 
representing specific body parts. PoseNet and MoveNet models are 
used to estimate the locations of these keypoints on the body.

Developer: Ekanayaka G.M.D.P.
Email: it19955650@my.sliit.lk
"""


# set log level to ignore some logs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# import libraries
import tensorflow as tf
import tensorflow_hub as hub

# Dictionary to map joints of body part
MOVENET_KEYPOINT_DICT = {
    'nose':0,
    'left_eye':1,
    'right_eye':2,
    'left_ear':3,
    'right_ear':4,
    'left_shoulder':5,
    'right_shoulder':6,
    'left_elbow':7,
    'right_elbow':8,
    'left_wrist':9,
    'right_wrist':10,
    'left_hip':11,
    'right_hip':12,
    'left_knee':13,
    'right_knee':14,
    'left_ankle':15,
    'right_ankle':16
}

WAIST_DETECTION_DICT = {
  "left_waist": 0,
  "right_waist": 1
}

module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/3")
input_size = 192


def predict(input_image):
  model = module.signatures['serving_default']

  # SavedModel format expects tensor type of int32.
  input_image = tf.cast(input_image, dtype=tf.int32)
  # Run model inference.
  outputs = model(input_image)
  # Output is a [1, 1, 17, 3] tensor.
  keypoint_with_scores = outputs['output_0'].numpy()
  return keypoint_with_scores


def get_list_of_keypoints(image):
  image = tf.image.decode_jpeg(image)

  # Resize and pad the image to keep the aspect ratio and fit the expected size.
  input_image = tf.expand_dims(image, axis=0)
  input_image = tf.image.resize_with_pad(input_image, input_size, input_size)

  # run model and return prediction
  return predict(input_image)

def get_labeled_keypoints(keypoints):
  kpnts = []
  for key, value in MOVENET_KEYPOINT_DICT.items():
    kpnts.append({key: keypoints[0][0][value].tolist()})
  return kpnts

def get_list_of_keypoints_by_path(image_path):
  image = tf.io.read_file(image_path)
  # Run model inference.
  return get_labeled_keypoints(get_list_of_keypoints(image))