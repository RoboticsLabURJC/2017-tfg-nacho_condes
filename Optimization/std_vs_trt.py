#
# Created on Dec, 2019
# @author: naxvm
#
import argparse

parser = argparse.ArgumentParser(description='Obtain inferences of the standard graph vs the TRT version.')
parser.add_argument('model', type=str, help='Model to optimize')
parser.add_argument('input_w', type=int, help='Width of the input images')
parser.add_argument('input_h', type=int, help='Height of the input images')
parser.add_argument('precision', type=str, help='Precision for the conversion', choices=['FP32', 'FP16'])
parser.add_argument('mss', type=int, help='Minimum segment size for the optimization')
parser.add_argument('mce', type=int, help='Maximum cached engines for the optimization')
parser.add_argument('allow_growth', type=bool, help='Allow growth or not for the TF session allocation')
parser.add_argument('rosbag_file', type=str, help='Location of the rosbag to perform the comparison')
args = parser.parse_args()

# If the arguments are OK, import the TF stuff
import utils2
from tf_trt_models.detection import build_detection_graph
import tensorflow as tf
from cprint import cprint
import rosbag
# from std_msgs.msg import Image
import cv_bridge
# import rospy


# rospy.init_node('models_comparison', anonymous=True)


# Parameters
DATA_DIR = 'dl_models/'

# TODO: extend to others than MobileNet-SSD

MODEL = args.model
INPUT_SHAPE = (args.input_w, args.input_h)

MODEL_DIR = DATA_DIR + MODEL + '/'

# CONFIG_FILE = MODEL_DIR + 'pipeline.config'
# CHECKPOINT_FILE = MODEL_DIR + 'model.ckpt'
BAGFILE = args.rosbag_file
#IMAGE_PATH = 'data/huskies.jpg'

PRECISION = args.precision
MSS = args.mss     # 3, 20, 50
MCE = args.mce     # 1, 3, 5
ALLOW_GROWTH = args.allow_growth

input_names = ['image_tensor']
output_names = ['detection_boxes', 'detection_classes', 'detection_scores', 'num_detections']

# # Build the frozen graph
# orig_graph, input_names, output_names = build_detection_graph(
#     config=CONFIG_FILE,
#     checkpoint=CHECKPOINT_FILE,
#     score_threshold=0.3,
#     batch_size=1,
# )

cprint.ok('Input names: ' + '\t'.join(input_names))
cprint.ok('Output names: ' + '\t'.join(output_names))


cprint.ok('Inferences with the standard graph...')

orig_graph_f =  MODEL_DIR + 'frozen_inference_graph.pb'
std_infs = utils2.inference_times_from_rosbag(BAGFILE, INPUT_SHAPE, orig_graph_f,
                                              input_names, output_names, ALLOW_GROWTH)


cprint.ok('Inferences with the standard graph done.')


# Measure inference times
cprint.ok('Inferences with the optimized graph...')
# It has to be already exported
trt_graph = f'{MODEL_DIR}optimizations/{PRECISION}_{MSS}_{MCE}.pb'
trt_infs = utils2.inference_times_from_rosbag(BAGFILE, INPUT_SHAPE, trt_graph,
                                              input_names, output_names, ALLOW_GROWTH)

cprint.ok('Inferences with the optimized graph done')

import pickle

cprint.ok('Saving results')
with open('trt_infs.pkl', 'wb') as f:
    pickle.dump(trt_infs, f)

with open('std_infs.pkl', 'wb') as f:
    pickle.dump(std_infs, f)

cprint.ok('All finished')

