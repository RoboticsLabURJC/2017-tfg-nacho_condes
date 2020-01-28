import pickle
import cv2
import rosbag
import argparse
import time
import cv_bridge
import numpy as np

parser = argparse.ArgumentParser(description='Visualize results')
parser.add_argument('rosbag_file', type=str, help='Location of the original rosbag')
parser.add_argument('input_width', type=int)
parser.add_argument('input_height', type=int)
parser.add_argument('std_inferences_pickle', type=str, help='File containing the pickled standard inferences')
parser.add_argument('trt_inferences_pickle', type=str, help='File containing the pickled TRT inferences')

args = parser.parse_args()

BAGFILE = args.rosbag_file
STD_INFS = args.std_inferences_pickle
TRT_INFS = args.trt_inferences_pickle

INPUT_SHAPE = (args.input_width, args.input_height)

bag = rosbag.Bag(BAGFILE)
bridge = cv_bridge.CvBridge()

with open(STD_INFS, 'rb') as f:
    std_infs = pickle.load(f)
with open(TRT_INFS, 'rb') as f:
    trt_infs = pickle.load(f)

for topic, data, ts in bag.read_messages('/camera/rgb/image_raw'):
    img = bridge.imgmsg_to_cv2(data, data.encoding)
    img_cp1 = img.copy()

    # Look for that ts in the saved inferences
    std_inf = std_infs[ts]
    trt_inf = trt_infs[ts]
    # scores, boxes, classes, elapsed
    # std:
    trusty_idxs = np.where(std_inf[0].squeeze() > 0.5)[0]
    trusty_bbs = std_inf[1].squeeze()[trusty_idxs]
    bbs = []
    for trusty_bb in trusty_bbs:
        bb = np.zeros(4)
        bb[[0,2]] = trusty_bb[[1,3]] * img_cp1.shape[0]
        bb[[1,3]] = trusty_bb[[0,2]] * img_cp1.shape[1]
        img_cp1 = cv2.rectangle(img_cp1, tuple(bb.astype(int)), (0, 255, 0), -1)

    # trt:
    trusty_idxs = np.where(trt_inf[0].squeeze() > 0.5)[0]
    trusty_bbs = trt_inf[1].squeeze()[trusty_idxs]
    bbs = []
    for trusty_bb in trusty_bbs:
        bb = np.zeros(4)
        bb[[0,2]] = trusty_bb[[1,3]] * img_cp1.shape[0]
        bb[[1,3]] = trusty_bb[[0,2]] * img_cp1.shape[1]
        img_cp1 = cv2.rectangle(img_cp1, tuple(bb.astype(int)), (255, 0, 0), 10)
    # print(str(trusty_bb) + ' -> ' + str(bb.astype(int)))
    cv2.imshow('result', cv2.cvtColor(img_cp1, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)

cv2.destroyAllWindows()







