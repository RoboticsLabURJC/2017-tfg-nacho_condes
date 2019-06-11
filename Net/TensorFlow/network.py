import tensorflow as tf
import tensorflow.contrib.tensorrt as trt # to solve compat. on bin graph
from tensorflow.python.platform import gfile
import numpy as np
from Camera import ROSCam
import cv2
from PIL import Image
from datetime import datetime


from Net.utils import label_map_util

LABELS_DICT = {'voc': 'Net/labels/pascal_label_map.pbtxt',
               'coco': 'Net/labels/mscoco_label_map.pbtxt',
               'kitti': 'Net/labels/kitti_label_map.txt',
               'oid': 'Net/labels/oid_bboc_trainable_label_map.pbtxt',
               'pet': 'Net/labels/pet_label_map.pbtxt'}

class TrackingNetwork():
    def __init__(self, net_model):
        self.framework = "TensorFlow"

        labels_file = LABELS_DICT['coco']
        label_map = label_map_util.load_labelmap(labels_file) # loads the labels map.
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes= 999999)
        category_index = label_map_util.create_category_index(categories)
        self.classes = {}
        # We build is as a dict because of gaps on the labels definitions
        for cat in category_index:
            self.classes[cat] = str(category_index[cat]['name'])


        print "Creating session..."
        conf = tf.ConfigProto(log_device_placement=False)
        conf.gpu_options.allow_growth = True

        self.sess = tf.Session(config=conf)
        print " Created"
        print "Loading the custom graph..."
        # Load the TRT frozen graph from disk
        CKPT = 'Net/TensorFlow/' + net_model
        self.sess.graph.as_default()
        graph_def = tf.GraphDef()
        with gfile.FastGFile(CKPT, 'rb') as fid:
            graph_def.ParseFromString(fid.read())
        print "Loaded..."
        tf.import_graph_def(graph_def, name='')
        print "Imported"

        # Set placeholders...
        self.image_tensor = self.sess.graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.sess.graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.sess.graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.sess.graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.sess.graph.get_tensor_by_name('num_detections:0')

        self.boxes = []
        self.scores = []
        self.predictions = []


        # Dummy initialization (otherwise it takes longer then)
        dummy_tensor = np.zeros((1,1,1,3), dtype=np.int32)
        self.sess.run(
                [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                feed_dict={self.image_tensor: dummy_tensor})

        self.confidence_threshold = 0.5

        print("Network ready!")

    def setCamera(self, cam):
        self.cam = cam
        self.original_height = ROSCam.IMAGE_HEIGHT
        self.original_width = ROSCam.IMAGE_WIDTH

    def setDepth(self, depth):
        self.depth = depth

    def predict(self):
        # Reshape the latest image
        input_image = Image.fromarray(self.cam.rgb_img)
        img_rsz = np.array(input_image.resize((300,300)))

        (boxes, scores, predictions, _) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: img_rsz[None, ...]})
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        predictions = np.squeeze(predictions)

        # We only keep the most confident predictions.
        mask1 = scores > self.confidence_threshold # bool array

        # We map the predictions into a mask (human or not)
        mask2 = []
        for idx in predictions:
            mask2.append(self.classes[int(idx)] == 'person')

        # Total mask: CONFIDENT PERSONS
        mask = np.logical_and(mask1, mask2)
        # Boxes containing only confident humans
        boxes = boxes[mask]
        # aux variable for avoiding race condition while int casting
        # Box format and reshaping...
        tmp_boxes = np.zeros([len(boxes), 4])
        tmp_boxes[:,[0,2]] = boxes[:,[1,3]] * self.original_width
        tmp_boxes[:,[3,1]] = boxes[:,[2,0]] * self.original_height
        self.boxes = tmp_boxes.astype(int)

        self.scores = scores[mask]
        self.predictions = []
        for idx in predictions[mask]:
            self.predictions.append(self.classes[idx])

