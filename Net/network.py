import tensorflow as tf
import tensorflow.contrib.tensorrt as trt # to solve compat. on bin graph
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

class DetectionNetwork():
    def __init__(self, net_model):
        labels_file = LABELS_DICT['coco']
        label_map = label_map_util.load_labelmap(labels_file) # loads the labels map.
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes= 999999)
        category_index = label_map_util.create_category_index(categories)
        self.classes = {k:str(v['name']) for k, v in category_index.items()}
        # Find person index
        for idx, class_ in self.classes.items():
            if class_ == 'person':
                self.person_class = idx
                break

        print("Creating session...")
        conf = tf.ConfigProto(log_device_placement=False)
        conf.gpu_options.allow_growth = False
        # conf.gpu_options.per_process_gpu_memory_fraction = 0.67 # leave mem for tf-rt

        self.sess = tf.Session(config=conf)
        print("Created")
        print("Loading the custom graph...")
        # Load the TRT frozen graph from disk
        CKPT = 'Net/Models/' + net_model
        self.sess.graph.as_default()
        graph_def = tf.compat.v1.GraphDef()
        with tf.gfile.GFile(CKPT, 'rb') as fid:
            graph_def.ParseFromString(fid.read())
        print("Loaded...")
        tf.import_graph_def(graph_def, name='')
        print("Imported")

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

    def predict(self, img):
        # Reshape the latest image
        orig_h, orig_w = img.shape[:2]
        input_image = Image.fromarray(img)
        img_rsz = np.array(input_image.resize((300,300)))

        (boxes, scores, predictions, _) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: img_rsz[None, ...]})
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        predictions = np.squeeze(predictions).astype(int)

        mask1 = scores > self.confidence_threshold # bool array
        mask2 = [idx == self.person_class for idx in predictions]

        # Total mask: CONFIDENT PERSONS
        mask = np.logical_and(mask1, mask2)
        # Boxes containing only confident humans
        boxes = boxes[mask]
        # Box format and reshaping...
        boxes_ = [[b[1] * orig_w, b[0] * orig_h,
                   b[3] * orig_w, b[2] * orig_h] for b in boxes]
        scores_ = scores[mask]

        return np.array(boxes_), np.array(scores_)
