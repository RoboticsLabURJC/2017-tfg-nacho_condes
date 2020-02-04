import tensorflow as tf
import tensorflow.contrib.tensorrt as trt # to solve compat. on bin graph
import numpy as np
from Camera import ROSCam
from Net.utils import label_map_util
import cv2
from os import path
from PIL import Image
from datetime import datetime
from cprint import cprint

LABELS_DICT = {'voc':   ('resources/labels/pascal_label_map.pbtxt',             20),
               'coco':  ('resources/labels/mscoco_label_map.pbtxt',             80),
               'kitti': ('resources/labels/kitti_label_map.txt',                8),
               'oid':   ('resources/labels/oid_bboc_trainable_label_map.pbtxt', 600),
               'pet':   ('resources/labels/pet_label_map.pbtxt',                37),
               }


class DetectionNetwork():
    def __init__(self, arch, input_shape, net_model_path=None, graph_def=None, dataset='coco', confidence_threshold=0.5, path_to_root=None):
        labels_file, max_num_classes = LABELS_DICT[dataset]
        # Append dir if provided (calling from another directory)
        if path_to_root is not None:
            labels_file = path.join(path_to_root, labels_file)
        label_map = label_map_util.load_labelmap(labels_file) # loads the labels map.
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=max_num_classes)
        category_index = label_map_util.create_category_index(categories)
        self.classes = {k:str(v['name']) for k, v in category_index.items()}
        # Find person index
        for idx, class_ in self.classes.items():
            if class_ == 'person':
                self.person_class = idx
                break


        # Graph load. We allocate the session attribute
        self.sess = None

        if net_model_path is not None:
            # Read the graph def from a .pb file
            graph_def = tf.compat.v1.GraphDef()
            cprint.info(f'Loading the graph def from {net_model_path}')
            with tf.io.gfile.GFile(net_model_path, 'rb') as f:
                graph_def.ParseFromString(f.read())

            self.load_graphdef(graph_def)

        elif graph_def is not None:
            cprint.info('Loading the provided graph def...')
            self.load_graphdef(graph_def)

        else:
            # No graph def was provided!
            cprint.error('The graph definition has not been loaded.')
            raise BaseException


        self.input_shape = input_shape
        self.arch = arch

        # Dummy tensor to be used for the first inference.
        dummy_tensor = np.zeros((1,*self.input_shape), dtype=np.int32)

        # Set placeholders, depending on the network architecture
        cprint.warn(f'This is the arch: {self.arch}')
        if self.arch == 'ssd':
            # Inputs
            self.image_tensor      = self.sess.graph.get_tensor_by_name('image_tensor:0')
            # Outputs
            self.detection_boxes   = self.sess.graph.get_tensor_by_name('detection_boxes:0')
            self.detection_scores  = self.sess.graph.get_tensor_by_name('detection_scores:0')
            self.detection_classes = self.sess.graph.get_tensor_by_name('detection_classes:0')
            self.num_detections    = self.sess.graph.get_tensor_by_name('num_detections:0')
            self.boxes = []
            self.scores = []
            self.predictions = []

            self.output_tensors = [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections]

            self.dummy_feed = {self.image_tensor: dummy_tensor}

        elif self.arch == 'yolov3':
            # Inputs
            self.input_data = self.sess.graph.get_tensor_by_name('input/input_data:0')
            # Outputs
            self.sbbox = self.sess.graph.get_tensor_by_name('pred_sbbox/concat_2:0')
            self.mbbox = self.sess.graph.get_tensor_by_name('pred_mbbox/concat_2:0')
            self.lbbox = self.sess.graph.get_tensor_by_name('pred_lbbox/concat_2:0')

            self.output_tensors = [self.sbbox, self.mbbox, self.lbbox]
            self.dummy_feed = {self.input_data: dummy_tensor}

        elif self.arch == 'face_yolo':
            # Inputs
            self.input = self.sess.graph.get_tensor_by_name('img:0')
            self.training = self.sess.graph.get_tensor_by_name('training:0')
            # Outputs
            self.prob = self.sess.graph.get_tensor_by_name('prob:0')
            self.x_center = self.sess.graph.get_tensor_by_name('x_center:0')
            self.y_center = self.sess.graph.get_tensor_by_name('y_center:0')
            self.w = self.sess.graph.get_tensor_by_name('w:0')
            self.h = self.sess.graph.get_tensor_by_name('h:0')

            self.output_tensors = [self.prob, self.x_center, self.y_center, self.w, self.h]
            self.dummy_feed = {self.input: dummy_tensor, self.training: False}


        elif self.arch == 'face_corrector':
            # Inputs
            self.input = self.sess.graph.get_tensor_by_name('img:0')
            self.training = self.sess.graph.get_tensor_by_name('training:0')
            # Outputs
            self.X = self.sess.graph.get_tensor_by_name('prob:0')
            self.Y = self.sess.graph.get_tensor_by_name('x_center:0')
            self.W = self.sess.graph.get_tensor_by_name('w:0')
            self.H = self.sess.graph.get_tensor_by_name('h:0')
            self.output_tensors = [self.X, self.Y, self.W, self.H]
            self.dummy_feed = {self.input: dummy_tensor, self.training: False}

        # First (slower) inference
        cprint.info("Performing first inference...")
        self._forward_pass(self.dummy_feed)

        self.confidence_threshold = confidence_threshold
        cprint.ok("Detection network ready!")

    def load_graphdef(self, graph_def):
        ''' Plug a graph def into the main graph of the detector session . '''
        conf = tf.compat.v1.ConfigProto(log_device_placement=False)
        conf.gpu_options.allow_growth = True
        conf.gpu_options.per_process_gpu_memory_fraction = 0.5
        graph = tf.compat.v1.Graph()
        with graph.as_default():
            tf.import_graph_def(graph_def, name='')

        # Now we use the previously allocated session attribute
        self.sess = tf.compat.v1.Session(graph=graph, config=conf)
        cprint.ok('Loaded the graph definition!')

    def _forward_pass(self, feed_dict):
        ''' Perform a forward pass of the provided feed_dict through the network. '''
        start = datetime.now()
        out = self.sess.run(self.output_tensors, feed_dict=feed_dict)
        elapsed = datetime.now() - start
        return out, elapsed

    def predict(self, img):
        if self.arch == 'ssd':
            # Reshape the latest image
            orig_h, orig_w = img.shape[:2]
            input_image = Image.fromarray(img)
            img_rsz = np.array(input_image.resize(self.input_shape[:2]))
            (boxes, scores, predictions, _), elapsed = self._forward_pass({self.image_tensor: img_rsz[None, ...]})
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

            return np.array(boxes_), np.array(scores_), elapsed
        else:
            cprint.warn(f'Implement predict for {self.arch}!!')
