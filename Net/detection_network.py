import tensorflow as tf
import tensorflow.contrib.tensorrt as trt # to solve compat. on bin graph
import numpy as np
import cv2
from os import path
from PIL import Image
from Net.utils import label_map_util, nms
from datetime import datetime
from cprint import cprint

LABELS_DICT = {'voc':   ('resources/labels/pascal_label_map.pbtxt',             20),
               'coco':  ('resources/labels/mscoco_label_map.pbtxt',             80),
               'kitti': ('resources/labels/kitti_label_map.txt',                8),
               'oid':   ('resources/labels/oid_bboc_trainable_label_map.pbtxt', 600),
               'pet':   ('resources/labels/pet_label_map.pbtxt',                37),
               }


class DetectionNetwork():
    def __init__(self, arch, input_shape, frozen_graph=None, graph_def=None, dataset='coco', confidence_threshold=0.5, path_to_root=None):
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

        if frozen_graph is not None:
            # Read the graph def from a .pb file
            graph_def = tf.compat.v1.GraphDef()
            cprint.info(f'Loading the graph def from {frozen_graph}')
            with tf.io.gfile.GFile(frozen_graph, 'rb') as f:
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

        elif self.arch in ['yolov3', 'yolov3tiny']:
            # Inputs
            self.inputs = self.sess.graph.get_tensor_by_name('inputs:0')
            # Outputs
            self.output_boxes = self.sess.graph.get_tensor_by_name('output_boxes:0')

            self.output_tensors = [self.output_boxes]
            self.dummy_feed = {self.inputs: dummy_tensor}

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
            self.X = self.sess.graph.get_tensor_by_name('X:0')
            self.Y = self.sess.graph.get_tensor_by_name('Y:0')
            self.W = self.sess.graph.get_tensor_by_name('W:0')
            self.H = self.sess.graph.get_tensor_by_name('H:0')
            self.output_tensors = [self.X, self.Y, self.W, self.H]
            self.dummy_feed = {self.input: dummy_tensor, self.training: False}

        elif self.arch == 'facenet':
            # Inputs
            self.input = self.sess.graph.get_tensor_by_name('input:0')
            self.phase_train = self.sess.graph.get_tensor_by_name('phase_train:0')
            # Outputs
            self.embeddings = self.sess.graph.get_tensor_by_name('embeddings:0')
            self.output_tensors = [self.embeddings]
            self.dummy_feed = {self.input: dummy_tensor, self.phase_train: False}

        else:
            cprint.fatal(f'Architecture {arch} is not supported', interrupt=True)
        # First (slower) inference
        cprint.info("Performing first inference...")
        self._forward_pass(self.dummy_feed)

        self.confidence_threshold = confidence_threshold
        cprint.ok("Detection network ready!")

    def load_graphdef(self, graph_def):
        ''' Plug a graph def into the main graph of the detector session . '''
        conf = tf.compat.v1.ConfigProto(log_device_placement=True)
        # conf.gpu_options.allow_growth = True
        conf.gpu_options.per_process_gpu_memory_fraction = 0.05
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
        # Reshape the latest image
        orig_h, orig_w = img.shape[:2]
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        input_image = Image.fromarray(img)
        img_rsz = np.array(input_image.resize(self.input_shape[:2]))

        if self.arch == 'ssd':
            (boxes, scores, predictions, _), elapsed = self._forward_pass({self.image_tensor: img_rsz[None, ...]})
            boxes = list(np.squeeze(boxes))
            scores = list(np.squeeze(scores))
            classes = list(np.squeeze(predictions).astype(int))

            boxes_full = []
            for box, prob, cls in zip(boxes, scores, classes):
                if prob >= self.confidence_threshold and cls == self.person_class:
                    # x, y, w, h, p
                    y1 = box[0] * orig_h
                    x1 = box[1] * orig_w
                    y2 = box[2] * orig_h
                    x2 = box[3] * orig_w

                    boxes_full.append([x1, y1, x2-x1, y2-y1, prob])

            return boxes_full, elapsed

        elif self.arch in ['yolov3', 'yolov3tiny']:
            detections, elapsed = self._forward_pass({self.inputs: img_rsz[None, ...]})
            # Nxkx(NUM_CLASSES + 4 + 1) tensor containing k detections for each n-th image
            # NMS
            detections_filtered = nms.non_max_suppression(detections[0], 0.5)
            # The key 0 contains the human detections.
            if not 0 in detections_filtered:
                return [], elapsed
            persons = detections_filtered[0]
            boxes_full = []
            for box, prob in persons:
                if prob >= self.confidence_threshold:
                    # x, y, w, h, p
                    x1 = box[0] / self.input_shape[1] * orig_w
                    y1 = box[1] / self.input_shape[0] * orig_h
                    x2 = box[2] / self.input_shape[1] * orig_w
                    y2 = box[3] / self.input_shape[0] * orig_h
                    boxes_full.append([x1, y1, x2-x1, y2-y1, prob])
            return boxes_full, elapsed
        else:
            cprint.warn(f'Implement predict for {self.arch}!!')
