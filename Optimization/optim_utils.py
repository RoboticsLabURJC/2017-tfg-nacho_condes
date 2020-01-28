#
# Created on Nov, 2019
# @author: naxvm
#

from datetime import datetime, timedelta
import yaml
from os import listdir, path
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert
import numpy as np
from PIL import Image
import rosbag
import cv_bridge
import cv2

META_KEY = '1 - Meta'
SUMMARY_KEY = '2 - Summary'
ITERS_KEY = '3 - Iterations'

FILENAME_FORMAT = '%Y%m%d %H%M%S.yml'

class BenchmarkWriter:
    ''' Class to save a benchmark result on a file. '''
    def __init__(self, logdir, network_model):
        self.logdir = logdir
        self.description = None
        self.network_model = network_model
        # self.check_last_commented()

    def check_last_commented(self):
        if not path.exists(self.logdir):
            print(f'Error: {self.path} does not exist.')
            return
        files = listdir(self.logdir)
        # Parse the filenames
        datetimes = list(map(lambda x: (datetime.strptime(x, FILENAME_FORMAT), x), files))
        datetimes.sort()
        # Read the most recent logging file
        latest_datetime, latest_file = datetimes[-1]
        with open(self.logdir + '/' + latest_file, 'r') as f:
            latest_log = yaml.safe_load(f)

        latest_descr = latest_log[META_KEY]['Description']
        
        if latest_descr != '':
            print(f'Latest description:\n{latest_descr}')
            self.description = input('Please enter the description for the new benchmark >> ')
            return True
        else:
            print(f'Error: the latest description on "{latest_file}" is empty')
            return False


    def write_log(self, elapsed_times, detections, optim_params=None, filename=None, write_iters=True):
        metrics = {}
        elapsed_ms = list(map(lambda x: x.seconds * 1000.0 + x.microseconds / 1000.0, elapsed_times))
        elapsed_mean = float(np.mean(elapsed_ms))
        elapsed_std = float(np.std(elapsed_ms))
        avg_fps = float(1000.0 / elapsed_mean)
        summary = {
            'AverageInferenceTime': elapsed_mean,
            'StdInferenceTime': elapsed_std,
            'AverageFPS': avg_fps,
        }
        if optim_params is not None:
            metrics[META_KEY] = {
                'OriginalModel': optim_params['model_name'],
                'PrecisionMode': optim_params['prec'],
                'MinimumSegmentSize': optim_params['mss'],
                'MaximumCachedEngines': optim_params['mce'],
            }
        else:
            metrics[META_KEY] = {
                'Datetime': datetime.now().strftime('%d-%m-%Y %H:%M:%S'),
                'Description': self.description,
                'NetworkModel': self.network_model,
            }
        metrics[SUMMARY_KEY] = summary
        if write_iters:
            metrics[ITERS_KEY] = {}

            for (idx, elp), dets in zip(enumerate(elapsed_ms), detections):
                det = {
                    'ElapsedMs':     elp,
                    'NumDetections': len(dets[0])
                }
                metrics[ITERS_KEY][idx + 1] = det
        if filename is None:
            filename = datetime.now().strftime(self.logdir + '/' + FILENAME_FORMAT)
        else:
            filename = self.logdir + '/' + filename
        with open(filename, 'w') as f:
            yaml.dump(metrics, f)

        print(f'Logfile saved on {filename}')



def optim_graph(graph, blacklist_names, precision_mode, mss, mce):
    ''' Returns the TRT converted graph given the input parameters. '''
    with tf.Session() as sess:
        converter = trt_convert.TrtGraphConverter(
            input_graph_def=graph,
            nodes_blacklist=blacklist_names,
            precision_mode=precision_mode,
            max_batch_size=1,
            minimum_segment_size=mss,
            maximum_cached_engines=mce,
            use_calibration=True)
        new_g = converter.convert()
    return new_g


def inference_times(image_path, input_shape, graph, input_names, output_names, allow_growth,n_samples=100, arch='ssd'):
    # TODO: merge both eval functions into one.
    image = Image.open(image_path)
    image_resized = np.array(image.resize(input_shape))

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = allow_growth
    tf_sess = tf.Session(config=tf_config)
    tf.import_graph_def(graph, name='')

    # Placeholders
    tf_input = tf_sess.graph.get_tensor_by_name('input:0')
    is_train = tf_sess.graph.get_tensor_by_name('phase_train:0')
    net_outputs = []
    for out_tensor in output_names:
        net_outputs.append(tf_sess.graph.get_tensor_by_name(out_tensor + ':0'))

    # First slower inference
    tf_sess.run(net_outputs, feed_dict={
                tf_input: image_resized[None, ...],
                is_train: False})

    # Benchmarking loop
    elapsed_times = []    
    detections = []
    for i in range(n_samples):
        start = datetime.now()
        if arch == 'ssd':
            _ = tf_sess.run(net_outputs, feed_dict={
                                        tf_input: image_resized[None, ...],
                                        is_train: False})
        elif arch == 'yolov3':
            _ = tf_sess.run(net_outputs, feed_dict={
                    tf_input: image_resized[None, ...],
                    is_train: False})
            boxes = np.squeeze(boxes)
            scores = np.array([0.0] * len(boxes))
        
        boxes = []
        scores  = []

        elapsed = datetime.now() - start
        elapsed_times.append(elapsed)
        detections.append([np.squeeze(boxes), np.squeeze(scores)])
    tf_sess.close()
    return elapsed_times, detections


def inference_times_from_rosbag(rosbag_filename, input_shape, graph_filename,
                                input_names, output_names, allow_growth):
    # TODO: merge both inference times methods into one
    # TODO: cope with non-MobileNet/SSD networks (I/O tensors)

    tf_config = tf.ConfigProto(log_device_placement=False)
    tf_config.gpu_options.allow_growth = allow_growth
    # tf_config.gpu_options.per_process_gpu_memory_fraction = 0.67
    tf_sess = tf.Session(config=tf_config)
    tf_sess.graph.as_default()
    graph_def = tf.compat.v1.GraphDef()
    with tf.gfile.GFile(graph_filename, 'rb') as fid:
        graph_def.ParseFromString(fid.read())
    tf.import_graph_def(graph_def, name='')

    # Placeholders
    tf_input = tf_sess.graph.get_tensor_by_name(input_names[0] + ':0')
    tf_boxes = tf_sess.graph.get_tensor_by_name(output_names[0] + ':0')
    tf_classes = tf_sess.graph.get_tensor_by_name(output_names[1] + ':0')
    tf_scores = tf_sess.graph.get_tensor_by_name(output_names[2] + ':0')
    tf_num_detections = tf_sess.graph.get_tensor_by_name(output_names[3] + ':0')

    # First slower inference
    dummy_input = np.zeros([input_shape[0], input_shape[1], 3])
    tf_sess.run([tf_scores, tf_boxes, tf_classes, tf_num_detections], feed_dict={
                tf_input: dummy_input[None, ...]})

    # rosbag iterator
    bag = rosbag.Bag(rosbag_filename)
    bridge = cv_bridge.CvBridge()

    inference_results = {}
    for topic, data, ts in bag.read_messages('/camera/rgb/image_raw'):
        img_arr = bridge.imgmsg_to_cv2(data, data.encoding)
        img_res = cv2.resize(img_arr, input_shape)

        start = datetime.now()
        # Run the inference!
        scores, boxes, classes, _ = tf_sess.run([tf_scores, tf_boxes, tf_classes, tf_num_detections], feed_dict={
                                        tf_input: img_res[None, ...]})
        elapsed = datetime.now() - start

        inference_results[ts] = (scores, boxes, classes, elapsed)

    # Free resources
    tf_sess.close()
    return inference_results
