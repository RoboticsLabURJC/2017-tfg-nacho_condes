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


    def write_log(self, elapsed_times, detections, optim_params=None, filename=None):
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
        metrics[ITERS_KEY] = {}

        for (idx, elp), dets in zip(enumerate(elapsed_ms), detections):
            det = {
                'ElapsedMs':     elp,
                'NumDetections': len(dets[0])
            }
            metrics['3 - Iterations'][idx + 1] = det
        if filename is None:
            filename = datetime.now().strftime(self.logdir + '/' + FILENAME_FORMAT)
        else:
            filename = self.logdir + '/' + filename
        with open(filename, 'w') as f:
            yaml.dump(metrics, f)

        print(f'Logfile saved on {filename}')


    def write_comparison(self, elapsed_times, detections, optim_params=None, filename=None):
        pass
        # metrics = {}
        # elapsed_ms = list(map(lambda x: x.seconds * 1000.0 + x.microseconds / 1000.0, elapsed_times))
        # elapsed_mean = float(np.mean(elapsed_ms))
        # elapsed_std = float(np.std(elapsed_ms))
        # avg_fps = float(1000.0 / elapsed_mean)
        # summary = {
        #     'AverageInferenceTime': elapsed_mean,
        #     'StdInferenceTime': elapsed_std,
        #     'AverageFPS': avg_fps,
        # }
        # if optim_params is not None:
        #     metrics[META_KEY] = {
        #         'OriginalModel': optim_params['model_name'],
        #         'PrecisionMode': optim_params['prec'],
        #         'MinimumSegmentSize': optim_params['mss'],
        #         'MaximumCachedEngines': optim_params['mce'],
        #     }
        # else:
        #     metrics[META_KEY] = {
        #         'Datetime': datetime.now().strftime('%d-%m-%Y %H:%M:%S'),
        #         'Description': self.description,
        #         'NetworkModel': self.network_model,
        #     }
        # metrics[SUMMARY_KEY] = summary
        # metrics[ITERS_KEY] = {}

        # for (idx, elp), dets in zip(enumerate(elapsed_ms), detections):
        #     det = {
        #         'ElapsedMs':     elp,
        #         'NumDetections': len(dets[0])
        #     }
        #     metrics['3 - Iterations'][idx + 1] = det
        # if filename is None:
        #     filename = datetime.now().strftime(self.logdir + '/' + FILENAME_FORMAT)
        # else:
        #     filename = self.logdir + '/' + filename
        # with open(filename, 'w') as f:
        #     yaml.dump(metrics, f)

        # print(f'Logfile saved on {filename}')




def optim_graph(graph, output_names, precision_mode, mss, mce):
    ''' Returns the TRT converted graph given the input parameters. '''
    with tf.Session() as sess:
        converter = trt_convert.TrtGraphConverter(
            input_graph_def=graph,
            nodes_blacklist=output_names,
            precision_mode=precision_mode,
            max_batch_size=1,
            minimum_segment_size=mss,
            maximum_cached_engines=mce,
            use_calibration=True)
        new_g = converter.convert()
    return new_g


def inference_times(image_path, input_shape, graph, input_names, output_names, allow_growth,n_samples=100):
    # TODO: merge both eval functions into one.
    image = Image.open(image_path)
    image_resized = np.array(image.resize(input_shape))

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = allow_growth
    tf_sess = tf.Session(config=tf_config)
    tf.import_graph_def(graph, name='')

    # Placeholders
    tf_input = tf_sess.graph.get_tensor_by_name(input_names[0] + ':0')
    tf_boxes = tf_sess.graph.get_tensor_by_name(output_names[0] + ':0')
    tf_classes = tf_sess.graph.get_tensor_by_name(output_names[1] + ':0')
    tf_scores = tf_sess.graph.get_tensor_by_name(output_names[2] + ':0')
    tf_num_detections = tf_sess.graph.get_tensor_by_name(output_names[3] + ':0')

    # First slower inference
    tf_sess.run([tf_scores, tf_boxes, tf_classes, tf_num_detections], feed_dict={
                tf_input: image_resized[None, ...]})

    # Benchmarking loop
    elapsed_times = []    
    detections = []
    for i in range(n_samples):
        start = datetime.now()
        scores, boxes, classes, _ = tf_sess.run([tf_scores, tf_boxes, tf_classes, tf_num_detections], feed_dict={
                                    tf_input: image_resized[None, ...]})
        elapsed = datetime.now() - start
        elapsed_times.append(elapsed)
        detections.append([np.squeeze(boxes), np.squeeze(scores)])
    tf_sess.close()
    return elapsed_times, detections


def inference_times_from_rosbag(img_list, input_shape, graph,
                                input_names, output_names, allow_growth):
    # TODO: merge both inference times methods into one
    # TODO: cope with non-MobileNet/SSD networks (I/O tensors)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = allow_growth
    tf_sess = tf.Session(config=tf_config)
    tf.import_graph_def(graph, name='')

    # Placeholders
    tf_input = tf_sess.graph.get_tensor_by_name(input_names[0] + ':0')
    tf_boxes = tf_sess.graph.get_tensor_by_name(output_names[0] + ':0')
    tf_classes = tf_sess.graph.get_tensor_by_name(output_names[1] + ':0')
    tf_scores = tf_sess.graph.get_tensor_by_name(output_names[2] + ':0')
    tf_num_detections = tf_sess.graph.get_tensor_by_name(output_names[3] + ':0')

    # First slower inference
    tf_sess.run([tf_scores, tf_boxes, tf_classes, tf_num_detections], feed_dict={
                tf_input: np.zeros(input_shape)[None, ...]})

    inference_results = []
    for img_arr, ts in img_list:
        img_res = cv2.resize(img_arr, input_shape)
        start = datetime.now()
        # Run the inference!
        scores, boxes, classes, _ = tf_sess.run([tf_scores, tf_boxes, tf_classes, tf_num_detections], feed_dict={
                                        tf_input: image_resized[None, ...]})
        elapsed = datetime.now() - start

        inference_results.append(img_arr, ts, scores, boxes, classes, elapsed)

    # Free resources
    tf_sess.close()
    return inference_results
