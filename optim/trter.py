#
# Created on Nov, 2019
# @author: naxvm
#
import argparse

parser = argparse.ArgumentParser(description='Optimize a TF model (TF-TRT engine')
parser.add_argument('model', type=str, help='Model to optimize')
parser.add_argument('input_w', type=int, help='Width of the input images')
parser.add_argument('input_h', type=int, help='Height of the input images')
parser.add_argument('precision', type=str, help='Precision for the conversion', choices=['FP32', 'FP16'])
parser.add_argument('mss', type=int, help='Minimum segment size for the optimization')
parser.add_argument('mce', type=int, help='Maximum cached engines for the optimization')
parser.add_argument('allow_growth', type=bool, help='Allow growth or not for the TF session allocation')
parser.add_argument('--net_config_file', type='str', default=None, help='Configuration file for the network architecture if pipeline file can not be provided')
args = parser.parse_args()

# If the arguments are OK, import the TF stuff
from utils import optim_graph, inference_times, BenchmarkWriter
from tf_trt_models.detection import build_detection_graph
import tensorflow as tf

# Parameters
DATA_DIR = 'dl_models/'

MODEL = args.model
INPUT_SHAPE = (args.input_w, args.input_h)

manual_load = args.net_config_file is not None

MODEL_DIR = DATA_DIR + MODEL + '/'

CONFIG_FILE = MODEL_DIR + 'pipeline.config'
CHECKPOINT_FILE = MODEL_DIR + 'model.ckpt'
IMAGE_PATH = 'data/huskies.jpg'

PRECISION = args.precision
MSS = args.mss     # 3, 20, 50
MCE = args.mce     # 1, 3, 5
ALLOW_GROWTH = args.allow_growth


# if manual_load:
#     orig_graph = tf.

# Build the frozen graph
orig_graph, input_names, output_names = build_detection_graph(
    config=CONFIG_FILE,
    checkpoint=CHECKPOINT_FILE,
    score_threshold=0.3,
    batch_size=1,
)

print('Input names:', '\t'.join(input_names))
print('Output names:', '\t'.join(output_names))

# Optimization
new_graph = optim_graph(orig_graph, output_names, PRECISION, MSS, MCE)

print('\n' * 20)

# Measure inference times
times, detections = inference_times(image_path=IMAGE_PATH,
                                    input_shape=INPUT_SHAPE,
                                    graph=new_graph,
                                    input_names=input_names,
                                    output_names=output_names,
                                    allow_growth=ALLOW_GROWTH)

# Write results

bm_writer = BenchmarkWriter(MODEL_DIR + 'optimizations', MODEL)
optim_params = {
    'model_name': MODEL,
    'prec': PRECISION,
    'mss': MSS,
    'mce': MCE,
}

print('Parameters',
      '\n==========',
      f'\n\tBase model: {MODEL}',
      f'\n\tPrecision mode: {MSS}',
      f'\n\tMCE: {MCE}')


filename = f'{PRECISION}_{MSS}_{MCE}'
bm_writer.write_log(times, detections, optim_params, filename + '.yml')

with open(MODEL_DIR + 'optimizations/' + filename + '.pb', 'wb') as f:
    f.write(new_graph.SerializeToString())

print(f"Graph saved on {MODEL_DIR + 'optimizations' + filename + '.pb'}")
