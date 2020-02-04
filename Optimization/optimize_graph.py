#
# Created on Jan. 2020
#  @author: naxvm
#

import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from cprint import cprint
from tf_trt_models.detection import build_detection_graph
from tensorflow.python.compiler.tensorrt import trt_convert
import argparse
import tensorflow as tf


MODELS_DIR = os.path.join('Optimization', 'dl_models')
FG_NAME = 'frozen_graph.pb'
FORCE_NMS_CPU = True # better performance except on JetPack 4.3

def writeNodes(model_path, graph_def):
    ''' Dump the node names in a graph into a text file. '''
    text_filename = os.path.join(model_path, 'nodes.txt')
    nodes = [n.name for n in graph_def.node]
    cprint.info(f'Dumping {len(nodes)} nodes into {text_filename}...')
    with open(text_filename, 'w') as f:
        [f.write(node + '\n') for node in nodes]
    cprint.info('Done!')


def loadFrozenGraph(model_path, write_nodes):
    ''' Load a frozen graph from a .pb file. '''
    model_path = os.path.join(MODELS_DIR, model_path)
    pb_path = os.path.join(model_path, FG_NAME)
    # Check the existance
    if not os.path.isfile(pb_path):
        cprint.fatal(f'Error: the file {pb_path} does not exist.', interrupt=True)

    cprint.info('Loading the frozen graph...')
    graph_def = tf.GraphDef()
    with tf.io.gfile.GFile(pb_path, 'rb') as fid:
        serialized_graph = fid.read()
        graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(graph_def, name='')
    cprint.ok('Loaded!')
    if write_nodes:
        writeNodes(model_path, graph_def)
    return graph_def

def loadCheckpoint(model_path, write_nodes):
    ''' Load a graph model from a training checkpoint. '''
    model_path = os.path.join(MODELS_DIR, model_path)
    if not os.path.exists(model_path):
        cprint.fatal(f'Error: the path {model_path} does not exist.', interrupt=True)

    config_file = os.path.join(model_path, 'pipeline.config')
    if not os.path.isfile(config_file):
        cprint.fatal(f'Error: the config file {config_file} does not exist.', interrupt=True)

    checkpoint_file = os.path.join(model_path, 'model.ckpt')
    if not os.path.isfile(checkpoint_file + '.meta'):
        cprint.fatal(f'Error: the checkpoint file {checkpoint_file} does not exist.', interrupt=True)

    graph_def, input_names, output_names = build_detection_graph(config=config_file, checkpoint=checkpoint_file,
                                                                 score_threshold=0.3, batch_size=1,
                                                                 force_nms_cpu=FORCE_NMS_CPU)
    if write_nodes:
        writeNodes(model_path, graph_def)

    return graph_def, input_names, output_names


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
            use_calibration=False)
        new_g = converter.convert()
    return new_g


def saveTrtGraph(graph, model_path, save_in):
    ''' Save the TRT optimized graph into a frozen graph file. '''
    opt_dir = os.path.dirname(save_in)
    if not os.path.exists(opt_dir):
        os.makedirs(opt_dir)

    with open(save_in, 'wb') as f:
        f.write(graph.SerializeToString())

    cprint.ok(f'Graph saved in {save_in}')


def add_arguments(parser):
    parser.add_argument('model_dir', type=str, help='Model to optimize')
    parser.add_argument('model_format', type=str, help='Format of the saved model', choices=['frozen', 'checkpoint'])
    parser.add_argument('input_w', type=int, help='Width of the input images')
    parser.add_argument('input_h', type=int, help='Height of the input images')
    parser.add_argument('precision', type=str, help='Precision for the conversion', choices=['FP32', 'FP16', 'INT8'])
    parser.add_argument('mss', type=int, help='Minimum segment size for the optimization')
    parser.add_argument('mce', type=int, help='Maximum cached engines for the optimization')
    parser.add_argument('allow_growth', type=bool, help='Whether allowing growth for the TF session allocation')
    parser.add_argument('arch', type=str, help='Architecture of the given model', choices=['ssd', 'yolov3'])
    parser.add_argument('save_in', type=str, help='Desired .pb in which freeze the optimized graph')
    parser.add_argument('--input_names', nargs='*', help='Input tensors')
    parser.add_argument('--output_names', nargs='*', help='Output tensors')
    parser.add_argument('--write_nodes', type=bool, help='Whether writing the node names into a file')


if __name__ == '__main__':
    # Parse the calling arguments
    parser = argparse.ArgumentParser(description='Optimize a TF model (TF-TRT engine)')
    add_arguments(parser)

    args = parser.parse_args()

    model_dir =        args.model_dir
    model_format =     args.model_format
    input_w =          args.input_w
    input_h =          args.input_h
    precision =        args.precision
    mss =              args.mss
    mce =              args.mce
    allow_growth =     args.allow_growth
    arch =             args.arch
    write_nodes =      args.write_nodes
    save_in =          args.save_in

    if model_format == 'frozen':
        # The input and output names have to be provided
        input_names = args.input_names
        output_names = args.output_names
        graph_def = loadFrozenGraph(model_dir, write_nodes)

    else:
        graph_def, input_names, output_names = loadCheckpoint(model_dir, write_nodes)

    cprint.ok('Graph loaded')
    # These nodes can't be optimized
    blacklist_nodes = input_names + output_names
    # Run the optimization!
    trt_graph = optim_graph(graph_def, blacklist_nodes, precision, mss, mce)
    if trt_graph is None:
        cprint.fatal('Error: optimization not completed.', interrupt=True)

    cprint.ok('Optimization done!')
    # And dump the graph into a new .pb file
    trt_path = saveTrtGraph(trt_graph, model_dir, save_in)
