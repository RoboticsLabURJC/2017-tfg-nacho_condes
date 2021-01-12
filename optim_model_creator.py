#
# Created on Feb, 2020
#
__author__ = 'Nacho Condés'


import argparse
import os
from cprint import cprint
from time import sleep
import yaml

description = ''' Perform the optimization of a given model using the TF-TRT interface converter to the TensorRT engine.
           Search over a grid of provided parameters, and run a benchmark on each configuration, dumping the result on a YML file
           besides the resulting frozen graph. '''

parser = argparse.ArgumentParser(description=description)
parser.add_argument('config_file', type=str, help='YML containing the parameters grid to try.')
args = parser.parse_args()

config_file = args.config_file
if not os.path.isfile(config_file):
    cprint.fatal(f'Error: the file {config_file} does not exist!', interrupt=True)

# Otherwise, parse it!
with open(config_file, 'r') as f:
    cfg = yaml.safe_load(f)

BAGS_DIR = 'bags'
MODELS_DIR = 'Optimization/dl_models'
OPTS_SUBDIR = 'optimizations'
OPTIMIZATION_SCRIPT = 'Optimization/optimize_graph.py'
BENCHMARKING_SCRIPT = 'benchmarkers.py'


model_name = cfg['ModelName']
saved_as = cfg['SavedAs']
input_w, input_h = cfg['InputWidth'], cfg['InputHeight']
arch = cfg['Architecture']
optim_params = cfg['OptimParams']

input_names = cfg['InputNames']
output_names = cfg['OutputNames']
write_nodes = cfg['WriteNodes']
rosbag_file = cfg['RosBag']

# Iteration over the parameters grid...
for format in optim_params['Formats']:
    for mss in optim_params['MSS']:
        for mce in optim_params['MCE']:
            # === Optimization ===
            # Where the file will be stored.
            filename = f'{format}_{mss}_{mce}'
            pb_name = os.path.join(MODELS_DIR, model_name, OPTS_SUBDIR, filename + '.pb')
            yml_name = os.path.join(MODELS_DIR, model_name, OPTS_SUBDIR, filename + '.yml')

            optim_command = (f"python3 {OPTIMIZATION_SCRIPT} {model_name} {saved_as} {input_w} " +
                             f"{input_h} {format} {mss} {mce} {optim_params['AllowGrowth']} " +
                             f"{arch} {pb_name} --input_names {' '.join(input_names)} " +
                             f"--output_names {' '.join(output_names)} " +
                             f"--write_nodes {write_nodes} "
                             # f"--benchmark_rosbag {rosbag_file} " +
            )
            cprint.info(optim_command)
            os.system(optim_command)

            sleep(2)

            # === Benchmark ===
            benchmark_command = (f"python3 {BENCHMARKING_SCRIPT} {pb_name} {arch} {input_w} {input_h} " +
                                 f"{rosbag_file} {yml_name}")
            cprint.info(benchmark_command)
            os.system(benchmark_command)
            cprint.info('\n' * 15)
