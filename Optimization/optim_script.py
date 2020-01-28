import os
from time import sleep
import yaml
import argparse
from cprint import cprint


parser = argparse.ArgumentParser(description='Run different optimizations on a saved graph.')

parser.add_argument('config_file', type=str, help='YML file describing the desired optimization.')

args = parser.parse_args()
config_file = args.config_file

if not os.path.isfile(config_file):
    cprint.fatal(f'Error: the file {config_file} does not exist!')
    exit()

# Parse the configuration file
with open(config_file, 'r') as f:
    cfg = yaml.safe_load(f)

model_name = cfg['ModelName']
input_w, input_h = cfg['InputWidth'], cfg['InputHeight']
arch = cfg['Architecture']
optim_params = cfg['OptimParams']

saved_as = cfg['SavedAs']

input_names = cfg['InputNames']
output_names = cfg['OutputNames']
write_nodes = cfg['WriteNodes']

# Iterate over the optimization grid!

for format in optim_params['Formats']:
    for mss in optim_params['MSS']:
        for mce in optim_params['MCE']:
            command = (f"python optimize_graph.py {model_name} {saved_as} {input_w} " +
                       f"{input_h} {format} {mss} {mce} {optim_params['AllowGrowth']} " +
                       f"{arch} --input_names {' '.join(input_names)} " +
                       f"--output_names {' '.join(output_names)} " +
                       f"--write_nodes {write_nodes} ")
            cprint.info(command)
            os.system(command)
            sleep(2)
            cprint.ok('Finished!!\n\n')
