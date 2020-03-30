import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yaml
import pickle
from PIL import Image
from Net.detection_network import DetectionNetwork
from Net.utils import nms
from Camera.ROSCam import ROSCam
from os import listdir, path, makedirs
from cprint import cprint



FILENAME_FORMAT = '%Y%m%d %H%M%S'
TO_MS = np.vectorize(lambda x: x.seconds * 1000.0 + x.microseconds / 1000.0) # Auxiliary vectorized function


class FollowPersonBenchmarker:
    '''Writer for a full-set benchmark using a certain configuration.'''
    def __init__(self, logdir):
        self.logdir = logdir
        self.description = None
        # Create the benchmark folder
        folder_name = datetime.now().strftime(FILENAME_FORMAT)
        self.dirname = path.join(self.logdir, folder_name)
        if not path.exists(self.dirname):
            makedirs(self.dirname)


    def write_benchmark(self, times_list, rosbag_file,
                        pdet_model, fenc_model,
                        t_pers_det, t_face_det, t_face_enc, ttfi,
                        write_iters=True):
        '''Write the metrics to the output file.'''

        benchmark = {}
        summary = {}
        summary['1.- Configuration'] = {
            '1.- Networks': {
                '1.- PersonDetectionModel': pdet_model,
                # '2.- FaceDetectionModel':   fdet_model,
                '2.- FaceEncodingModel':    fenc_model,
            },
            '2.- RosbagFile': rosbag_file,
        }
        summary['2.- LoadTimes'] = {
            '1.- PersonDetectionNetworkLoad': f'{TO_MS(t_pers_det):.4f} ms',
            '2.- FaceDetectionNetworkLoad':   f'{TO_MS(t_face_det):.4f} ms',
            '3.- FaceEncodingNetworkLoad':    f'{TO_MS(t_face_enc):.4f} ms',
            '4.- TTFI':                       f'{TO_MS(ttfi):.4f} ms',
        }

        # Process the measured times
        import pickle
        with open('times_list.pkl', 'wb') as f:
            pickle.dump(times_list, f)
        # Drop the first (slower) inferences
        times_raw = np.array(times_list)[2:, :]

        pdets_raw = np.array(list(times_raw[:, 0]))
        total_pdets = pdets_raw.copy()
        total_pdets[:, 0] = TO_MS(total_pdets[:, 0])

        fdets_raw = np.array(list(times_raw[:, 1]))
        total_fdets = fdets_raw.copy()
        total_fdets[:, 0] = TO_MS(fdets_raw[:, 0])

        fencs_raw = np.array(list(times_raw[:, 2]))
        total_fencs = fencs_raw.copy()
        total_fencs[:, 0] = TO_MS(total_fencs[:, 0])
        total_fencs_flt = total_fencs[total_fencs[:, 1] > 0]  # Just times belonging to a face filtering
        # total_fencs_flt = list(filter(lambda x: x[1] > 0, total_fencs))

        iters_raw = times_raw[:, 3]
        total_iters = TO_MS(iters_raw)[1:]

        summary['3.- Stats'] = {
            '1.- PersonDetection': {
                '1.- Mean': f'{total_pdets.mean():.4f} ms',
                '2.- Std':  f'{total_pdets.std():.4f} ms',
            },
            '2.- FaceDetection': {
                '1.- Mean': f'{total_fdets.mean():.4f} ms',
                '2.- Std':  f'{total_fdets.std():.4f} ms',
            },
            '3.- FaceEncoding': {
                '1.- Mean': f'{total_fencs_flt.mean():.4f} ms',
                '2.- Std':  f'{total_fencs_flt.std():.4f} ms',
            },
            '4.- IterationTime': {
                '1.- Mean': f'{total_iters.mean():.4f} ms',
                '2.- Std': f'{total_iters.std():.4f} ms',
            }
        }
        benchmark['1.- Summary'] = summary

        if write_iters:
            iterations = []
            for it_time, pdet, fdet, fenc in zip(total_iters, total_pdets, total_fdets, total_fencs):
                iteration = {}
                # Persons detection
                persons_detection = {}
                n_persons = len(pdet)
                persons_detection['1.- NumDetections'] = int(n_persons)
                if n_persons == 0:
                    continue
                persons_detection['2.- TotalTime'] = f'{pdet.sum():.4f} ms'
                # Faces detection
                faces_detection = {}
                n_faces = fdet[1]
                faces_detection['1.- NumDetections'] = int(n_faces)
                if n_faces == 0:
                    continue
                faces_detection['2.- TotalTime'] = f'{fdet.sum():.4f} ms'
                # Faces encoding
                faces_encoding = {}
                n_faces = fenc[1]
                faces_encoding['1.- NumDetections'] = int(n_faces)
                if n_faces == 0:
                    continue
                faces_encoding['2.- TotalTime'] = f'{fenc.sum():.4f} ms'

                iteration = {
                    '1.- PersonsDetection':   persons_detection,
                    '2.- FacesDetection':     faces_detection,
                    '3.- FacesEncoding':      faces_encoding,
                    '4.- TotalIterationTime': f'{it_time:.4f} ms',
                }
                iterations.append(iteration)

            benchmark['2.- Iterations'] = iterations

        benchmark_name = path.join(self.dirname, 'benchmark.yml')

        # Dump
        with open(benchmark_name, 'w') as f:
            yaml.dump(benchmark, f)

        print(f'Saved on {benchmark_name}')
        # Graphs
        #  Total iteration time
        fig, ax = plt.subplots()
        ax.plot(total_iters)
        ax.set_ylim([0, total_iters.mean() + 2 * total_iters.std()])
        ax.set_title('Total iteration time')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Time (ms)')
        figname = path.join(self.dirname, 'iterations.png')
        fig.savefig(figname)

        #  Person detection time
        fig, ax = plt.subplots()
        ax.plot(total_pdets[:, 0])
        ax.set_ylim([0, total_pdets.mean() + 2 * total_pdets.std()])
        ax.set_title('Person detection time')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Time (ms)')
        figname = path.join(self.dirname, 'person_detections.png')
        fig.savefig(figname)

        #  Face detection time
        fig, ax = plt.subplots()
        ax.plot(total_fdets[:, 0])
        ax.set_ylim([0, total_fdets.mean() + 2 * total_fdets.std()])
        ax.set_title('Face detection time')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Time (ms)')
        figname = path.join(self.dirname, 'face_detections.png')
        fig.savefig(figname)

        #  Face encoding time
        fig, ax = plt.subplots()
        ax.plot(total_fencs[:, 0])
        ax.set_ylim([0, total_fencs.mean() + 2 * total_fencs.std()])
        ax.set_title('Face encoding time')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Time (ms)')
        figname = path.join(self.dirname, 'face_encoding.png')
        fig.savefig(figname)

        # Save the times matrix (for further inspection)
        dump_file = path.join(self.dirname, 'times.pkl')
        with open(dump_file, 'wb') as f:
            pickle.dump([total_iters, total_pdets, total_fdets, total_fencs], f)

class SingleModelBenchmarker:
    ''' Writer for a single model benchmark using. '''
    def __init__(self, save_in):
        self.save_in = save_in

    def write_benchmark(self, total_times, model_name, rosbag_file, arch, write_iters=True):
        # Convert the lapse measurements to milliseconds
        total_ms = np.array(total_times)
        total_ms[:, 0] = TO_MS(total_ms[:, 0])
        # Filter the non-empty inferences
        nonempty = total_ms[total_ms[:, 1] > 0]

        dic = {}
        # Metadata
        dic['1.- Meta'] = {
            '1.- ModelName': model_name,
            '2.- ROSBag': rosbag_file,
            '3.- Architecture': arch,
        }
        # Stats
        stats_total = {
            'Mean': f'{total_ms[:, 0].mean():.4f} ms',
            'Std':  f'{total_ms[:, 0].std():.4f} ms',
        }

        stats_nonempty = {
            'Mean': f'{nonempty[:, 0].mean():.4f} ms',
            'Std':  f'{nonempty[:, 0].std():.4f} ms',
        }

        dic['2.- Stats'] = {
            '1.- Total': stats_total,
            '2.- NonEmpty': stats_nonempty,
        }

        if write_iters:
            iters = {}
            for idx, iteration in enumerate(total_ms):
                iters[idx] = {
                    'InferenceTime': f'{iteration[0]:.4f} ms',
                    'NumDetections': iteration[1]
                }
            dic['3.- Iterations'] = iters

        # Finally, dump the results into the requested file.
        with open(self.save_in, 'w') as f:
            yaml.dump(dic, f)
        cprint.ok(f'Benchmark written in {self.save_in}!')




if __name__ == '__main__':
    description = ''' If this script is called, it will perform inferences using
    a provided model on a test rosbag, and will store the results into a YML file. '''

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('pb_file', type=str, help='.pb file containing the frozen graph to test')
    parser.add_argument('arch', type=str, help='Detection architecture of the provided network')
    parser.add_argument('input_width', type=int, help='Width of the network input')
    parser.add_argument('input_height', type=int, help='Height of the network input')
    parser.add_argument('rosbag_file', type=str, help='ROSBag to perform the test on')
    parser.add_argument('save_in', type=str, help='File in which write the output result')
    # Parse the args
    args = parser.parse_args()

    # print('\n' * 10, listdir('.'), '\n' * 20)
    pb_file = args.pb_file
    rosbag_file = args.rosbag_file

    # Check the existance of the files
    if not path.isfile(pb_file):
        cprint.fatal(f'Error: the provided frozen graph {pb_file} does not exist', interrupt=True)
    if not path.isfile(rosbag_file):
        cprint.fatal(f'Error: the provided ROSBag {rosbag_file} does not exist', interrupt=True)


    save_in = args.save_in
    arch = args.arch
    input_w, input_h = args.input_width, args.input_height

    # Create the ROSCam to open the ROSBag
    topics = {'RGB':   '/camera/rgb/image_raw',
              'Depth': '/camera/depth_registered/image_raw'}
    cam = ROSCam(topics, rosbag_file)

    # Load the model into a network object to perform inferences
    input_shape = (input_h, input_w, 3)
    net = DetectionNetwork(arch, input_shape, pb_file)

    total_times = []
    # Iterate the rosbag
    bag_len = cam.getBagLength(topics)
    img_count = 0
    while True:
        cprint.info(f'\tImage {img_count}/{bag_len}')
        img_count += 1
        try:
            image, _ = cam.getImages()
        except StopIteration:
            cprint.ok('ROSBag completed!')
            break

        image = np.array(Image.fromarray(image).resize(input_shape[:2]))

        if arch in ['ssd', 'yolov3', 'yolov3tiny']:
            dets, elapsed = net.predict(image)
            n_dets = len(dets)

        elif arch in ['face_yolo', 'face_corrector']:
            feed_dict = {net.input: image[None, ...], net.training: False}
            out, elapsed = net._forward_pass(feed_dict)
            n_dets = len(out[0])

        elif arch == 'facenet':
            feed_dict = {net.input: image[None, ...], net.phase_train: False}
            out, elapsed = net._forward_pass(feed_dict)
            n_dets = len(out[0])


        else:
            cprint.fatal(f'{arch} benchmarking not implemented yet!', interrupt=True)

        total_times.append([elapsed, n_dets])

    # The benchmark is finished. We log the results now.
    net.sess.close()
    writer = SingleModelBenchmarker(save_in)
    writer.write_benchmark(total_times, pb_file, rosbag_file, arch, write_iters=True)
