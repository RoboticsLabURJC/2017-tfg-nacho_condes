import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yaml
import pickle
from PIL import Image
from Perception.Net.detection_network import DetectionNetwork
from Perception.Net.utils import nms
from Perception.Camera.ROSCam import ROSCam
from os import listdir, path, makedirs
from scipy.stats import median_absolute_deviation as mad
from cprint import cprint



FILENAME_FORMAT = '%Y%m%d %H%M%S'
TO_MS = lambda x: x.seconds*1000.0 + x.microseconds/1000.0 # Auxiliary vectorized function
TO_MS_VEC = np.vectorize(TO_MS)


class FollowPersonBenchmarker:
    """Writer for a full-set benchmark using a certain configuration."""
    def __init__(self, logdir):
        self.logdir = logdir
        # Sections
        self.config = None
        self.load_times = None
        self.detection_stats = None
        self.tracking_stats = None
        self.iterations = None

        self.plot_times = {}
        # Create the benchmark folder
        folder_name = datetime.now().strftime(FILENAME_FORMAT)
        self.dirname = path.join(self.logdir, folder_name)
        if not path.exists(self.dirname):
            makedirs(self.dirname)


    def makeConfig(self, pdet_model, fenc_model, rosbag_file, xcfg, wcfg, ptcfg):
        """Build the config section for the benchmark report."""
        config = {
            '1.- Networks': {
                '1.- PersonDetectionmodel': pdet_model,
                '2.- FaceEncodingModel': fenc_model,
            },
            '2.- RosbagFile': rosbag_file,
            '3.- Tracker': {
                    '1.- Patience': ptcfg['Patience'],
                    '2.- RefSimThr': ptcfg['RefSimThr'],
                    '3.- SamePersonThr': ptcfg['SamePersonThr'],
            },
            '4.- XController': {
                '1.- Kp': xcfg['Kp'],
                '2.- Ki': xcfg['Ki'],
                '3.- Kd': xcfg['Kd'],
            },
            '5.- WController': {
                '1.- Kp': wcfg['Kp'],
                '2.- Ki': wcfg['Ki'],
                '3.- Kd': wcfg['Kd'],
            },
        }
        self.config = config

    def makeLoadTimes(self, t_pers_det, t_face_det, t_face_enc, ttfi):
        """Build the load times section for the benchmark report."""

        load_times = {
            '1.- PersonDetectionNetworkLoad': TO_MS(t_pers_det),
            '2.- FaceDetectionNetworkLoad':   TO_MS(t_face_det),
            '3.- FaceEncodingNetworkLoad':    TO_MS(t_face_enc),
            '4.- TTFI':                       TO_MS(ttfi),
        }
        self.load_times = load_times

    def makeDetectionStats(self, frames_times):
        """Build the detection statistics section for the benchmark report."""

        # Convert the times to an array
        measured_times = np.array(list(frames_times.values()))

        # Split into the components
        ## Person detection times
        pdet_times = np.array([TO_MS(dt) for dt, _ in measured_times[:, 0]])
        self.plot_times['pdet'] = pdet_times
        ## Face detection times
        fdet_times = np.array([TO_MS(dt) for dt, _ in measured_times[:, 1]])
        self.plot_times['fdet'] = fdet_times
        ## Face encoding times (and filtered version with only iterations including faces)
        fenc_times = np.array([TO_MS(dt) for dt, _ in measured_times[:, 2]])
        self.plot_times['fenc'] = fenc_times
        fenc_times_flt = np.array([TO_MS(dt) for dt, count in measured_times[:, 2] if count>0])
        ## Iteration times
        iter_times = np.array(list(map(TO_MS, measured_times[:, 3])))
        self.plot_times['iter'] = iter_times

        # Build the stats!
        # Median + Median Absolute Deviation
        detection_stats = {
            '1.- PersonDetection': {
                '1.- Median': f'{np.median(pdet_times):.4f} ms',
                '2.- MAD':    f'{mad(pdet_times):.4f} ms',
            },
            '2.- FaceDetection': {
                '1.- Median': f'{np.median(fdet_times):.4f} ms',
                '2.- MAD':  f'{mad(fdet_times):.4f} ms',
            },
            '3.- FaceEncoding': {
                '1.- Median': f'{np.median(fenc_times_flt):.4f} ms',
                '2.- MAD':  f'{mad(fenc_times_flt):.4f} ms',
            },
            '4.- NeuralTime': {
                '1.- Median': f'{np.median(iter_times):.4f} ms',
                '2.- MAD': f'{mad(iter_times):.4f} ms',
            }
        }
        self.detection_stats = detection_stats

    def makeTrackingStats(self, tracked_persons, frames_with_ref):
        """Build the tracking statistics section for the benchmark report."""

        tracking_stats = {
            '1.- TrackedPersons': tracked_persons,
            '2.- FramesWithRef': frames_with_ref,
        }
        self.tracking_stats = tracking_stats

    def makeIters(self, frames_times, frames_numtrackings, frames_errors, ref_coords, frames_responses):
        """Write the iterations for each processed frame in the benchmark."""

        iterations = []
        for frame, times in frames_times.items():
            # Fill selectively for each frame
            frame_info = {}

            frame_info['1.- Frame'] = frame

            frame_info['2.- PersonDetection'] = {
                    '1.- Elapsed': f'{TO_MS(times[0][0]):.4f} ms',
                    '2.- Number': times[0][1],
            }

            frame_info['3.- FaceDetection'] = {
                    '1.- Elapsed': f'{TO_MS(times[1][0]):.4f} ms',
                    '2.- Number': times[1][1],
            }

            frame_info['4.- FaceEncoding'] = {
                    '1.- Elapsed': f'{TO_MS(times[2][0]):.4f} ms',
                    '2.- Number': times[2][1],
            }

            frame_info['5.- NeuralTime'] = f'{TO_MS(times[3]):.4f} ms'

            # From now on, each frame might not having been tracked
            val = frames_numtrackings.get(frame, '')
            frame_info['6.- Tracking'] = {
                '1.- TrackedPersons': val,
            }

            errors = frames_errors.get(frame, ['', ''])
            responses = frames_responses.get(frame, ['', ''])
            coords = ref_coords.get(frame, ['', '', '', '', ''])
            frame_info['7.- XControl'] = {
                '1.- Error': f'{errors[1]}',
                '2.- Response': f'{responses[1]}',
            }

            frame_info['8.- WControl'] = {
                '1.- Error': f'{errors[0]}',
                '2.- Response': f'{responses[0]}',
            }
            frame_info['9.- RefCoords'] = {
                '1.- X': f'{coords[0]}',
                '2.- Y': f'{coords[1]}',
                '3.- W': f'{coords[2]}',
                '4.- H': f'{coords[3]}',
            }
            iterations.append(frame_info)

        self.iterations = iterations


    def writeBenchmark(self):
        """Write the metrics to the output file."""

        # Build the entire structure
        benchmark = {
            '1.- Summary': {
                '1.- Config': self.config,
                '2.- LoadTimes': self.load_times,
                '3.- DetectionStats': self.detection_stats,
                '4.- TrackingStats': self.tracking_stats,
            },
            '2.- Iterations': self.iterations
        }

        benchmark_name = path.join(self.dirname, 'benchmark.yml')

        # Dump
        with open(benchmark_name, 'w') as f:
            yaml.dump(benchmark, f)

        print(f'Saved on {benchmark_name}')

        # Graphs spanning ± 2σ from mean
        ## Total iteration time
        fig, ax = plt.subplots()
        times = self.plot_times['iter']
        ax.plot(times)
        ylim = [max([0, times.mean() - 2*times.std()]), times.mean() + 2*times.std()]
        ax.set_ylim(ylim)
        ax.set_title('Total iteration time')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Time (ms)')
        figname = path.join(self.dirname, 'iterations.png')
        fig.savefig(figname)

        ## Person detection time
        fig, ax = plt.subplots()
        times = self.plot_times['pdet']
        ax.plot(times)
        ylim = [max([0, times.mean() - 2*times.std()]), times.mean() + 2*times.std()]
        ax.set_ylim(ylim)
        ax.set_title('Person detection time')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Time (ms)')
        figname = path.join(self.dirname, 'person_detections.png')
        fig.savefig(figname)

        ## Face detection time
        fig, ax = plt.subplots()
        times = self.plot_times['fdet']
        ax.plot(times)
        ylim = [max([0, times.mean() - 2*times.std()]), times.mean() + 2*times.std()]
        ax.set_ylim(ylim)
        ax.set_title('Face detection time')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Time (ms)')
        figname = path.join(self.dirname, 'face_detections.png')
        fig.savefig(figname)

        ## Face encoding time
        fig, ax = plt.subplots()
        times = self.plot_times['fenc']
        ax.plot(times)
        ylim = [max([0, times.mean() - 2*times.std()]), times.mean() + 2*times.std()]
        ax.set_ylim(ylim)
        ax.set_title('Face encoding time')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Time (ms)')
        figname = path.join(self.dirname, 'face_encoding.png')
        fig.savefig(figname)

        # Save the times dict (for further inspection)
        dump_file = path.join(self.dirname, 'plot_times.pkl')
        with open(dump_file, 'wb') as f:
            pickle.dump(self.plot_times, f)

class SingleModelBenchmarker:
    ''' Writer for a single model benchmark using. '''
    def __init__(self, save_in):
        self.save_in = save_in

    def write_benchmark(self, total_times, model_name, rosbag_file, arch, write_iters=True):
        # Convert the lapse measurements to milliseconds
        total_ms = np.array(total_times)
        total_ms[:, 0] = TO_MS_VEC(total_ms[:, 0])
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
        cprint.info(f'\rImage {img_count}/{bag_len}', end='', flush=True)
        print()
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
