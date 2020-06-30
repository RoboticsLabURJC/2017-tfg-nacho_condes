import pickle
from labelmesequence import LabelMeSequence
from Perception.Net.detection_network import DetectionNetwork
from benchmarkers import TO_MS
import numpy as np
import utils

STANDARD_PATH = 'Optimization/dl_models/ssd_mobilenet_v1_0.75_depth_coco/frozen_inference_graph.pb'
TRT_PATH = 'Optimization/dl_models/ssd_mobilenet_v1_0.75_depth_coco/optimizations/FP16_50_1.pb'
VIDEO_PATH = 'own_videos/test'
GT_DETECTIONS_PATH = 'test3/gt_detections.pkl'
GT_TIMES_PATH = 'test3/gt_times.pkl'
TRT_DETECTIONS_PATH = 'test3/trt_detections.pkl'
TRT_TIMES_PATH = 'test3/trt_times.pkl'


video = LabelMeSequence(VIDEO_PATH)


def save_inferences(graph_path, detections_path, times_path):
    """Save inferences for the comparison"""
    net = DetectionNetwork('ssd', (300, 300, 3), frozen_graph=graph_path)
    detections = []
    el_times = []
    for frix, (image, _) in enumerate(video):
        print(f'Frame {frix}/{len(video)}')
        persons, elapsed = net.predict(image)
        detections.append(persons)
        el_times.append(TO_MS(elapsed))

    with open(detections_path, 'wb') as f:
        pickle.dump(detections, f)

    with open(times_path, 'wb') as f:
        pickle.dump(el_times, f)

def computeIoU(gt_detections_path, trt_detections_path, gt_times_path, trt_times_path):
    """Extract the IoU between persons pairs."""
    # Load the files
    with open(gt_detections_path, 'rb') as f:
        gt_detections = pickle.load(f)
    with open(gt_times_path, 'rb') as f:
        gt_times = pickle.load(f)
    with open(trt_detections_path, 'rb') as f:
        trt_detections = pickle.load(f)
    with open(trt_times_path, 'rb') as f:
        trt_times = pickle.load(f)

    times = np.zeros((len(gt_times), 2))
    times[:, 0] = gt_times
    times[:, 1] = trt_times
    np.savetxt('test3/times.csv', times, delimiter=';')
    # And the IoU
    JIs_avg = []
    for gt_persons, trt_persons in zip(gt_detections, trt_detections):
        # print('gt_persons', gt_persons)
        # print('trt_persons', trt_persons)
        # break
        JIs_iter = []
        for trt_person in trt_persons:
            JIs = [utils.jaccardIndex(gt_person[:4], trt_person[:4]) for gt_person in gt_persons]
            if len(JIs) > 0:
                JIs_iter.append(max(JIs))
        if len(JIs_iter) > 0:
            JIs_avg.append(np.mean(JIs_iter))
        else:
            JIs_avg.append(np.nan)
    np.savetxt('test3/JIs.csv', np.array(JIs_avg), delimiter=';')


if __name__ == '__main__':
    # Save standard inferences
    # save_inferences(STANDARD_PATH, GT_DETECTIONS_PATH, GT_TIMES_PATH)
    # save_inferences(TRT_PATH, TRT_DETECTIONS_PATH, TRT_TIMES_PATH)
    computeIoU(GT_DETECTIONS_PATH, TRT_DETECTIONS_PATH, GT_TIMES_PATH, TRT_TIMES_PATH)