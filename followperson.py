#
# Created on Dec, 2019
#
#
__author__ = '@naxvm'


import argparse
import sys
from datetime import datetime, timedelta
from os import path

import numpy as np
import yaml

import cv2
import rospy
import utils
from cprint import cprint
from faced.detector import FaceDetector
from imageio import imread
from benchmarkers import FollowPersonBenchmarker
# from Tracking.motors import Motors
from Tracking import tracking
from Perception.Camera.ROSCam import ROSCam, IMAGE_HEIGHT, IMAGE_WIDTH
from Perception.Net.detection_network import DetectionNetwork
from Perception.Net.networks_controller import NetworksController
from Perception.Net.facenet import FaceNet
from Perception.Net.utils import visualization_utils as vis_utils

MAX_ITERS = None

if __name__ == '__main__':
    # Parameter parsing
    parser = argparse.ArgumentParser(description='Run the main followperson script with the provided configuration')
    parser.add_argument('config_file', type=str, help='Path for the YML configuration file')
    args = parser.parse_args()
    with open(args.config_file, 'r') as f:
        cfg = yaml.safe_load(f)

    rospy.init_node(cfg['NodeName'])
    # Requested behavioral
    benchmark = cfg['Benchmark']
    nets = cfg['Networks']


    # Instantiations
    if benchmark:
        cam = ROSCam(cfg['Topics'], cfg['RosbagFile'])
        n_images = cam.getBagLength(cfg['Topics'])
        benchmarker = FollowPersonBenchmarker(cfg['LogDir'])
        # Save the video output
        # save_video = cfg['SaveVideo']
        v_path = path.join(benchmarker.dirname, 'output.mp4')
        v_out = cv2.VideoWriter(v_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (2*IMAGE_WIDTH, 2*IMAGE_HEIGHT))

    else:
        cam = ROSCam(cfg['Topics'])


    if benchmark: zero_time = datetime.now(); step_time = zero_time

    # Person detection network (SSD or YOLO)
    input_shape = (nets['DetectionHeight'], nets['DetectionWidth'], 3)
    pers_det = DetectionNetwork(nets['Arch'], input_shape, nets['DetectionModel'])
    if benchmark: t_pers_det = datetime.now() - step_time; step_time = datetime.now()


    # Face detection network. The frozen graphs can't be overridden as they are included in the
    # faced package. Use symlinks in order to exchange them for others.
    face_det = FaceDetector()
    if benchmark: t_face_det = datetime.now() - step_time; step_time = datetime.now()

    # FaceNet embedding encoder.
    face_enc = FaceNet(nets['FaceEncoderModel'])
    face_img = imread(cfg['RefFace'])
    fbox = face_det.predict(face_img)
    ref_face = utils.crop_face(face_img, fbox)
    face_enc.set_reference_face(ref_face)
    if benchmark: t_face_enc = datetime.now() - step_time; step_time = datetime.now()

    # The networks are ready to be threaded!!
    nets_c = NetworksController(pers_det, face_det, face_enc, benchmark=True)
    # Configure the controller
    nets_c.setCamera(cam)
    nets_c.daemon = True
    nets_c.start()
    if benchmark: ttfi = datetime.now() - zero_time



    # Person tracker
    p_tracker = tracking.PersonTracker(same_person_thr=60)
    # Motors instance
    # motors = Motors(cfg['Topics']['Motors'])

    iteration = 0
    elapsed_times = []

    def shtdn_hook():
        rospy.loginfo("\nCleaning and exiting...")
        nets_c.close_all()
        cv2.destroyAllWindows()
        if benchmark:
            v_out.release()

    # Register shutdown hook
    rospy.on_shutdown(shtdn_hook)

    while not rospy.is_shutdown():
        if not nets_c.is_activated:
            rospy.signal_shutdown('ROSBag completed!')

        ################
        ### TRACKING ###
        ################
        # Forward step in the tracking using the detections
        cprint.info('=====')
        cprint.info(f'Detections: {len(nets_c.persons)}|{len(nets_c.faces)}')
        p_tracker.handleDetections(nets_c.persons)
        cprint.info(f'Tracked {len(p_tracker.persons)} persons')
        p_tracker.handleFaces(nets_c.faces, nets_c.similarities, nets_c.image)

        # And process the similarities
        p_tracker.checkRef()

        # Persons ready to be fetched
        persons = p_tracker.persons

        # Move towards the ref person
        for person in persons:
            if person.is_ref:
                w_error = utils.computeWError(person.coords, IMAGE_WIDTH)
                x_error = utils.computeXError(person.coords, nets_c.depth)
                cprint(f'w: {w_error}\tx: {x_error}')

        ###############
        ### DRAWING ###
        ###############
        # Draw the images.
        image = nets_c.image
        depth = 255.0 * (1 - nets_c.depth / 6.0) # 8-bit quantization of the effective Xtion range
        transformed = np.copy(image)

        for person in persons:
            x1, y1, x2, y2 = person.coords[0], person.coords[1], person.coords[0]+person.coords[2], person.coords[1]+person.coords[3]
            if person.is_ref:
                color = 'green'
            else:
                color = 'red'
            vis_utils.draw_bounding_box_on_image_array(transformed, y1, x1, y2, x2, color=color, use_normalized_coordinates=False)

            for face in person.ftrk.faces:
                x1, y1, x2, y2 = face.coords[0]-face.coords[2]//2, face.coords[1]-face.coords[3]//2, face.coords[0]+face.coords[2]//2, face.coords[1]+face.coords[3]//2
                vis_utils.draw_bounding_box_on_image_array(transformed, y1, x1, y2, x2, color='blue', use_normalized_coordinates=False)

        # Show the images
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        transformed = cv2.cvtColor(transformed, cv2.COLOR_RGB2BGR)
        depth = cv2.applyColorMap(depth.astype(np.uint8), cv2.COLORMAP_JET)
        inputs = np.vstack((image, depth))

        moves = np.zeros_like(image)
        outputs = np.vstack((moves, transformed))

        total_out = np.hstack((inputs, outputs))
        cv2.imshow('Output', cv2.resize(total_out, dsize=(IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_CUBIC))
        cv2.waitKey(1)
        if benchmark: v_out.write(total_out)


    # Finish the execution
    if benchmark:
        benchmarker.write_benchmark(nets_c.total_times,
                                    cfg['RosbagFile'],
                                    cfg['Networks']['DetectionModel'],
                                    cfg['Networks']['FaceEncoderModel'],
                                    t_pers_det, t_face_det, t_face_enc, ttfi)

    rospy.signal_shutdown("Finished!!")
