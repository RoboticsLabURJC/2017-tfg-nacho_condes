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
from Camera.ROSCam import ROSCam, IMAGE_HEIGHT, IMAGE_WIDTH
from cprint import cprint
from faced.detector import FaceDetector
from imageio import imread
from logs.benchmarkers import FollowPersonBenchmarker
from Motors.motors import Motors
from Net.detection_network import DetectionNetwork
from Net.facenet import FaceNet
from Net.utils import visualization_utils as vis_utils

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
    benchmark = cfg['Mode'].lower() == 'benchmark'
    nets = cfg['Networks']

    # Instantiations
    if benchmark:
        cam = ROSCam(cfg['Topics'], cfg['RosbagFile'])
        n_images = cam.getBagLength(cfg['Topics'])
        benchmarker = FollowPersonBenchmarker(cfg['LogDir'])

        # Check if we have to save the video output
        save_video = cfg['SaveVideo']
        if save_video:
            v_path = path.join(benchmarker.dirname, 'output.mp4')
            v_out = cv2.VideoWriter(v_path, cv2.VideoWriter_fourcc(*'MP4V'), 20.0, (IMAGE_WIDTH, IMAGE_HEIGHT))
    else:
        cam = ROSCam(cfg['Topics'])

    if benchmark: zero_time = datetime.now(); step_time = zero_time

    # Person detection network (SSD or YOLO)
    input_shape = (nets['DetectionHeight'], nets['DetectionWidth'], 3)
    pers_det = DetectionNetwork(nets['Arch'], input_shape, nets['DetectionModel'])

    # Face detection network. The frozen graphs can't be overridden as they are included in the
    # faced package. Use symlinks in order to exchange them for anothers.
    if benchmark: t_pers_det = datetime.now() - step_time; step_time = datetime.now()
    face_det = FaceDetector()

    # FaceNet embedding encoder.
    if benchmark: t_face_det = datetime.now() - step_time; step_time = datetime.now()
    face_enc = FaceNet(nets['FaceEncoderModel'])

    # Now we extract the reference face
    face_img = imread(cfg['RefFace'])
    fbox = face_det.predict(face_img)
    ref_face = utils.crop_face(face_img, fbox)
    # and plug it into the encoder
    face_enc.set_reference_face(ref_face)
    if benchmark: t_face_enc = datetime.now() - step_time; step_time = datetime.now()

    # Motors instance
    # motors = Motors(cfg['Topics']['Velocity'])
    display_imgs = cfg['DisplayImages']

    iteration = 0
    elapsed_times = []

    def shtdn_hook():
        pers_det.sess.close()
        face_det.sess.close()
        face_enc.sess.close()
        if benchmark and save_video:
            v_out.release()
        rospy.loginfo("Cleaning and exiting...")

    # Register shutdown hook
    rospy.on_shutdown(shtdn_hook)


    if benchmark: ttfi = datetime.now() - zero_time; total_times = []


    while not rospy.is_shutdown():
        if benchmark: times = []
        # Fetch the images
        try:
            image, depth = cam.getImages()
        except StopIteration:
            rospy.signal_shutdown("rosbag completed!!")
            break
        if benchmark: step_time = datetime.now(); iter_start = step_time

        # Make an inference on the current image
        persons, elapsed = pers_det.predict(image)

        if benchmark: times.append([elapsed, len(persons)])
        # Detect and crop
        if benchmark: step_time = datetime.now()
        face_detections = face_det.predict(image)
        if benchmark: elapsed = datetime.now() - step_time; times.append([elapsed, len(face_detections) if isinstance(face_detections, list) else 1])

        # Filter just confident faces
        # TODO: keep only faces inside persons
        faces_flt = list(filter(lambda f: f[-1] > 0.9, face_detections))
        # TODO: adapt crop_face in order to use the common bb format
        faces_cropped = [utils.crop_face(image, fdet) for fdet in faces_flt]
        # # elapsed_3 = motors.move(image, depth, persons, scores, faces)


        if benchmark: step_time = datetime.now()
        similarities = face_enc.distancesToRef(faces_cropped)
        # for idx, sim in enumerate(similarities):
        #     print(idx, '\t', sim)
        if benchmark: elapsed = datetime.now() - step_time; times.append([elapsed, len(similarities) if isinstance(similarities, list) else 1])


    # motors.setNetworks(network, siamese_network)
    # motors.setCamera(cam)
    # if device_type.lower() == 'kobuki':
    #     motors.setDepth(depth)

        # for face in faces:
        #     for person in boxes:
        #         # We check the face center belongs to a person
        #         horz = person[0] <= face[0] <= person[2]
        #         vert = person[1] <= face[1] <= person[3]
        #         print horz, vert


        if display_imgs:
            display_start = datetime.now()
            img_cp = np.copy(image)

            for person in persons:
                x1, y1, x2, y2 = person[0], person[1], person[0]+person[2], person[1]+person[3]
                vis_utils.draw_bounding_box_on_image_array(img_cp, y1, x1, y2, x2, use_normalized_coordinates=False)

            for face in faces_flt:
                x1, y1, x2, y2 = face[0]-face[2]//2, face[1]-face[3]//2, face[0]+face[2]//2, face[1]+face[3]//2
                vis_utils.draw_bounding_box_on_image_array(img_cp, y1, x1, y2, x2, color='green', use_normalized_coordinates=False)

            print(f'Persons: {len(persons)}\tFaces: {len(faces_flt)}')
            transformed = cv2.cvtColor(img_cp, cv2.COLOR_RGB2BGR)
            cv2.imshow('Image', transformed)
            # Write to the video output
            if benchmark and save_video: v_out.write(transformed)
            cv2.waitKey(30)
            display_elapsed = [datetime.now() - display_start]
        else:
            display_elapsed = []


        if benchmark: iter_elapsed = [datetime.now() - iter_start]; total_times.append(iter_elapsed + times + display_elapsed)


        if benchmark:
            iteration += 1
            n_image = f'Image {iteration}/{n_images}'
            print(n_image)
            print('*' * len(n_image))

        # Stop conditions
        if MAX_ITERS is not None and iteration == MAX_ITERS: break
        if iteration == n_images: break

    # Finish the loop
    if benchmark:
        benchmarker.write_benchmark(total_times,
                                    cfg['RosbagFile'],
                                    cfg['Networks']['DetectionModel'],
                                    cfg['Networks']['FaceEncoderModel'],
                                    t_pers_det, t_face_det, t_face_enc, ttfi,
                                    display_imgs, write_iters=True)

    if display_imgs: cv2.destroyAllWindows()

    rospy.signal_shutdown("Finished!!")
