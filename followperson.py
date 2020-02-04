#
# Created on Dec, 2019
#
# @author: naxvm
#

import argparse
import sys
from datetime import datetime, timedelta

import numpy as np

import cv2
import rospy
import yaml
from Camera.ROSCam import ROSCam
from faced.detector import FaceDetector
from imageio import imread
from Motors.motors import Motors
from Net.detection_network import DetectionNetwork
from Net.facenet import FaceNet
import utils
from logs.benchmarkers import FollowPersonBenchmarker
from cprint import cprint

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
    else:
        cam = ROSCam(cfg['Topics'])

    if benchmark:
        # Tick the zero time in order to measure instantiation times.
        zero_time = datetime.now()
        step_time = zero_time

    # Person detection network (SSD or (TODO) YOLO)
    input_shape = (nets['DetectionHeight'], nets['DetectionWidth'], 3)
    pers_det = DetectionNetwork(nets['Arch'], input_shape, nets['DetectionModel'])
    if benchmark:
        t_pers_det = datetime.now() - step_time
        step_time = datetime.now()
    # Face detection network. The frozen graphs can't be overridden as they are included in the
    # faced package. Use symlinks in order to exchange them for anothers.
    face_det = FaceDetector()
    if benchmark:
        t_face_det = datetime.now() - step_time
        step_time = datetime.now()

    # FaceNet embedding encoder.
    face_enc = FaceNet(nets['FaceEncoderModel'])

    # Now we extract the reference face
    face_img = imread(cfg['RefFace'])
    fbox = face_det.predict(face_img)
    ref_face = utils.crop_face(face_img, fbox)
    # and plug it into the encoder

    face_enc.set_reference_face(ref_face)
    if benchmark:
        t_face_enc = datetime.now() - step_time
        step_time = datetime.now()

    # Motors instance
    # motors = Motors(cfg['Topics']['Velocity'])
    display_imgs = cfg['DisplayImages']

    iteration = 0
    elapsed_times = []

    def shtdn_hook():
        pers_det.sess.close()
        face_det.sess.close()
        face_enc.sess.close()
        rospy.loginfo("Cleaning and exiting...")

    # Register shutdown hook
    rospy.on_shutdown(shtdn_hook)

    if benchmark:
        ttfi = datetime.now() - zero_time
        total_times = []

    while not rospy.is_shutdown():
        if benchmark:
            times = []
        # Fetch the images
        try:
            image, depth = cam.getImages()
        except StopIteration:
            rospy.signal_shutdown("rosbag completed!!")
            break
        if benchmark:
            # Measure the elapsed time for the entire iteration and for the
            step_time = datetime.now()
            iter_start = step_time
        # Make an inference on the current image
        persons, scores, elapsed = pers_det.predict(image)
        if benchmark:
            times.append([elapsed, len(persons)])

        # Detect and crop
        if benchmark:
            step_time = datetime.now()
        face_detections = face_det.predict(image)
        if benchmark:
            elapsed = datetime.now() - step_time
            times.append([elapsed, len(face_detections)])

        # Filter just confident faces
        # # TODO: keep only faces inside persons
        faces_flt = filter(lambda f: f[-1] > 0.9, face_detections)
        faces_cropped = [utils.crop_face(image, fdet) for fdet in faces_flt]

        # # elapsed_3 = motors.move(image, depth, persons, scores, faces)


        if benchmark:
            step_time = datetime.now()
        similarities = list(map(face_enc.distanceToRef, faces_cropped))
        # for idx, sim in enumerate(similarities):
        #     print(idx, '\t', sim)
        if benchmark:
            elapsed = datetime.now() - step_time
            times.append([elapsed, len(similarities)])


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
            transformed = cv2.cvtColor(img_cp, cv2.COLOR_RGB2BGR)
            if faces_cropped:
                f = cv2.cvtColor(faces_cropped[0], cv2.COLOR_RGB2BGR)
                cv2.imshow('Face', f)
            cv2.imshow('Image', transformed)
            cv2.waitKey(30)
            display_elapsed = [datetime.now() - display_start]
        else:
            display_elapsed = []


        if benchmark:
            iter_elapsed = [datetime.now() - iter_start]
            total_times.append(iter_elapsed + times + display_elapsed)


        if benchmark:
            iteration += 1
            n_image = f'Image {iteration}/{n_images}'
            print(n_image)
            print('*' * len(n_image))

        if iteration == n_images:
            rospy.signal_shutdown("Finished!!")


    # Finish the loop
    if benchmark:
        benchmarker = FollowPersonBenchmarker(cfg['LogDir'])
        benchmarker.write_benchmark(total_times,
                                    cfg['RosbagFile'],
                                    cfg['Networks']['DetectionModel'],
                                    cfg['Networks']['FaceEncoderModel'],
                                    t_pers_det, t_face_det, t_face_enc, ttfi,
                                    display_imgs, write_iters=True)

    if display_imgs:
        cv2.destroyAllWindows()
