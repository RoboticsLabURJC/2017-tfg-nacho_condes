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
from detector import FaceDetector # based on a symlink to faced library path
from imageio import imread
from Motors.motors import Motors
from Net.detection_network import DetectionNetwork
from Net.facenet import FaceNet
import utils


if __name__ == '__main__':
    # Parameter parsing
    parser = argparse.ArgumentParser(description='Run the main following script')
    parser.add_argument('config_file', type=str, help='Path for the YML configuration file')
    args = parser.parse_args()
    with open(args.config_file, 'r') as f:
        cfg = yaml.safe_load(f)

    rospy.init_node(cfg['NodeName'])

    # Requested behavioral
    benchmark = cfg['Mode'].lower() == 'benchmark'

    # Instantiations
    if benchmark:
        cam = ROSCam(cfg['Topics'], cfg['RosbagFile'])
    else:
        cam = ROSCam(cfg['Topics'])

    from cprint import cprint

    # Person detection network (SSD or (TODO) YOLO)
    cprint.info('Loading object detector...')
    obj_det = DetectionNetwork(cfg['Networks']['DetectionModel'])
    cprint.ok('Object detector loaded')

    # Face detection network. The frozen graphs can't be overridden as they are included in the
    # faced package. Use symlinks in order to exchange them for anothers.
    face_det = FaceDetector()

    # FaceNet embedding encoder.
    cprint.info('Loading face encoder...')
    face_enc = FaceNet(cfg['Networks']['FaceEncoderModel'])
    cprint.ok('Face encoder loaded')


    # Now we extract the reference face
    face_img = imread(cfg['RefFace'])
    fbox = face_det.predict(face_img)
    ref_face = utils.crop_face(face_img, fbox)
    # and plug it into the encoder
    face_enc.set_reference_face(ref_face)

    # Motors instance
    # motors = Motors(cfg['Topics']['Velocity'])
    display_imgs = True
    MAX_ITER = 1000

    iteration = 0
    elapsed_times = []

    def shtdn_hook():
        obj_det.sess.close()
        face_det.sess.close()
        face_enc.sess.close()
        print("Cleaning and exiting...")
    # Register shutdown hook
    rospy.on_shutdown(shtdn_hook)

    if benchmark:
        total_times = []

    while not rospy.is_shutdown():
        if benchmark:
            times = []
        # Fetch the images
        image, depth = cam.getImages()

        if benchmark:
            start_time = datetime.now()
            iter_time = datetime.now()
        # Make an inference on the current image
        persons, scores, _ = obj_det.predict(image)
        if benchmark:
            times.append(datetime.now() - start_time)

        # Detect and crop
        if benchmark:
            start_time = datetime.now()
        face_detections = face_det.predict(image)
        if benchmark:
            times.append(datetime.now() - start_time)

        # Filter just confident faces
        # # TODO: keep only faces inside persons
        faces_flt = filter(lambda f: f[-1] > 0.9, face_detections)
        faces_cropped = [utils.crop_face(image, fdet) for fdet in faces_flt]

        # # elapsed_3 = motors.move(image, depth, persons, scores, faces)


        if benchmark:
            start_time = datetime.now()
        sims = map(face_enc.distanceToRef, faces_cropped)
        for idx, sim in enumerate(sims):
            print(idx, '\t', sim)
        if benchmark:
            times.append(datetime.now() - start_time)
            times.append(datetime.now() - iter_time)


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

        if benchmark:
            total_times.append(times)

        if display_imgs:
            img_cp = np.copy(image)
            transformed = cv2.cvtColor(img_cp, cv2.COLOR_RGB2BGR)
            if len(faces_cropped) > 0:
                f = cv2.cvtColor(faces_cropped[0], cv2.COLOR_RGB2BGR)
                cv2.imshow('Face', f)
            cv2.imshow('Image', transformed)
            cv2.waitKey(30)

        iteration += 1
        print(iteration)
        print('*' * 20)

        if iteration == MAX_ITER:
            rospy.signal_shutdown("Finished!!")

    if benchmark:
        # Vectorized conversion to ms
        times_ms = utils.TO_MS(total_times)
        t_mean = times_ms.mean(axis=0)
        t_std = times_ms.std(axis=0)

        np.set_printoptions(precision=3)
        print('Mean:', times_ms.mean(axis=0), ' (ms)')
        print('Dev.:', times_ms.std(axis=0),  ' (ms)')
        print('Max.:', times_ms.max(axis=0),  ' (ms)')
        print('Min.:', times_ms.min(axis=0),  ' (ms)')

    if display_imgs:
        cv2.destroyAllWindows()
