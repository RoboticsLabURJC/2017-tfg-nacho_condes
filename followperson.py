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
from Actuation import tracking
from Actuation.pid_controller import PIDController
from benchmarkers import FollowPersonBenchmarker, TO_MS
from cprint import cprint  # this import is added from the GitHub source as the pip version is outdated
# https://github.com/EVasseure/cprint
from geometry_msgs.msg import Twist
from kobuki_msgs.msg import Sound
from Perception.Camera.ROSCam import IMAGE_HEIGHT, IMAGE_WIDTH, ROSCam
from Perception.Net.networks_controller import NetworksController
from Perception.Net.utils import visualization_utils as vis_utils
from time import sleep
XLIM = 0.7
WLIM = 1

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
    nets_cfg = cfg['Networks']

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


    # Create the networks controller
    # It configures itself after starting
    nets_c = NetworksController(nets_cfg, cfg['RefFace'], benchmark=True)
    nets_c.setCamera(cam)
    nets_c.daemon = True
    nets_c.start()


    # Person tracker
    ptcfg = cfg['PersonTracker']
    p_tracker = tracking.PersonTracker(ptcfg['Patience'], ptcfg['RefSimThr'], ptcfg['SamePersonThr'])


    # PID controllers
    xcfg = cfg['XController']
    x_pid = PIDController(xcfg['Kp'], xcfg['Ki'], xcfg['Kd'], K_loss=0.75,
                          limit=XLIM, stop_range=(xcfg['Min'], xcfg['Max']),
                          soften=True, verbose=False)

    wcfg = cfg['WController']
    w_pid = PIDController(wcfg['Kp'], wcfg['Ki'], wcfg['Kd'], K_loss=0.75,
                          limit=WLIM, stop_range=(wcfg['Min'], wcfg['Max']),
                          soften=True, verbose=True)


    # Twist messages publisher for moving the robot
    if not benchmark:
        tw_pub = rospy.Publisher(cfg['Topics']['Motors'], Twist, queue_size=1)
        sn_pub = rospy.Publisher(cfg['Topics']['Sound'], Sound, queue_size=1)


    # Everything ready. Wait for the controller to be set.
    while not nets_c.is_activated:
        sleep(1)

    if benchmark:
        # Save the configuration on the benchmarker
        benchmarker.makeConfig(nets_cfg['DetectionModel'], nets_cfg['FaceEncoderModel'], cfg['RosbagFile'], xcfg, wcfg, ptcfg)
        benchmarker.makeLoadTimes(nets_c.t_pers_det, nets_c.t_face_det, nets_c.t_face_enc, nets_c.ttfi)

    # Data structures to save the results
    iteration = 0
    frames_with_ref = 0
    sent_responses = {}
    num_trackings = {}
    ref_errors = {}

    ref_tracked = False
    fps_str = 'N/A'

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

        image = nets_c.image
        depth = nets_c.depth
        frame_counter = nets_c.frame_counter

        ################
        ### TRACKING ###
        ################
        # Forward step in the tracking using the detections
        cprint.info('=====')
        cprint.info(f'Detections: {len(nets_c.persons)}|{len(nets_c.faces)}')
        p_tracker.handleDetections(nets_c.persons)
        cprint.info(f'Tracked {len(p_tracker.persons)} persons')
        p_tracker.handleFaces(nets_c.faces, nets_c.similarities, image)

        # And process the similarities
        p_tracker.checkRef()

        # Persons ready to be fetched
        persons = p_tracker.persons


        ################
        #### MOVING ####
        ################
        # Compute errors
        ref_found = False
        for person in persons:
            if person.is_ref:
                w_error = utils.computeWError(person.coords, IMAGE_WIDTH)
                x_error = utils.computeXError(person.coords, depth)
                ref_found = True
                frames_with_ref += 1
                break
        # Compute a suitable response with the PID controllers
        if ref_found:
            if not ref_tracked and not benchmark:
                # Sound if just found
                sn_pub.publish(Sound.CLEANINGEND)
            ref_tracked = True
            w_response = w_pid.computeResponse(w_error)
            x_response = x_pid.computeResponse(x_error)
            cprint.info(f'w: {w_error:.3f} => {w_response:.3f}')
            cprint.info(f'x: {x_error:.3f} => {x_response:.3f}')
            # Send the response to the robot
            if not benchmark:
                msg = Twist()
                msg.linear.x  = x_response
                msg.angular.z = w_response
                tw_pub.publish(msg)
        else:
            if ref_tracked and not benchmark:
                # Sound if just lost
                sn_pub.publish(Sound.CLEANINGSTART)

            ref_tracked = False
            w_error = None
            x_error = None
            w_response = w_pid.lostResponse()
            x_response = x_pid.lostResponse()

        if benchmark:
            num_trackings[frame_counter] = len(persons)
            ref_errors[frame_counter] = (w_error, x_error)
            sent_responses[frame_counter] = (w_response, x_response)

        ###############
        ### DRAWING ###
        ###############
        # Draw the images.
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

        moves = utils.movesImage(image.shape, XLIM, x_response, WLIM, w_response)
        outputs = np.vstack((moves, transformed))

        total_out = np.hstack((inputs, outputs))
        cv2.putText(total_out, f'Frame #{frame_counter}', (2*IMAGE_WIDTH-280,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        if frame_counter % 10 == 0:
            # Update the fps
            last_elapsed = nets_c.last_elapsed
            fps_str = f'{1000/TO_MS(last_elapsed):.2f} fps'

        cv2.putText(total_out, f'Neural rate: {fps_str}', (2*IMAGE_WIDTH-280,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1)

        cv2.imshow('Output', cv2.resize(total_out, dsize=(IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_CUBIC))
        cv2.waitKey(1)
        if benchmark: v_out.write(total_out)


    # Finish the execution
    if benchmark:
        benchmarker.makeDetectionStats(nets_c.total_times)
        benchmarker.makeTrackingStats(p_tracker.tracked_counter, frames_with_ref)
        benchmarker.makeIters(nets_c.total_times, num_trackings, ref_errors, sent_responses)
        benchmarker.writeBenchmark()

    rospy.signal_shutdown("Finished!!")
