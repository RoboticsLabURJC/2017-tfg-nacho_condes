#! /usr/bin/python3

#
# Created on Dec, 2019
#
#
__author__ = '@naxvm'

import argparse
from os import path
from time import sleep

import cv2
import numpy as np
import rospy
import yaml
from geometry_msgs.msg import Twist
from kobuki_msgs.msg import Sound

import utils
from Actuation.pid_controller import PIDController
from benchmarkers import TO_MS, FollowPersonBenchmarker
from cprint import cprint
from Perception.Camera.labeled_cam import IMAGE_HEIGHT, IMAGE_WIDTH, LabeledCam
from Perception.Net.networks_controller import NetworksController
from Perception.Net.utils import visualization_utils as vis_utils
from Perception.people_tracker import PeopleTracker

if __name__ == '__main__':
    # Parameter parsing
    parser = argparse.ArgumentParser(description='Run the main followperson script with the provided configuration')
    parser.add_argument('config_file', type=str, help='Path for the YML configuration file')
    args = parser.parse_args()
    with open(args.config_file, 'r') as f:
        cfg = yaml.safe_load(f)

    exec_cfg = cfg['Execution']
    nets_cfg = cfg['Networks']
    ptrk_cfg = cfg['PeopleTracker']
    rgbd_cfg = cfg['Camera']
    xctr_cfg = cfg['XController']
    wctr_cfg = cfg['WController']

    # Instantiations
    if exec_cfg['Benchmark']:
        cam = LabeledCam(rgbd_cfg['SequenceDir'], rgbd_cfg['Topics'], rate=rgbd_cfg['Rate'])
        n_images = len(cam)
        benchmarker = FollowPersonBenchmarker(exec_cfg['LogDir'])
        # Save the video output
        v_path = path.join(benchmarker.dirname, 'output.mp4')
        v_out = cv2.VideoWriter(v_path, cv2.VideoWriter_fourcc(*'mp4v'), float(rgbd_cfg['Rate']), (2 * IMAGE_WIDTH, 2 * IMAGE_HEIGHT))

    else:
        cam = LabeledCam(cfg['Topics']) # TODO: Change back to ROSCam!
    rospy.init_node(exec_cfg['NodeName'])
    # Create the networks controller (thread running on the GPU)
    # It configures itself after starting
    nets_c = NetworksController(benchmark=exec_cfg['Benchmark'], debug=exec_cfg['Debug'], **nets_cfg)

    # Person tracker (thread running on the CPU)
    # p_tracker = PeopleTracker(ptrk_cfg, debug=exec_cfg['Debug'])
    # p_tracker.setCam(cam)
    # sleep(2)

    # # Link the networks to the tracker, to update the references with the inferences
    # nets_c.setTracker(p_tracker)
    nets_c.start()

    # PID controllers
    # x_pid = PIDController(**xctr_cfg)
    # w_pid = PIDController(**wctr_cfg)


    # Twist messages publisher for moving the robot
    # if not exec_cfg['Benchmark']:
    #     tw_pub = rospy.Publisher(cfg['Topics']['Motors'], Twist, queue_size=1)
    #     sn_pub = rospy.Publisher(cfg['Topics']['Sound'], Sound, queue_size=1)

    # Everything ready. Wait for the thread to be set.
    # while not (p_tracker.is_activated and nets_c.is_activated):
    while not nets_c.is_activated:
        sleep(1)
    sleep(2)

    if exec_cfg['Benchmark']:
        # Save the configuration on the benchmarker
        benchmarker.make_config(nets_cfg['DetectionModel'], nets_cfg['FaceEncoderModel'], rgbd_cfg['SequenceDir'], xctr_cfg, wctr_cfg,
                               ptrk_cfg)
        benchmarker.make_load_times(nets_c.t_pers_det, nets_c.t_face_det, nets_c.t_face_enc, nets_c.ttfi)

    # Data structures to save the results
    frame_counter = 0
    frames_with_ref = 0
    sent_responses = {}
    num_trackings = {}
    ref_errors = {}
    ref_coords = {}

    ref_tracked = False
    fps_str = 'N/A'
    show_images = True


    def shtdn_hook():
        rospy.loginfo("\nCleaning and exiting...")
        # p_tracker.is_activated = False
        nets_c.close_all()
        global show_images
        show_images = False
        cv2.destroyAllWindows()
        if exec_cfg['Benchmark']:
            v_out.release()


    # Register shutdown hook
    rospy.on_shutdown(shtdn_hook)
    # if exec_cfg['Debug']:
    #     counter = 30000

    # Start the video sequence
    if exec_cfg['Benchmark']:
        cam.init_cam()

    while not rospy.is_shutdown():
        print('=== Main loop iteration ===')

        if not nets_c.is_activated:
            rospy.signal_shutdown('ROSBag completed!')

        # Get the data
        print('Getting data...')
        try:
            image, depth, labels = cam.get_data()
        except StopIteration:
            print('Sequence finished!')
            break

        # result = nets_c.predict(image)

        cv2.imshow('image', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)



        ########################################################
        # frame_counter = p_tracker.frame_counter

        ################
        ### TRACKING ###
        ################
        # Forward step in the tracking using the detections
        # print('=====')
        # print('-main-')
        # print(f'Detections: {len(nets_c.persons)}|{len(nets_c.faces)}')
        # print(f'Tracker: {len(p_tracker.persons)}')
        # persons = p_tracker.persons

        ################
        #### MOVING ####
        ################
        # Compute errors
    #     ref_found = False
    #     person = None
    #     for person in persons:
    #         if person.is_ref:
    #             w_error = utils.computeWError(person.coords, IMAGE_WIDTH)
    #             # The depth might not be sampleable
    #             new_x_error = utils.computeXError(person.coords, depth)
    #             if new_x_error is not None:
    #                 x_error = new_x_error
    #             ref_found = True
    #             frames_with_ref += 1
    #             break
    #     # Compute a suitable response with the PID controllers
    #     if ref_found:
    #         if not ref_tracked and not exec_cfg['Benchmark']:
    #             # Sound if just found
    #             sn_pub.publish(Sound.CLEANINGEND)
    #         ref_tracked = True
    #         w_response = w_pid.computeResponse(w_error)
    #         x_response = x_pid.computeResponse(x_error)
    #         # cprint.info(f'w: {w_error:.3f} => {w_response:.3f}')
    #         # cprint.info(f'x: {x_error:.3f} => {x_response:.3f}')
    #         # Send the response to the robot
    #         if not exec_cfg['Benchmark']:
    #             msg = Twist()
    #             msg.linear.x = x_response
    #             msg.angular.z = w_response
    #             tw_pub.publish(msg)
    #     else:
    #         if ref_tracked and not exec_cfg['Benchmark']:
    #             # Sound if just lost
    #             sn_pub.publish(Sound.CLEANINGSTART)

    #         ref_tracked = False
    #         w_error = None
    #         x_error = None
    #         w_response = w_pid.lostResponse()
    #         x_response = x_pid.lostResponse()

    #     if exec_cfg['Benchmark']:
    #         num_trackings[frame_counter] = len(persons)
    #         ref_errors[frame_counter] = (w_error, x_error)
    #         sent_responses[frame_counter] = (w_response, x_response)
    #         if person is not None:
    #             ref_coords[frame_counter] = person.coords

    #     ###############
    #     ### DRAWING ###
    #     ###############
    #     # Draw the images.
    #     depth = 255.0 * (1 - depth / 6.0)  # 8-bit quantization of the effective Xtion range
    #     transformed = np.copy(image)

    #     for person in persons:
    #         x1, y1, x2, y2 = utils.corner2Corners(person.coords)
    #         color = utils.BOX_COLOR[person.is_ref]
    #         vis_utils.draw_bounding_box_on_image_array(transformed, y1, x1, y2, x2, color=color,
    #                                                    use_normalized_coordinates=False)
    #         face = person.face
    #         if face is not None:
    #             x1, y1, x2, y2 = utils.center2Corners(face.coords)
    #             vis_utils.draw_bounding_box_on_image_array(transformed, y1, x1, y2, x2, color='blue',
    #                                                        use_normalized_coordinates=False)
    #     for kp in p_tracker.keypoints.astype(int):
    #         try:
    #             x, y = kp
    #         except TypeError:
    #             pass
    #         cv2.circle(transformed, (int(x), int(y)), 5, (255, 255, 0))

    #     # Show the images
    #     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #     transformed = cv2.cvtColor(transformed, cv2.COLOR_RGB2BGR)
    #     depth = cv2.applyColorMap(depth.astype(np.uint8), cv2.COLORMAP_JET)
    #     inputs = np.vstack((image, depth))

    #     moves = utils.movesImage(image.shape, XLIM, x_response, WLIM, w_response)
    #     outputs = np.vstack((moves, transformed))

    #     total_out = np.hstack((inputs, outputs))
    #     # print('frame_counter:', frame_counter)
    #     cv2.putText(total_out, f'Frame #{frame_counter}', (2 * IMAGE_WIDTH - 280, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
    #                 (255, 255, 255), 2)
    #     if frame_counter % 10 == 0:
    #         # Update the fps
    #         last_elapsed = nets_c.last_elapsed
    #         if last_elapsed != 0:
    #             fps_str = f'{1000 / TO_MS(last_elapsed):.2f} fps'

    #     cv2.putText(total_out, f'Neural rate: {fps_str}', (2 * IMAGE_WIDTH - 280, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
    #                 (255, 255, 255), 1)
    #     if show_images:
    #         cv2.imshow('Output', cv2.resize(total_out, dsize=(IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_CUBIC))
    #         cv2.waitKey(1)
    #     if exec_cfg['Benchmark']: v_out.write(total_out)
    #     # elapsed_ = time.time() - start
    # # Finish the execution
    # if exec_cfg['Benchmark']:
    #     benchmarker.makeDetectionStats(nets_c.total_times)
    #     benchmarker.makeTrackingStats(p_tracker.tracked_counter, frames_with_ref)
    #     benchmarker.makeIters(frame_counter, nets_c.total_times, num_trackings, ref_errors, ref_coords, sent_responses)
    #     benchmarker.writeBenchmark()
    rospy.signal_shutdown("Finished!!")
