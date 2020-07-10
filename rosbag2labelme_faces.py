from Perception.Camera.ROSCam import ROSCam
from faced import FaceDetector
import os
import utils
import base64
import cv2
import numpy as np
import json

topics = {
    'RGB': '/camera/rgb/image_raw',
    'Depth': '/camera/depth_registered/image_raw'
}
ROSBAG_PATH = 'resources/bags/test.bag'
OUT_DIR = 'own_videos/test_face'
SSD_PATH = 'Optimization/dl_models/ssd_mobilenet_v1_0.75_depth_coco/frozen_inference_graph.pb'

cam = ROSCam(topics, ROSBAG_PATH)
net = FaceDetector()



frame_ix = -1
while True:
    frame_ix += 1
    print(f'\r{frame_ix}', end='', flush=True)
    im_path = os.path.join(OUT_DIR, f'{frame_ix}.jpg')
    try:
        image, _ = cam.getImages()
    except StopIteration:
        break
    faces = net.predict(image)
    if len(faces) != 0:
        probs = [face[4] for face in faces]
        best_face_idx = np.argmax(probs)
        bbox = faces[best_face_idx]
        bbox = utils.center2Corners(bbox).tolist()
        dictio = {'version': '4.4.0',
                'flags': {},
                'shapes': [
                    {
                    'label': '0',
                    'points': [
                        [
                        bbox[0],
                        bbox[1],
                    ],
                    [
                        bbox[2],
                        bbox[3],

                    ]
                    ],
                    "group_id": None,
                    "shape_type": "rectangle",
                    "flags": {}
                    }
                ],
                "imagePath": im_path,
                "imageData": str(base64.b64encode(cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))[1]))[1:-1]
                }
        json_name = f'{str(frame_ix).zfill(4)}.json'
        # print(json_name)
        # print(dictio)
        with open(os.path.join(OUT_DIR, json_name), 'w') as f:
            f.write(json.dumps(dictio, ensure_ascii=True))


    cv2.imwrite(os.path.join(OUT_DIR, f'{str(frame_ix).zfill(4)}.jpg'), cv2.cvtColor(image, cv2.COLOR_BGR2RGB))