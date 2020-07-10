from Perception.Camera.ROSCam import ROSCam
from Perception.Net.detection_network import DetectionNetwork
import os
import utils
import base64
import cv2
import json

topics = {
    'RGB': '/camera/rgb/image_raw',
    'Depth': '/camera/depth_registered/image_raw'
}
ROSBAG_PATH = 'resources/bags/test4.bag'
OUT_DIR = 'own_videos/test4'
SSD_PATH = 'Optimization/dl_models/ssd_mobilenet_v1_0.75_depth_coco/frozen_inference_graph.pb'

cam = ROSCam(topics, ROSBAG_PATH)
net = DetectionNetwork('ssd', (300, 300, 3), frozen_graph=SSD_PATH)



frame_ix = -1
while True:
    print(f'\r{frame_ix}', end='', flush=True)
    frame_ix += 1
    im_path = os.path.join(OUT_DIR, f'{frame_ix}.jpg')
    try:
        image, _ = cam.getImages()
    except StopIteration:
        break
    persons, _ = net.predict(image)

    if len(persons) != 0:
        bbox = persons[0]
        bbox = utils.corner2Corners(bbox)
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
        with open(os.path.join(OUT_DIR, json_name), 'w') as f:
            f.write(json.dumps(dictio, ensure_ascii=True))


    cv2.imwrite(os.path.join(OUT_DIR, f'{str(frame_ix).zfill(4)}.jpg'), cv2.cvtColor(image, cv2.COLOR_BGR2RGB))