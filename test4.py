from labelmesequence import LabelMeSequence
from faced import FaceDetector
from Perception.Net.facenet import FaceNet
import utils
import pickle
from imageio import imread
import numpy as np
import cv2

FACENET_PATH = 'Optimization/dl_models/facenet_inception_resnet_vggface2/frozen_graph.pb'
VIDEO_PATH = 'own_videos/2persons_faces'
REF_FACE_PATH = 'resources/ref_face/refface.jpg'
OUT_FILE_PATH = 'test4/distances_list.pkl'

if __name__ == '__main__':

    fdet = FaceDetector()
    fenc = FaceNet(FACENET_PATH)
    ref_img = imread(REF_FACE_PATH)
    ref_box = fdet.predict(ref_img)
    ref_face = utils.crop_face(ref_img, ref_box)
    fenc.setReferenceFace(ref_face)

    video = LabelMeSequence(VIDEO_PATH)
    distances_list = []
    for image, labels in video:
        distances = {}
        if labels is not None:
            for label, face in labels.items():
                face = utils.corner2Center(face)
                cropped = utils.crop_face(image, face)
                distances[label] = fenc.distancesToRef([cropped])[0]
            # print(distances)
            distances_list.append(distances)
    import pickle
    with open(OUT_FILE_PATH, 'wb') as f:
        pickle.dump(distances_list, f)