import numpy as np
import tensorflow as tf
#from scipy import misc
import cv2
import matplotlib.pyplot as plt
from tensorflow.python.platform import gfile

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

class SiameseNetwork:
    '''
    Class to abstract a siamese network, useful to compare faces between
    them, and compute their L2 (euclidean) distance.
    '''
    def __init__(self, model_name):
        # Load the siamese network model
        model_path = 'Net/TensorFlow/' + model_name + '.pb'
        with tf.device('/gpu:0'):
            siamese_graph = tf.Graph()
            with siamese_graph.as_default():
                graph_def = tf.GraphDef()
                with gfile.FastGFile(model_path, 'rb') as fid:
                #with tf.gfile.GFile(model_path, 'rb') as fid:
                    graph_def.ParseFromString(fid.read())
                    tf.import_graph_def(graph_def, input_map=None, name='')

            # Instance of the session, placeholders and embeddings (output)
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
            self.sess = tf.Session(graph=siamese_graph,
                                   config=tf.ConfigProto(gpu_options=gpu_options,
                                                         log_device_placement=True))

            self.inputs_tensor = siamese_graph.get_tensor_by_name('input:0')
            self.phase_train_placeholder = siamese_graph.get_tensor_by_name('phase_train:0')
            self.embeddings = siamese_graph.get_tensor_by_name('embeddings:0')

        print("Siamese network ready!")

    def getFace(self, person_img, margin=40, square_size=128):
        '''
        This method looks for a face in a given image, and returns it with a certain
        preprocessing.
        '''
        image_size = np.asarray(person_img.shape)[0:2]
        face_boxes = face_cascade.detectMultiScale(person_img, 1.3, 5)
        dets = []
        # Now we transform and store the detections
        [dets.append([x, y, x+w, y+h]) for (x, y, w, h) in face_boxes]

        for det in dets:
            box = np.zeros(4, dtype=np.int32)
            # Check that the box+margin does not go outside of the image
            box[0] = np.maximum(det[0] - margin/2, 0)
            box[1] = np.maximum(det[1] - margin/2, 0)
            box[2] = np.minimum(det[2] + margin/2, image_size[1])
            box[3] = np.minimum(det[3] + margin/2, image_size[0])

            face = person_img[box[1]:box[3], box[0]:box[2], :]
            # Squared crop
            square_face = cv2.resize(face, (square_size, square_size), interpolation=cv2.INTER_LINEAR)
            # Now we normalize (whiten) the image
            [mean, std] = [np.mean(square_face), np.std(square_face)]
            whitened_face = np.multiply(np.subtract(square_face, mean), 1.0/std)

        return whitened_face


    def compareFaces(self, face1, face2):
        '''
        Get the L2 distance between two faces (already processed).
        '''
        # We feed the siamese network with both faces
        img_list = [face1, face2]
        images = np.stack(img_list)

        feed_dict = {self.inputs_tensor: images, self.phase_train_placeholder: False}
        with tf.device('/gpu:0'):
            emb = self.sess.run(self.embeddings, feed_dict=feed_dict)
        # Compute the distance between the output features
        dist = np.sqrt(np.sum(np.square(np.subtract(emb[0,:], emb[1,:]))))

        return dist
