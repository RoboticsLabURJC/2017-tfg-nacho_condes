import numpy as np
import tensorflow as tf
import cv2
from tensorflow.python.platform import gfile
from imageio import imread

# frontal_face_cascade = cv2.CascadeClassifier('resources/haarcascade_frontalface_default.xml')

class FaceTrackingNetwork:
    '''
    Class to abstract a siamese network, useful to compare faces between
    them, and compute their L2 (euclidean) distance.
    '''
    def __init__(self, model_name, mom_path, face_detector):
        # Load the siamese network model
        model_path = 'Net/Models/' + model_name
        with tf.device('/cpu:0'):
            siamese_graph = tf.Graph()
            with siamese_graph.as_default():
                graph_def = tf.GraphDef()
                with gfile.FastGFile(model_path, 'rb') as fid:
                    graph_def.ParseFromString(fid.read())
                    tf.import_graph_def(graph_def, input_map=None, name='')

            # Instance of the session, placeholders and embeddings (output)
            #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.05)
            self.sess = tf.Session(graph=siamese_graph,
                                   config=tf.ConfigProto(#gpu_options=gpu_options,
                                                         log_device_placement=False))

            self.inputs_tensor = siamese_graph.get_tensor_by_name('input:0')
            self.phase_train_placeholder = siamese_graph.get_tensor_by_name('phase_train:0')
            self.embeddings = siamese_graph.get_tensor_by_name('embeddings:0')

        mom_img = imread(mom_path)
        self.face_detector = face_detector
        self.mom_face, _ = self.getFace(mom_img)

        # Dummy initialization...


        dummy_tensor = np.zeros((160, 160, 3))
        _ = self.distanceToMom(dummy_tensor)

        print("MobileNet ready!")

    def prewhiten(self, face, square_size=160):
        ''' Function to preprocess a certain face, to be fed to the
        siamese network. '''
        # Squared crop
        if 0 not in face.shape:
            square_face = cv2.resize(face, (square_size, square_size), interpolation=cv2.INTER_CUBIC)
            blurred_face = cv2.blur(square_face, (5,5))
            # Now we normalize (whiten) the image
            [mean, std] = [np.mean(blurred_face), np.std(square_face)]
            whitened_face = np.multiply(np.subtract(blurred_face, mean), 1.0/std)

            return whitened_face

    def getFace(self, person_img, margin=2):
        '''
        This method looks for a face in a given image, and returns it with a certain
        preprocessing.
        '''

        def highestIdx(face_boxes):
            ''' Returns the index which belongs to the highest bounding box.'''
            high_idx = None
            high_y = np.infty

            for idx in range(len(face_boxes)):
                face = face_boxes[idx]
                y = face[1]
                if y < high_y:
                    high_y = y
                    high_idx = idx
            return high_idx


        image_size = np.asarray(person_img.shape)[0:2]
        # Gray conversion for the face detection.
        gray_img = cv2.cvtColor(person_img, cv2.COLOR_BGR2GRAY)

        # We look for frontal faces
        #frontal_face_boxes = frontal_face_cascade.detectMultiScale(gray_img,
        #                                           scaleFactor=1.1,
        #                                           minNeighbors=2)
        #                                           

        # face_boxes = frontal_face_boxes
        face_boxes = self.face_detector.predict(person_img)
        n_faces = np.asarray(face_boxes).shape[0]

        if n_faces == 0:
            # No face detected
            return [], []

        elif n_faces > 1:
            # More than 1 detected face. We will keep
            # the highest one.
            idx = highestIdx(face_boxes)

            n_faces = 1
            # We imput the "1" value to enter on the next condition
            face_boxes = face_boxes[idx]

        if n_faces == 1:
            # Now we transform the detection
            x, y, w, h, _ = np.squeeze(face_boxes)
            det = [x, y, x+w, y+h]

            box = np.zeros(4, dtype=np.int32)
            # Check that the box+margin does not go outside of the image
            box[0] = np.maximum(det[0] - margin/2, 0)
            box[1] = np.maximum(det[1] - margin/2, 0)
            box[2] = np.minimum(det[2] + margin/2, image_size[1])
            box[3] = np.minimum(det[3] + margin/2, image_size[0])
            face = person_img[box[1]:box[3], box[0]:box[2], :]
            whitened_face = self.prewhiten(face)

            return whitened_face, box


    def compareFaces(self, face1, face2):
        '''
        Get the L2 distance between two faces (already processed).
        '''
        # We feed the siamese network with both faces
        img_list = [face1, face2]
        images = np.stack(img_list)
        feed_dict = {self.inputs_tensor: images, self.phase_train_placeholder: False}
        with tf.device('/cpu:0'):
            emb = self.sess.run(self.embeddings, feed_dict=feed_dict)
        # Compute the distance between the output features
        dist = np.sqrt(np.sum(np.square(np.subtract(emb[0,:], emb[1,:]))))

        return dist

    def distanceToMom(self, face_query, preprocess=True):
        '''
        Checks if a given face corresponds to mom's.
        '''
        if preprocess:
            target = self.prewhiten(face_query)
        else:
            target = face_query
        if target is None:
            return np.infty
        
        dist = self.compareFaces(target, self.mom_face)
        return dist
