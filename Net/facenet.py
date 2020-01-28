import numpy as np
import tensorflow as tf
import cv2
from imageio import imread
from cprint import cprint

SQUARE_SIZE = 160

class FaceNet:
    '''
    Class to abstract an embedding network. Used to compare faces similarity.
    '''

    def __init__(self, model_path):
        # Load the embedding network model
        conf = tf.compat.v1.ConfigProto(log_device_placement=False)
        conf.gpu_options.allow_growth = True

        detection_graph = tf.compat.v1.Graph()
        with detection_graph.as_default():
            fn_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                fn_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(fn_graph_def, name='')

        self.sess = tf.Session(graph=detection_graph, config=conf)

        # Instance of the placeholders and embedding tensors

        self.input       = self.sess.graph.get_tensor_by_name('input:0')
        self.phase_train = self.sess.graph.get_tensor_by_name('phase_train:0')
        self.embeddings  = self.sess.graph.get_tensor_by_name('embeddings:0')

        cprint.info("FaceNet ready!")


    def set_reference_face(self, ref_crop):
        ''' Set the reference face (previously cropped by the detector). '''

        # Set the reference face
        self.ref_face = self.preprocess(ref_crop)

        # Dummy initialization...
        dummy_tensor = np.random.randn(SQUARE_SIZE, SQUARE_SIZE, 3)
        _ = self.distanceToRef(dummy_tensor, preprocess=False)

    def preprocess(self, face):
        ''' Function to preprocess a face. '''
        # Squared crop
        prep_face = cv2.resize(face, dsize=(SQUARE_SIZE, SQUARE_SIZE), interpolation=cv2.INTER_CUBIC)
        # prep_face = cv2.blur(prep_face, (5,5))
        # Normalize the distribution
        prep_face = (prep_face - prep_face.mean()) / prep_face.std()

        return prep_face


    def distanceToRef(self, face, preprocess=True):
        '''
        Compute the distance between the embeddings
        of a face and the reference ones.
        '''
        if face is None:
            return np.infty
        target = self.preprocess(face) if preprocess else face

        # Embeddings computation
        faces = np.stack([target, self.ref_face])
        feed_dict = {self.input:       faces,
                     self.phase_train: False}

        emb = self.sess.run(self.embeddings, feed_dict=feed_dict)
        # Distance norm
        return np.linalg.norm(emb[0, :] - emb[1, :])
