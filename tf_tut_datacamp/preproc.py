import numpy as np
from skimage import transform # we will change the size
from skimage.color import rgb2gray
import data_gather as dg # service for gathering data
import matplotlib.pyplot as plt

def preproc():
    images, labels = dg.data_gather()

    images28 = [transform.resize(image, (28,28)) for image in images] # scales the images and normalize them!

    images28 = np.array(images28) # rgb2gray expects an array, so we convert the images to it
    images28 = rgb2gray(images28) # conversion to grayscale

    return images28, labels
