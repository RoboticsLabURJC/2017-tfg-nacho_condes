#
# Created on Apr 23, 2017
#
# @author: dpascualhe
#
# It classifies a given image and shows the activation between layers
# and their weights.
#

import math
import os

import cv2
from keras import backend as K
from keras.models import load_model, Model
from keras.utils import vis_utils

import matplotlib.pyplot as plt
import numpy as np
from docutils.nodes import subtitle
    
def plot_data(data_name, layer, nb_data, data, id, min, max, title,
             interp="none", color="jet"):
    side = math.sqrt(nb_data)
        
    # Plot
    fig = plt.figure(data_name + " - " + layer.get_config()["name"])
    plot = plt.subplot(math.ceil(side), math.ceil(side), id)
    im = plt.imshow(data, interpolation=interp, vmin=min, vmax=max)
    frame = plt.gca()
        
    # Configuring plot
    if side > 15:
        title_filter, title_channel = title.split("; ")
        if (id == 1 or ((id-1) % math.ceil(side) == 0)):
            plt.ylabel(title_filter)
        if (id <= math.ceil(side)):
            plot.set_title(title_channel)
    else:
        plot.set_title(title, fontsize=8)
    plt.set_cmap(color)
    cbar_ax = fig.add_axes([0.92, 0.10, 0.03, 0.7])
    fig.colorbar(im, cbar_ax)
        
    for xlabel_i in frame.axes.get_xticklabels():
        xlabel_i.set_visible(False)
        xlabel_i.set_fontsize(0.0)
    for xlabel_i in frame.axes.get_yticklabels():
        xlabel_i.set_fontsize(0.0)
        xlabel_i.set_visible(False)
    for tick in frame.axes.get_xticklines():
        tick.set_visible(False)
    for tick in frame.axes.get_yticklines():
        tick.set_visible(False)

if __name__ == "__main__":  

    model = load_model("Nets/0-1_tuned/net_earlystopping.h5")
    im = cv2.imread("Datasets/Samples/0-1.png")
    
    # We adapt the image shape.
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = im.reshape(1, im.shape[0], im.shape[1], 1)
    
    interp="none"
    color="jet"
    
    pred = np.argmax(model.predict(im))
    print "\nDigit prediction: ", pred, "\n"
        
    for i, layer in enumerate(model.layers):
        if layer.get_config()["name"][:6] == "conv2d":
            # Getting weights
            shape = layer.get_weights()[0].shape
            weights = layer.get_weights()[0].reshape(shape[2], shape[0],
                                                     shape[1], shape[3])
            biases = layer.get_weights()[1]
            
            min_weights = np.amin(weights)
            max_weights = np.amax(weights)
            print "-------------", layer.get_config()["name"], \
                  "-------------"
            print "Filters:"
            print "    Width: ", weights.shape[1]
            print "    Height: ", weights.shape[2]
            print "    Depth: ", weights.shape[0]
            print "    Number of filters: ", weights.shape[3]
            print "    Min-max value: ", min_weights, "/", max_weights
            print "    Biases:"
            for i in range(np.size(biases)):
                print "        Filter ", i, ": ", biases[i]
            print "\n"
        
            # Getting activations
            truncated = Model(inputs=model.inputs,
                              outputs=layer.output)
            activations = truncated.predict(im)
            min_activations = np.amin(activations)
            max_activations = np.amax(activations)
            print "Activations:"
            print "    Width: ", activations.shape[1]
            print "    Height: ", activations.shape[2]
            print "    Depth: ", activations.shape[0]
            print "    Number of activation maps: ", activations.shape[3]
            print "    Min-max value: ", min_activations, "/", \
                  max_activations, "\n"
        
            nb_filters = weights.shape[3]
            filter_depth = weights.shape[0]
            nb_channels = nb_filters * filter_depth
            for filter_id in range(nb_filters):
                for depth_id in range(filter_depth):
                    id = depth_id + 1 + (filter_id*filter_depth)
                    # Weights
                    filter = weights[depth_id][:, :, filter_id]
                    plot_data("Weights", layer, nb_channels, filter, id,
                              min_weights, max_weights, "Filter" +
                              str(filter_id + 1) + "; Channel" +
                              str(depth_id + 1), interp, color)
            
                    #=======================================================
                    # # Weights gradient
                    # sobelx = cv2.Sobel(filter, cv2.CV_32F, 1, 0, ksize=5)
                    # sobely = cv2.Sobel(filter, cv2.CV_32F, 0, 1, ksize=5)
                    # sobel = abs(sobelx + sobely)
                    # min_sobel = np.amin(sobel)
                    # max_sobel = np.amax(sobel)
                    # self.plot_data("Weights gradient", layer, nb_channels,
                    #                sobel, id, min_sobel, max_sobel,
                    #                str(filter_id + 1) +
                    #                "; " + str(depth_id + 1),
                    #                interp, color)
                    #=======================================================
                        
                # Activation maps
                activation_map = activations[0][:, :, filter_id]
                activation_relu = activation_map
                activation_relu[activation_map<0] = 0
                plot_data("Activation maps", layer, nb_filters, activation_map,
                          filter_id + 1, min_activations, max_activations,
                          "Activation map " + str(filter_id + 1),
                          interp, color)
                plot_data("Activation maps+ReLU", layer, nb_filters,
                          activation_relu, filter_id + 1, min_activations,
                          max_activations, "Activation map " +
                          str(filter_id + 1), interp, color)

            plt.show()
