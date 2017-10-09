#
# Created on Mar 12, 2017
#
# @author: dpascualhe
#
# It trains and tests a convolutional neural network with an augmented
# MNIST dataset.
#

import os
import sys

import numpy as np
from sklearn import metrics
from keras.utils import vis_utils
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential, load_model

from DataManager.netdata import NetData
from CustomEvaluation.customevaluation import CustomEvaluation
from CustomEvaluation.customcallback import LearningCurves

# Seed for the computer pseudorandom number generator.
np.random.seed(123)

if __name__ == "__main__":  
    nb_epoch = 100
    batch_size = 128
    nb_classes = 10
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        
    im_rows, im_cols = 28, 28
    nb_filters = 32
    kernel_size = (3, 3)
    pool_size = (2, 2)

    verbose = 0
    training = 0
    while training != "y" and training != "n":
        training = raw_input("Do you want to train the model?(y/n)")
    while verbose != "y" and verbose != "n":
        verbose = raw_input("Verbose?(y/n)")

    data = NetData(im_rows, im_cols, nb_classes, verbose)
    
    if training == "y":
        dropout = 0
        while dropout != "y" and dropout != "n":
            dropout = raw_input("Dropout?(y/n)")
        train_ds = raw_input("Train dataset path: ")
        while not os.path.isfile(train_ds):
            train_ds = raw_input("Enter a valid path: ")
        val_ds = raw_input("Validation dataset path: ")
        while not os.path.isfile(val_ds):
            val_ds = raw_input("Enter a valid path: ")
            
    test_ds = raw_input("Test dataset path: ")
    while not os.path.isfile(test_ds):
        test_ds = raw_input("Enter a valid path: ")
        
    if training == "y":    
        # We load and reshape data in a way that it can work as input of
        # our model.
        (X_train, Y_train) = data.load(train_ds)
        (x_train, y_train), input_shape = data.adapt(X_train, Y_train)
        
        (X_val, Y_val) = data.load(val_ds)
        (x_val, y_val), input_shape = data.adapt(X_val, Y_val)
        
        # We add layers to our model.
        model = Sequential()
        model.add(Conv2D(16, kernel_size, activation="relu",
                         input_shape=input_shape))
        model.add(Conv2D(nb_filters, kernel_size, activation="relu"))
        model.add(MaxPooling2D(pool_size=pool_size))
        if dropout == "y":
            model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        if dropout == "y":
            model.add(Dropout(0.5))
        model.add(Dense(nb_classes, activation="softmax"))
        
        # We compile our model.
        model.compile(loss="categorical_crossentropy", optimizer="adadelta",
                      metrics=["accuracy"])
            
        # We train the model and save data to plot a learning curve.
        learning_curves = LearningCurves()
        early_stopping = EarlyStopping(monitor='val_loss', patience=2)
        checkpoint = ModelCheckpoint("net.h5", verbose=1, monitor='val_loss',
                                     save_best_only=True)
        
        validation = model.fit(x_train, y_train,
                               batch_size=batch_size, 
                               epochs=nb_epoch,
                               validation_data=(x_val, y_val),
                               callbacks=[learning_curves, early_stopping,
                                          checkpoint])
        vis_utils.plot_model(model, "net.png", show_shapes=True)

    # We load and reshape test data and model.
    (X_test, Y_test) = data.load(test_ds)
    (x_test, y_test), input_shape = data.adapt(X_test, Y_test)

    model = load_model("net.h5")
    
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test score:", score[0])
    print("Test accuracy:", score[1])
    
    # We log the results.
    y_proba = model.predict_proba(x_test, batch_size=batch_size, verbose=1)
    y_test = np.argmax(y_test, axis=1)
    
    if training == "n":
        results = CustomEvaluation(y_test, y_proba, training)
    else:
        train_loss = learning_curves.loss
        train_acc = learning_curves.accuracy
        val_loss = validation.history["val_loss"]
        val_acc = validation.history["val_acc"]
        results = CustomEvaluation(y_test, y_proba, training, train_loss,
                                   train_acc, val_loss, val_acc)
    
    results_dict = results.dictionary()
    results.log(results_dict)
    
