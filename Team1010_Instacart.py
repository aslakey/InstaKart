# -*- coding: utf-8 -*-
'''
authors: Aslakey FPiasevoli
Instacart Neural net for Team1010

A multilabel Feed Forward Neural Network with a sigmoid cross entropy loss.
Deneral Structure:

N_Items_Model                 Dept_Model
    |                             |
number_input     item_input   dept_input
    |                |            |
    |             FF Layer        |
     \                \          /
       \               concatenate
         \                  |
           \             FF Layer
             \             /
               concatenate
                    |
                 FF Layer
                    |
                 FF Layer
                    |
                  Output
Where item_input will have the same dimensions as output, a reduced featurespace 
consisting of the most commonly reordered items.
'''
import time
import numpy as np
from matplotlib import pyplot as plt
from keras.utils import np_utils
import keras.callbacks as cb
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras import backend as K
from sklearn.model_selection import train_test_split
import pandas as pd

_EPSILON = K.epsilon()
_SEED = 7

def load_data(filename, seed = 7):
    '''
    Expects data to be in a certain format (to be determined) Does a 20% split using seed
    as RandomState.

    Args:
        filename (str) - name of data file
        seed (int) - seed to use to split test and train

    Returns: 
        test_Y, test_Number, test_Item, test_Department, train_Y, [train_Number, train_Item, train_Department - 
            as numpy matrices
    Raises:
        AssertionError - if test_Y, train_Y, train_Item, test_Item do not all have the same shape
    '''
    data = pd.read_csv(filename, error_bad_lines=False,header=None)
    train, test = train_test_split(data, test_size = 0.2, random_state = seed)

    #test_Y, test_X = test[0].as_matrix(), test.ix[:, test.columns != 0].as_matrix()
    #train_Y, train_X = train[0].as_matrix(), train.ix[:, train.columns != 0].as_matrix()

    assert test_Y.shape == test_Item.shape
    assert test_Y.shape == train_Y.shape
    assert test_Y.shape == train_Item.shape
    
    return (test_Y, test_Number, test_Item, test_Department, train_Y, train_Number, train_Item, train_Department)

class LossHistory(cb.Callback):
    '''
    On start of training, initializes empty loss array,
    Returns loss array with batch losses
    '''
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        batch_loss = logs.get('loss')
        self.losses.append(batch_loss)

def _sigmoid_cross_entropy(y_true, y_pred):
    y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
    out = -(y_true * K.log(y_pred) + (1.0 - y_true) * K.log(1.0 - y_pred))
    return K.mean(out, axis=-1)

def init_model(numer_dim, item_dim, dept_dim):
    '''
    Creates model Graph and compiles it
    '''
    start_time = time.time()
    print ('Compiling Model ... ')
    number_branch = Sequential()
    number_branch.add(Dense(32, input_dim=number_dim))

    item_branch = Sequential()
    item_branch.add(Dense(32, input_dim=item_dim))

    dept_branch = Sequential()
    dept_branch.add(Dense(32, input_dim=dept_dim))

    merged = Merge([number_branch, item_branch, dept_branch], mode='concat')

    final_model = Sequential()
    final_model.add(merged)
    final_model.add(Dense(10, activation='softmax'))


    model = Sequential()
    model.add(Dense(500, input_dim=350))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(300))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(15))
    model.add(Activation('softmax'))

    rms = RMSprop()

    #compile Graph
    model.compile(loss=_sigmoid_cross_entropy, optimizer=rms, metrics=['accuracy'])
    print ('Model compield in {0} seconds'.format(time.time() - start_time))
    return model

def run_network(data=None, model=None, epochs=20, batch=200):
    try:
        start_time = time.time()
        if data is None:
            filename = "FILENAME HERE"
            test_Y, test_Number, test_Item, test_Department, train_Y, train_Number, train_Item, train_Department = load_data(filename)
        else:
            test_Y, test_Number, test_Item, test_Department, train_Y, train_Number, train_Item, train_Department = data

        if model is None:
            model = init_model()

        #loss history
        history = LossHistory()

        #Saving checkpoint
        filepath="best_model_weights.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint] + history

        print ('Training model...')
        model.fit([train_Number, train_Item, train_Department], train_Y, epochs=epochs, batch_size=batch,
                  callbacks=callbacks_list,
                  validation_data=([test_Number, test_Item, test_Department], test_Y), verbose=2)

        print ("Training duration : {0}".format(time.time() - start_time))
        score = model.evaluate([test_Number, test_Item, test_Department], test_Y, batch_size=16)

        print ("Network's test score [loss, accuracy]: {0}".format(score))
        return model, history.losses

    except KeyboardInterrupt:
        print (' KeyboardInterrupt')
        return model, history.losses

def plot_losses(losses):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(losses)
    ax.set_title('Loss per batch')
    fig.show()