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
import os
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
from pandas.util.testing import assert_frame_equal


_EPSILON = K.epsilon()
_SEED = 7


def load_validation():
    '''load_validation
    Simply returns the first set of trianing files as validation data
    '''
    test_file = '14'
    cwd = os.getcwd()

    test_file = '14'
    folder = "/processed_training_files/"

    test_Y = pd.read_csv(cwd+folder+'real_train_file_'+test_file+'.csv')
    test_Number = pd.read_csv(cwd+folder+'ni_train_file_'+test_file+'.txt')
    test_Item = pd.read_csv(cwd+folder+'pct_train_file_'+test_file+'.csv')
    test_Department = pd.read_csv(cwd+folder+'dept_train_file_'+test_file+'.csv')

    test_Y = test_Y.sort_values(by= ['user_id'])
    test_Number = test_Number.sort_values(by= ['user_id'])
    test_Item = test_Item.sort_values(by= ['user_id'])
    test_Department = test_Department.sort_values(by= ['user_id'])

    assert_frame_equal(test_Y[['user_id']].reset_index(drop=True), test_Number[['user_id']].reset_index(drop=True))
    assert_frame_equal(test_Y[['user_id']].reset_index(drop=True), test_Item[['user_id']].reset_index(drop=True))
    assert_frame_equal(test_Y[['user_id']].reset_index(drop=True), test_Department[['user_id']].reset_index(drop=True))

    return(test_Y, test_Number, test_Item, test_Department)

def load_batches(epoch):
    '''
    Expects data to be in a certain format (to be determined) Does a 20% split using seed
    as RandomState.

    Args:
        filename (str) - name of data file
        seed (int) - seed to use to split test and train

    Returns: 
        test_Y, test_Number, test_Item, test_Department - 
            as numpy matrices
    Raises:
        AssertionError - if test_Y, train_Y, train_Item, test_Item do not all have the same shape
    '''
    if epoch < 1 or epoch > 13:
            raise ValueError("Epoch can only be between 1 and 13")

    cwd = os.getcwd()

    test_file = str(epoch)

    train_Y = pd.read_csv(cwd+folder+'real_train_file_'+test_file+'.csv')
    train_Number = pd.read_csv(cwd+folder+'ni_train_file_'+test_file+'.txt')
    train_Item = pd.read_csv(cwd+folder+'pct_train_file_'+test_file+'.csv')
    train_Department = pd.read_csv(cwd+folder+'dept_train_file_'+test_file+'.csv')

    train_Y = train_Y.sort_values(by= ['user_id'])
    train_Number = train_Number.sort_values(by= ['user_id'])
    train_Item = train_Item.sort_values(by= ['user_id'])
    train_Department = train_Department.sort_values(by= ['user_id'])

    assert_frame_equal(train_Y[['user_id']].reset_index(drop=True), train_Number[['user_id']].reset_index(drop=True))
    assert_frame_equal(train_Y[['user_id']].reset_index(drop=True), train_Item[['user_id']].reset_index(drop=True))
    assert_frame_equal(train_Y[['user_id']].reset_index(drop=True), train_Department[['user_id']].reset_index(drop=True))

    return(train_Y, train_Number, train_Item, train_Department)

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


def init_model(numer_dim = 12, item_dim = 20000, dept_dim = 23):
    '''
    Creates model Graph and compiles it
    '''
    start_time = time.time()
    print ('Compiling Model ... ')

    #Number Branch
    number_branch = Sequential()
    number_branch.add(Dense(32, input_dim=number_dim))
    number_branch.add(Activation('relu'))
    number_branch.add(Dropout(0.4))

    #Item Branch
    item_branch = Sequential()
    item_branch.add(Dense(16000, input_dim=item_dim))
    item_branch.add(Activation('relu'))
    item_branch.add(Dropout(0.4))

    #Department Branch
    dept_branch = Sequential()
    dept_branch.add(Dense(32, input_dim=dept_dim))
    dept_branch.add(Activation('relu'))
    dept_branch.add(Dropout(0.4))

    merged1 = Merge([item_branch, dept_branch], mode='concat')

    item_dept_model = Sequential()
    item_dept_model.add(merged1)
    item_dept_model.add(Dense(12000, input_dim=item_dim))
    item_dept_model.add(Activation('relu'))
    item_dept_model.add(Dropout(0.4))

    merged2 = Merge([number_branch, item_dept_model], mode='concat')

    final_model = Sequential()
    final_model.add(merged2)
    final_model.add(Dense(8000))
    final_model.add(Activation('relu'))
    final_model.add(Dropout(0.4))
    final_model.add(Dense(item_dim))
    final_model.add(Activation('sigmoid'))

    #recommended param values, but we can test a little here (especially lr)
    rms = RMSprop(lr=0.01, epsilon=1e-08, decay=0.0)

    #compile Graph
    model.compile(loss='binary_crossentropy', optimizer=rms, metrics=['accuracy'])
    print ('Model compield in {0} seconds'.format(time.time() - start_time))
    return model

def run_network(valid_data=None, model=None, epochs=13, batch=32):
    try:
        start_time = time.time()
        if valid_data is None:
            filename = "FILENAME HERE"
            test_Y, test_Number, test_Item, test_Department = load_validation(filename)
        else:
            test_Y, test_Number, test_Item, test_Department = valid_data

        if model is None:
            model = init_model()

        #loss history
        history = LossHistory()

        #Saving checkpoint
        filepath="best_model_weights.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint] + history

        print ('Training model...')
        #More code to use gernerators: https://keras.io/models/sequential/ and https://github.com/fchollet/keras/issues/68
        #one epoch will be one file. 1000 users, 131 files
        for e in range(epochs):
            print("epoch %d" % e)
            for train_Y, train_Number, train_Item, train_Department in load_batches(e): # these are chunks of ~1k users
                model.fit([train_Number, train_Item, train_Department], train_Y, 
                            epochs=1, 
                            batch_size=batch,
                            callbacks=callbacks_list,
                            validation_data=([test_Number, test_Item, test_Department], test_Y), 
                            verbose=2)
                '''
        model.fit_generator(generator(file_path = path),
                            steps_per_epoch = samples_per_epoch//batch,
                            epochs=epochs, 
                            batch_size=batch,
                            callbacks=callbacks_list,
                            validation_data=([test_Number, test_Item, test_Department], test_Y), 
                            verbose=2)
                '''

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








