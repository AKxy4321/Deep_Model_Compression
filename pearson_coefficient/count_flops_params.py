import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import preprocessing

import keras
from keras import backend as K
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, GlobalAveragePooling2D,BatchNormalization,Activation,AveragePooling2D
from keras.models import load_model
from keras.callbacks import Callback
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers
from keras.callbacks import ModelCheckpoint


def count_conv_params_flops(conv_layer):
    # out shape is  n_cells_dim1 * (n_cells_dim2 * n_cells_dim3)
    '''
    Arguments:
        conv layer 
    Return:
        Number of Parameters, Number of Flops
    '''

    out_shape = conv_layer.output_shape

    n_cells_total = np.prod(out_shape[1:-1])

    n_conv_params_total = conv_layer.count_params()
    # print(n_conv_params_total,len(conv_layer.get_weights()[0]),)
    conv_flops =  (n_conv_params_total * n_cells_total - len(conv_layer.get_weights()[1]) *n_cells_total)

 

    return n_conv_params_total, conv_flops


def count_dense_params_flops(dense_layer):
    # out shape is  n_cells_dim1 * (n_cells_dim2 * n_cells_dim3)
    '''
    Arguments:
      dense layer 
    Return:
        Number of Parameters, Number of Flops
    '''

    out_shape = dense_layer.output_shape
    n_cells_total = np.prod(out_shape[1:-1])

    n_dense_params_total = dense_layer.count_params()

    dense_flops =  (n_dense_params_total - len(dense_layer.get_weights()[1]) * n_cells_total)


    return n_dense_params_total, dense_flops




def count_model_params_flops(model,first_time):

    '''
    Arguments:
        model -> your model
        first_time -> boolean variable
        first_time = True => model is not pruned 
        first_time = False => model is pruned
    Return:
        Number of parmaters, Number of Flops
    '''
    total_params = 0
    total_flops = 0
    model_layers = model.layers
    for index,layer in enumerate(model_layers):
        if any(conv_type in str(type(layer)) for conv_type in ['Conv1D', 'Conv2D', 'Conv3D']):
            
            params, flops = count_conv_params_flops(layer)
            print(index,layer.name,params,flops)
            total_params += params
            total_flops += flops
        elif 'Dense' in str(type(layer)):
            
            params, flops = count_dense_params_flops(layer)
            print(index,layer.name,params,flops)
            total_params += params
            total_flops += flops
    return total_params, int(total_flops)


if __name__ == "__main__":
	model = keras.Sequential()
	model.add(Conv2D(filters=20, kernel_size=(5, 5), activation='relu', input_shape=(28,28,1)))
	model.add(MaxPooling2D())
	model.add(Conv2D(filters=50, kernel_size=(5, 5), activation='relu'))
	model.add(MaxPooling2D())
	model.add(Flatten())
	model.add(Dense(units=500, activation='relu'))
	model.add(Dense(units=10, activation = 'softmax'))
	
	total_flops, total_parameters = count_model_params_flops(model,True)
	print(total_flops,total_parameters) 
	