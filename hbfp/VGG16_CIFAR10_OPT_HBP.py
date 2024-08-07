import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

import keras
from keras.datasets import cifar10
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, GlobalAveragePooling2D,BatchNormalization,Activation
from keras.models import load_model
from keras.callbacks import Callback
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.layers.core import Lambda
from keras import regularizers
from keras.callbacks import ModelCheckpoint
from sklearn import preprocessing

# !pip install kerassurgeon
from kerassurgeon import identify 
from kerassurgeon.operations import delete_channels,delete_layer
from kerassurgeon import Surgeon
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def my_get_all_conv_layers(model , first_time):

    '''
    Arguments:
        model -> your model
        first_time -> type boolean 
            first_time = True => model is not pruned 
            first_time = False => model is pruned
    Return:
        List of Indices containing convolution layers
    '''

    all_conv_layers = list()
    for i,each_layer in enumerate(model.layers):
        if (each_layer.name[0:6] == 'conv2d'):
            all_conv_layers.append(i)
    return all_conv_layers if (first_time==True) else all_conv_layers[1:]


def my_get_all_dense_layers(model):
    '''
    Arguments:
        model -> your model        
    Return:
        List of Indices containing fully connected layers
    '''
    all_dense_layers = list()
    for i,each_layer in enumerate(model.layers):
        if (each_layer.name[0:5] == 'dense'):
            all_dense_layers.append(i)
    return all_dense_layers


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
            # print(index,layer.name,params,flops)
            total_params += params
            total_flops += flops
        elif 'Dense' in str(type(layer)):
            
            params, flops = count_dense_params_flops(layer)
            # print(index,layer.name,params,flops)
            total_params += params
            total_flops += flops
    return total_params, int(total_flops)


def my_get_weights_in_conv_layers(model,first_time):

    '''
    Arguments:
        model -> your model
        first_time -> boolean variable
            first_time = True => model is not pruned 
            first_time = False => model is pruned
    Return:
        List containing weight tensors of each layer
    '''
    
    weights = list()
    all_conv_layers = my_get_all_conv_layers(model,first_time)
    layer_wise_weights = list() 
    for i in all_conv_layers:
          weights.append(model.layers[i].get_weights()[0])  
    return weights

def my_get_l1_norms_filters_per_epoch(weight_list_per_epoch):

    '''
    Arguments:
        List
    Return:
        Number of parmaters, Number of Flops
    '''
    
    # weight_list_per_epoch = my_get_weights_in_conv_layers(model,first_time)
    l1_norms_filters_per_epoch = list()
    

    for index in range(len(weight_list_per_epoch)):

        epochs = np.array(weight_list_per_epoch[index]).shape[0]
        h , w , d = np.array(weight_list_per_epoch[index]).shape[1], np.array(weight_list_per_epoch[index]).shape[2] , np.array(weight_list_per_epoch[index]).shape[3]


        l1_norms_filters_per_epoch.append(np.sum(np.abs(np.array(weight_list_per_epoch[index])).reshape(epochs,h*w*d,-1),axis=1))
    return l1_norms_filters_per_epoch


def my_in_conv_layers_get_sum_of_l1_norms_sorted_indices(weight_list_per_epoch):
    '''
        Arguments:
            weight List 
        Return:
            layer_wise_filter_sorted_indices
            
    '''
    layer_wise_filter_sorted_indices = list()
    layer_wise_filter_sorted_values = list()
    l1_norms_filters_per_epoch = my_get_l1_norms_filters_per_epoch(weight_list_per_epoch)
    sum_l1_norms = list()
    
    for i in l1_norms_filters_per_epoch:
        sum_l1_norms.append(np.sum(i,axis=0))
    
    layer_wise_filter_sorted_indices = list()
    
    for i in sum_l1_norms:
        a = pd.Series(i).sort_values().index
        layer_wise_filter_sorted_indices.append(a.tolist())
    return layer_wise_filter_sorted_indices


def my_get_percent_prune_filter_indices(layer_wise_filter_sorted_indices,percentage):    
    """
    Arguments:
        layer_wise_filter_sorted_indices:filters to be 
        percentage:percentage of filters to be pruned
    Return:
        prune_filter_indices: indices of filter to be pruned
    """

    prune_filter_indices = list()
    for i in range(len(layer_wise_filter_sorted_indices)):
        prune_filter_indices.append(int(len(layer_wise_filter_sorted_indices[i]) * (percentage/100))   )
    return prune_filter_indices


def my_get_distance_matrix(l1_norm_matrix):
    """
    Arguments:
        l1_norm_matrix:matrix that stores the l1 norms of filters
    Return:
        distance_matrix: matrix that stores the manhattan distance between filters 
    """
    distance_matrix = []
    for i,v1 in enumerate(l1_norm_matrix):
        distance_matrix.append([])
        for v2 in l1_norm_matrix:
            distance_matrix[i].append(np.sum(abs(v1-v2)))
    return np.array(distance_matrix)
    

def my_get_distance_matrix_list(l1_norm_matrix_list):
    """
    Arguments:
        l1_norm_matrix_list:
    Return:
        distance_matrix_list:
    """ 
    distance_matrix_list = []
    for l1_norm_matrix in l1_norm_matrix_list:
        distance_matrix_list.append(my_get_distance_matrix(l1_norm_matrix.T))
    return distance_matrix_list


def my_get_episodes(distance_matrix,percentage):
    """
    Arguments:
        distance_matrix:
        percentage:Percentage of filters to be pruned
    Return:
    episodes:list of filter indices
    """
    distance_matrix_flatten = pd.Series(distance_matrix.flatten())
    
    distance_matrix_flatten = distance_matrix_flatten.sort_values().index.to_list()
    
    episodes = list()
    n = distance_matrix.shape[0]
    for i in distance_matrix_flatten:
        episodes.append((i//n,i%n))
    k = int((n * percentage)/100)
    li = list()
    for i in range(2*k):
        if i%2!=0:
            li.append(episodes[n+i])
    return li


def my_get_episodes_for_all_layers(distance_matrix_list,percentage):
    """
    Arguments:
        distance_matrix_list:matrix containing the manhattan distance of all layers
        percentage:percentage of filters to be pruned
    Return:
        all_episodes:all the selected filter pairs
    """
    all_episodes = list()
    for matrix in distance_matrix_list:
        all_episodes.append(my_get_episodes(matrix,percentage))
    return all_episodes


def my_get_filter_pruning_indices(episodes_for_all_layers,l1_norms_list):
    """
    Arguments:
        episodes_for_all_layers:list of selected filter pairs 
        l1_norm_matrix_list:list of l1 norm matrices of all the filters of each layer
    Return:
        filter_pruning_indices:list of indices of filters to be pruned
    """
    filter_pruning_indices = list()
    for layer_index,episode_layer in enumerate(episodes_for_all_layers):
        a = set()
        sum_l1_norms = np.sum(l1_norms_list[layer_index],axis=0,keepdims=True)

        for episode in episode_layer:
            ep1 = sum_l1_norms.T[episode[0]]
            ep2 = sum_l1_norms.T[episode[1]]
            if ep1 >= ep2:
                a.add(episode[0])
            else:
                a.add(episode[1])

        a = list(a)
        filter_pruning_indices.append(a)
    return filter_pruning_indices


def my_delete_filters(model,weight_list_per_epoch,percentage,first_time):
    """
    Arguments:
        model:CNN Model
        wieight_list_per_epoch:History
        percentage:Percentage to be pruned
        first_time:Boolean Variable
            first_time -> boolean variable
            first_time = True => model is not pruned 
            first_time = False => model is pruned
    Return:
        model_new:input model after pruning

    """
    l1_norms = my_get_l1_norms_filters_per_epoch(weight_list_per_epoch)
    distance_matrix_list = my_get_distance_matrix_list(l1_norms)
    episodes_for_all_layers = my_get_episodes_for_all_layers(distance_matrix_list,percentage)
    filter_pruning_indices = my_get_filter_pruning_indices(episodes_for_all_layers,l1_norms)
    all_conv_layers = my_get_all_conv_layers(model,first_time)

    surgeon = Surgeon(model)
    for index,value in enumerate(all_conv_layers):
        print(index,value,filter_pruning_indices[index])
        surgeon.add_job('delete_channels',model.layers[value],channels = filter_pruning_indices[index])

    model_new = surgeon.operate()
    return model_new


class Get_Weights(Callback):
    def __init__(self,first_time):
        super(Get_Weights, self).__init__()
        self.weight_list = [] #Using a list of list to store weight tensors per epoch
        self.first_time = first_time
    def on_epoch_end(self,epoch,logs=None):
        if epoch == 0:
            all_conv_layers = my_get_all_conv_layers(self.model,self.first_time)
            for i in range(len(all_conv_layers)):
                self.weight_list.append([]) # appending empty lists for later appending weight tensors 
        
        for index,each_weight in enumerate(my_get_weights_in_conv_layers(self.model,self.first_time)):
                self.weight_list[index].append(each_weight)  


class cifar10vgg:

    def __init__(self,first_time,epochs,train=True):
        self.epochs = epochs
        self.first_time = first_time
        self.num_classes = 10
        self.weight_decay = 0.0005
        self.x_shape = [32,32,3]
        self.history = 0
        self.weight_list_per_epoch = None
        self.model = self.build_model()
        if train:
            self.model, self.history ,self.weight_list_per_epoch = self.train(self.model)
        else:
            #change

            self.model.load_weights(os.path.join('.', 'models', 'cifarvgg10.h5'))


    def build_model(self):
        # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.

        model = Sequential()
        weight_decay = self.weight_decay

        model.add(Conv2D(64, (3, 3), padding='same',
                         input_shape=self.x_shape,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes))
        model.add(Activation('softmax'))
        return model


    def normalize(self,X_train,X_test):
        #this function normalize inputs for zero mean and unit variance
        # it is used when training a model.
        # Input: training set and test set
        # Output: normalized training set and test set according to the trianing set statistics.
        mean = np.mean(X_train,axis=(0,1,2,3))
        std = np.std(X_train, axis=(0, 1, 2, 3))
        X_train = (X_train-mean)/(std+1e-7)
        X_test = (X_test-mean)/(std+1e-7)
        return X_train, X_test


    def train(self,model):

        #training parameters
        batch_size = 128
        maxepoches = 10
        learning_rate = 0.1
        lr_decay = 1e-6
        lr_drop = 20
        # The data, shuffled and split between train and test sets:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train, x_test = self.normalize(x_train, x_test)

        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)

        def lr_scheduler(epoch):
            return learning_rate * (0.5 ** (epoch // lr_drop))
        reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

        #data augmentation
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        # (std, mean, and principal components if ZCA whitening is applied).
        # datagen.fit(x_train)

        #optimization details
        sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])

        gw = Get_Weights(self.first_time)

        # training process in a for loop with learning rate drop every 25 epoches.

        history = model.fit_generator(datagen.flow(x_train, y_train,
                                         batch_size=batch_size),
                            steps_per_epoch=x_train.shape[0] // batch_size,
                            epochs=self.epochs,
                            validation_data=(x_test, y_test),callbacks=[reduce_lr,gw],verbose=1)

        return model, history,gw.weight_list


def normalize(X_train,X_test):
    #this function normalize inputs for zero mean and unit variance
    # it is used when training a model.
    # Input: training set and test set
    # Output: normalized training set and test set according to the trianing set statistics.
    mean = np.mean(X_train,axis=(0,1,2,3))
    std = np.std(X_train, axis=(0, 1, 2, 3))
    X_train = (X_train-mean)/(std+1e-7)
    X_test = (X_test-mean)/(std+1e-7)
    return X_train, X_test


def train(model,epochs):
    """
    Arguments:
        model:model to be trained
        epochs:number of epochs to be trained
        first_tim:
    Return:
        model:trained/fine-tuned Model,
        history: accuracies and losses (keras history)
        weight_list_per_epoch = all weight tensors per epochs in a list
    """
    #training parameters
    batch_size = 128
    learning_rate = 0.01
    lr_decay = 1e-6
    lr_drop = 20

    num_classes = 10
    weight_decay = 0.0005
    x_shape = [32,32,3]

    # The data, shuffled and split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train, x_test = normalize(x_train, x_test)

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    def lr_scheduler(epoch):
        return learning_rate * (0.5 ** (epoch // lr_drop))
    reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)


    #data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
    # (std, mean, and principal components if ZCA whitening is applied).
    # datagen.fit(x_train)

    #optimization details
    sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])

    gw = Get_Weights(False)

    # training process in a for loop with learning rate drop every 25 epoches.

    history = model.fit_generator(datagen.flow(x_train, y_train,
                                        batch_size=batch_size),
                        steps_per_epoch=x_train.shape[0] // batch_size,
                        epochs=epochs,
                        validation_data=(x_test, y_test),callbacks=[reduce_lr,gw],verbose=1)

    return model, history,gw.weight_list

from keras import backend as K

def custom_loss(lmbda , regularizer_value):
  def loss(y_true , y_pred):
    # print(type(K.categorical_crossentropy(y_true ,y_pred)),K.categorical_crossentropy(y_true ,y_pred),regularizer_value)
    return K.categorical_crossentropy(y_true ,y_pred) + lmbda * regularizer_value
  return loss


def my_get_l1_norms_filters(model,first_time):
    """
    Arguments:
        model:initial model
        weight_list_per_epoch:weight tensors at every epoch
        percentage:percentage of filter to be pruned
        first_time:type bool
    Return:
        regularizer_value
    """

    conv_layers = my_get_all_conv_layers(model,first_time)
    l1_norms = list()
    for index,layer_index in enumerate(conv_layers):
        l1_norms.append([])
        # print(layer_index)
        weights = model.layers[layer_index].get_weights()[0]
        num_filters = len(weights[0,0,0,:])
        for i in range(num_filters):
            weights_sum = np.sum(abs(weights[:,:,:,i]))
            l1_norms[index].append(weights_sum)
    return l1_norms


def my_get_regularizer_value(model,weight_list_per_epoch,percentage,first_time):
    """
    Arguments:
        model:initial model
        weight_list_per_epoch:weight tensors at every epoch
        percentage:percentage of filter to be pruned
        first_time:type bool
    Return:
        regularizer_value
    """
    l1_norms_per_epoch = my_get_l1_norms_filters_per_epoch(weight_list_per_epoch)
    distance_matrix_list = my_get_distance_matrix_list(l1_norms_per_epoch)
    episodes_for_all_layers = my_get_episodes_for_all_layers(distance_matrix_list,percentage)
    l1_norms = my_get_l1_norms_filters(model,first_time)
    # print(episodes_for_all_layers)
    regularizer_value = 0
    for layer_index,layer in enumerate(episodes_for_all_layers):
        for episode in layer:
            # print(episode[1],episode[0])
            regularizer_value += abs(l1_norms[layer_index][episode[1]] - l1_norms[layer_index][episode[0]])
    regularizer_value = np.exp(-1*(regularizer_value))
    # print(regularizer_value)    
    return regularizer_value


def optimize(model,weight_list_per_epoch,epochs,percentage,first_time):
    """
    Arguments:
        model:inital model
        weight_list_per_epoch: weight tensors at every epoch
        epochs:number of epochs to be trained on custom regularizer
        percentage:percentage of filters to be pruned
        first_time:type bool
    Return:
        model:optimized model
        hisory: accuracies and losses over the process keras library
    """
    batch_size = 128
    learning_rate = 0.01
    lr_decay = 1e-6
    lr_drop = 20

    num_classes = 10
    weight_decay = 0.0005
    x_shape = [32,32,3]

    # The data, shuffled and split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train, x_test = normalize(x_train, x_test)

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    def lr_scheduler(epoch):
        return learning_rate * (0.5 ** (epoch // lr_drop))
    reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

    #data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
    # (std, mean, and principal components if ZCA whitening is applied).
    # datagen.fit(x_train)

    regularizer_value = my_get_regularizer_value(model,weight_list_per_epoch,percentage,first_time)
    print("INITIAL REGULARIZER VALUE ",my_get_regularizer_value(model,weight_list_per_epoch,percentage,first_time))
    model_loss = custom_loss(lmbda= 0.1 , regularizer_value=regularizer_value)
    sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
    model.compile(loss=model_loss, optimizer=sgd,metrics=['accuracy'])
    
    history = model.fit_generator(datagen.flow(x_train, y_train,
                                        batch_size=batch_size),
                        steps_per_epoch=x_train.shape[0] // batch_size,
                        epochs=epochs,
                        validation_data=(x_test, y_test),callbacks=[reduce_lr],verbose=1)
    
    print("FINAL REGULARIZER VALUE ",my_get_regularizer_value(model,weight_list_per_epoch,percentage,first_time))
    return model,history


#this dictionary is to log the parameters and is later converted into a dataframe.
log_dict = dict()
log_dict['train_loss'] = []
log_dict['train_acc'] = []
log_dict['val_loss'] = []
log_dict['val_acc'] = []
log_dict['total_params'] = []
log_dict['total_flops'] = []
choice = input("USE PRETRAINED VGG16 FOR CIFAR10 [Y/N] : ")

if choice == 'Y':
    my_vgg = cifar10vgg(first_time=True,epochs=0,train=False)
    model = my_vgg.model

    learning_rate = 0.1
    lr_decay = 1e-6
    lr_drop = 20
    
    def lr_scheduler(epoch):
        return learning_rate * (0.5 ** (epoch // lr_drop))
    reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

    sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])

    weight_list_per_epoch = list()
    data = np.load(os.path.join('.', 'models', 'vgg_weights.npz'))
    for i in range(13):
        weight_list_per_epoch.append(data['w_{}'.format(i+1)])

    (x_train,y_train),(x_test,y_test) = cifar10.load_data()
    x_train,x_test = normalize(x_train,x_test)
    y_train = keras.utils.to_categorical(y_train,10)
    y_test = keras.utils.to_categorical(y_test,10)
    train_loss,train_acc = model.evaluate(x_train,y_train)
    val_loss,val_acc = model.evaluate(x_test,y_test)
    validation_accuracy = val_acc
    log_dict['train_loss'].append(train_loss)
    log_dict['train_acc'].append(train_acc)
    log_dict['val_loss'].append(val_loss)
    log_dict['val_acc'].append(val_acc)

elif choice == 'N':
    # train for first time
    my_vgg = cifar10vgg(first_time=True,epochs=250)
    model, history ,weight_list_per_epoch= my_vgg.model, my_vgg.history, my_vgg.weight_list_per_epoch

    model.save_weights(os.path.join('.', 'models', 'cifarvgg10.h5'))
    #save the weights of training process
    np.savez(os.path.join('.', 'models', 'vgg_weights.npz')
            ,w_1=weight_list_per_epoch[0],
            w_2=weight_list_per_epoch[1],
            w_3=weight_list_per_epoch[2],
            w_4=weight_list_per_epoch[3],
            w_5=weight_list_per_epoch[4],
            w_6=weight_list_per_epoch[5],
            w_7=weight_list_per_epoch[6],
            w_8=weight_list_per_epoch[7],
            w_9=weight_list_per_epoch[8],
            w_10=weight_list_per_epoch[9],
            w_11=weight_list_per_epoch[10],
            w_12=weight_list_per_epoch[11],
            w_13=weight_list_per_epoch[12])
    best_acc_index = history.history['val_acc'].index(max(history.history['val_acc']))
    log_dict['train_loss'].append(history.history['loss'][best_acc_index])
    log_dict['train_acc'].append(history.history['acc'][best_acc_index])
    log_dict['val_loss'].append(history.history['val_loss'][best_acc_index])
    log_dict['val_acc'].append(history.history['val_acc'][best_acc_index])
    validation_accuracy = max(history.history['val_acc'])

a,b = count_model_params_flops(model,True)
log_dict['total_params'].append(a)
log_dict['total_flops'].append(b)

log_df = pd.DataFrame(log_dict)
log_df.to_csv(os.path.join('.', 'results', 'VGG16.csv'))

print("Initial Validation Accuracy = {}".format(validation_accuracy) )
max_val_acc = validation_accuracy
count = 0

while validation_accuracy - max_val_acc >= -0.02 :

# while count <= 2  :

    print("ITERATION {} ".format(count+1))
    
    if max_val_acc < validation_accuracy:
        max_val_acc = validation_accuracy
        
    if count < 1:
        print('OPTIMIZATION')
        model,_ = optimize(model,weight_list_per_epoch,50,10,True)
        model = my_delete_filters(model,weight_list_per_epoch,10,True)
        print('FINETUNING')
        model,history,weight_list_per_epoch = train(model,150)
   
    elif count <= 3:
        print('OPTIMIZATION')
        model,_ =optimize(model,weight_list_per_epoch,50,10,False)
        model = my_delete_filters(model,weight_list_per_epoch,10,False)
        print('FINETUNING')
        model,history,weight_list_per_epoch = train(model,150)
    else:
        print('OPTIMIZATION')   
        model,_ =optimize(model,weight_list_per_epoch,10,10,False)
        model = my_delete_filters(model,weight_list_per_epoch,10,False)
        print('FINETUNING')
        model,history,weight_list_per_epoch = train(model,200)

    a,b = count_model_params_flops(model,False)
    
    validation_accuracy = max(history.history['val_acc'])
    best_acc_index = history.history['val_acc'].index(max(history.history['val_acc']))
    log_dict['train_loss'].append(history.history['loss'][best_acc_index])
    log_dict['train_acc'].append(history.history['acc'][best_acc_index])
    log_dict['val_loss'].append(history.history['val_loss'][best_acc_index])
    log_dict['val_acc'].append(history.history['val_acc'][best_acc_index])
    log_dict['total_params'].append(a)
    log_dict['total_flops'].append(b)
    log_df = pd.DataFrame(log_dict)
    log_df.to_csv(os.path.join('.', 'results', 'VGG16.csv'))
    print("VALIDATION ACCURACY AFTER {} ITERATIONS = {}".format(count+1,validation_accuracy))
    count+=1

log_df = pd.DataFrame(log_dict)
log_df.to_csv(os.path.join('.', 'results', 'VGG16.csv'))