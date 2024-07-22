# %tensorflow_version 1.x
# !pip install tesnsorflow 1.x
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import preprocessing

import keras
# from keras import backend as K
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
from sklearn.metrics.pairwise import cosine_similarity


# !pip install kerassurgeon
from kerassurgeon import identify 
from kerassurgeon.operations import delete_channels, delete_layer
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
    
    cosine_similarities_filters_per_epoch = []
    epochs = np.array(weight_list_per_epoch[0]).shape[0]
    for weight_array in weight_list_per_epoch:
        epoch_cosine_similarities = []
        for epochs in weight_array:
            num_filters = epochs.shape[3]
            h, w, d = epochs.shape[0], epochs.shape[1], epochs.shape[2]
            flattened_filters = epochs.reshape(-1, num_filters).T
            cosine_sim = cosine_similarity(flattened_filters)
            summed_cosine_similarities = np.sum(cosine_sim, axis=1) - 1
            epoch_cosine_similarities.append(summed_cosine_similarities.tolist())
        cosine_similarities_filters_per_epoch.append(np.array(epoch_cosine_similarities))

    # print(cosine_similarities_filters_per_epoch)
    return cosine_similarities_filters_per_epoch                            # Return as numpy array so that transpose can be implemented

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
        prune_filter_indices.append(int(len(layer_wise_filter_sorted_indices[i]) * (percentage/100)))
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
            distance_matrix[i].append(np.sum((v1-v2)**2))
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


def my_get_filter_pruning_indices(episodes_for_all_layers,l1_norm_matrix_list):

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
        sum_l1_norms = np.sum(l1_norm_matrix_list[layer_index],axis=0,keepdims=True)

        for episode in episode_layer:
            ep1 = sum_l1_norms.T[episode[0]]
            ep2 = sum_l1_norms.T[episode[1]]
            if ep1 >= ep2:
                a.add(episode[0])
            else:
                a.add(episode[1])
            a.add(episode[0])
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
    conv_flops = 2 * (n_conv_params_total * n_cells_total - len(conv_layer.get_weights()[1]) *n_cells_total)

 

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

    dense_flops = 2* (n_dense_params_total - len(dense_layer.get_weights()[1]) * n_cells_total)


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

model = keras.Sequential()

model.add(Conv2D(filters=20, kernel_size=(5, 5), activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D())

model.add(Conv2D(filters=50, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(units=500, activation='relu'))

model.add(Dense(units=10, activation = 'softmax'))

def train(model,epochs,first_time):
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
    (x_train, y_train), (x_test, y_test) = mnist.load_data()


    img_rows, img_cols = 28, 28
    batch_size = 128
    num_classes = 10

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255




    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)


    # Compile the model
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    sgd = optimizers.SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc']) 

    gw = Get_Weights(first_time)
    history = model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=2,
            callbacks=[gw],
            validation_data=(x_test, y_test))

    return model,history,gw.weight_list

model,history,weight_list_per_epoch = train(model,10,True)
initial_flops = count_model_params_flops(model,True)[1]
log_dict = dict()
log_dict['train_loss'] = []
log_dict['train_acc'] = []
log_dict['val_loss'] = []
log_dict['val_acc'] = []
log_dict['total_params'] = []
log_dict['total_flops'] = []
log_dict['filters_in_conv1'] = []
log_dict['filters_in_conv2'] = []

best_acc_index = history.history['val_acc'].index(max(history.history['val_acc']))
log_dict['train_loss'].append(history.history['loss'][best_acc_index])
log_dict['train_acc'].append(history.history['acc'][best_acc_index])
log_dict['val_loss'].append(history.history['val_loss'][best_acc_index])
log_dict['val_acc'].append(history.history['val_acc'][best_acc_index])
a,b = count_model_params_flops(model,True)
log_dict['total_params'].append(a)
log_dict['total_flops'].append(b)
log_dict['filters_in_conv1'].append(model.layers[0].get_weights()[0].shape[-1])
log_dict['filters_in_conv2'].append(model.layers[2].get_weights()[0].shape[-1])
al = history

from keras import backend as K
def custom_loss(lmbda , regularizer_value):
  def loss(y_true , y_pred):
    # print(type(K.categorical_crossentropy(y_true ,y_pred)),K.categorical_crossentropy(y_true ,y_pred),regularizer_value)
    return K.categorical_crossentropy(y_true ,y_pred) + lmbda * regularizer_value
  return loss

def my_get_l1_norms_filters(model,first_time):
    """
    Arguments:
        model:

        first_time : type boolean 
            first_time = True => model is not pruned 
            first_time = False => model is pruned
        Return:
            l1_norms of all filters of every layer as a list
    """
    conv_layers = my_get_all_conv_layers(model,first_time)
    l1_norms = list()
    for index,layer_index in enumerate(conv_layers):
        l1_norms.append([])
        # print(layer_index)
        weights = model.layers[layer_index].get_weights()[0]
        num_filters = len(weights[0,0,0,:])
        for i in range(num_filters):
            weights_sum = np.sum(weights[:,:,:,i])
            l1_norms[index].append(weights_sum)
    return l1_norms

def my_get_cosine_similarity_filters(model, first_time):
    """
    Arguments:
        model:

        first_time : type boolean 
            first_time = True => model is not pruned 
            first_time = False => model is pruned
        Return:
            Cosine similarity sums of all filters of every layer as a list
    """
    conv_layers = my_get_all_conv_layers(model, first_time)
    cosine_sums = list()
    for index, layer_index in enumerate(conv_layers):
        cosine_sums.append([])
        weights = model.layers[layer_index].get_weights()[0]
        num_filters = len(weights[0,0,0,:])
        filter_vectors = [weights[:,:,:,i].flatten() for i in range(num_filters)]
        
        for i in range(num_filters):
            similarities = cosine_similarity([filter_vectors[i]], filter_vectors)[0]
            cosine_sum = np.sum(similarities)
            cosine_sums[index].append(cosine_sum)
    return cosine_sums

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
    l1_norms = my_get_cosine_similarity_filters(model,first_time)
    print(episodes_for_all_layers)
    regularizer_value = 0
    for layer_index,layer in enumerate(episodes_for_all_layers):
        for episode in layer:
            # print(episode[1],episode[0])
            regularizer_value += abs(l1_norms[layer_index][episode[1]] - l1_norms[layer_index][episode[0]])
    regularizer_value = np.exp((regularizer_value))
    print(regularizer_value)    
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
    (x_train, y_train), (x_test, y_test) = mnist.load_data()


    img_rows, img_cols = 28, 28
    batch_size = 128
    num_classes = 10

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    regularizer_value = my_get_regularizer_value(model,weight_list_per_epoch,percentage,first_time)
    print("INITIAL REGULARIZER VALUE ",my_get_regularizer_value(model,weight_list_per_epoch,percentage,first_time))
    model_loss = custom_loss(lmbda= 0.1 , regularizer_value=regularizer_value)
    # print('model loss',model_loss)
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss=model_loss,optimizer=adam,metrics=['acc'])

    history = model.fit(x_train , y_train,epochs=epochs,batch_size = batch_size,validation_data=(x_test, y_test),verbose=1)
    print("FINAL REGULARIZER VALUE ",my_get_regularizer_value(model,weight_list_per_epoch,percentage,first_time))
    return model,history

count_model_params_flops(model,True)

print('Validation acc ',max(history.history['val_acc']))

#stop pruning if the acc drops by 5% from maximum acc ever obtained. 
validation_accuracy = max(history.history['val_acc'])
print("Initial Validation acc = {}".format(validation_accuracy) )
max_val_acc = validation_accuracy
count = 0
all_models = list()
a,b = count_model_params_flops(model,False)
print(a,b)
while validation_accuracy - max_val_acc >= -0.01:


    print("ITERATION {} ".format(count+1))
    all_models.append(model)
    if max_val_acc < validation_accuracy:
        max_val_acc = validation_accuracy
        

    if count < 1:
        optimize(model,weight_list_per_epoch,10,50,True)
        model = my_delete_filters(model,weight_list_per_epoch,50,True)
        model,history,weight_list_per_epoch = train(model,10,False)
   
    else:
        optimize(model,weight_list_per_epoch,10,30,False)
        model = my_delete_filters(model,weight_list_per_epoch,30,False)
        model,history,weight_list_per_epoch = train(model,20,False)

    a,b = count_model_params_flops(model,False)
    print(a,b)
    
    # al+=history
    validation_accuracy = max(history.history['val_acc'])
    best_acc_index = history.history['val_acc'].index(max(history.history['val_acc']))
    log_dict['train_loss'].append(history.history['loss'][best_acc_index])
    log_dict['train_acc'].append(history.history['acc'][best_acc_index])
    log_dict['val_loss'].append(history.history['val_loss'][best_acc_index])
    log_dict['val_acc'].append(history.history['val_acc'][best_acc_index])
    a,b = count_model_params_flops(model,False)
    log_dict['total_params'].append(a)
    log_dict['total_flops'].append(b)
    log_dict['filters_in_conv1'].append(model.layers[1].get_weights()[0].shape[-1])
    log_dict['filters_in_conv2'].append(model.layers[3].get_weights()[0].shape[-1])
    print("VALIDATION acc AFTER {} ITERATIONS = {}".format(count+1,validation_accuracy))
    count+=1

model.summary()

l1_norms = my_get_l1_norms_filters_per_epoch(weight_list_per_epoch)
distance_matrix_list = my_get_distance_matrix_list(l1_norms)
episodes_for_all_layers = my_get_episodes_for_all_layers(distance_matrix_list,95)
print(episodes_for_all_layers)
filter_pruning_indices = my_get_filter_pruning_indices(episodes_for_all_layers,l1_norms)
print(filter_pruning_indices[0],filter_pruning_indices[1])

optimize(model,weight_list_per_epoch,20,40,False)

# all_conv_layers = my_get_all_conv_layers(model,first_time)

surgeon = Surgeon(model)
surgeon.add_job('delete_channels',model.layers[1],channels = filter_pruning_indices[0][:1])
surgeon.add_job('delete_channels',model.layers[3],channels =filter_pruning_indices[1][:1])
model = surgeon.operate()

model.summary()

model,history,weight_list_per_epoch = train(model,60,False)

best_acc_index = history.history['val_acc'].index(max(history.history['val_acc']))
log_dict['train_loss'].append(history.history['loss'][best_acc_index])
log_dict['train_acc'].append(history.history['acc'][best_acc_index])
log_dict['val_loss'].append(history.history['val_loss'][best_acc_index])
log_dict['val_acc'].append(history.history['val_acc'][best_acc_index])
a,b = count_model_params_flops(model,False)
log_dict['total_params'].append(a)
log_dict['total_flops'].append(b)
log_dict['filters_in_conv1'].append(model.layers[1].get_weights()[0].shape[-1])
log_dict['filters_in_conv2'].append(model.layers[3].get_weights()[0].shape[-1])
print("Final Validation acc = ",(max(history.history['val_acc'])*100))
output_path = os.path.join('cosine_similarity', 'results', 'cosine_pruned_filters.txt')
with open(output_path, 'a') as f:
    f.write('\n Final Validation accuracy:  ' + str(max(history.history['val_acc'])*100) + '\n')


log_df = pd.DataFrame(log_dict)
log_df

log_df.to_csv(os.path.join('.', 'results', 'lenet5_2_results.csv'))

