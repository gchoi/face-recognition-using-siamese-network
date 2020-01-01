#%% Import libraries
from keras import backend as K
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.models import Sequential


#%% function: build_base_network()
def build_base_network(input_shape):
  seq = Sequential()

  nb_filter = [6, 12]
  kernel_size = 3
  
  #convolutional layer 1
  seq.add(Conv2D(nb_filter[0],
                 (kernel_size, kernel_size),
                 input_shape=input_shape,
                 padding='valid',
                 data_format="channels_first"))
  seq.add(Activation('relu'))
  seq.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first")) 
  seq.add(Dropout(.25))

  #convolutional layer 2
  seq.add(Conv2D(nb_filter[1],
                 (kernel_size, kernel_size),
                 input_shape=input_shape,
                 padding='valid',
                 data_format="channels_first"))
  seq.add(Activation('relu'))
  seq.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))
  seq.add(Dropout(.25))

  #flatten 
  seq.add(Flatten())
  seq.add(Dense(128, activation='relu'))
  seq.add(Dropout(0.1))
  seq.add(Dense(50, activation='relu'))
  
  return seq


#%% function: euclidean_distance()
def euclidean_distance(vects):
  x, y = vects
  return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


#%% function: eucl_dist_output_shape()
def eucl_dist_output_shape(shapes):
  shape1, shape2 = shapes
  return (shape1[0], 1)


#%% function: contrastive_loss()
def contrastive_loss(y_true, y_pred):
  margin = 1
  return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


#%% function: compute_accuracy()
def compute_accuracy(predictions, labels):
  return labels[predictions.ravel() < 0.5].mean()