#%% Import libraries
import os
import numpy as np

from sklearn.model_selection import train_test_split
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
import siamese_network as SN

import matplotlib.pyplot as plt


#%% Define some params
DATA_PATH = './dataset/numpy'
X_data_name = 'X_data.npy'
Y_data_name = 'Y_data.npy'

CHKPT_PATH = './checkpoint'
CHK_PT_NAME = 'chkpt-model.ckpt'

MODEL_PATH = './models'
MODEL_FILENAME = 'siamese-face-model.h5'

TEST_SIZE = .25
NUM_EPOCHS = 50
BATCH_SIZE = 128
PATIENCE = 10


#%% Main
X = np.load(os.path.join(DATA_PATH, X_data_name))
Y = np.load(os.path.join(DATA_PATH, Y_data_name))

print(X.shape)
print(Y.shape)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=TEST_SIZE)

input_dim = x_train.shape[2:]
print(len(input_dim))

img_a = Input(shape=input_dim)
img_b = Input(shape=input_dim)

base_network = SN.build_base_network(input_dim)
base_network.summary()

feat_vecs_a = base_network(img_a)
feat_vecs_b = base_network(img_b)

print(feat_vecs_a)
print(feat_vecs_b)

distance = Lambda(SN.euclidean_distance,
                  output_shape=SN.eucl_dist_output_shape)([feat_vecs_a, feat_vecs_b])


#%% Train
earlystopper = EarlyStopping(patience=PATIENCE, verbose=1)

if not os.path.exists(CHKPT_PATH):
  os.mkdir(CHKPT_PATH)
  
checkpt_path = os.path.join(CHKPT_PATH, CHK_PT_NAME)
print('Check Point File Path: %s' % (checkpt_path))

checkpointer = ModelCheckpoint(checkpt_path, verbose=1, save_best_only=True)

rms = RMSprop()
model = Model(inputs=[img_a, img_b], outputs=distance)
model.summary()

model.compile(loss=SN.contrastive_loss, optimizer=rms)

img_1 = x_train[:, 0]
img_2 = x_train[:, 1] 

# elapsed time checking
import time
start_time = time.time()

hist = model.fit([img_1, img_2],
                 y_train,
                 validation_split=TEST_SIZE,
                 batch_size=BATCH_SIZE,
                 verbose=2,
                 epochs=NUM_EPOCHS,
                 callbacks=[earlystopper, checkpointer])

print("\n\n Elapsed Time for Training: %s seconds ---" % (time.time() - start_time))


#%% Save the trained model
if not os.path.exists(MODEL_PATH):
  os.mkdir(MODEL_PATH)

model.save(os.path.join(MODEL_PATH, MODEL_FILENAME))


#%% Plot the Change in Loss over Epochs
for key in ['loss', 'val_loss']:
  plt.plot(hist.history[key], label=key)

plt.legend()
plt.show()

pred = model.predict([x_test[:, 0], x_test[:, 1]])

SN.compute_accuracy(pred, y_test)