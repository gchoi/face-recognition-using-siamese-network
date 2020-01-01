#%% Import libraries
import os
import numpy as np
from keras.models import load_model
import siamese_network as SN
import helper


#%% Define some params
DATA_PATH = "./dataset/faces"
MODEL_PATH = './models'
MODEL_NAME = 'siamese-face-model.h5'
NUM_TRIALS = 500 ## Trials for Testing Accuracy


#%% Main
## load the model
model = load_model(os.path.join(MODEL_PATH, MODEL_NAME),
                   custom_objects={'contrastive_loss': SN.contrastive_loss})

## test with many randomly selected images
matching = 0
for tiral in range(NUM_TRIALS):
    print("Now trial #%d of %d" % (tiral, NUM_TRIALS))

    pathlist = os.listdir(DATA_PATH)
    category = np.random.randint(len(pathlist))
    
    cur_path = os.path.join(DATA_PATH, pathlist[category])
    filelist = os.listdir(cur_path)
    index = np.random.randint(len(filelist))

    ref_image = helper.get_image(DATA_PATH, category, index)
    
    results = []
    for cat in range(len(pathlist)):
        filelist = os.listdir(os.path.join(DATA_PATH, pathlist[cat]))
        idx = np.random.randint(len(filelist))
        cur_image = helper.get_image(DATA_PATH, cat, idx)
        
        dist = model.predict([ref_image, cur_image])[0][0]
        results.append(dist)
    
    if category == np.argmin(results):
        matching += 1

print("Accuracy: %5.2f %%\n" % (100.0 * matching / NUM_TRIALS))

## select an image randomly (with only 1 image)
print("\n.... Now predict with the randomly selected image ....")
pathlist = os.listdir(DATA_PATH)
category = np.random.randint(len(pathlist))
cur_path = os.path.join(DATA_PATH, pathlist[category])
filelist = os.listdir(cur_path)
index = np.random.randint(len(filelist))

ref_image = helper.get_image(DATA_PATH, category, index)

results = []
for cat in range(len(pathlist)):
    filelist = os.listdir(os.path.join(DATA_PATH, pathlist[cat]))
    idx = np.random.randint(len(filelist))
    cur_image = helper.get_image(DATA_PATH, cat, idx)
    
    dist = model.predict([ref_image, cur_image])[0][0]
    results.append(dist)

print("Selected Category: %d" % (category))
print("Predicted Category: %d" % (np.argmin(results)))