#%% Import packages
import os
from PIL import Image
import helper


#%% Deine some params
SOURCE_BASE_PATH = "./downloads"
TARGET_BASE_PATH = "./test/extracted-faces"


#%% Main
idx = 0
for cur_path in os.listdir(SOURCE_BASE_PATH):
    src_sub_path = os.path.join(SOURCE_BASE_PATH, cur_path)
    tgt_sub_path = os.path.join(TARGET_BASE_PATH, cur_path)
    
    if not os.path.isdir(tgt_sub_path):
        os.mkdir(tgt_sub_path)
    
    jdx = 0
    for filename in os.listdir(src_sub_path):
        print("Progress - idx: %s / %03d" % (cur_path, jdx))
        src_file_path = os.path.join(src_sub_path, filename)
        pixels = helper.extract_face(src_file_path)
        img = Image.fromarray(pixels, mode='RGB')
        img.save(os.path.join(tgt_sub_path, filename))
        
        jdx += 1
    
    idx += 1