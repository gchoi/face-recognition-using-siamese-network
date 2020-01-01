#%% Import libraries
import os
import cv2
import numpy as np
from PIL import Image
from mtcnn.mtcnn import MTCNN
from numpy import asarray


#%% function: read_image()
def read_image(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    return img


#%% function: get_image()
def get_image(data_path, category, index):
    pathlist = os.listdir(data_path)
    if category > len(pathlist):
        return None
    
    cur_path = os.path.join(data_path, pathlist[category])
    
    filelist = os.listdir(cur_path)
    if index > len(filelist):
        return None
    
    filepath = os.path.join(cur_path, filelist[index])
    
    image = read_image(filepath)
    size = 2
    image = image[::size, ::size]
    image = image.reshape(1, 1, image.shape[0], image.shape[1])
    image = image / 255
    
    return image


#%% function: get_image_from_filename()
def get_image_from_filename(img_filename):
    image = read_image(img_filename)
    size = 2
    image = image[::size, ::size]
    image = image.reshape(1, 1, image.shape[0], image.shape[1])
    image = image / 255

    return image


#%% function: get_data()
def get_data(data_path, size=2, total_sample_size=10000):
    sub_paths = os.listdir(data_path)
    filename = os.listdir(os.path.join(data_path, sub_paths[0]))[0]
    
    # read the image
    image = read_image(os.path.join(data_path, sub_paths[0], filename))
    
    # reduce the size
    image = image[::size, ::size]
    
    # get the new size
    dim1 = image.shape[0]
    dim2 = image.shape[1]
    
    # initialize the numpy array with the shape of [total_sample, no_of_pairs, dim1, dim2]
    x_similar_pair = np.zeros([total_sample_size, 2, 1, dim1, dim2]) # 2 is for pairs
    y_similar = np.zeros([total_sample_size, 1])
    
    print("\n......... NOW MAKING SIMILAR PAIRS .........\n")
    count = 0
    for i in range(len(sub_paths)):
        for j in range(int(total_sample_size / len(sub_paths))):
            cur_path = os.path.join(data_path, sub_paths[i])
            filelist = os.listdir(cur_path)
            
            ind1 = 0
            ind2 = 0            
            
            # read images from same directory (similar pair)
            while ind1 == ind2:
                ind1 = np.random.randint(len(filelist))
                ind2 = np.random.randint(len(filelist))
                
            # read the two images
            img1 = read_image(os.path.join(cur_path, filelist[ind1]))
            img2 = read_image(os.path.join(cur_path, filelist[ind2]))
            
            # reduce the size
            img1 = img1[::size, ::size]
            img2 = img2[::size, ::size]
            
            # store the images to the initialized numpy array
            x_similar_pair[count, 0, 0, :, :] = img1
            x_similar_pair[count, 1, 0, :, :] = img2
            
            # as we are drawing images from the same directory we assign label as 1. (similar pair)
            y_similar[count] = 1
            count += 1
            
            # display the progress
            if count % 500 == 0:
                print("[SIMILAR PAIRS] %d / %d" % (count, total_sample_size))
    
    print("[SIMILAR PAIRS] %d / %d" % (total_sample_size, total_sample_size))
    
    # initialize the numpy array with the shape of [total_sample, no_of_pairs, dim1, dim2]
    x_dissimilar_pair = np.zeros([total_sample_size, 2, 1, dim1, dim2])
    y_dissimilar = np.zeros([total_sample_size, 1])
    
    print("\n......... NOW MAKING DISSIMILAR PAIRS .........\n")
    pathlist = os.listdir(data_path)
    count = 0
    for i in range(int(total_sample_size / len(sub_paths))):
        for j in range(len(sub_paths)):
            ind1_1 = 0
            ind1_2 = 0
            
            # read images from different directory (dissimilar pair)
            while ind1_1 == ind1_2:
                ind1_1 = np.random.randint(len(pathlist))
                ind1_2 = np.random.randint(len(pathlist))
            
            flist1 = os.listdir(os.path.join(data_path, pathlist[ind1_1]))
            flist2 = os.listdir(os.path.join(data_path, pathlist[ind1_2]))
            
            ind2_1 = np.random.randint(len(flist1))
            ind2_2 = np.random.randint(len(flist2))
            
            img1 = read_image(os.path.join(data_path, pathlist[ind1_1], flist1[ind2_1]))
            img2 = read_image(os.path.join(data_path, pathlist[ind1_2], flist2[ind2_2]))
            
            img1 = img1[::size, ::size]
            img2 = img2[::size, ::size]
            
            x_dissimilar_pair[count, 0, 0, :, :] = img1
            x_dissimilar_pair[count, 1, 0, :, :] = img2
            
            # as we are drawing images from the different directory we assign label as 0. (imposite pair)
            y_dissimilar[count] = 0
            count += 1
            
            # display the progress
            if count % 500 == 0:
                print("[DISSIMILAR PAIRS] %d / %d" % (count, total_sample_size))
            
    print("[DISSIMILAR PAIRS] %d / %d" % (total_sample_size, total_sample_size))
    
    ## concatenate, similar pairs and dissimlar pair to get the whole data
    ## and save as numpy data
    X = np.concatenate([x_similar_pair, x_dissimilar_pair], axis=0) / 255
    Y = np.concatenate([y_similar, y_dissimilar], axis=0)

    return X, Y


#%% function: extract_face()
# extract a single face from a given photograph
def extract_face(filename, required_size=(160, 160)):
	# load image from file
	image = Image.open(filename)

  # convert to RGB, if needed
	image = image.convert('RGB')

  # convert to array
	pixels = asarray(image)

  # create the detector, using default weights
	detector = MTCNN()

  # detect faces in the image
	results = detector.detect_faces(pixels)

  # extract the bounding box from the first face
	x1, y1, width, height = results[0]['box']

  # bug fix
	x1, y1 = abs(x1), abs(y1)
	x2, y2 = x1 + width, y1 + height

  # extract the face
	face = pixels[y1:y2, x1:x2]

  # resize pixels to the model size
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)

	return face_array