import numpy as np
import pandas as pd

import matplotlib.pyplot as plt 

import cv2

import os

NUM_CLASSES = 16

# Fixed for Cats & Dogs color images
CHANNELS = 3

IMAGE_RESIZE = 224
RESNET50_POOLING_AVERAGE = 'avg'
DENSE_LAYER_ACTIVATION = 'softmax'
OBJECTIVE_FUNCTION = 'categorical_crossentropy'

# Common accuracy metric for all outputs, but can use different metrics for different output
LOSS_METRICS = ['accuracy']

# EARLY_STOP_PATIENCE must be < NUM_EPOCHS
NUM_EPOCHS = 3
EARLY_STOP_PATIENCE = 3

# These steps value should be proper FACTOR of no.-of-images in train & valid folders respectively
# Training images processed in each step would be no.-of-train-images / STEPS_PER_EPOCH_TRAINING
STEPS_PER_EPOCH_TRAINING = 10
STEPS_PER_EPOCH_VALIDATION = 10

# These steps value should be proper FACTOR of no.-of-images in train & valid folders respectively
# NOTE that these BATCH* are for Keras ImageDataGenerator batching to fill epoch step input
BATCH_SIZE_TRAINING = 100
BATCH_SIZE_VALIDATION = 100

# Using 1 to easily manage mapping between test_generator & prediction for submission preparation
BATCH_SIZE_TESTING = 1

from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense


resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

#Still not talking about our train/test data or any pre-processing.

model = Sequential()

# 1st layer as the lumpsum weights from resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
# NOTE that this layer will be set below as NOT TRAINABLE, i.e., use it as is
model.add(ResNet50(include_top = False, pooling = RESNET50_POOLING_AVERAGE, weights = resnet_weights_path))

# model.add(Dense(256,activation="relu"))
# 2nd layer as Dense for 2-class classification, i.e., dog or cat using SoftMax activation
model.add(Dense(NUM_CLASSES, activation = DENSE_LAYER_ACTIVATION))

# Say not to train first layer (ResNet) model as it is already trained
model.layers[0].trainable = False

model.summary()

from tensorflow.python.keras import optimizers

sgd = optimizers.SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
model.compile(optimizer = sgd, loss = OBJECTIVE_FUNCTION, metrics = LOSS_METRICS)

from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator

image_height = 200
image_width = 300

data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

model.load_weights("../working/best.hdf5")

test_generator = data_generator.flow_from_directory(
    directory = '../dataset/sample_test/',
    target_size = (image_height, image_width),
    batch_size = BATCH_SIZE_TESTING,
    class_mode = None,
    shuffle = False,
    seed = 123
)

test_generator.reset()
pred = model.predict_generator(test_generator, steps = len(test_generator), verbose = 1)
print(pred)

import scipy.stats as ss

pred2 = np.argsort(-pred, axis=1) #descending order of labels

print(pred2)

classDict= {0:'3m_high_tack_spray_adhesive',
    1:'aunt_jemima_original_syrup',
    2:'campbells_chicken_noodle_soup',
    3:'cheez_it_white_cheddar',
    4:'cholula_chipotle_hot_sauce',
    5:'clif_crunch_chocolate_chip',
    6:'coca_cola_glass_bottle',
    7:'detergent',
    8:'expo_marker_red',
    9:'listerine_green',
    10:'nice_honey_roasted_almonds',
    11:'nutrigrain_apple_cinnamon',
    12:'palmolive_green',
    13:'pringles_bbq',
    14:'vo5_extra_body_volumizing_shampoo',
    15:'vo5_split_ends_anti_breakage_shampoo'}

i=0

ImageNames=np.load("Image_Name_resnet.npy")
for fname in test_generator.filenames:
    fname2 = '../results/' + fname.split('test/')[1].split('.jpg')[0] + '.txt'
    fp = open(fname2, 'w')
    for j in pred2[i]:
        folderName = classDict[j]
        # for file_ in os.listdir('../dataset/train/' + folderName):
        for file_ in ImageNames[j]:
            fp.write(folderName + '_' + file_ + '\n')
    fp.close()
    i+=1

fp.close()