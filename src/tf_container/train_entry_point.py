#!/usr/bin/env python3

import container_support as cs

import os
import json
import re
import zipfile
from keras.preprocessing.image import load_img, img_to_array
import numpy as np

from keras.layers import Input, Dense, merge
from keras.models import Model
from keras.layers import Convolution2D, MaxPooling2D, Reshape, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import callbacks
from tensorflow.python.client import device_lib

def train():
    env = cs.TrainingEnvironment()

    print(device_lib.list_local_devices())
    os.system('mkdir -p logs')

    # ### Loading the files ###
    # ** You need to copy all your files to the directory where you are runing this notebook into a folder named "data" **

    numbers = re.compile(r'(\d+)')
    data = []
    def get_data(root,f):
        d = json.load(open(os.path.join(root,f)))
        if ('pilot/throttle' in d):
            return [d['user/mode'],d['user/throttle'],d['user/angle'],root,d['cam/image_array'],d['pilot/throttle'],d['pilot/angle']]
        else:
            return [d['user/mode'],d['user/throttle'],d['user/angle'],root,d['cam/image_array']]
    def numericalSort(value):
        parts = numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
        return parts
    def unzip_file(root,f):
        zip_ref = zipfile.ZipFile(os.path.join(root,f), 'r')
        zip_ref.extractall(root)
        zip_ref.close()

    for root, dirs, files in os.walk('/opt/ml/input/data/train'):
        for f in files: 
            if f.endswith('.zip'):
                unzip_file(root, f)

    for root, dirs, files in os.walk('/opt/ml/input/data/train'):
        data.extend([get_data(root,f) for f in sorted(files, key=numericalSort) if f.startswith('record') and f.endswith('.json')])

    # Normalize / correct data
    data = [d for d in data if d[1] > 0.1]
    for d in data:
        if d[1] < 0.2:
            d[1] = 0.2

    # ### Loading throttle and angle ###

    angle = [d[2] for d in data]
    throttle = [d[1] for d in data]
    angle_array = np.array(angle)
    throttle_array = np.array(throttle)
    if (len(data[0]) > 5):
        pilot_angle = [d[6] for d in data]
        pilot_throttle = [d[5] for d in data]
        pilot_angle_array = np.array(pilot_angle)
        pilot_throttle_array = np.array(pilot_throttle)
    else:
        pilot_angle = []
        pilot_throttle = []


    # ### Loading images ###
    images = np.array([img_to_array(load_img(os.path.join(d[3],d[4]))) for d in data],'f')

    # slide images vs orders
    if env.hyperparameters.get('with_slide', False):
        images = images[:len(images)-2]
        angle_array = angle_array[2:]
        throttle_array = throttle_array[2:]

    # ### Start training ###
    def linear_bin(a):
        a = a + 1
        b = round(a / (2/14))
        arr = np.zeros(15)
        arr[int(b)] = 1
        return arr

    logs = callbacks.TensorBoard(log_dir='logs', histogram_freq=0, write_graph=True, write_images=True)
    save_best = callbacks.ModelCheckpoint('/opt/ml/model/model_cat', monitor='angle_out_loss', verbose=1, save_best_only=True, mode='min')
    early_stop = callbacks.EarlyStopping(monitor='angle_out_loss', 
                                                    min_delta=.0005, 
                                                    patience=10, 
                                                    verbose=1, 
                                                    mode='auto')
    img_in = Input(shape=(120, 160, 3), name='img_in')                      # First layer, input layer, Shape comes from camera.py resolution, RGB
    x = img_in
    x = Convolution2D(24, (5,5), strides=(2,2), activation='relu')(x)       # 24 features, 5 pixel x 5 pixel kernel (convolution, feauture) window, 2wx2h stride, relu activation
    x = Convolution2D(32, (5,5), strides=(2,2), activation='relu')(x)       # 32 features, 5px5p kernel window, 2wx2h stride, relu activatiion
    x = Convolution2D(64, (5,5), strides=(2,2), activation='relu')(x)       # 64 features, 5px5p kernal window, 2wx2h stride, relu
    x = Convolution2D(64, (3,3), strides=(2,2), activation='relu')(x)       # 64 features, 3px3p kernal window, 2wx2h stride, relu
    x = Convolution2D(64, (3,3), strides=(1,1), activation='relu')(x)       # 64 features, 3px3p kernal window, 1wx1h stride, relu

    # Possibly add MaxPooling (will make it less sensitive to position in image).  Camera angle fixed, so may not to be needed

    x = Flatten(name='flattened')(x)                                        # Flatten to 1D (Fully connected)
    x = Dense(100, activation='relu')(x)                                    # Classify the data into 100 features, make all negatives 0
    x = Dropout(.1)(x)
    x = Dense(50, activation='relu')(x)
    x = Dropout(.1)(x)                                                      # Randomly drop out 10% of the neurons (Prevent overfitting)
    #categorical output of the angle
    callbacks_list = [save_best, early_stop, logs]
    angle_out = Dense(15, activation='softmax', name='angle_out')(x)        # Connect every input with every output and output 15 hidden units. Use Softmax to give percentage. 15 categories and find best one based off percentage 0.0-1.0

    #continous output of throttle
    throttle_out = Dense(1, activation='relu', name='throttle_out')(x)      # Reduce to 1 number, Positive number only
    angle_cat_array = np.array([linear_bin(a) for a in angle_array])
    model = Model(inputs=[img_in], outputs=[angle_out, throttle_out])
    model.compile(optimizer='adam',
                loss={'angle_out': 'categorical_crossentropy', 
                        'throttle_out': 'mean_absolute_error'},
                loss_weights={'angle_out': 0.9, 'throttle_out': .001})
    model.fit({'img_in':images},{'angle_out': angle_cat_array, 'throttle_out': throttle_array}, batch_size=32, epochs=100, verbose=1, validation_split=0.2, shuffle=True, callbacks=callbacks_list)
