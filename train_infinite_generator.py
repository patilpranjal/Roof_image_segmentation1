# coding: utf-8
'''
    - train "ZF_UNET_224" CNN with random images
'''

__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import os
import cv2
import random
import numpy as np
import pandas as pd
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import __version__
from zf_unet_224_model import *
#import matplotlib("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import PIL
from PIL import Image
from numpy import array
import scipy
from skimage import io



data_path = "airs-dataset/"

def random_selector(data_path):
    train_data_path = os.path.join(data_path, 'train-input')
    train_data_path1 = os.path.join(data_path, 'train-output')
    images = os.listdir(train_data_path)
    mask = os.listdir(train_data_path1)
    total = len(images)
    a = random.randint(0, total-1)


    img =  cv2.imread(os.path.join(train_data_path, images[a]))
    mask = cv2.imread(os.path.join(train_data_path1, mask[a]), cv2.IMREAD_GRAYSCALE)


    x = random.randint(0, 1276)
    y = random.randint(0, 1276)
    res = img[x:x+224, y:y+224,:]
    res1 = mask[x:x+224, y:y+224]


    return res, res1


def batch_generator(batch_size):
    while True:
        image_list = []
        mask_list = []
        for i in range(batch_size):
            img, mask = random_selector(data_path)
            image_list.append(img)
            mask_list.append([mask])

        image_list = np.array(image_list, dtype=np.float32)
        if K.image_dim_ordering() == 'th':
            image_list = image_list.transpose((0, 3, 1, 2))
        image_list = preprocess_input(image_list)
        mask_list = np.array(mask_list, dtype=np.float32)
        mask_list /= 76
        mask_list = np.transpose(mask_list,(0,2, 3, 1))


        yield image_list, mask_list


def train_unet():
    out_model_path = 'zf_unet_224.h5'
    epochs = 50
    patience = 20
    batch_size = 16
    optim_type = 'Adam'
    learning_rate = 0.001
    model = ZF_UNET_224()
    if os.path.isfile(out_model_path):
        model.load_weights(out_model_path)


    optim = Adam(lr=learning_rate)
    model.compile(optimizer=optim, loss="binary_crossentropy", metrics=["accuracy"])


    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-9, epsilon=0.00001, verbose=1, mode='min'),
        EarlyStopping(monitor='val_loss', patience=patience, verbose=0),
        ModelCheckpoint('zf_unet_224_temp.h5', monitor='val_loss', save_best_only=True, verbose=0),
    ]

    print('Start training...')
    history = model.fit_generator(
        generator=batch_generator(batch_size),
        epochs=epochs,
        steps_per_epoch= 100,
        validation_data=batch_generator(batch_size),
        validation_steps=100,
        verbose=2,
        callbacks=callbacks)

    model.save_weights(out_model_path)
    pd.DataFrame(history.history).to_csv('zf_unet_224_train.csv', index=False)
    print('Training is finished (weights zf_unet_224.h5 and log zf_unet_224_train.csv are generated )...')

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

if __name__ == '__main__':
    if K.backend() == 'tensorflow':
        try:
            from tensorflow import __version__ as __tensorflow_version__
            print('Tensorflow version: {}'.format(__tensorflow_version__))
        except:
            print('Tensorflow is unavailable...')
    else:
        try:
            from theano.version import version as __theano_version__
            print('Theano version: {}'.format(__theano_version__))
        except:
            print('Theano is unavailable...')
    print('Keras version {}'.format(__version__))
    print('Dim ordering:', K.image_dim_ordering())
    train_unet()

