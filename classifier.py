from keras.models import Sequential
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras import Model, Input
from keras.layers import (Conv2D, MaxPooling2D, Flatten, Reshape, GlobalMaxPooling2D,
                          Activation)

import config
from detection import get_object_boxes, cut_objects_from_image
import matplotlib.pyplot as plt
import numpy as np
from data_handler import get_example_data
from skimage.color import rgb2gray


def make_model(num_classes):  # using basic model, maybeTODO go to VGG format
    imodel = Sequential()
    imodel.add(Reshape((128, 128, 3), input_shape=(128, 128, 3)))
    imodel.add(Conv2D(4, kernel_size=(3, 3), strides=(1, 1),
                      activation='relu', padding='same'))
    imodel.add(BatchNormalization())
    imodel.add(MaxPooling2D())

    imodel.add(Conv2D(8, kernel_size=(3, 3), strides=(1, 1),
                      activation='relu', padding='same'))
    imodel.add(BatchNormalization())
    imodel.add(MaxPooling2D())
    imodel.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1),
                      activation='relu', padding='same'))
    imodel.add(BatchNormalization())
    imodel.add(MaxPooling2D())
    imodel.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
                      activation='relu', padding='same'))
    imodel.add(BatchNormalization())
    imodel.add(MaxPooling2D())
    imodel.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1),
                      activation='relu', padding='same'))
    imodel.add(BatchNormalization())
    imodel.add(MaxPooling2D())
    imodel.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1),
                      activation='relu', padding='same'))
    imodel.add(BatchNormalization())
    imodel.add(Flatten())

    iclf = Sequential()
    iclf.add(imodel)
    iclf.add(Dense(num_classes, activation='softmax'))

    iclf.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])

    return iclf


def get_best_model(classes=8):  # upload best model from repo
    model = make_model(num_classes=classes)
    model.load_weights('model\\best').expect_partial()
    return model


def run_classify_example(classes=8):  # run example of objects classification
    model = make_model(num_classes=classes)
    model.load_weights('best').expect_partial()
    data = get_example_data()
    datas = data[1:]
    indx_to_name = config.indx_to_object_name
    for img in datas:
        plt.imshow(img)
        plt.show()
        objects_boxes = get_object_boxes(rgb2gray(img))
        objects = cut_objects_from_image(img, objects_boxes)
        res = model.predict(np.array(objects))
        for indx, obj in enumerate(objects):
            plt.imshow(obj)
            plt.show()
            obj_indx = np.argmax(res[indx])
            print(res[indx], indx_to_name[obj_indx])


