from keras import Model, Input
from keras.layers import (Conv2D, MaxPooling2D, Flatten, Reshape, GlobalMaxPooling2D,
                          Activation)
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from skimage.color import rgb2gray

from data_handler import get_train_data, get_example_data
from detection import get_object_boxes, cut_objects_from_image


def make_fashion_model(num_classes):  # using basic model, maybeTODO go to VGG format
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


def plot_accuracy(history):
    acc = 'acc' if 'acc' in history else 'accuracy'
    val_acc = 'val_' + acc

    plt.plot(range(len(history[acc])), history[acc], color='b')
    plt.plot(range(len(history[val_acc])), history[val_acc], color='g')

    return history[val_acc][-1]


def train_model():  # using augmentation
    aug = ImageDataGenerator(featurewise_center=False,
                             samplewise_center=False,
                             featurewise_std_normalization=False,
                             samplewise_std_normalization=False,
                             rotation_range=60,
                             width_shift_range=0,
                             height_shift_range=0,
                             brightness_range=None,
                             shear_range=0.0,
                             zoom_range=[0.9, 1.05],
                             channel_shift_range=0.0,
                             fill_mode='nearest',
                             cval=0.0,
                             horizontal_flip=False,
                             vertical_flip=False,
                             rescale=None)
    batch_size = 4
    X_train, X_test, y_train, y_test, classes = get_train_data()
    fashion_clf = make_fashion_model(num_classes=classes)  # TODO rename all model stuff from lab3

    train_X_train = X_train.copy()
    aug.fit(train_X_train)

    gen = aug.flow(train_X_train, y_train,
                   batch_size=batch_size)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='best',
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    hist = fashion_clf.fit(gen,
                           steps_per_epoch=len(train_X_train) // batch_size,
                           epochs=50,
                           validation_data=(X_test, y_test), callbacks=[model_checkpoint_callback])
    plot_accuracy(hist.history)
    print(max(hist.history['val_accuracy']))


def get_best_model(classes=8):  # upload best model from repo
    model = make_fashion_model(num_classes=classes)
    model.load_weights('best').expect_partial()
    return model


def run_classify_example(classes=8):  # run example of objects classification
    fashion_clf = make_fashion_model(num_classes=classes)
    fashion_clf.load_weights('best').expect_partial()
    data = get_example_data()
    datas = data[1:]
    for img in datas:
        plt.imshow(img)
        plt.show()
        objects_boxes = get_object_boxes(rgb2gray(img))
        objects = cut_objects_from_image(img, objects_boxes)
        res = fashion_clf.predict(np.array(objects))
        for indx, obj in enumerate(objects):
            plt.imshow(obj)
            plt.show()
            obj_indx = np.argmax(res[indx])
            print(res[indx], indx_to_name[obj_indx])


indx_to_name = {0: "Значок", 1: "Пульт", 2: "Зажигалка", 3: "Медиатор", 4: "Шахматный конь", 5: "Крышка",
                6: "Кубик", 7: "Батарейка"}
