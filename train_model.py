import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from data_handler import get_train_data
from classifier import make_model
from sklearn.model_selection import train_test_split
import numpy as np


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
    X, Y, classes = get_train_data()
    X_train, X_test, y_train, y_test = train_test_split(
        np.array(X), np.array(Y), test_size=0.4, shuffle=True)
    model = make_model(num_classes=classes)

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

    hist = model.fit(gen,
                     steps_per_epoch=len(train_X_train) // batch_size,
                     epochs=50,
                     validation_data=(X_test, y_test), callbacks=[model_checkpoint_callback])
    plot_accuracy(hist.history)
    print(max(hist.history['val_accuracy']))
