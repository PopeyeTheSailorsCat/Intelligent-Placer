from sklearn.model_selection import train_test_split
import os
from imageio import imread, imsave
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray


def show_example_data(data):
    fix, ax = plt.subplots(4, 3, figsize=(30, 30))
    for i, elem in enumerate(data):
        ax.flat[i].imshow(elem)


def get_example_data(show_data=False):
    example_data = []
    path = 'inputs_example'
    for file in os.listdir(path):
        img = imread(os.path.join(path, file))
        example_data.append(img)
    if show_data:
        show_example_data(example_data)
    return example_data


def get_path_data(path, show_data=False):  # get images from directory
    example_data = []
    path = path
    for file in os.listdir(path):
        img = imread(os.path.join(path, file))
        example_data.append(img)
    if show_data:
        show_example_data(example_data)
    return example_data


def generate_train_img():  # using for CNN training.
    # find, cut and resize images for CNN training later.
    from detection import cut_objects_from_image
    from detection import get_object_boxes
    # Генерируем тренировочные наборы
    cls_path = 'classifyer_imgs'
    path = "6"
    cut_path = "6_cut"
    data = get_path_data(os.path.join(cls_path, path))
    print(len(data))
    for indx, img in enumerate(data):
        plt.imshow(img)
        plt.show()
        objects_boxes = get_object_boxes(rgb2gray(img))
        objects = cut_objects_from_image(img, objects_boxes)
        for j, obj in enumerate(objects):
            imsave(os.path.join(cut_path, f"{indx}_{j}.jpg"), obj)


def get_train_data():  # get training data from repos for CNN training
    X = []
    Y = []
    cls_path = 'classifyer_imgs'
    folders = ['1_cut', "2_cut", "3_cut", "4_cut", "5_cut", "6_cut", "7_cut", "8_cut"]

    for indx, folder in enumerate(folders):
        for file in os.listdir(os.path.join(cls_path, folder)):
            img = imread(os.path.join(cls_path, folder, file))
            X.append(img)
            Y.append(create_vector(len(folders), indx))

    # count how many images of each class we have
    from collections import Counter
    counter = Counter()
    for vector in Y:
        counter[np.argmax(vector)] += 1

    print(counter)
    X_train, X_test, y_train, y_test = train_test_split(  # TODO get this to classifier
        np.array(X), np.array(Y), test_size=0.4, shuffle=True)
    return X_train, X_test, y_train, y_test, len(folders)


def create_vector(size, elem):  # for training CNN data creation. Create onehot vector.
    arr = np.zeros(size)
    arr[elem] = 1
    return arr


def create_objects_structure():  # using for good objects structure creation
    from detection import get_object_boxes, cut_figure
    path = "imgs_for_structure"
    save_path = "objects_figure"

    for indx, file in enumerate(os.listdir(path)):
        img = imread(os.path.join(path, file))
        objects_boxes = get_object_boxes(rgb2gray(img))
        for obj in objects_boxes:
            res = cut_figure(img, objects_boxes)
            plt.imshow(res, cmap='gray')
            plt.show()
            imsave(os.path.join(save_path, f"{indx}.jpg"), 255 * res.astype("uint8"))
