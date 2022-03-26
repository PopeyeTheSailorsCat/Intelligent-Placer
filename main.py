import os
from skimage.color import rgb2gray, gray2rgb
import matplotlib.pyplot as plt
import numpy as np

import config
from data_handler import get_example_data
from detection import get_object_and_figure_boxes, cut_objects_from_image, cut_figure
from imageio import imread
from classifier import get_best_model
import cv2 as cv
from utils import patch_rectangle


def cut_figure_example():  # example of cutting figure structure
    data = get_example_data()
    img = data[3]
    edge_boxes, objects_boxes = get_object_and_figure_boxes(rgb2gray(img))
    plt.imshow(img)
    plt.show()
    figure = cut_figure(img, edge_boxes)
    plt.imshow(figure, cmap='gray')


def classify_objects(cut_objects, classes=8):  # interface of using CNN
    model = get_best_model(classes)  # get best model from directory
    res = model.predict(np.array(cut_objects))
    return res


def extract_fig_and_objects(img, show_boxes=False):
    """
    This function , find figure location  and extract figure structure
    Also it finds, extract and classifies objects.
    :param img:
    :param show_boxes:
    :return: figure - figure structure in matrix,
    :return:  figure edge_boxes location on image
    :return: class_objects - matrix(num_on_img, num_in_task) where every row is line with probabilities of classes
    """
    edge_boxes, objects_boxes = get_object_and_figure_boxes(rgb2gray(img))  # get bboxes for figure and objects
    if show_boxes:
        fig, ax = plt.subplots()
        ax.imshow(img)
        for box in objects_boxes:
            ax.add_patch(patch_rectangle(box, 'r', img.shape))

        # TODO check we only have one figure
        for box in edge_boxes:
            ax.add_patch(ax.add_patch(patch_rectangle(box, 'b',img.shape)))
        plt.show()
    cut_objects = cut_objects_from_image(img, objects_boxes)  # get objects images from bboxes
    figure = cut_figure(img, edge_boxes)  # get figure structure from its bboxes(find edges and fill holes)
    class_objects = classify_objects(cut_objects)  # using CNN to classify object on images
    return figure, edge_boxes[:2], class_objects


def extract_fig_and_objects_example():  # example for figure and object classification function
    data = get_example_data()
    indx_to_name = config.indx_to_object_name
    for elem in data:
        figure, _, class_objects = extract_fig_and_objects(elem, show_boxes=True)
        plt.imshow(figure, cmap='gray')
        plt.show()
        for obj in class_objects:
            obj_indx = np.argmax(obj)
            print(round(obj[obj_indx], 3), indx_to_name[obj_indx])


def slide_obj_over_fig(main_figure, obj, object_area):  # Brute force solution
    # just slide object structure image above figure image to find place, where we can place object.
    fig_y, fig_x = main_figure.shape
    obj_y, obj_x = obj.shape
    y_grid, x_grid = config.grid_search_y_positions, config.grid_search_x_positions
    for pos_y in range(0, fig_y - obj_y, y_grid):  # grid for y
        for pos_x in range(0, fig_x - obj_x, x_grid):  # grid for x
            roi = main_figure[pos_y:pos_y + obj_y, pos_x:pos_x + obj_x].astype(int)  # take area from figure
            intersect = cv.bitwise_and(roi, obj.astype(int))  # check how we can place object
            if np.sum(intersect) == object_area:  # confirm we can place whole object
                main_figure[pos_y:pos_y + obj_y, pos_x:pos_x + obj_x] = cv.bitwise_and(roi, 255 - obj)
                # paint object on figure, so we can't use this area again
                # we find location for this object on this figure, so we need to return location on figure and
                # figure without used area
                return (pos_y, pos_x), main_figure

    return (-1, -1), main_figure  # if we don't find place


def run_epoch_stuff(path=config.easy_test_img):  # function to show example working for first step

    figures = config.objects_figures_folder
    objects_struct_paths = config.objects_structure_files  # files with "good" object images

    img = imread(path)
    indx_to_name = config.indx_to_object_name
    figure, fig_location, class_objects = extract_fig_and_objects(img, show_boxes=True)

    plt.imshow(figure, cmap='gray')
    plt.show()
    objects_structures = []

    for obj in class_objects:  # iteration though CNN prediction about class
        obj_indx = np.argmax(obj)
        struct = imread(os.path.join(figures, objects_struct_paths[obj_indx]))  # getting this object "good" struct
        plt.imshow(struct, cmap='gray')
        plt.show()
        struct = struct / 255 > 0  # this make me feel safe about mask. Some of them I fix in Paint...
        objects_structures.append(struct)
        print(round(obj[obj_indx], 3), indx_to_name[obj_indx])  # show our confidence in prediction
        # TODO work bad confidence cases

    object_areas = []
    for elem in objects_structures:  # getting area of objects, maybe TODO deal with big thirst
        area = np.sum(elem)
        object_areas.append(area)

    object_loc = []  # TODO REWORK WHOLE
    for obj, area in zip(objects_structures, object_areas):  # now we try to put objects inside figure
        (y, x), figure = slide_obj_over_fig(figure, obj, area)
        object_loc.append([y, x])
        if x != -1:  # cant put object inside.
            plt.imshow(figure, cmap='gray')
            plt.show()
        else:
            print("nope")

    y_fig, x_fig, *_ = fig_location[0]
    fig_y, fig_x = figure.shape
    img[y_fig:y_fig + fig_y, x_fig:x_fig + fig_x, :] = gray2rgb(255 * figure)  # plot results on image
    plt.imshow(img)
    plt.show()


run_epoch_stuff()

# extract_fig_and_objects_example()
