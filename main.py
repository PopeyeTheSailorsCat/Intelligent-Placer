import os

from skimage.color import rgb2gray
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from data_handler import get_example_data
from detection import get_object_and_figure_boxes, cut_objects_from_image, cut_figure
from imageio import imread
import cv2 as cv


def cut_figure_example():  # example of cutting figure structure
    data = get_example_data()
    img = data[3]
    edge_boxes, objects_boxes = get_object_and_figure_boxes(rgb2gray(img))
    plt.imshow(img)
    plt.show()
    figure = cut_figure(img, edge_boxes)
    plt.imshow(figure, cmap='gray')


def classify_objects(cut_objects, classes=8):  # interface of using CNN
    from classyfier import get_best_model
    model = get_best_model(classes)  # get best model from directory # TODO add best model to repo
    res = model.predict(np.array(cut_objects))
    return res


def extract_fig_and_objects(img, show_boxes=False):
    """
    This function , find figure location  and extract figure structure
    Also it finds, extract and classifies objects. Maybe too much for one func # TODO split function
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
            min_row, min_col, max_row, max_col = box
            # max used so we don't paint rectangle outside of image
            # (max(min_col - 20, 0) mean that we want to see some area around in case we lose some part of object
            # max_row - min_row + 50 add another area below and right of boxes
            ax.add_patch(patches.Rectangle((max(min_col - 20, 0), max(min_row - 20, 0)), max_col - min_col + 30,
                                           max_row - min_row + 50, linewidth=1, edgecolor='r', facecolor='none'))
        # TODO check we only have one figure
        for box in edge_boxes:
            min_row, min_col, max_row, max_col = box
            # max used so we don't paint rectangle outside of image
            # (max(min_col - 20, 0) mean that we want to see some area around in case we lose some part of object
            # max_row - min_row + 50 add another area below and right of boxes
            ax.add_patch(patches.Rectangle((max(min_col - 20, 0), max(min_row - 20, 0)), max_col - min_col + 30,
                                           max_row - min_row + 50, linewidth=1, edgecolor='b', facecolor='none'))
        plt.show()
    cut_objects = cut_objects_from_image(img, objects_boxes)  # get objects images from bboxes
    figure = cut_figure(img, edge_boxes)  # get figure structure from its bboxes(find edges and fill holes)
    class_objects = classify_objects(cut_objects)  # using CNN to classify object on images
    return figure, edge_boxes[:2], class_objects


def extract_fig_and_objects_example():  # example for figure and object classification function
    data = get_example_data()
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
    for pos_y in range(0, fig_y - obj_y, 20):  # grid for y
        for pos_x in range(0, fig_x - obj_x, 20):  # grid for x
            roi = main_figure[pos_y:pos_y + obj_y, pos_x:pos_x + obj_x].astype(int)  # take area from figure
            intersect = cv.bitwise_and(roi, obj.astype(int))  # check how we can place object
            if np.sum(intersect) == object_area:  # confirm we can place whole object
                main_figure[pos_y:pos_y + obj_y, pos_x:pos_x + obj_x] = obj.astype(bool)  # paint object on figure
                # so we can't use this area again
                # we find location for this object on this figure, so we need to return location on figure and
                # figure without used area
                return (pos_y, pos_x), main_figure

    return (-1, -1), main_figure  # if we don't find place


def run_epoch_stuff():  # function to show example working for first step
    path = 'work/easy_img.jpg'
    figures = "objects_figure"
    objects_struct_paths = {0: "0.jpg", 1: "2.jpg", 2: "5.jpg", 3: "4.jpg", 4: "1.jpg", 5: "3.jpg", 6: "8.jpg",
                            7: "7.jpg"}  # files with "good" object images

    img = imread(path)

    figure, fig_location, class_objects = extract_fig_and_objects(img, show_boxes=True)

    plt.imshow(figure, cmap='gray')
    plt.show()
    objects_structures = []

    for obj in class_objects:  # iteration though CNN prediction about class
        obj_indx = np.argmax(obj)
        struct = imread(os.path.join(figures, objects_struct_paths[obj_indx]))  # getting this object "good" struct
        plt.imshow(struct, cmap='gray')
        plt.show()
        struct = struct / 255 > 0  # maybe TODO, but this make me feel safe about mask
        objects_structures.append(struct)
        print(round(obj[obj_indx], 3), indx_to_name[obj_indx])  # show our confidence in prediction
        # TODO work bad confidence cases

    object_areas = []
    for elem in objects_structures:  # getting area of objects, maybe TODO deal with big thirst
        area = np.sum(elem)
        object_areas.append(area)

    object_loc = []
    for obj, area in zip(objects_structures, object_areas):  # now we try to put objects inside figure
        (y, x), figure = slide_obj_over_fig(figure, obj, area)
        object_loc.append([y, x])
        if x != -1:  # cant pu object inside.
            plt.imshow(figure, cmap='gray')
        else:
            print("nope")

    for obj, location in zip(objects_structures, object_loc):  # plotting object images above figure
        y_fig, x_fig, *_ = fig_location[0]
        y, x = location
        obj_y, obj_x = obj.shape
        # there is a better way to do this. # TODO do it better
        # here is main code of putting object image above figure image on initial image
        # (x/y)_fig - place where our figure locate on image
        # (y,x) - location of our object on image
        # obj_(x,y) - size of object
        # 255 * object <- paint white mask in place where is this object structure
        img[y_fig + y:y_fig + y + obj_y, x_fig + x:x_fig + x + obj_x, 0] = 255 * obj.astype(int)
        img[y_fig + y:y_fig + y + obj_y, x_fig + x:x_fig + x + obj_x, 1] = 255 * obj.astype(int)
        img[y_fig + y:y_fig + y + obj_y, x_fig + x:x_fig + x + obj_x, 2] = 255 * obj.astype(int)

    plt.imshow(img)
    plt.show()


indx_to_name = {0: "Значок", 1: "Пульт", 2: "Зажигалка", 3: "Медиатор", 4: "Шахматный конь", 5: "Крышка",
                6: "Кубик", 7: "Батарейка"}
run_epoch_stuff()

# extract_fig_and_objects_example()
