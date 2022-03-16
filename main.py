import os

from skimage.color import rgb2gray
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from data_handler import get_example_data
from detection import get_object_and_figure_boxes, cut_objects_from_image, cut_figure
from imageio import imread
import cv2 as cv


def cut_figure_example():
    data = get_example_data()
    img = data[3]
    edge_boxes, objects_boxes = get_object_and_figure_boxes(rgb2gray(img))
    plt.imshow(img)
    plt.show()
    figure = cut_figure(img, edge_boxes)
    plt.imshow(figure, cmap='gray')


def classify_objects(cut_objects, classes=8):
    from classyfier import get_best_model
    model = get_best_model(classes)
    res = model.predict(np.array(cut_objects))
    return res


def extract_fig_and_objects(img, show_boxes=False):
    edge_boxes, objects_boxes = get_object_and_figure_boxes(rgb2gray(img))
    if show_boxes:
        fig, ax = plt.subplots()
        ax.imshow(img)
        for box in objects_boxes:
            min_row, min_col, max_row, max_col = box
            ax.add_patch(patches.Rectangle((max(min_col - 20, 0), max(min_row - 20, 0)), max_col - min_col + 30,
                                           max_row - min_row + 50, linewidth=1, edgecolor='r', facecolor='none'))
        # TODO ПРОВЕРКА ЧТО ФИГУРА ТОЛЬКО ОДНА
        for box in edge_boxes:
            min_row, min_col, max_row, max_col = box
            ax.add_patch(patches.Rectangle((max(min_col - 20, 0), max(min_row - 20, 0)), max_col - min_col + 30,
                                           max_row - min_row + 50, linewidth=1, edgecolor='b', facecolor='none'))
        plt.show()
    cut_objects = cut_objects_from_image(img, objects_boxes)
    figure = cut_figure(img, edge_boxes)
    class_objects = classify_objects(cut_objects)
    return figure, edge_boxes[:2], class_objects


def extract_fig_and_objects_example():
    data = get_example_data()
    for elem in data:
        figure, _, class_objects = extract_fig_and_objects(elem, show_boxes=True)
        plt.imshow(figure, cmap='gray')
        plt.show()
        for obj in class_objects:
            obj_indx = np.argmax(obj)
            print(round(obj[obj_indx], 3), indx_to_name[obj_indx])


def slide_obj_over_fig(main_figure, obj, object_area):
    fig_y, fig_x = main_figure.shape
    obj_y, obj_x = obj.shape
    for pos_y in range(0, fig_y - obj_y, 20):
        for pos_x in range(0, fig_x - obj_x, 20):
            roi = main_figure[pos_y:pos_y + obj_y, pos_x:pos_x + obj_x].astype(int)
            intersect = cv.bitwise_and(roi, obj.astype(int))
            if np.sum(intersect) == object_area:
                main_figure[pos_y:pos_y + obj_y, pos_x:pos_x + obj_x] = obj.astype(bool)
                return (pos_y, pos_x), main_figure

    return (-1, -1), main_figure


def run_epoch_stuff():
    path = 'work/easy_img.jpg'
    figures = "objects_figure"
    figure_struct_paths = {0: "0.jpg", 1: "2.jpg", 2: "5.jpg", 3: "4.jpg", 4: "1.jpg", 5: "3.jpg", 6: "8.jpgк",
                           7: "7.jpg"}
    img = imread(path)

    figure, fig_location, class_objects = extract_fig_and_objects(img, show_boxes=True)

    plt.imshow(figure, cmap='gray')
    plt.show()
    objects_structures = []
    for obj in class_objects:
        obj_indx = np.argmax(obj)
        struct = imread(os.path.join(figures, figure_struct_paths[obj_indx]))
        plt.imshow(struct, cmap='gray')
        plt.show()
        struct = struct / 255 > 0
        objects_structures.append(struct)
        print(round(obj[obj_indx], 3), indx_to_name[obj_indx])

    object_areas = []
    for elem in objects_structures:
        area = np.sum(elem)
        object_areas.append(area)

    object_loc = []
    for obj, area in zip(objects_structures, object_areas):
        (y, x), figure = slide_obj_over_fig(figure, obj, area)
        object_loc.append([y, x])
        if x != -1:
            plt.imshow(figure, cmap='gray')
        else:
            print("nope")

    for obj, location in zip(objects_structures, object_loc):
        y_fig, x_fig, *_ = fig_location[0]
        y, x = location
        obj_y, obj_x = obj.shape
        img[y_fig + y:y_fig + y + obj_y, x_fig + x:x_fig + x + obj_x, 0] = 255 * obj.astype(int)
        img[y_fig + y:y_fig + y + obj_y, x_fig + x:x_fig + x + obj_x, 1] = 255 * obj.astype(int)
        img[y_fig + y:y_fig + y + obj_y, x_fig + x:x_fig + x + obj_x, 2] = 255 * obj.astype(int)

    plt.imshow(img)
    plt.show()


indx_to_name = {0: "Значок", 1: "Пульт", 2: "Зажигалка", 3: "Медиатор", 4: "Шахматный конь", 5: "Крышка",
                6: "Кубик", 7: "Батарейка"}
run_epoch_stuff()

# extract_fig_and_objects_example()
