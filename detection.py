from skimage.feature import canny
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import binary_closing, binary_opening
from skimage.color import rgb2gray
from skimage.filters import threshold_minimum
from skimage.measure import regionprops
from skimage.measure import label as sk_measure_label
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
from utils import patch_rectangle
import config


def get_largest_area_boxes(mask):
    """
    Function find all connected components, filter all low area and return bboxes of remaining
    :param mask:
    :return:
    """
    result_boxes = []
    labels = sk_measure_label(~mask)  # splitting the mask into connectivity components
    props = regionprops(
        labels)  # finding the properties of each area (center position, area, bbox, intensity interval, etc.)
    for prop in props:
        img_threshold_percent = config.object_no_less_than_percent_of_img
        if prop.area > img_threshold_percent * mask.shape[0] * mask.shape[1]:  # clear all low area
            box = prop.bbox
            result_boxes.append(box)

    return result_boxes


def get_rect(x, y, width, height, angle):  # LMAO stackoverflow
    # https://stackoverflow.com/questions/12638790/drawing-a-rectangle-inside-a-2d-numpy-array
    rect = np.array([(0, 0), (width, 0), (width, height), (0, height), (0, 0)])
    theta = (np.pi / 180.0) * angle
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    offset = np.array([x, y])
    transformed_rect = np.dot(rect, R) + offset
    return transformed_rect


def get_object_boxes(gray_img):
    upper_threshold = threshold_minimum(gray_img)  # get only objects on image
    mask = gray_img >= upper_threshold  # In the case of uniform illumination, the minimum threshold value leaves only
    # the shapes

    return get_largest_area_boxes(mask)  # return all objects bboxes.


def get_object_and_figure_boxes(main_img):
    gray_img = rgb2gray(main_img)
    objects_boxes = get_object_boxes(gray_img)  # Get objects bbox

    edges = canny(gray_img, sigma=1.5, low_threshold=0.1)  # get all edges on image
    # they contain objects edges too.
    # so we use data from object bboxes and paint on mask black bboxes, so only figure edges stay
    working_img = Image.fromarray(edges)
    draw = ImageDraw.Draw(working_img)
    for box in objects_boxes:  # paint black boxes where object bbox exist
        min_row, min_col, max_row, max_col = box
        # max used so we don't paint rectangle outside of image
        # (max(min_col - 20, 0) mean that we want to see some area around in case we lose some part of object
        # max_row - min_row + 50 add another area below and right of boxes
        rect = get_rect(max(min_col - 20, 0), max(min_row - 20, 0), width=max_col - min_col + 20,
                        height=max_row - min_row + 20, angle=0)
        draw.polygon([tuple(p) for p in rect], fill=0)  # stack overflow

    edges = np.asarray(working_img)  # now we have mask only with figure edges
    edges = binary_closing(edges, selem=np.ones((4, 4)))  # if we have breaks in the contour this will close them
    my_edge_segmentation = binary_fill_holes(binary_closing(edges, selem=np.ones((7, 7))))  # fill figure
    my_edge_segmentation = binary_opening(my_edge_segmentation, selem=np.ones((10, 10)))  # if some noises survive
    edge_boxes = get_largest_area_boxes(~my_edge_segmentation)  # get figure bbox.

    return edge_boxes, objects_boxes  #


def get_bboxes_example_show(show_data=True):  # example of usage get_object_and_figure_boxes
    # we find and plot all boxes of objects and box on figure
    from data_handler import get_example_data
    data = get_example_data(show_data=show_data)
    fix, ax = plt.subplots(4, 3, figsize=(30, 30))

    for indx, main_img in enumerate(data):
        edge_boxes, objects_boxes = get_object_and_figure_boxes(main_img)
        ax.flat[indx].imshow(main_img, cmap="gray")
        for box in objects_boxes:
            ax.flat[indx].add_patch(patch_rectangle(box, 'r'))
        for box in edge_boxes:
            ax.flat[indx].add_patch(patch_rectangle(box, 'b'))
    fix.tight_layout()
    plt.show()


def cut_objects_from_image(image, bboxes, resize=True):
    """
    this function cut and resize object from image using object bbox
    :param image:
    :param bboxes:
    :param resize:
    :return:
    """
    add_area = 10  # if we lost some part of object during detection
    images = []
    img_row, img_col, _ = image.shape
    for box in bboxes:
        min_row, min_col, max_row, max_col = box
        roi = image[max(min_row - add_area, 0): min(max_row + add_area, img_row),
              max(min_col - add_area, 0): min(max_col + add_area, img_col)]
        im = Image.fromarray(np.uint8(roi))
        expected_size = config.CNN_expected_img_size
        if resize:
            im = im.resize((expected_size, expected_size), Image.ANTIALIAS)
        img = np.array(im)
        images.append(img)

    return images


def cut_figure(img, edge_boxes):  # using bbox of figure we get good cut of it
    cut = cut_objects_from_image(img, edge_boxes, resize=False)[0]  # maybe TODO to check if we find figure
    edges = canny(rgb2gray(cut), sigma=1.5, low_threshold=0.1)
    figure = binary_fill_holes(binary_closing(edges, selem=np.ones((7, 7))))
    return figure
