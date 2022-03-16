from skimage.feature import canny
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import binary_closing, binary_opening
from skimage.color import rgb2gray
from skimage.filters import threshold_minimum
from skimage.measure import regionprops
from skimage.measure import label as sk_measure_label
import matplotlib.patches as patches
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np


def get_largest_area_boxes(mask):
    result_boxes = []
    labels = sk_measure_label(~mask)  # разбиение маски на компоненты связности
    props = regionprops(
        labels)  # нахождение свойств каждой области (положение центра, площадь, bbox, интервал интенсивностей и т.д.)
    areas = [prop.area for prop in props]  # нас интересуют площади компонент связности
    for prop in props:
        if prop.area > 1200:  # TODO ЭТо потенцияально зависящая от размеров изображения дырааааа.
            box = prop.bbox
            result_boxes.append(box)

    return result_boxes


def get_rect(x, y, width, height, angle):
    rect = np.array([(0, 0), (width, 0), (width, height), (0, height), (0, 0)])
    theta = (np.pi / 180.0) * angle
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    offset = np.array([x, y])
    transformed_rect = np.dot(rect, R) + offset
    return transformed_rect


def get_object_boxes(gray_img):
    upper_threshold = threshold_minimum(gray_img)  # get only objects on image
    mask = gray_img >= upper_threshold
    return get_largest_area_boxes(mask)


def get_object_and_figure_boxes(main_img):
    gray_img = rgb2gray(main_img)
    objects_boxes = get_object_boxes(gray_img)  # Get object bbox

    edges = canny(gray_img, sigma=1.5, low_threshold=0.1)

    working_img = Image.fromarray(edges)
    draw = ImageDraw.Draw(working_img)
    for box in objects_boxes:  # paint black boxes where object bbox exist
        min_row, min_col, max_row, max_col = box
        rect = get_rect(max(min_col - 20, 0), max(min_row - 20, 0), width=max_col - min_col + 20,
                        height=max_row - min_row + 20, angle=0)
        draw.polygon([tuple(p) for p in rect], fill=0)
    edges = np.asarray(working_img)
    edges = binary_closing(edges, selem=np.ones((4, 4)))
    my_edge_segmentation = binary_fill_holes(binary_closing(edges, selem=np.ones((7, 7))))
    my_edge_segmentation = binary_opening(my_edge_segmentation, selem=np.ones((10, 10)))
    edge_boxes = get_largest_area_boxes(~my_edge_segmentation)

    return edge_boxes, objects_boxes


def get_bboxes_example_show(show_data=True):
    from data_handler import get_example_data
    data = get_example_data(show_data=show_data)
    fix, ax = plt.subplots(4, 3, figsize=(30, 30))

    for indx, main_img in enumerate(data):
        edge_boxes, objects_boxes = get_object_and_figure_boxes(main_img)
        ax.flat[indx].imshow(main_img, cmap="gray")
        for box in objects_boxes:
            min_row, min_col, max_row, max_col = box
            ax.flat[indx].add_patch(
                patches.Rectangle((max(min_col - 20, 0), max(min_row - 20, 0)), max_col - min_col + 30,
                                  max_row - min_row + 50, linewidth=1, edgecolor='r', facecolor='none'))
        for box in edge_boxes:
            min_row, min_col, max_row, max_col = box
            ax.flat[indx].add_patch(
                patches.Rectangle((max(min_col - 20, 0), max(min_row - 20, 0)), max_col - min_col + 30,
                                  max_row - min_row + 50, linewidth=1, edgecolor='b', facecolor='none'))
    fix.tight_layout()
    plt.show()


def cut_objects_from_image(image, bboxes, resize=True):
    add_area = 10
    images = []
    img_row, img_col, _ = image.shape
    for box in bboxes:
        min_row, min_col, max_row, max_col = box
        roi = image[max(min_row - add_area, 0): min(max_row + add_area, img_row),
              max(min_col - add_area, 0): min(max_col + add_area, img_col)]
        im = Image.fromarray(np.uint8(roi))
        if resize:
            im = im.resize((128, 128), Image.ANTIALIAS)
        img = np.array(im)
        images.append(img)

    return images


def cut_figure(img, edge_boxes):
    cut = cut_objects_from_image(img, edge_boxes, resize=False)[0]
    edges = canny(rgb2gray(cut), sigma=1.5, low_threshold=0.1)
    figure = binary_fill_holes(binary_closing(edges, selem=np.ones((7, 7))))
    return figure
