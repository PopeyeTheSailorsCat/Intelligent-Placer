import matplotlib.patches as patches
import config


def patch_rectangle(box, edge_color, img_shape):
    min_row, min_col, max_row, max_col = box
    img_row, img_col, _ = img_shape
    add_area_x =int(img_col * config.additional_x_for_cut)
    add_area_y = int(img_row * config.additional_y_for_cut)
    # max used so we don't paint rectangle outside of image
    # (max(min_col - 20, 0) mean that we want to see some area around in case we lose some part of object
    # max_row - min_row + 50 add another area below and right of boxes
    return patches.Rectangle((max(min_col - add_area_x, 0), max(min_row - add_area_y, 0)),
                             max_col - min_col + add_area_x,
                             max_row - min_row + add_area_y, linewidth=1, edgecolor=edge_color, facecolor='none')
