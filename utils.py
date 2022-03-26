import matplotlib.patches as patches


def patch_rectangle(box, edge_color):
    min_row, min_col, max_row, max_col = box
    # max used so we don't paint rectangle outside of image
    # (max(min_col - 20, 0) mean that we want to see some area around in case we lose some part of object
    # max_row - min_row + 50 add another area below and right of boxes
    return patches.Rectangle((max(min_col - 20, 0), max(min_row - 20, 0)), max_col - min_col + 30,
                             max_row - min_row + 50, linewidth=1, edgecolor=edge_color, facecolor='none')