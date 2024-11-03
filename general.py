def maxmin_corner(corner):
    x_max = max(corner[0][0], corner[1][0], corner[2][0], corner[3][0])
    y_max = max(corner[0][1], corner[1][1], corner[2][1], corner[3][1])
    x_min = min(corner[0][0], corner[1][0], corner[2][0], corner[3][0])
    y_min = min(corner[0][1], corner[1][1], corner[2][1], corner[3][1])

    return x_max, y_max, x_min, y_min

def xywh2xyX4(bbox):
    x,y,w,h = bbox

    top_left = (x - w//2, y - h//2)
    top_right = (x + w//2, y - h//2)
    bottom_right = (x + w//2, y + h//2)
    bottom_left = (x - w//2, y + h//2)
    
    return top_left, top_right, bottom_right, bottom_left

def xyX42xywh(corner):
    x_max, y_max, x_min, y_min = maxmin_corner(corner)

    x = (x_max + x_min)/2
    y = (y_max + y_min)/2
    w = x_max - x_min
    h = y_max - y_min

    return x, y, w, h

def adjust_corner(corner, width, height):
    x_max, y_max, x_min, y_min = maxmin_corner(corner)

    x_max = width if x_max > width else 0 if x_max < 0 else x_max
    y_max = height if y_max > height else 0 if y_max < 0 else y_max
    x_min = width if x_min > width else 0 if x_min < 0 else x_min
    y_min = height if y_min > height else 0 if y_min < 0 else y_min

    top_left = (x_min, y_min)
    top_right = (x_max, y_min)
    bottom_right = (x_max, y_max)
    bottom_left = (x_min, y_max)

    return top_left, top_right, bottom_right, bottom_left

def denormalize(xywh, width, height):
    de_xywh = []
    for coor, scale in zip(xywh, (width, height, width, height)):
        de_xywh.append(round(coor * scale))
    
    return de_xywh

def normalize(xywh, width, height):
    nor_xywh = []
    for coor, scale in zip(xywh, (width, height, width, height)):
        nor_xywh.append(round(coor / scale, 6))
    
    return nor_xywh