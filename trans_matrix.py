import cv2
import numpy as np

def trans_matrix(process):
    width = process['width']
    height = process['height']
    if process['process'] == 'rotate':
        scale = process['rotate']['scale']
        angle = process['rotate']['angle']
        center = (width//2, height//2)
        trans = cv2.getRotationMatrix2D(center, angle, scale)

    elif process['process'] == 'flip':
        src = np.array([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0]], np.float32)
        dest = dest = src.copy()
        flipcode = process['flip']['flipcode']
        if flipcode == 0:
            dest[:,1] = height - src[:,1] 
        
        elif flipcode == 1:
            dest[:,0] = width - src[:,0]
       
        elif flipcode == -1:
            dest[:,0] = width - src[:,0]
            dest[:,1] = height - src[:,1] 

        trans = cv2.getAffineTransform(src, dest)

    elif process['process'] == 'shear':
        src = np.array([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0]], np.float32)
        dest = src.copy()
        shear_point = process['shear']['shear_point']
        shear_factor = process['shear']['shear_factor']
        if shear_point == 0:   
            dest[:,0] += (shear_factor * (height - src[:,1])).astype(np.float32)

        elif shear_point == 1:
            dest[:,1] += (shear_factor * (width - src[:,0])).astype(np.float32)

        elif shear_point == 2:
            dest[:,0] += (shear_factor * src[:, 1]).astype(np.float32)

        elif shear_point == 3:
            dest[:,1] += (shear_factor * src[:,0]).astype(np.float32)
        
        trans = cv2.getAffineTransform(src, dest)

    return trans