import cv2
import numpy as np

def resize_img(source_img: np.array):
    shape_of_src = source_img.shape[:2]
    if max(shape_of_src) > 1000:
        resize_coefficient = 1000/max(shape_of_src)
        result_size = tuple([int(shape_of_src)])
        result_size = (int(shape_of_src[1]*resize_coefficient), 
                       int(shape_of_src[0]*resize_coefficient))
        result_img = cv2.resize(source_img, 
                                result_size,
                                interpolation=cv2.INTER_CUBIC)
        return result_img
    return source_img

img = cv2.imread('images\\for_tests\\1.jpg')
res = resize_img(img)
cv2.imshow('test', res)
cv2.waitKey(0)
