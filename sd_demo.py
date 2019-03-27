import cv2
import MR
import scipy as sp
import sys
import numpy

def get_salient_object_rect(binary_image):
    x_axis = []
    y_axis = []
    for y in range(binary_image.shape[0]):
        for x in range(binary_image.shape[1]):
            if binary_image[y][x] == 255:
                x_axis.append(x)
                y_axis.append(y)
    return min(x_axis), min(y_axis), max(x_axis), max(y_axis)

global mr_sal
mr_sal = MR.MR_saliency()

def main(image):
    image_data = cv2.imread(image)
    shape = image_data.shape
    sal = mr_sal.saliency(image_data)
    print sal.shape
    sal = cv2.resize(sal, (shape[1],shape[0])).astype(sp.uint8)
    avg =  numpy.mean(sal)
    print "average pixel value: ", avg
    retval, threshold = cv2.threshold(sal, avg, 255, cv2.THRESH_BINARY_INV)
    min_x, min_y, max_x, max_y = get_salient_object_rect(threshold)
    print min_x, min_y, max_x, max_y
    cv2.rectangle(image_data, (min_x, min_y), (max_x, max_y), (0,0,255), 2)
    cv2.imshow("threshold", threshold)
    print sal.shape
    # sal = sal.astype(sp.uint8)
    sal = cv2.normalize(sal, None, 0, 255, cv2.NORM_MINMAX)
    outsal = cv2.applyColorMap(sal, cv2.COLORMAP_HSV)
    cv2.imshow("original", image_data)
    cv2.imshow("saliency", outsal)
    cv2.waitKey(0)

if __name__ == "__main__" :
    img = "astronaut.png"
    main(img)