# this is the culmination of all files in this project
# from a pool of test images it classifies as 1, 2, 3, 4, 5

import cv2
import numpy as np
import os

# get image from pool image folder --> lets have 10 images in here testing 1, 2, 3 rn
images = []
imagelink = []
for image in os.scandir('/Users/amasud7/Desktop/code/roboProj1/pool_images'):
    if image.path.endswith('.jpg'):
        imagelink.append(image.path)
        images.append(cv2.imread(image.path))


# create dict to store template and its corresponding hu moments --> dont forget to add 4
template = {1: tuple([0.20190165, 0.46352852, 1.51366024, 1.6986363, 3.30562525, 1.9312427, -4.51126758]),
            2: tuple([0.37610695, 1.11190971, 2.66961032, 3.30066991, 6.57430505, 4.32652557, 6.35262385]),
             3: tuple([0.38479041, 1.08707333, 2.3122054,  3.03535197, -5.71430297, -3.58084251, -6.52324638]),
              5:tuple([0.40817217, 1.20667722, 3.41604175, 4.11077181, 8.02564175, -5.41968741, -8.02374956])}


# compute moment of image
def get_contour(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY) 
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    child_contours = [] 
    for i, h in enumerate(hierarchy[0]): 
        if h[2] == -1: 
            child_contours.append(contours[i])

    areas = [cv2.contourArea(c) for c in child_contours]
    max_area_index = areas.index(max(areas))
    return child_contours[max_area_index]


def hu_moment(contour):
    moments = cv2.moments(contour)
    img_hu_moments = cv2.HuMoments(moments).flatten()

    for i in range(len(img_hu_moments)):
        img_hu_moments[i] = -1 * np.copysign(1.0, img_hu_moments[i]) * np.log10(abs(img_hu_moments[i]))

    return img_hu_moments # hu moments of image that needs to be classified


# perform euclidian distance between image and template
results = []
distances = []
for image in images:
    # need to make sure image is 640x640 bc that is what templates are
    image = cv2.resize(image, (640, 640))
    img_contour = get_contour(image)
    img_hu_moments = hu_moment(img_contour)

    # store distances in list
    min_distance = float('inf')
    for key, value in template.items(): # .items() returns key and value
        distance = np.linalg.norm(value - img_hu_moments)
        distances.append(distance)
        if distance < min_distance and distance < 10:
            min_distance = distance
            min_key = key
    # cv2.imshow('Image', image)
    # cv2.waitKey(0)
    # print(min_key)
    results.append(min_key)

# print(results)
# print(imagelink)
print(distances)




# if distance is less than threshold (10, might be different for different numbers) classify as that number or not that number.

