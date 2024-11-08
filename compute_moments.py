import cv2
import numpy as np
import os


def get_contour(image): # process image and return major contour aka number
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY) 

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # change to NONE for more points

    child_contours = [] 
    for i, h in enumerate(hierarchy[0]): 
        if h[2] == -1: 
            child_contours.append(contours[i])


    # calculate area of each contour to find the largest one
    areas = [cv2.contourArea(c) for c in child_contours]

    # index of max area contour
    max_area_index = areas.index(max(areas))

    # find out how to make contour into shape


    return child_contours[max_area_index]


# def compute_hu_moments(contour): # pass in contour to find hu moment
#     moments = cv2.moments(contour)
#     hu_moments = cv2.HuMoments(moments).flatten()
#     return hu_moments

def euclidian_distance(contour): # computes euclidian distance between contour and template
    # store hu moments of all templates somehow --> currently testing with this
    # temp_hu_moments = [0.38479041, 1.08707333, 2.3122054,  3.03535197, -5.71430297, -3.58084251, -6.52324638] # standard_3 resized to 640x640
    # temp_hu_moments = [0.20190165, 0.46352852, 1.51366024, 1.6986363, 3.30562525, 1.9312427, -4.51126758] # hero_1 resized to 640x640
    # temp_hu_moments = [0.40817217, 1.20667722, 3.41604175, 4.11077181, 8.02564175, -5.41968741, -8.02374956] # standard_5 resized to 640x640
    temp_hu_moments =  [0.37610695, 1.11190971, 2.66961032, 3.30066991, 6.57430505, 4.32652557, 6.35262385] # engineer_2 resized to 640x640

    moments = cv2.moments(contour)
    img_hu_moments = cv2.HuMoments(moments).flatten()

    # proper log transform?
    for i in range(len(img_hu_moments)):
        img_hu_moments[i] = -1 * np.copysign(1.0, img_hu_moments[i]) * np.log10(abs(img_hu_moments[i]))
    

    # euclidian distance computation
    difference = np.sum(np.abs(-np.sign(temp_hu_moments) * np.log10(np.abs(temp_hu_moments)) - 
                               -np.sign(img_hu_moments) * np.log10(np.abs(img_hu_moments))))
    distance = np.linalg.norm(temp_hu_moments - img_hu_moments)

    return distance


def main():
    # framework to test to find correct threshold value
    # loop through all images in directory
    difference_list = []
    difference_list2 = []
    image_num = 0
    for image in os.scandir('/Users/amasud7/Desktop/code/roboProj1/test_2'):
        if image.path.endswith('.jpg'):
            # for each image compute euclidean distance and push back to list
            
            img = cv2.imread(image.path)
            image_num += 1
            difference_list2.append(euclidian_distance(get_contour(img)))
            difference_list.append('Image ' + str(image_num) + ': ' + str(euclidian_distance(get_contour(img))))


            # difference_list.append(cv2.matchShapes(contour, get_contour(cv2.imread('images/standard_3.jpg')), cv2.CONTOURS_MATCH_I1, 0.0))

    print(np.average(difference_list2))
    # print(difference_list)

# image = '/Users/amasud7/Desktop/code/roboProj1/test_3/image1.jpg'
# img = cv2.imread(image)

# contour = get_contour(img)
# difference = euclidian_distance(contour)

# print(difference)

main()
# threshhold under 10 is looking pretty good
# 5 does not seem to work well (15.00) --> use orb to distinguish between the two??







# # testing hu moments with these lines of code


# # preprocess image
# image = cv2.imread('./images/engineer_2.jpg')

# # change size to be the same as samples (640x640)
# resize_img = cv2.resize(image, (640, 640))

# gray = cv2.cvtColor(resize_img, cv2.COLOR_BGR2GRAY)
# blurred = cv2.GaussianBlur(gray, (7, 7), 0)
# _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)

# # inner contour of image

# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# child_contours = [] 
# for i, h in enumerate(hierarchy[0]): 
#     if h[2] == -1: 
#         child_contours.append(contours[i])


# # calculate area of each contour to find the largest one
# areas = [cv2.contourArea(c) for c in child_contours]

# # index of max area contour
# max_area_index = areas.index(max(areas))

# # draw only the max area contour
# cv2.drawContours(image, child_contours, max_area_index, (0, 255, 0), 3)




# # calculate moments
# moments = cv2.moments(child_contours[max_area_index]) # we only want moment of innermost contour --> so we pass child_contours[max_area_index]
# hu_moments = cv2.HuMoments(moments).flatten()

# # cv2.imshow('Image with Contours', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# # log transform
# for i in range(len(hu_moments)):
#     hu_moments[i] = -1 * np.copysign(1.0, hu_moments[i]) * np.log10(abs(hu_moments[i]))


# # log = np.log10(np.abs(hu_moments))
# print(hu_moments)

