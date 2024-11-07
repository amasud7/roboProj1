import cv2
import numpy as np
import os


def get_contour(image): # process image and return major contour aka number
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 65, 255, cv2.THRESH_BINARY) 

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) # change to not simple for more points

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
    # temp_hu_moments = [-0.42033896, -1.2800598, -2.37100151, -3.00287561, -5.72290071, -3.64337416, -6.11469886] # this is for standard_3
    temp_hu_moments =[-0.3762888, -1.06797808, -2.28293167, -3.00242794, -5.65180683, -3.53765858, -6.40381786] # standard_3 resized to 640x640
    # temp_hu_moments = [-0.41209761, -1.0100353,  -1.99089754, -2.54051226, -4.84663678, -3.0727363, -5.1911946 ] # hero_1

    moments = cv2.moments(contour)
    img_hu_moments = cv2.HuMoments(moments).flatten()

    # proper log transform?
    

    # euclidian distance computation
    difference = np.sum(np.abs(-np.sign(temp_hu_moments) * np.log10(np.abs(temp_hu_moments)) - 
                               -np.sign(img_hu_moments) * np.log10(np.abs(img_hu_moments))))

    return difference


def main():
    # framework to test to find correct threshold value
    # loop through all images in directory
    difference_list = []
    difference_list2 = []
    image_num = 0
    for image in os.scandir('/Users/amasud7/Desktop/code/roboProj1/test_1'):
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








# testing hu moments with these lines of code


# # preprocess image
# image = cv2.imread('images/standard_3.jpg')

# # change size to be the same as samples (640x640)
# resize_img = cv2.resize(image, (640, 640))

# gray = cv2.cvtColor(resize_img, cv2.COLOR_BGR2GRAY)
# _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

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

# log = np.log10(np.abs(hu_moments))
# print(log)
# # template = [0.42033896, 1.2800598, 2.37100151, 3.00287561, -5.72290071, -3.64337416, -6.11469886] # standard_3
# # template = [-0.3762888, -1.06797808, -2.28293167, -3.00242794, -5.65180683, -3.53765858, -6.40381786] # resized 640x640 standard_3

# # template = [-0.41209761, -1.0100353,  -1.99089754, -2.54051226, -4.84663678, -3.0727363,
# #  -5.1911946 ] # hero_1



# # raw = [0.44113424, 0.9791763,  3.75028028, 3.88864883, 7.70819348, 4.38124476, 9.42470464] # (raw_3)


# # # difference = np.sum(np.abs(-np.sign(template) * np.log10(np.abs(template)) - 
# #                             #    -np.sign(raw) * np.log10(np.abs(raw))))

