# trying countours to detect shapes and then return numbers related to those shapes
import cv2
import numpy as np


# Load the image
image = cv2.imread('./images/hero_1.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# threshold
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY) # threhshold value, 65 is for real images, 150 for template images


# Find contours
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# we only want to print contours that dont have children --> these are the innermost shapes that dont have anymore shapes within them
child_contours = [] # list of all innermost contours
for i, h in enumerate(hierarchy[0]): # [Next, Previous, First_Child, Parent], [0, 1, 2 ,3]
    if h[2] == -1: # if there is no next child, print the contour
        child_contours.append(contours[i])
        # cv2.drawContours(image, contours, i, (0, 255, 0), 3)

# calculate area of each contour to find the largest one
areas = [cv2.contourArea(c) for c in child_contours]

# index of max area contour
max_area_index = areas.index(max(areas))



# draw only the max area contour
cv2.drawContours(image, child_contours, max_area_index, (0, 255, 0), 3) #thickness=cv2.FILLED


# trying Hu-moments to compare images
# hu moment of template image

cv2.imshow('Image with Contours', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

