import cv2
import numpy as np
from compute_moments import get_contour

# orb uses FAST and BRIEF to make fast and efficient feature matching




# load two images to perform feature matching
template = cv2.imread('./images/standard_3.jpg')
resize_template = cv2.resize(template, (640, 640))

img = cv2.imread('/Users/amasud7/Desktop/code/roboProj1/test_3/image28.jpg')



# create orb object
orb = cv2.ORB_create()

# detect keypoints and compute descriptors
kp1, des1 = orb.detectAndCompute(resize_template, None)
kp2, des2 = orb.detectAndCompute(img, None)  

# using brute force mathcher to match keypoints
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) 
matches = bf.match(des1, des2) # --> performs actual matching

# sort matches
matches = sorted(matches, key=lambda x: x.distance)

# ratio test
# good_matches = []
# for m in matches:
#     if m.distance < 0.75 * matches[1].distance:
#         good_matches.append(m)


good_matches = matches[:30]  # Keep only the top 30 matches

# draw matches
results = cv2.drawMatches(resize_template, kp1, img, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# display results
cv2.imshow('Results', results)
cv2.waitKey(0)
cv2.destroyAllWindows()

