import cv2
import numpy as np

# load two images to perform feature matching
img = cv2.imread('./test_4/image19.jpg')
resize_img = cv2.resize(img, (640, 640))
resize_img = cv2.GaussianBlur(resize_img, (7, 7), 0)
resize_img = cv2.addWeighted(resize_img, 1.5, resize_img, -0.5, 0)

template = cv2.imread('./images/standard_4.jpg')
resize_template = cv2.resize(template, (640, 640)) 
resize_template = cv2.GaussianBlur(resize_template, (9, 9), 10.0)
resize_template = cv2.addWeighted(resize_template, 1.5, resize_template, -0.5, 0)


# create orb/beblid object
descriptor = cv2.xfeatures2d.BEBLID_create(0.75) # similar to orb but faster and more accurate
detector = cv2.ORB_create()

# detect keypoints and compute descriptors
kpts1 = detector.detect(resize_img, None)
kpts2 = detector.detect(resize_template, None)

kp1, des1 = descriptor.compute(resize_img, kpts1)
kp2, des2 = descriptor.compute(resize_template, kpts2)

# match with Brute force matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# sort matches by distance
matches = sorted(matches, key=lambda x: x.distance)

# finding hompgraphy matrix
#RANSAC(Random Sample Consensus) is a method used to fit noisy data. classifies inliers and outliers
src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2) # gets (x, y) coordinates of the keypoints in the first image
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

try:
    h_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
except:
    print('at least 4 matches were not found')


# finding inliers
inliers1 = []
inliers2 = []
good_matches = []
inlier_threshold = 0.5 # Distance threshold to identify inliers with homography check

# Iterate over the matches
for i, match in enumerate(matches):
    # Get the keypoints from the first and second images
    pt1 = np.array(kp1[match.queryIdx].pt)
    pt2 = np.array(kp2[match.trainIdx].pt)

    # Create the homogeneous point for the first image
    col = np.ones((3, 1), dtype=np.float64)
    col[0:2, 0] = pt1

    # Project from image 1 to image 2 using the homography matrix
    col = np.dot(h_matrix, col)
    col /= col[2, 0]

    # Calculate the Euclidean distance between the projected point and the actual point in the second image
    dist = np.sqrt(pow(col[0, 0] - pt2[0], 2) + pow(col[1, 0] - pt2[1], 2))

    # Check if the distance is within the inlier threshold
    if dist < inlier_threshold:
        good_matches.append(cv2.DMatch(len(inliers1), len(inliers2), 0))
        inliers1.append(kp1[match.queryIdx])
        inliers2.append(kp2[match.trainIdx])

# draw matches
results = cv2.drawMatches(resize_img, inliers1, resize_template, inliers2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
print(len(matches))
print(len(inliers1))


# display results
# cv2.imshow('Results', results)
cv2.waitKey(0)
cv2.destroyAllWindows()

