import numpy as np
import cv2


# function for messing with frame --> denoise, find color, shape, etc
def process(frame):

    frame = cv2.medianBlur(frame, 5)
     # converting image to hsv format
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # lower and upper bounds for color (hue[0, 179], saturation[0, 255], value[0, 255])
    lower = np.array([82, 50, 70])
    upper = np.array([105 ,255, 255])

    # creating mask (hsv image, lower bound, upper bound)
    mask = cv2.inRange(hsv, lower, upper)

    # keep only the color region
    color_only = cv2.bitwise_and(frame, frame, mask=mask)

    # invert the image
    inverted_image = cv2.bitwise_not(color_only)

    # convert image to grayscale
    gray_image = cv2.cvtColor(inverted_image, cv2.COLOR_BGR2GRAY)

    # threshhold of image / binarize image --> all pixels are either 0 for black or 255 for white.
    ret, thresh = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

    # get rid of noise
    kernel = np.ones((3, 3), np.uint8)
    thresh_image = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel) 
    thresh_image = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # use gray scale image to find contours --> countours is a list of all contours found in image
    contours, hierarchy = cv2.findContours(thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # make an new image for the corners
    

    # using hierarchy to find corners
    for i, contour in enumerate(contours):
        if hierarchy[0][i][2] < 0:
                cv2.drawContours(frame, [contour], -1, (0, 0, 255), 3) # red

    # need another image with just the corner contours --> then pass into convex hull

    for i, contour in enumerate(contours):
        if hierarchy[0][i][2] == -1: # makes sure there is no children --> innermost contour
            convex_hull = cv2.convexHull(contour)
            cv2.drawContours(frame, [convex_hull], -1, (255, 0, 0), 3)

    
# this is currently how im trying to use convexHull to get a bounding box around the corners, but right now its making a bounding box around each corner that it detects instead of just one big square.
# Just wondering how I would use convexHull to get a bounding box that goes around all 4 corners instead of each one individually
    # return processed images --> mask: is just color detection, contours_frame: is the contours
    return mask, frame





# opening cam
cam = cv2.VideoCapture(0)

# starting infinite loop for video feed
while True:
    # read() returns a boolean value saying whether image is being captured --> stored in cap
    # the second value returned by read() is the frame --> stored in frame
    cap, frame = cam.read()

    # calling process function
    mask_frame, contour_frame = process(frame)


    # displaying image
    cv2.imshow('mask', mask_frame)
    cv2.imshow('contour', contour_frame)

    # termination of webcam
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()