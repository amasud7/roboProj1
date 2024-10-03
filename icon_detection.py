import cv2
import numpy as np
import pytesseract;
from pytesseract import Output

# Read the image
img = cv2.imread('images/standard_5_balancing.jpg')

# convert bgr --> rgb because tesseract requires RGB colorspace for OCR computation
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# grey scale image
grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Remove Noise 
denoised_img = cv2.medianBlur(grey_img, 5)

# Thresholding
ret, thresh_img = cv2.threshold(denoised_img, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

# custom config for tesseract parameters

# --oem:
# 0: Legacy engine only
# 1: Neural nets LSTM engine only
# 2: Legacy and LSTM engines
# 3: Default, based on what is available

# --psm:
#   0    Orientation and script detection (OSD) only.
#   1    Automatic page segmentation with OSD.
#   2    Automatic page segmentation, but no OSD, or OCR. (not implemented)
#   3    Fully automatic page segmentation, but no OSD. (Default)
#   4    Assume a single column of text of variable sizes.
#   5    Assume a single uniform block of vertically aligned text.
#   6    Assume a single uniform block of text.
#   7    Treat the image as a single text line.
#   8    Treat the image as a single word.
#   9    Treat the image as a single word in a circle.
#  10    Treat the image as a single character.
#  11    Sparse text. Find as much text as possible in no particular order.
#  12    Sparse text with OSD.
#  13    Raw line. Treat the image as a single text line,
#        bypassing hacks that are Tesseract-specific.


# outputbase digits: just for finding digits
custom_config = r'--oem 3 --psm 9 outputbase digits'

output = pytesseract.image_to_string(thresh_img, config=custom_config)
print(output)


# Check if the image was loaded successfully
if img is None:
    print("Error: Could not load image")
else:
    # Display the image
    
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



