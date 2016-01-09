# from https://github.com/shantnu/FaceDetect/blob/master/face_detect.py

import cv2
import sys
import common
import Image
import common
import numpy as np

# Get user supplied values
imagePath = sys.argv[1]
cascPath = sys.argv[2]

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = Image.open(imagePath)
gray = np.array(common.imgToGray(common.imgResizeAndKeepRatio(image, 512)))
image = gray

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=11,
    minSize=(20, 20),
    flags = cv2.CASCADE_SCALE_IMAGE
)

print "Found {0} faces!".format(len(faces))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces found", image)
cv2.waitKey(0)
