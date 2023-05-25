import cv2 as cv
import pytesseract

# This set is using 1920x1080 px images.
# This file is for valorant


# So it looks right now that the stuff shown here: https://stackoverflow.com/questions/62042172/how-to-remove-noise-in-image-opencv-python
# the binary output on an inverted image gives really readable text in white boxes.
# looking for the white boxes and just cutting out around htem, and using them for the input.

# The other thing to note is that I should be able to cut exact boxes around all of the text
# I just need to figure out how to get the bound correctly.
# just going through and cutting exact rectangles out around the data I need and then running
# Tesseract on each of the images.
# NOTE: as long as the aspect ratio doesn't change, all of the locations should stay consistent
#       when compared to %width and height of the image. This means I Just need to go through and
#       figure out all the boxes.
#       something like go get all of the rows, then break each row down into all the columns
#       then run each snippet through tesseract and save the text to the desired dict key
