from PIL import Image, ImageOps
import cv2 as cv
from matplotlib import pyplot as plt
import pytesseract
from pytesseract import Output
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

IMAGE_FP = "csbtest.png"

img = cv.imread(IMAGE_FP, cv.IMREAD_GRAYSCALE)

# img = cv.imread(IMAGE_FP)

img = cv.bitwise_not(img)
# cv.imshow("Invert1",img_not)
# cv.waitKey(0)
# cv.destroyAllWindows()




# d = pytesseract.image_to_data(img, output_type=Output.DICT)
# n_boxes = len(d['level'])
# for i in range(n_boxes):
#     (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
#     cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# cv.imshow('img', img)
# cv.waitKey(0)




image = cv.imread('csbtest.png')
image=cv.cvtColor(image,cv.COLOR_BGR2GRAY)
se=cv.getStructuringElement(cv.MORPH_RECT , (8,8))
bg=cv.morphologyEx(image, cv.MORPH_DILATE, se)
out_gray=cv.divide(image, bg, scale=255)
out_binary=cv.threshold(out_gray, 0, 255, cv.THRESH_OTSU )[1] 


nbin = cv.bitwise_not(out_binary)
pytesseract.image_to_string(out_gray)
# d = pytesseract.image_to_data(nbin, output_type=Output.DICT)
# n_boxes = len(d['level'])
# for i in range(n_boxes):
#     (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
#     cv.rectangle(nbin, (x, y), (x + w, y + h), (0, 255, 0), 2)



# cv.imshow("bin", nbin)
# cv.imshow('binary', out_binary)  
# cv.imwrite('binary.png',out_binary)

# cv.imshow('gray', out_gray)  
# cv.imwrite('gray.png',out_gray)

# cv.waitKey(0)

# img = cv.imread(IMAGE_FP)

# dst = cv.fastNlMeansDenoisingColored(img,None,10,10,7,21)

# img = cv.medianBlur(img,5)
# converted_img = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
# cdst = cv.fastNlMeansDenoisingColored(converted_img,None,10,10,7,21)


# ret,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
# th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
#             cv.THRESH_BINARY,11,2)
# th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
#             cv.THRESH_BINARY,11,2)
# titles = ['Original Image', 'Global Thresholding (v = 127)',
#             'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
# images = [img, th1, th2, th3]
# for i in range(4):
#     plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])
# plt.show()


# for i in range(4):
#     print(pytesseract.image_to_string(images[i]))
#     print("####")

# img = Image.open(IMAGE_FP)
# img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
#             cv.THRESH_BINARY,11,2)
# If you don't have tess
# eract executable in your PATH, include the following:


# Simple image to string
# print(pytesseract.image_to_string(th2))

# In order to bypass the image conversions of pytesseract, just use relative or absolute image path
# NOTE: In this case you should provide tesseract supported images or tesseract will return error
# print(pytesseract.image_to_string('test.png'))

# List of available languages
# print(pytesseract.get_languages(config=''))

# French text image to string
# print(pytesseract.image_to_string(Image.open('test-european.jpg'), lang='fra'))

# Batch processing with a single file containing the list of multiple image file paths
# print(pytesseract.image_to_string('images.txt'))

# Timeout/terminate the tesseract job after a period of time
# try:
#     print(pytesseract.image_to_string('test.jpg', timeout=2)) # Timeout after 2 seconds
#     print(pytesseract.image_to_string('test.jpg', timeout=0.5)) # Timeout after half a second
# except RuntimeError as timeout_error:
#     # Tesseract processing is terminated
#     print("runtime error")
#     pass

# Get bounding box estimates
# print(pytesseract.image_to_boxes(Image.open('kda.png')))

# Get verbose data including boxes, confidences, line and page numbers
# print(pytesseract.image_to_data(Image.open('sbtest.png')))

# Get information about orientation and script detection
# print(pytesseract.image_to_osd(Image.open('test.png')))

# Get a searchable PDF
# pdf = pytesseract.image_to_pdf_or_hocr('test.png', extension='pdf')
# with open('test.pdf', 'w+b') as f:
#     f.write(pdf) # pdf type is bytes by default

# Get HOCR output
# hocr = pytesseract.image_to_pdf_or_hocr('test.png', extension='hocr')

# Get ALTO XML output
# xml = pytesseract.image_to_alto_xml('test.png')