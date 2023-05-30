import cv2 as cv2
import pytesseract
import numpy as np

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


# Overwatch2 scoreboard rectagnles for 1080p
# NOTE: these are y, x tuples
blue_top_left = (313, 159)
blue_bottom_right = (313, 500)
blue_top_right = (1162, 159)
blue_bottom_left = (1162, 500)

# red team
red_top_right = (313, 612)
red_bottom_right = (313, 919)
red_top_left = (1162, 919)
red_bottom_left = (1162, 612)

# each column should be this size on 1080p images
column_yx_size = (60, 845)

# cuts each column out of a scoreboard display
# TODO: make this determine the regions based on img display ratio
# NOTE: current implimentation is only good for 1080p images
def cut_out_columns():
    img = cv2.imread("test_videos/extracted_frames/ow_scoreboard.png")
    blue_cols = []
    red_cols = []
    blue_y = 195
    red_y = 612
    x = 315

    for i in range(5):
        print("blue_y: " + str(blue_y))
        # blue_cols[i] = cut_out(img, x, blue_y)
        blue_cols.append(cut_out(img, x, blue_y))
        blue_y += 60 
        print("red_y: " + str(red_y))
        red_cols.append(cut_out(img, x, red_y))
        # red_cols[i] = cut_out(img, x, red_y)
        # red_cols[i] = img[red_y : red_y + 60, x : x + 850]
        red_y += 60
    return (blue_cols, red_cols)


def cut_out(img, x, y):
    """ 
    
    """
    return img[y:y+ 60, x:x + 850]

# o = cut_out_columns()
# for i in range(5):
#     cv2.imshow(str(i), o[1][i])
#     cv2.waitKey(0)


# for kda its 700 - 850 on the x axis so starting at 690 and incrementing ~50 should get the 3 numbers
# damage is 850-950
# healing is 950 -1050
# and mit is 1050 - 1150

# the info to grab a column just needs to be one single y,x pair and the height and width of the
# area you are grabbing.
# for player 1 on blue this is x=315, y=195, for player 1 on red it is x=315, y=612
# then for each blue player it should be y+60
# 
# for 16:9 display ratios the start should be:
# blue: x=0.164 and y=0.180
# red: x=0.164 and y=0.566



# chat can overlap the names, but the names should always come up in the same order
# (?) is there a way to detect if the name is valie
# names can also be screwed up becuase various characters


# NOTE: this doesn't extract names.
def get_rows_from_column(img):
    rows = []
    current_x = 365  # the start of th kda
    next = 60
    for i in range(6):
          # spacing for kda
        if i == 3: # after kda increase spacing 
            next = 100 
        rows.append(img[0 : 60, current_x:current_x + next])
        current_x += next
    return rows



# get image
# img = cv2.imread("test_videos/extracted_frames/ow_scoreboard.png")
# get player 1 col
# test_column = cut_out(img, 315, 195)
# cv2.imwrite("test_column", test_column)
# cv2.imwrite("test_videos/extracted_frames/test_column.png" ,test_column)
# cut the col up into values
# 
# for this image x's values are 
# kda: 365 - 545
# dmg: 540 - 640
# healing: 640 - 740
# mit: 740 - 840
# y should always be 60

img = cv2.imread("/home/evie/Projects/Statminer/test_videos/extracted_frames/test_column.png")
# im = img

# row, col = im.shape[:2]
# bottom = im[row-2:row, 0:col]
# mean = cv2.mean(bottom)[0]

# border_size = 10
# border = cv2.copyMakeBorder(
#     im,
#     top=border_size,
#     bottom=border_size,
#     left=border_size,
#     right=border_size,
#     borderType=cv2.BORDER_CONSTANT,
#     value=[mean, mean, mean]
# )

# cv2.imshow('image', im)
# cv2.imshow('bottom', bottom)
# cv2.imshow('border', border)
# cv2.waitKey(0)





rows = get_rows_from_column(img)
rows.append(img)
read = []

def proc_col(col):
    rows = get_rows_from_column(col)
    rows.append(col)
    for i in range (7):
        out_img = cv2.bitwise_not(rows[i])
        # out_img = rows[i]
        # image=cv2.cvtColor(rows[i],cv2.COLOR_BGR2GRAY)

        image=cv2.cvtColor(out_img,cv2.COLOR_BGR2GRAY)
        se=cv2.getStructuringElement(cv2.MORPH_RECT , (8,8))
        bg=cv2.morphologyEx(image, cv2.MORPH_DILATE, se)
        out_gray=cv2.divide(image, bg, scale=255)


        out_binary=cv2.threshold(out_gray, 0, 255, cv2.THRESH_OTSU )[1] 


        scale_percent = 150 # percent of original size
        width = int(out_binary.shape[1] * scale_percent / 100)
        height = int(out_binary.shape[0] * scale_percent / 100)
        dim = (width, height)

        # # resize image
        # resized = cv2.resize(out_gray, dim, interpolation = cv2.INTER_AREA)    
        resized = cv2.resize(out_binary, dim, interpolation = cv2.INTER_AREA)    

        #   Creating kernel
        kernel = np.ones((2, 2), np.uint8)

        # Using cv2.erode() method 
        # im = cv2.erode(out_gray, kernel, iterations=1) 
        # im = cv2.erode(resized, kernel, iterations=1) 
        # im = cv2.bitwise_not(out_gray)
        # im = out_gray
        # im = out_gray
        im = resized
        # im = out_binary
        # im = out_img
        # im = rows[i]
        row, col = im.shape[:2]
        bottom = im[row-2:row, 0:col]
        mean = cv2.mean(bottom)[0]

        border_size = 1
        border = cv2.copyMakeBorder(
        im,
        top=border_size,
        bottom=border_size,
        left=border_size,
        right=border_size,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
        # value=[255,255,255]
        )
        print(str(i) + ": " + pytesseract.image_to_string(border))
        if i == 6:
            cv2.imshow(str(i),  border)



teams = cut_out_columns()
def proc_team(team_cols):
    for c in team_cols:
        print("##########")
        proc_col(c)

proc_team(teams[1])



# for i in range (7):
#     out_img = cv2.bitwise_not(rows[i])
#     # out_img = rows[i]
#     # image=cv2.cvtColor(rows[i],cv2.COLOR_BGR2GRAY)

#     image=cv2.cvtColor(out_img,cv2.COLOR_BGR2GRAY)
#     se=cv2.getStructuringElement(cv2.MORPH_RECT , (8,8))
#     bg=cv2.morphologyEx(image, cv2.MORPH_DILATE, se)
#     out_gray=cv2.divide(image, bg, scale=255)

    
#     out_binary=cv2.threshold(out_gray, 0, 255, cv2.THRESH_OTSU )[1] 

  
#     scale_percent = 300 # percent of original size
#     width = int(out_binary.shape[1] * scale_percent / 100)
#     height = int(out_binary.shape[0] * scale_percent / 100)
#     dim = (width, height)
  
# # # resize image
#     resized = cv2.resize(out_gray, dim, interpolation = cv2.INTER_AREA)    
#     # resized = cv2.resize(out_binary, dim, interpolation = cv2.INTER_AREA)    

# #   Creating kernel
#     kernel = np.ones((2, 2), np.uint8)
  
#     # Using cv2.erode() method 
#     # im = cv2.erode(out_gray, kernel, iterations=1) 
#     im = cv2.erode(resized, kernel, iterations=1) 
#     # im = cv2.bitwise_not(out_gray)
#     # im = out_gray
#     # im = out_gray
#     im = resized
#     # im = out_binary
#     # im = out_img
#     # im = rows[i]
#     row, col = im.shape[:2]
#     bottom = im[row-2:row, 0:col]
#     mean = cv2.mean(bottom)[0]

#     border_size = 1
#     border = cv2.copyMakeBorder(
#     im,
#     top=border_size,
#     bottom=border_size,
#     left=border_size,
#     right=border_size,
#     borderType=cv2.BORDER_CONSTANT,
#     value=[0, 0, 0]
#     # value=[255,255,255]
#     )



#     print(str(i) + ": " + pytesseract.image_to_string(border))

#     cv2.imshow(str(i),  border)

cv2.waitKey(0)
cv2.destroyAllWindows()
