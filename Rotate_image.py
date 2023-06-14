import cv2
import numpy as np

image=cv2.imread("OpenVtuber/input/IMG_4714.JPG")
image=cv2.rotate(image,cv2.ROTATE_180)

cv2.imwrite("OpenVtuber/output/Rotate_result.jpg",image)
cv2.waitKey()