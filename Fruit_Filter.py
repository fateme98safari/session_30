import numpy as np
import cv2
import tensorflow as tf
from functools import partial
import time
from TFLiteFaceDetector import UltraLightFaceDetecion
from TFLiteFaceAlignment import CoordinateAlignmentModel




def zoom_effect(my_landmarks,pred,image):
#     for i,p in enumerate(np.round(pred).astype(np.int)):
#         cv2.circle(image, tuple(p), 1, color, 1, cv2.LINE_AA)
#         cv2.putText(image,str(i),tuple(p),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255))
    landmarks=[]           
    for i in my_landmarks:
            landmarks.append(pred[i])
    landmarks=np.array( landmarks,dtype=int)

    x, y, w, h=cv2.boundingRect(landmarks)
    mask=np.zeros(image.shape,dtype=np.uint8)
    cv2.drawContours(mask,[landmarks],-1,(255,255,255),-1)
    mask=mask//255
    result=image * mask 
    result=result[y:y+h,x:x+w]
    result_big=cv2.resize(result,(0,0),fx=2,fy=2)
    for i in range(h*2):
        for j in range(w*2):
            if result_big[i][j][0] == 0 and result_big[i][j][1] == 0 and result_big[i][j][2] == 0:
                result_big[i][j] = apple_image[int(y-h//2)+i, int(x-w//2)+j]

    apple_image[int(y-h//2):int(y-h//2)+h*2, int(x-w//2):int(x-w//2)+w*2]=result_big
    return apple_image
    


fd = UltraLightFaceDetecion("OpenVtuber/weights/RFB-320.tflite", conf_threshold=0.88)
fa = CoordinateAlignmentModel("OpenVtuber/weights/coor_2d106.tflite")

    # cap = cv2.VideoCapture(0)
image=cv2.imread("OpenVtuber\input\IMG_4714.JPG")
apple_image=cv2.imread("OpenVtuber/input/apple_158989157.jpg")

color = (125, 255, 125)

    # while True:
    #     ret, frame = cap.read()

    #     if not ret:
    #         break

start_time = time.perf_counter()

boxes, scores = fd.inference(image)

for pred in fa.get_landmarks(image, boxes):
      
        # for i,p in enumerate(np.round(pred).astype(np.int)):
        #         cv2.circle(image, tuple(p), 1, color, 1, cv2.LINE_AA)
        #         cv2.putText(image,str(i),tuple(p),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255))
        L_eye_landmarks=[35,36,33,37,39,42,40,41]
        R_eye_landmarks=[89,90,87,91,93,96,94,95]
        lips_landmarks=[52,55,56,53,59,58,61,68,67,63,64]

zoom_effect(L_eye_landmarks,pred,image)
zoom_effect(R_eye_landmarks,pred,image)
zoom_effect(lips_landmarks,pred,image)




cv2.imshow("result",apple_image)

cv2.waitKey()
cv2.imwrite("OpenVtuber/output/Fruit_filter1.jpg",apple_image)

        # if cv2.waitKey(1) == ord('q'):
        #     break