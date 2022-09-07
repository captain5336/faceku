import os
from tokenize import Name
from FaceKu import FaceKu
import cv2
import time
import numpy as np


# 实例化人脸库对象
fk = FaceKu()

camera_index = 0  # 0 is default windows camera
video_capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
input_width = 200

if (video_capture.isOpened() == False):
    print("Camera is unable to open.")

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    # Only process every other frame of video to save time
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    imgx = frame[:, :, ::-1]
    width = imgx.shape[1]
    ratio = input_width/width
    input_height = int(imgx.shape[0]*ratio)
    imgx = cv2.resize(imgx, (input_width, input_height), cv2.INTER_LINEAR)
    # 人脸识别
    res_comp = fk.face_compare(imgx)
    pred_name = res_comp['name_predict']
    bbox = res_comp['bbox']
    top, right, bottom, left = bbox
    top = int(top/ratio) + 15
    right = int(right/ratio)
    bottom = int(bottom/ratio)
    left = int(left/ratio)
    # Draw a box around the face
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    # Draw a label with a name below the face
    cv2.rectangle(frame, (left, bottom - 35),
                  (right, bottom), (0, 0, 255), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, pred_name, (left + 6, bottom - 6),
                font, 1.0, (255, 255, 255), 1)

    cv2.putText(frame, pred_name, (20, 50), font, 1.0, (255, 255, 255), 1)
    cv2.imshow('Video', frame)
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
