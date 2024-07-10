import cv2 as cv


def test_cam(source):
   cap = cv.VideoCapture(source)
   if cap is None or not cap.isOpened():
       return False
   return True







