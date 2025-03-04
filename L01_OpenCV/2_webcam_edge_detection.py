import cv2
import numpy as np
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,100,200)
    combined = np.hstack((frame, cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)))
    cv2.imshow('Video Display', combined)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    

cap.release()
cv2.destroyAllWindows()