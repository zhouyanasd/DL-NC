import numpy as np
import cv2

cap = cv2.VideoCapture("/Users/scarecrow/PycharmProjects/DL-NC/Brain scripts/Data/KHT/person25_boxing_d4_uncomp.avi")
print(cap.isOpened())

while(cap.isOpened()):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame',gray)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()