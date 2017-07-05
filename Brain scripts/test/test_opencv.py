import numpy as np
import cv2

cap = cv2.VideoCapture("../Data/KHT/person25_boxing_d4_uncomp.avi")

while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow('image', gray)
        k = cv2.waitKey(20)
        if (k & 0xff == ord('q')):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()

# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
#
# # 图6-1中的矩阵
# img = np.array([
#     [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
#     [[255, 255, 0], [255, 0, 255], [0, 255, 255]],
#     [[255, 255, 255], [128, 128, 128], [0, 0, 0]],
# ], dtype=np.uint8)
#
# # 用matplotlib存储
# plt.imsave('img_pyplot.jpg', img)
#
# # 用OpenCV存储
# cv2.imwrite('img_cv2.jpg', img)
