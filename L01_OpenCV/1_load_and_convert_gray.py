import cv2 as cv
import sys
import numpy as np


img = cv.imread('soccer.jpg')

#img 원본 이미지 #gray 흑백 이미지
if img is None:
    sys.exit('파일이 존재하지 않습니다')

img = cv.resize(img, dsize=(0,0), fx=0.5,fy=0.5)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray_3ch = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

stacked = np.hstack((img, gray_3ch))

print(img.shape)
print(gray.shape)
print(gray_3ch.shape)
print(gray_3ch[0,0,0],gray_3ch[0,0,1],gray_3ch[0,0,2])
# # 결과 출력
cv.imshow("Original & Grayscale", stacked)


while(True):
    if cv.waitKey(1)==ord('q'):
        cv.destroyAllWindows()
        break

