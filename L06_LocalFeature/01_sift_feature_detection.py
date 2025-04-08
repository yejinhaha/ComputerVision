import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('./src/mot_color70.jpg')
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#SIFT 객체 생성 (nfeatures 조절 가능)
sift = cv.SIFT_create(nfeatures=1000)

#특징점 검출, 디스크립터 계산
keypoints, descriptors = sift.detectAndCompute(img_gray, None)

img_sift = cv.drawKeypoints(
    img, keypoints, None,
    flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv.cvtColor(img_sift, cv.COLOR_BGR2RGB))
plt.title('SIFT Keypoints')
plt.axis('off')

plt.tight_layout()
plt.show()
