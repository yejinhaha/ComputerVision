import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 이미지 로드
image_path = "./src/JohnHancocksSignature.png" 
image = cv.imread(image_path, cv.IMREAD_UNCHANGED)

# 임계값을 적용하여 이진화

t, bin_image = cv.threshold(image[:,:,3], 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

# 사각형(5x5) 커널 생성
kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))

# 모폴로지 연산 적용
dilation = cv.morphologyEx(bin_image, cv.MORPH_DILATE, kernel)  # 팽창
erosion = cv.morphologyEx(bin_image, cv.MORPH_ERODE, kernel)  # 침식
opening = cv.morphologyEx(bin_image, cv.MORPH_OPEN, kernel)  # 열림
closing = cv.morphologyEx(bin_image, cv.MORPH_CLOSE, kernel)  # 닫힘

# 결과를 한 줄로 정렬
result_images = np.hstack((bin_image, dilation, erosion, opening, closing))
titles = ["Original", "Dilation", "Erosion", "Open", "Close"]

# 결과 출력
plt.figure(figsize=(15, 5))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(result_images[:, i * bin_image.shape[1]:(i+1) * bin_image.shape[1]], cmap="gray")
    plt.title(titles[i])
    plt.axis("off")

plt.show()