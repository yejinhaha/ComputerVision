import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = "./src/mistyroad.jpg"  
image = cv.imread(image_path)

# 그레이스케일로 변환
gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# 특정 임계값 설정
threshold_value = 127
_, bin_image = cv.threshold(gray_image, threshold_value, 255, cv.THRESH_BINARY)


# 히스토그램 계산
histogram = cv.calcHist([gray_image], [0], None, [256], [0, 256])

# Display results
plt.figure(figsize=(10, 5))

# 시각화
plt.subplot(1, 2, 1)
plt.imshow(gray_image, cmap="gray")
plt.title("grayscale Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.plot(histogram, color="black")
plt.title("Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.show()