import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 불러오기 (컬러)
img = cv2.imread('./src/edgeDetectionImage.jpg')  

# 그레이스케일로 변환
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Sobel 에지 검출 (X, Y 방향)
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

# 에지 강도 계산 (벡터 크기: magnitude)
magnitude = cv2.magnitude(sobel_x, sobel_y)

# 정수형 변환 (시각화)
magnitude_uint8 = cv2.convertScaleAbs(magnitude)

# 결과 시각화
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.title('Original (Grayscale)')
plt.imshow(gray, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Sobel Edge Magnitude')
plt.imshow(magnitude_uint8, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Sobel X + Y Combined')
combined = cv2.convertScaleAbs(sobel_x) + cv2.convertScaleAbs(sobel_y)
plt.imshow(combined, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()