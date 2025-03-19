import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 이미지 불러오기
image_path = "./src/tree.png"  # 파일 경로 설정
image = cv.imread(image_path)

# 원본 이미지 크기 가져오기
rows, cols = image.shape[:2]

# 45도 회전 변환 행렬 생성 
rotation_matrix = cv.getRotationMatrix2D((cols / 2, rows / 2), 45, 1.5)

# 회전 및 확대 적용 (선형 보간 적용)
rotated_scaled_image = cv.warpAffine(image, rotation_matrix, (int(cols * 1.5), int(rows * 1.5)), flags=cv.INTER_LINEAR)

# 결과 출력
plt.figure(figsize=(10, 5))

# 원본 이미지 표시
plt.subplot(1, 2, 1)
plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

# 변환된 이미지 표시
plt.subplot(1, 2, 2)
plt.imshow(cv.cvtColor(rotated_scaled_image, cv.COLOR_BGR2RGB))
plt.title("Rotation & Scaled Image")
plt.axis("off")

plt.show()