import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('./src/dabo.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 2. Canny 에지 맵 생성
edges = cv2.Canny(gray, 100, 200)

# 3. Hough 변환으로 직선 검출
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, 
                        minLineLength=150, maxLineGap=10)

# 4. 검출된 직선 원본 이미지에 빨간 선으로 표시
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 빨간색, 두께 2

# 5. 시각화
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Detected Lines with Canny and Hough Transform')
plt.axis('off')
plt.show()