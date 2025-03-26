import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 이미지 불러오기
image = cv2.imread('./src/dabo.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 2. 마우스로 ROI 선택
# ROI: (x, y, w, h)
rect = cv2.selectROI("Select ROI", image, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Select ROI")  # ROI 선택 창 닫기

# 3. 마스크 및 모델 초기화
mask = np.zeros(image.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# 4. GrabCut 실행
cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

# 5. 마스크 처리 (배경: 0, 전경: 1)
mask2 = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype('uint8')
result = image_rgb * mask2[:, :, np.newaxis]

# 6. 시각화
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(mask, cmap='gray')
plt.title("GrabCut Mask")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(result)
plt.title("Foreground Extracted")
plt.axis("off")

plt.tight_layout()
plt.show()