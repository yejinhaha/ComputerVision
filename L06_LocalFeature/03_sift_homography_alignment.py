import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

# 1. 이미지 불러오기
img1 = cv.imread('./src/img1.jpg')  # 기준 이미지 (정렬 대상)
img2 = cv.imread('./src/img2.jpg')  # 이동 이미지 (정렬할 이미지)

gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

#SIFT 특징점 검출
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

#BFMatcher로 매칭 + knnMatch
bf = cv.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

#Lowe’s ratio test 적용
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

#매칭 점 수 확인
if len(good_matches) > 10:
    #대응점 추출
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    #호모그래피 계산 (RANSAC 사용)
    H, mask = cv.findHomography(dst_pts, src_pts, cv.RANSAC, 5.0)

    #이미지 변환 (img2 → img1과 정렬)
    height, width, _ = img1.shape
    aligned_img = cv.warpPerspective(img2, H, (width, height))

    #결과 비교 시각화
    plt.figure(figsize=(20, 10))

    plt.subplot(1, 3, 1)
    plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
    plt.title('Base Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB))
    plt.title('Unaligned Image')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(cv.cvtColor(aligned_img, cv.COLOR_BGR2RGB))
    plt.title('Aligned Image (warpPerspective)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

else:
    print(f"Not enough good matches found: {len(good_matches)} (need at least 10)")


# 기존 코드에서 만든 것
# img1 = base image
# aligned_img = img2을 img1에 맞춰 정렬한 결과

#두 이미지를 같은 크기로 만듦 (이미 맞춰져 있으면 생략 가능)
h1, w1 = img1.shape[:2]
h2, w2 = aligned_img.shape[:2]

#출력 이미지의 크기 설정 (폭은 두 배로 확장)
result_width = w1 + w2
result_height = max(h1, h2)

#파노라마용 빈 이미지 생성
panorama = np.zeros((result_height, result_width, 3), dtype=np.uint8)

#base 이미지 복사
panorama[0:h1, 0:w1] = img1

#정렬된 이미지 겹쳐 붙이기 (단순 덮기)
# → 겹치는 부분을 블렌딩 추가 필요
for y in range(h2):
    for x in range(w2):
        # aligned_img 픽셀이 유효한 영역일 때만 붙이기
        if not np.all(aligned_img[y, x] == 0):
            panorama[y, x] = aligned_img[y, x]

# 겹치는 부분만 블렌딩
blended = img1.copy()
alpha = 0.5  # 투명도 조절 (0.0~1.0)

# 블렌딩 적용: aligned_img의 유효 픽셀만 base 이미지와 평균 처리
mask = np.any(aligned_img > 0, axis=2)
blended[mask] = cv.addWeighted(img1[mask], alpha, aligned_img[mask], 1 - alpha, 0)

plt.figure(figsize=(20, 10))
plt.imshow(cv.cvtColor(panorama, cv.COLOR_BGR2RGB))
plt.title('Panorama-style Composition')
plt.axis('off')
plt.show()

plt.subplot(1, 2, 2)
plt.imshow(cv.cvtColor(blended, cv.COLOR_BGR2RGB))
plt.title('Blended Image (Overlay)')
plt.axis('off')

plt.tight_layout()
plt.show()

