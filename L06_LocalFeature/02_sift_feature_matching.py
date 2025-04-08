import cv2 as cv
import matplotlib.pyplot as plt

img1 = cv.imread('./src/mot_color70.jpg')
img2 = cv.imread('./src/mot_color83.jpg')
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

#SIFT 객체 생성
sift = cv.SIFT_create()

#특징점 검출 및 디스크립터 계산
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

#=====1======= : BFMatcher
#BFMatcher 생성 (L2 거리, crossCheck로 정밀한 매칭)
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
#매칭 수행
matches = bf.match(des1, des2)
#매칭 결과 정렬 (거리 기준 오름차순)
matches = sorted(matches, key=lambda x: x.distance)
#매칭 결과 시각화 (상위 50개만 표시)
matched_img = cv.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)



#=====2====== : Flann
# #FLANN 매칭을 위한 파라미터 설정
# FLANN_INDEX_KDTREE = 1  # 알고리즘 타입: KD-Tree
# index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
# search_params = dict(checks=50)  # 탐색 반복 횟수

# #FLANN 매처 생성
# flann = cv.FlannBasedMatcher(index_params, search_params)

# #knnMatch 사용 (각 특징점당 2개의 최근접 이웃)
# matches = flann.knnMatch(des1, des2, k=2)
# #Lowe's ratio test 적용
# good_matches = []
# for m, n in matches:
#     if m.distance < 0.7 * n.distance:
#         good_matches.append(m)

# matched_img = cv.drawMatches(img1, kp1, img2, kp2, good_matches, None,
#                              flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)



plt.figure(figsize=(20, 10))
plt.imshow(cv.cvtColor(matched_img, cv.COLOR_BGR2RGB))
plt.title('SIFT Feature Matching')
#plt.title(f'SIFT Matching with FLANN (Good matches: {len(good_matches)})')
plt.axis('off')
plt.show()
