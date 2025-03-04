import cv2 as cv
import numpy as np
import sys

# 마우스 드래그 관련 변수
isDragging = False
x0, y0, x1, y1 = -1, -1, -1, -1
roi = None  # 선택한 ROI 저장 변수

def draw(event, x, y, flags, param):
    global isDragging, x0, y0, x1, y1, roi, img

    if event == cv.EVENT_LBUTTONDOWN:  # 마우스 왼쪽 버튼 클릭 시작
        isDragging = True
        x0, y0 = x, y

    elif event == cv.EVENT_MOUSEMOVE:  # 마우스 이동 시 사각형 표시
        if isDragging:
            img_draw = img.copy()
            cv.rectangle(img_draw, (x0, y0), (x, y), (0, 0, 255), 2)
            cv.imshow('img', img_draw)

    elif event == cv.EVENT_LBUTTONUP:  # 마우스 왼쪽 버튼 클릭 해제
        if isDragging:
            isDragging = False
            x1, y1 = x, y

            # ROI 영역 계산
            x_start, x_end = min(x0, x1), max(x0, x1)
            y_start, y_end = min(y0, y1), max(y0, y1)

            if x_end - x_start > 0 and y_end - y_start > 0:
                roi = img[y_start:y_end, x_start:x_end]
                cv.imshow('ROI', roi)
                cv.moveWindow('ROI', 0, 0)
                print(f'ROI 선택 완료: x={x_start}, y={y_start}, w={x_end-x_start}, h={y_end-y_start}')
            else:
                print('잘못된 영역 선택. 다시 시도하세요.')

# 이미지 불러오기 및 예외 처리
img = cv.imread('0_LOVOT.png')
if img is None:
    sys.exit('파일을 찾을 수 없음')

cv.imshow('img', img)
cv.setMouseCallback('img', draw)

while True:
    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):  # 'q' 키를 누르면 종료
        break
    elif key == ord('r'):  # 'r' 키를 누르면 이미지 초기화
        print("ROI 선택 리셋")
        img = cv.imread('0_LOVOT.png')
        cv.imshow('img', img)
        roi = None
    elif key == ord('s'):  # 's' 키를 누르면 ROI 저장
        if roi is not None:
            cv.imwrite('saved_roi.png', roi)
            print("ROI가 saved_roi.png로 저장되었습니다.")
        else:
            print("저장할 ROI가 없습니다.")

cv.destroyAllWindows()
