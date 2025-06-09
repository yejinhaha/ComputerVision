import cv2
import numpy as np
import sys
from sort import Sort  # sort.py에 정의된 Sort 클래스

#YOLOv4 모델 로드
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255)

#COCO 클래스 로드
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

colors = np.random.uniform(0, 255, size=(len(classes), 3))
sort = Sort()

#비디오 열기
cap = cv2.VideoCapture('./src/slow_traffic_small.mp4')
if not cap.isOpened():
    print("비디오 파일을 열 수 없습니다.")
    sys.exit()

#리사이즈 크기 정의 (예: 640x360)
resize_width, resize_height = 640, 360

#결과 저장용 비디오 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 또는 'XVID'
out = cv2.VideoWriter('output_tracked.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (resize_width, resize_height))

# OpenCV 창 크기 설정
cv2.namedWindow("Person Tracking by SORT", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Person Tracking by SORT", resize_width, resize_height)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 💡 YOLO 객체 검출
    class_ids, scores, boxes = model.detect(frame, confThreshold=0.5, nmsThreshold=0.4)

    # 전체 클래스 대상 (모든 객체 추적)
    objects = []
    for class_id, score, box in zip(class_ids, scores, boxes):
        x, y, w, h = box
        objects.append([x, y, x + w, y + h, score])

    # persons = []
    # for class_id, score, box in zip(class_ids, scores, boxes):
    #     if class_id == 0:  # 사람 class만 추적
    #         x, y, w, h = box
    #         persons.append([x, y, x + w, y + h, score])

    # SORT로 추적
    if len(objects) == 0:
        tracks = sort.update()
    else:
        tracks = sort.update(np.array(objects))

    # if len(persons) == 0:
    #     tracks = sort.update()
    # else:
    #     tracks = sort.update(np.array(persons))
    # for class_id, score, box in zip(class_ids, scores, boxes):
    #     x, y, w, h = box
    #     label = f"{classes[class_id]}"
    #     color = colors[class_id]
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    #     cv2.putText(frame, label, (x, y - 10),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    #     objects.append([x, y, x + w, y + h, score])

    for track in tracks:
        x1, y1, x2, y2, track_id = track.astype(int)
        color = colors[int(track_id) % len(colors)]
        # label = f"{classes[class_id]}: {score:.2f}"
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID: {int(track_id)}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

        # cv2.putText(frame, (x, y - 10),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[class_id], 2)
    #프레임 리사이즈
    resized_frame = cv2.resize(frame, (resize_width, resize_height))

    #결과 저장
    out.write(resized_frame)

    #결과 출력
    cv2.imshow("Tracking by SORT", resized_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

#종료
cap.release()
out.release()
cv2.destroyAllWindows()
