import cv2
from ultralytics import YOLO  

# 1. 모델 로드 (가중치 경로 입력)
model_path = r"E:\AI_KDT7\12.Transfer_learning\mini\runs\detect\train\weights\best.pt"
model = YOLO(model_path)

# 2. 이미지 불러오기
img_path = r"E:\AI_KDT7\12.Transfer_learning\mini\vision\H_8205.20-0000_02_846.png"  # 예측할 이미지 경로
img = cv2.imread(img_path)

# 3. 예측 실행
results = model(img)

# 4. 결과 이미지 얻기 (바운딩 박스와 라벨이 그려진 이미지)
result_img = results[0].plot()  # 첫 번째 결과 이미지

# 5. 결과 이미지 출력 및 저장
cv2.imshow("YOLOv5 Prediction", result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 저장 원할 경우
cv2.imwrite("output.jpg", result_img)