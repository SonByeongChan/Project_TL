# import torch
# import os
# import yaml
# from pathlib import Path
# from ultralytics import YOLO  # YOLOv5 모델을 가져옵니다.

# os.environ['KMP_DUPLICATE_LIB_OK']='True'

# def main():
#     # 설정
#     data_dir = r'D:\AI_KDT7\12.Transfer_learning\mini\dataset'  # 데이터셋 경로
#     train_images_dir = os.path.join(data_dir, 'train')  # 훈련 이미지 경로
#     val_images_dir = os.path.join(data_dir, 'val')  # 검증 이미지 경로

#     # 클래스 이름
#     class_names = ['Firecracker', 'Hammer', 'NailClippers', 'Spanner', 'Thinner', 'ZippoOil']

#     # 데이터셋 설정
#     data_yaml = {
#         'train': str(Path(train_images_dir)),
#         'val': str(Path(val_images_dir)),
#         'nc': len(class_names),
#         'names': class_names
#     }

#     # YAML 파일로 저장
#     with open('data.yaml', 'w') as f:
#         yaml.dump(data_yaml, f)

#     # YOLOv5 모델 로드
#     model = YOLO("yolov5n.yaml")  # 모델을 YAML 파일로부터 로드합니다.

#     # 훈련 설정
#     epochs = 200
#     device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#     print(f'Using device: {device}')

#     # 모든 가중치 고정 해제
#     for param in model.parameters():
#         param.requires_grad = True  # 모든 가중치 학습 가능하도록 설정

#     # 모델 훈련
#     results = model.train(data='data.yaml', epochs=epochs, imgsz=640, batch=16, device=device)

#     # 평가
#     metrics = model.val(data='data.yaml', imgsz=640)

#     # IoU, Recall, Precision, F1 Score, Confusion Matrix
#     iou = metrics['metrics']['IoU']
#     recall = metrics['metrics']['recall']
#     precision = metrics['metrics']['precision']
#     f1 = metrics['metrics']['f1']
#     confusion_matrix = metrics['metrics']['confusion_matrix']

#     print(f'IoU: {iou:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, F1: {f1:.4f}')
#     print(f'Confusion Matrix:\n{confusion_matrix}')

#     print('Training complete.')

# if __name__ == '__main__':
#     main()

import torch
from pathlib import Path


def predict(model, image_path):
    # 이미지를 모델에 입력하여 예측을 수행합니다.
    results = model(image_path)
    return results

def main():
    model_path = r'D:\AI_KDT7\12.Transfer_learning\mini\runs\detect\train\weights\best.pt'  # 저장된 모델 경로
    image_path = r'D:\AI_KDT7\12.Transfer_learning\mini\xray_img\Astrophysics\[Astro]Hammer\Hammer\Single_Other\H_8205.20-0000_01_430.png'  # 추론할 이미지 경로

    # 모델 불러오기
    model = load_model(model_path)

    # 예측 수행
    results = predict(model, image_path)

    # 결과 출력
    results.print()  # 예측 결과 출력
    results.show()   # 예측 결과 이미지 표시

    # 결과를 저장하고 싶다면
    results.save(Path('output/'))  # 'output/' 폴더에 결과 저장

if __name__ == '__main__':
    main()
