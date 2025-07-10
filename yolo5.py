import os
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm  # tqdm 임포트

# 1. XML -> YOLO 라벨 변환 함수
def convert_bbox(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x_center = (box[0] + box[1]) / 2.0 - 1
    y_center = (box[2] + box[3]) / 2.0 - 1
    width = box[1] - box[0]
    height = box[3] - box[2]
    x_center *= dw
    width *= dw
    y_center *= dh
    height *= dh
    return (x_center, y_center, width, height)

def xml_to_yolo_labels(xml_folder, save_folder, classes):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    xml_files = [f for f in os.listdir(xml_folder) if f.endswith('.xml')]
    for xml_file in xml_files:
        tree = ET.parse(os.path.join(xml_folder, xml_file))
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        txt_path = os.path.join(save_folder, xml_file.replace('.xml', '.txt'))
        with open(txt_path, 'w') as f:
            for obj in root.iter('object'):
                cls = obj.find('name').text
                if cls not in classes:
                    continue
                cls_id = classes.index(cls)
                xmlbox = obj.find('bndbox')
                b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
                     float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                bb = convert_bbox((w, h), b)
                f.write(f"{cls_id} {' '.join(map(str, bb))}\n")
    print(f"XML to YOLO 라벨 변환 완료. 총 {len(xml_files)} 파일 처리됨.")

# 2. 커스텀 Dataset
class YoloDataset(Dataset):
    def __init__(self, img_dir, label_dir, classes, img_size=640, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.classes = classes
        self.img_size = img_size
        self.transform = transform
        self.img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg','.png','.jpeg'))]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        label_path = os.path.join(self.label_dir, self.img_files[idx].replace(os.path.splitext(self.img_files[idx])[1], '.txt'))

        image = Image.open(img_path).convert('RGB')
        w, h = image.size

        # YOLO는 입력 크기 고정 필요 (letterbox resize)
        image = image.resize((self.img_size, self.img_size))

        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    cls_id, x, y, bw, bh = line.strip().split()
                    cls_id = int(cls_id)
                    x, y, bw, bh = map(float, (x,y,bw,bh))
                    xmin = (x - bw/2) * self.img_size
                    ymin = (y - bh/2) * self.img_size
                    xmax = (x + bw/2) * self.img_size
                    ymax = (y + bh/2) * self.img_size
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(cls_id)

        boxes = torch.tensor(boxes) if boxes else torch.zeros((0,4), dtype=torch.float32)
        labels = torch.tensor(labels) if labels else torch.zeros((0,), dtype=torch.int64)

        if self.transform:
            image = self.transform(image)
        else:
            image = T.ToTensor()(image)

        target = {'boxes': boxes, 'labels': labels}
        return image, target

# 3. 모델 (YOLOv5 간단 버전 예시)
from yolov5.models.yolo import Model  # yolov5 repo가 로컬에 있어야 함

def create_model(num_classes):
    model = Model(cfg='yolov5s.yaml', ch=3, nc=num_classes)  # 초기 가중치 없이 생성
    return model

# 4. 평가 지표 함수 (간단히 유지)
def evaluate(model, dataloader, device, classes):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for imgs, targets in dataloader:
            imgs = imgs.to(device)
            preds = model(imgs)
            for t in targets:
                all_targets.extend(t['labels'].cpu().numpy())
            all_preds.extend([0]*len(all_targets))  # 임시값

    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro', zero_division=0)
    cm = confusion_matrix(all_targets, all_preds)
    return precision, recall, f1, cm

# 5. 학습 루프에 tqdm 적용
def train(model, train_loader, val_loader, device, epochs, save_dir):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()  # 단순화

    best_f1 = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        # tqdm으로 학습 배치 진행 상황 표시
        loop = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{epochs}]', leave=False)
        for imgs, targets in loop:
            imgs = imgs.to(device)
            labels = torch.cat([t['labels'] for t in targets]).to(device)  # 단순화

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=total_loss/((loop.n)+1))

        precision, recall, f1, cm = evaluate(model, val_loader, device, classes)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f} - Precision: {precision:.4f} - Recall: {recall:.4f} - F1: {f1:.4f}")
        print("Confusion Matrix:")
        print(cm)

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), os.path.join(save_dir, f"best_model_epoch_{epoch+1}.pt"))
            print("모델 저장 완료")
