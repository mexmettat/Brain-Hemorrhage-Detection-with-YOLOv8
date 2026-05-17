import os
import random
import cv2
import matplotlib.pyplot as plt
import yaml
import numpy as np

def load_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data

def get_yolo_boxes(label_path, img_w, img_h):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                cls_id = int(parts[0])
                x_center = float(parts[1]) * img_w
                y_center = float(parts[2]) * img_h
                w = float(parts[3]) * img_w
                h = float(parts[4]) * img_h
                x1 = int(x_center - w / 2)
                y1 = int(y_center - h / 2)
                x2 = int(x_center + w / 2)
                y2 = int(y_center + h / 2)
                boxes.append((cls_id, x1, y1, x2, y2))
    return boxes

def create_gallery():
    yaml_path = 'data.yaml'
    data = load_yaml(yaml_path)
    class_names = data.get('names', [])
    
    train_images_dir = 'train/images'
    train_labels_dir = 'train/labels'
    
    if not os.path.exists(train_images_dir):
        print(f"HATA: {train_images_dir} bulunamadı.")
        return

    all_images = [f for f in os.listdir(train_images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    if len(all_images) == 0:
        print("HATA: Eğitim klasöründe hiç resim bulunamadı.")
        return

    random.seed(42)
    selected_images = random.sample(all_images, min(20, len(all_images)))
    
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    axes = axes.flatten()
    
    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255)]
    
    for i, img_name in enumerate(selected_images):
        img_path = os.path.join(train_images_dir, img_name)
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(train_labels_dir, label_name)
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        
        boxes = get_yolo_boxes(label_path, w, h)
        
        for box in boxes:
            cls_id, x1, y1, x2, y2 = box
            color = colors[cls_id % len(colors)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cls_name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
            (tw, th), _ = cv2.getTextSize(cls_name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x1, max(y1-20, 0)), (x1+tw, max(y1, 20)), color, -1)
            cv2.putText(img, cls_name, (x1, max(y1-5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f"{img_name} (BBox: {len(boxes)})", fontsize=10)
        
    plt.suptitle("Label Dogrulama Galerisi (Rastgele 20 Goruntu)", fontsize=24, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig('label_gallery_v2.png', dpi=300)
    print("\n[BASARILI] 'label_gallery_v2.png' basariyla olusturuldu. Rapora ekleyebilirsiniz.")

if __name__ == '__main__':
    create_gallery()
