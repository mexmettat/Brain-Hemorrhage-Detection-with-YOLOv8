import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import yaml

def bb_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def load_gt_boxes(label_path, w, h):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls_id = int(parts[0])
                xc = float(parts[1]) * w
                yc = float(parts[2]) * h
                bw = float(parts[3]) * w
                bh = float(parts[4]) * h
                x1 = int(xc - bw / 2)
                y1 = int(yc - bh / 2)
                x2 = int(xc + bw / 2)
                y2 = int(yc + bh / 2)
                boxes.append({'box': [x1, y1, x2, y2], 'cls_id': cls_id})
    return boxes

def save_analysis_image(img_path, filename, pred_boxes, gt_boxes, class_names, title):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Draw Ground Truths (Green)
    for gbox in gt_boxes:
        b = gbox['box']
        cls_name = class_names[gbox['cls_id']]
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
        cv2.putText(img, f"GT: {cls_name}", (b[0], max(15, b[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
    # Draw Predictions (Red)
    for pbox in pred_boxes:
        b = pbox['box']
        cls_name = class_names[pbox['cls_id']]
        conf = pbox['conf']
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (255, 0, 0), 2)
        text = f"Pred: {cls_name} ({conf:.2f})"
        cv2.putText(img, text, (b[0], min(img.shape[0]-10, b[3]+20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.title(title, fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def main():
    model_path = "runs/detect/baseline/weights/best.pt"
    if not os.path.exists(model_path):
        model_path = "runs/detect/runs/baseline/weights/best.pt" # Fallback
        if not os.path.exists(model_path):
            print(f"HATA: Model agirliklari bulunamadi.")
            return
        
    model = YOLO(model_path)
    val_images_dir = "valid/images"
    val_labels_dir = "valid/labels"
    
    with open("data.yaml", "r") as f:
        data_yaml = yaml.safe_load(f)
    class_names = data_yaml.get('names', [])
    
    fps = []
    fns = []
    
    print("Hata analizi yapiliyor (False Positives ve False Negatives araniyor)...")
    
    for img_name in os.listdir(val_images_dir):
        if len(fps) >= 6 and len(fns) >= 6:
            break
            
        if not img_name.endswith(('.jpg', '.png', '.jpeg')):
            continue
            
        img_path = os.path.join(val_images_dir, img_name)
        label_path = os.path.join(val_labels_dir, os.path.splitext(img_name)[0] + '.txt')
        
        img = cv2.imread(img_path)
        if img is None: continue
        h, w, _ = img.shape
        
        gt_boxes = load_gt_boxes(label_path, w, h)
        
        results = model.predict(img_path, conf=0.25, verbose=False)
        pred_boxes = []
        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                pred_boxes.append({'box': [int(x1), int(y1), int(x2), int(y2)], 'cls_id': cls_id, 'conf': conf})
                
        matched_gt = set()
        matched_pred = set()
        
        # Match Predictions to GT
        has_fp = False
        has_fn = False
        
        for p_idx, pbox in enumerate(pred_boxes):
            best_iou = 0
            best_g_idx = -1
            for g_idx, gbox in enumerate(gt_boxes):
                if g_idx in matched_gt:
                    continue
                # Sınıfların ayni olmasi zorunludur!
                if pbox['cls_id'] != gbox['cls_id']:
                    continue
                    
                iou = bb_iou(pbox['box'], gbox['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_g_idx = g_idx
            
            if best_iou >= 0.5: # Standart mAP@0.5 IoU threshold
                matched_pred.add(p_idx)
                matched_gt.add(best_g_idx)
            else:
                has_fp = True
                    
        # Saf False Negative: Modelin hicbir tahminde bulunamadigi (len(pred_boxes) == 0) ama gercekte kanama olan (len(gt_boxes) > 0) durumlar
        if len(pred_boxes) == 0 and len(gt_boxes) > 0:
            has_fn = True
                
        if has_fp and len(fps) < 6:
            fps.append({'img_path': img_path, 'preds': pred_boxes, 'gt': gt_boxes})
            
        if has_fn and len(fns) < 6:
            fns.append({'img_path': img_path, 'preds': pred_boxes, 'gt': gt_boxes})

    os.makedirs("error_analysis", exist_ok=True)
    
    for i, fp in enumerate(fps):
        save_analysis_image(fp['img_path'], f"error_analysis/fp_{i+1}.png", fp['preds'], fp['gt'], class_names, f"False Positive {i+1} (Red: Prediction, Green: GT)")
        
    for i, fn in enumerate(fns):
        save_analysis_image(fn['img_path'], f"error_analysis/fn_{i+1}.png", fn['preds'], fn['gt'], class_names, f"False Negative {i+1} (Red: Prediction, Green: GT)")
        
    print(f"\n[BASARILI] {len(fps)} False Positive ve {len(fns)} False Negative analizi tamamlandi.")
    print("Incelenen resimler 'error_analysis' klasorune kaydedildi. Bunlari raporunuza ekleyebilirsiniz.")

if __name__ == "__main__":
    main()
