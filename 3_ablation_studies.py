import os
from ultralytics import YOLO

def train_ablation_960():
    print("\n--- ABLATION A: Goruntu Boyutunu 960 Yaparak Egitim ---")
    model = YOLO("yolov8n.pt")
    model.train(
        data="data.yaml",
        epochs=50,
        imgsz=960,
        seed=42,
        patience=10,
        project="runs/detect",
        name="ablation_960",
        exist_ok=True
    )

def evaluate_ablation_conf_sweep():
    print("\n--- ABLATION B: Confidence Sweep (Esik Degeri Ablasyonu) ---")
    # Using the baseline trained weights
    best_model_path = "runs/detect/baseline/weights/best.pt"
    
    if not os.path.exists(best_model_path):
        # Fallback to older directory structure if needed
        best_model_path = "runs/detect/runs/baseline/weights/best.pt"
        if not os.path.exists(best_model_path):
            print(f"HATA: Model agirliklari bulunamadi. Once baseline egitimi tamamlayin.")
            return
        
    model = YOLO(best_model_path)
    
    confs = [0.001, 0.01, 0.05, 0.1, 0.2]
    best_f1 = 0
    best_conf = 0
    results_list = []
    
    for c in confs:
        print(f"\nDegerlendiriliyor: conf={c}")
        # Validate the model on validation set with specific confidence
        metrics = model.val(data="data.yaml", conf=c, split="val", verbose=False)
        
        map50 = metrics.box.map50
        map50_95 = metrics.box.map
        p = metrics.box.mp
        r = metrics.box.mr
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0
        
        results_list.append((c, map50, map50_95, p, r, f1))
        
        if f1 > best_f1:
            best_f1 = f1
            best_conf = c
            
    print("\n" + "="*70)
    print("ABLATION B SONUCLARI (Confidence Sweep)")
    print("="*70)
    print(f"{'Conf':<10} | {'mAP@0.5':<10} | {'mAP@0.5:0.95':<15} | {'Precision':<10} | {'Recall':<10} | {'F1':<10}")
    print("-" * 70)
    for res in results_list:
        print(f"{res[0]:<10.3f} | {res[1]:<10.4f} | {res[2]:<15.4f} | {res[3]:<10.4f} | {res[4]:<10.4f} | {res[5]:<10.4f}")
    print("="*70)
    print(f"Raporlanacak En Iyi F1 Skoru veren Conf Degeri: {best_conf} (F1: {best_f1:.4f})")

if __name__ == "__main__":
    # Yalnizca calismasini istediginiz fonksiyonun basindaki '#' isaretini kaldirin.
    
    # 1. Adim: 960 boyutunda egitim yapmak icin (Uzun surer, projede onceden yapilmissa tekrar etmeyin)
    # train_ablation_960()
    
    # 2. Adim: Baseline model uzerinden Conf Sweep yapmak icin
    evaluate_ablation_conf_sweep()
