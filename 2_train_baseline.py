from ultralytics import YOLO

def main():
    print("--- 2. BASELINE EGITIM BASLIYOR ---")
    
    # Load a pre-trained small YOLOv8 model as baseline
    model = YOLO("yolov8n.pt")
    
    # Train the model with exact parameters from the assignment
    # imgsz=640, epochs=50, patience=10 for early stopping, seed=42
    results = model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        seed=42,
        patience=10,
        name="baseline",
        exist_ok=True
    )
    
    print("--- BASELINE EGITIM TAMAMLANDI ---")
    print("Sonuclar, PR egrisi ve kayip (loss) grafikleri 'runs/detect/baseline' klasorune kaydedildi.")
    print("mAP@0.5 ve mAP@0.5:0.95 metriklerini 'runs/detect/baseline/results.csv' dosyasinda bulabilirsiniz.")

if __name__ == "__main__":
    main()
