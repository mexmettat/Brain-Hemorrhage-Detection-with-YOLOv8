from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")
    
    # Train with 960 imgsz
    model.train(
        data="data.yaml",
        epochs=50,
        imgsz=960,
        seed=42,
        patience=10,
        name="ablation_960"
    )

if __name__ == "__main__":
    main()
