from ultralytics import YOLO

def main():
    model = YOLO("runs/detect/baseline/weights/best.pt")
    confs = [0.001, 0.01, 0.05, 0.1, 0.2]
    best_f1 = 0
    best_conf = 0
    
    results_list = []
    
    for c in confs:
        print(f"\n--- Evaluating with conf={c} ---")
        metrics = model.val(data="data.yaml", conf=c)
        
        map50 = metrics.box.map50
        map50_95 = metrics.box.map
        p = metrics.box.mp
        r = metrics.box.mr
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0
        
        results_list.append((c, map50, map50_95, p, r, f1))
        
        if f1 > best_f1:
            best_f1 = f1
            best_conf = c
            
    print("\n=== ABLATION RESULTS ===")
    print("Conf\tmAP@0.5\tmAP@0.5:0.95\tPrecision\tRecall\tF1")
    for res in results_list:
        print(f"{res[0]:.3f}\t{res[1]:.4f}\t\t{res[2]:.4f}\t\t{res[3]:.4f}\t\t{res[4]:.4f}\t{res[5]:.4f}")
    
    print(f"\nBest Conf for F1: {best_conf} (F1: {best_f1:.4f})")

if __name__ == "__main__":
    main()
