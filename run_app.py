import subprocess
import time
import sys
import os
import threading
import webbrowser
import urllib.request
import urllib.error

def start_server():
    import uvicorn
    # log_level="warning" to keep the terminal clean
    uvicorn.run("app:app", host="127.0.0.1", port=8000, log_level="warning")

def wait_for_server(url, timeout=60):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            urllib.request.urlopen(url)
            return True
        except urllib.error.URLError:
            time.sleep(1)
    return False

if __name__ == "__main__":
    print("Yapay zeka modeli yükleniyor ve sunucu başlatılıyor...")
    print("Bu işlem bilgisayarınızın hızına bağlı olarak 10-20 saniye sürebilir, lütfen bekleyin...")
    
    # Start the FastAPI server in a separate thread
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    
    url = "http://127.0.0.1:8000/static/index.html"
    
    # Wait until the server actually responds before opening the browser
    if wait_for_server(url, timeout=60):
        print(f"Sunucu hazır! Arayüz açılıyor...")
        
        # Try to open as a standalone desktop app using Edge or Chrome
        try:
            if os.name == 'nt':
                # Edge is built-in on Windows
                edge_cmd = f'start msedge --app="{url}"'
                chrome_cmd = f'start chrome --app="{url}"'
                
                # Try Edge first
                res = subprocess.run(edge_cmd, shell=True, capture_output=True)
                if res.returncode != 0:
                    # If Edge fails, try Chrome
                    subprocess.run(chrome_cmd, shell=True)
            else:
                webbrowser.open(url)
        except Exception:
            # Fallback to default browser
            webbrowser.open(url)
        
        print("\n" + "="*50)
        print("✅ Uygulama başarıyla çalışıyor!")
        print("❌ Kapatmak için bu pencerede CTRL+C tuşlarına basabilirsiniz.")
        print("="*50 + "\n")
    else:
        print("\nHATA: Sunucu başlatılamadı. Model yüklenirken bir hata olmuş olabilir.")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nUygulama kapatılıyor...")
        sys.exit(0)

