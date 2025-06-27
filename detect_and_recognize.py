from ultralytics import YOLO
from easyocr import Reader
import time
import cv2
import os
import csv
import re
import collections
from datetime import datetime, timedelta
import gc  # Garbage collector için
import psutil  # Sistem kaynak kullanımını izlemek için (pip install psutil)

# Yapılandırma
CONFIDENCE_THRESHOLD = 0.6  # YOLO modeli güven eşiği
OCR_CONFIDENCE_THRESHOLD = 0.6  # OCR güven eşiği
COLOR = (0, 255, 0)
MODEL_PATH = r"C:\Users\eakkaya\Desktop\bestpt\bestnano3.pt"
TURKISH_PLATE_PATTERN = r"^\d{2}[A-Z]{1,3}\d{2,4}$"
MIN_PLATE_LENGTH = 7
MAX_PLATE_LENGTH = 8
SAME_PLATE_COOLDOWN = 10  # aynı plaka tekrar işlenmeden önceki saniye
CSV_PATH = "detected_plates.csv"
PROCESS_EVERY_N_FRAMES = 20  # Her 60 karede bir OCR işlemi yapılacak (değer artırıldı)
YOLO_PROCESS_EVERY_N_FRAMES = 20  # Her 60 karede bir YOLO çalıştır (değer artırıldı)
VIDEO_WIDTH = 640  # Video genişliği (çözünürlük düşürüldü)
VIDEO_HEIGHT = 480  # Video yüksekliği (çözünürlük düşürüldü)
MEMORY_CLEANUP_INTERVAL = 3600  # Her 1 saatte bir bellek temizliği (saniye cinsinden)
MAX_PLATE_HISTORY = 1000  # Maksimum saklanacak plaka sayısı
LOG_SYSTEM_RESOURCES = True  # Sistem kaynaklarını logla

class PlateDetector:
    def __init__(self, model_path, use_gpu=False):  # GPU varsayılan olarak etkin
        print("Plaka tanıma sistemi başlatılıyor...")
        
        # Sistem kaynakları izleme için başlangıç zamanı
        self.start_time = time.time()
        self.last_cleanup_time = self.start_time
        self.last_resources_log_time = self.start_time
        
        try:
            self.model = YOLO(model_path)
            print("YOLO modeli başarıyla yüklendi.")
        except Exception as e:
            print(f"YOLO modeli yüklenirken hata: {e}")
            raise
            
        try:
            self.reader = Reader(['tr'], gpu=use_gpu)  # Türkçe dil kullanımı
            print("OCR modeli başarıyla yüklendi.")
        except Exception as e:
            print(f"OCR modeli yüklenirken hata: {e}")
            raise
            
        self.detected_plates = {}  # Plaka metni ve zaman damgası saklamak için sözlük
        self.plate_counts = collections.defaultdict(int)  # Her plakanın kaç kez tespit edildiğini saymak için
        self.already_saved_plates = set()  # Kaydedilen plakaları takip etmek için küme
        self.frame_count = 0  # İşlenen kare sayısını tutmak için sayaç
        self.last_plate_regions = []  # Son tespit edilen plaka bölgelerini saklamak için
        self.use_gpu = use_gpu
        
        # CSV dosyası yoksa oluştur
        if not os.path.exists(CSV_PATH):
            with open(CSV_PATH, "w", newline='', encoding='utf-8') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(["timestamp", "plate_number", "confidence"])
                
        # Log dosyası oluştur
        self.log_file = open("system_resources.log", "a", encoding='utf-8')
        self.log_file.write(f"\n--- Sistem başlatıldı: {datetime.now()} ---\n")
        self.log_file.write(f"GPU kullanımı: {'Etkin' if use_gpu else 'Devre dışı'}\n")
        
        print(f"Başlatma tamamlandı. Tespit başlıyor... (GPU: {'Etkin' if use_gpu else 'Devre dışı'})")

    def __del__(self):
        """Sınıf yok edildiğinde kaynakları temizle"""
        if hasattr(self, 'log_file') and self.log_file:
            self.log_file.close()

    def is_valid_turkish_plate(self, plate_text):
        # Plaka metnini temizle
        plate_text = plate_text.upper().replace(" ", "")
        
        # Uzunluk kontrolü
        if len(plate_text) < MIN_PLATE_LENGTH or len(plate_text) > MAX_PLATE_LENGTH:
            return False
        
        # Türk plaka formatına uygunluk kontrolü
        if re.match(TURKISH_PLATE_PATTERN, plate_text):
            return True
        return False

    def detect_plates(self, frame):
        try:
            start = time.time()
            # Kareyi model üzerinden geçirerek tespitleri al
            detections = self.model.predict(frame, conf=CONFIDENCE_THRESHOLD)[0].boxes.data
            
            # Tespit edilen tüm plaka bölgelerini saklamak için liste
            plate_regions = []

            # Herhangi bir tespit yapıldı mı kontrol et
            if len(detections) > 0:
                # Her tespiti işle
                for detection in detections:
                    # Koordinatları ve güven değerini çıkar
                    xmin, ymin, xmax, ymax = int(detection[0]), int(detection[1]), int(detection[2]), int(detection[3])
                    confidence = float(detection[4])
                    
                    # Çerçeve çiz
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), COLOR, 2)
                    text = f"Plaka: {confidence:.2f}"
                    cv2.putText(frame, text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR, 2)
                    
                    # OCR işlemi için plaka bölgesini sakla
                    plate_regions.append({
                        'roi': frame[ymin:ymax, xmin:xmax].copy(),  # Kopyasını al (bellek sorunlarını önlemek için)
                        'coords': [xmin, ymin, xmax, ymax],
                        'confidence': confidence
                    })
            
            end = time.time()
            detection_time = (end - start) * 1000
            return plate_regions, detection_time
        except Exception as e:
            print(f"Plaka tespiti sırasında hata: {e}")
            return [], 0
    
    def recognize_plate_text(self, plate_regions):
        try:
            start = time.time()
            results = []
            
            for plate_info in plate_regions:
                roi = plate_info['roi']
                coords = plate_info['coords']
                confidence = plate_info['confidence']
                
                # Plaka bölgesinin geçerli olup olmadığını kontrol et
                if roi is None or roi.size == 0:
                    continue
                
                # Farklı görüntü işleme seçenekleri uygula
                processed_images = []
                
                # Orijinal görüntü
                processed_images.append(("original", roi))
                
                # Ön işlemli görüntü
                preprocessed = self.preprocess_plate_image(roi)
                if preprocessed is not None:
                    processed_images.append(("preprocessed", preprocessed))
                
                # Kontrast artırılmış görüntü
                contrast_enhanced = cv2.convertScaleAbs(roi, alpha=1.5, beta=0)
                processed_images.append(("contrast", contrast_enhanced))
                
                best_text = None
                best_confidence = -1
                
                # Her işlenmiş görüntü için OCR dene
                for img_type, img in processed_images:
                    # OCR parametrelerini değiştirerek (threshold, width_ths değerini azaltarak) dene
                    detections = self.reader.readtext(img, detail=1, 
                                                     paragraph=False,
                                                     decoder='greedy',
                                                     beamWidth=5,
                                                     batch_size=8,
                                                     width_ths=0.7,  # Daha hoşgörülü karakter genişliği
                                                     height_ths=0.7)  # Daha hoşgörülü karakter yüksekliği
                    
                    for detection in detections:
                        bbox, text, ocr_confidence = detection
                        
                        # Plaka metnini temizle ve doğrula
                        cleaned_plate = text.upper().replace(" ", "")
                        
                        # OCR güven değeri eşiğini biraz düşür (güneş ışığında daha toleranslı olmak için)
                        effective_threshold = OCR_CONFIDENCE_THRESHOLD * 0.8  # %20 daha toleranslı
                        
                        if ocr_confidence > effective_threshold and self.is_valid_turkish_plate(cleaned_plate):
                            if ocr_confidence > best_confidence:
                                best_confidence = ocr_confidence
                                best_text = cleaned_plate
                
                # En iyi sonucu bulduk mu?
                if best_text and best_confidence > 0:
                    results.append({
                        'coords': coords,
                        'text': best_text,
                        'confidence': best_confidence
                    })
            
            end = time.time()
            recognition_time = (end - start) * 1000
            return results, recognition_time
        except Exception as e:
            print(f"Plaka tanıma sırasında hata: {e}")
            return [], 0

    def preprocess_plate_image(self, plate_img):
        """Plaka görüntüsünü OCR için optimize eder"""
        if plate_img is None or plate_img.size == 0:
            return None
            
        # Görüntüyü gri tona çevir
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        
        # Adaptif histogram eşitleme uygula (kontrast iyileştirme)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Adaptif eşikleme uygula
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 19, 9
        )
        
        # Morfolojik işlemler (gürültü azaltma)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
        
        # İşlenmiş görüntüyü döndür
        return morph

    def cleanup_memory(self):
        """Bellek temizliği ve kaynak yönetimi"""
        current_time = time.time()
        
        # Bellek temizliği yapılacak zamanı kontrol et
        if current_time - self.last_cleanup_time >= MEMORY_CLEANUP_INTERVAL:
            print("Bellek temizliği yapılıyor...")
            
            # Eski plakaları temizle (son 10 dakika içindekiler hariç)
            cleanup_threshold = datetime.now() - timedelta(minutes=10)
            plates_to_remove = []
            
            for plate, detected_time in self.detected_plates.items():
                if detected_time < cleanup_threshold:
                    plates_to_remove.append(plate)
            
            # Eski plakaları sözlükten çıkar
            for plate in plates_to_remove:
                del self.detected_plates[plate]
                
            # Maksimum plaka sayısını kontrol et
            if len(self.detected_plates) > MAX_PLATE_HISTORY:
                # En eski plakaları sil
                sorted_plates = sorted(self.detected_plates.items(), key=lambda x: x[1])
                plates_to_remove = sorted_plates[:len(sorted_plates) - MAX_PLATE_HISTORY]
                
                for plate, _ in plates_to_remove:
                    del self.detected_plates[plate]
            
            # Garbage collector çağır
            gc.collect()
            
            self.last_cleanup_time = current_time
            print(f"Bellek temizliği tamamlandı. Kalan plaka sayısı: {len(self.detected_plates)}")
            
            # Son kare bölgelerini temizle
            self.last_plate_regions = []
    
    def log_system_resources(self):
        """Sistem kaynaklarını logla"""
        if not LOG_SYSTEM_RESOURCES:
            return
            
        current_time = time.time()
        
        # Her 5 dakikada bir kaynak kullanımını logla
        if current_time - self.last_resources_log_time >= 300:  # 5 dakika = 300 saniye
            process = psutil.Process(os.getpid())
            
            # Bellek kullanımı
            memory_info = process.memory_info()
            memory_usage_mb = memory_info.rss / 1024 / 1024
            
            # CPU kullanımı
            cpu_percent = process.cpu_percent(interval=1)
            
            # Çalışma süresi
            uptime = current_time - self.start_time
            hours, remainder = divmod(uptime, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            log_message = (
                f"\n--- {datetime.now()} ---\n"
                f"Çalışma süresi: {int(hours)} saat, {int(minutes)} dakika, {int(seconds)} saniye\n"
                f"Bellek kullanımı: {memory_usage_mb:.2f} MB\n"
                f"CPU kullanımı: {cpu_percent:.1f}%\n"
                f"İşlenen toplam kare: {self.frame_count}\n"
                f"Tespit edilen toplam plaka: {len(self.already_saved_plates)}\n"
                f"Plaka sözlüğü boyutu: {len(self.detected_plates)}\n"
                f"---------------------------\n"
            )
            
            print(log_message)
            self.log_file.write(log_message)
            self.log_file.flush()  # Dosyaya hemen yaz
            
            self.last_resources_log_time = current_time

    def process_frame(self, frame):
        self.frame_count += 1
        recognition_time = 0
        detection_time = 0
        
        # Bellek temizliği ve kaynak yönetimi
        self.cleanup_memory()
        self.log_system_resources()
        
        # YOLO'yu belirli aralıklarla çalıştır
        if self.frame_count % YOLO_PROCESS_EVERY_N_FRAMES == 0:
            plate_regions, detection_time = self.detect_plates(frame)
            self.last_plate_regions = plate_regions
        else:
            plate_regions = self.last_plate_regions
        
        # Plaka bölgesi yoksa işlemi sonlandır
        if not plate_regions:
            return frame, detection_time, recognition_time
        
        # Her N karede bir OCR işlemi yap
        if self.frame_count % PROCESS_EVERY_N_FRAMES == 0:
            # Tespit edilen plakalardan metin okuması yap
            recognized_plates, recognition_time = self.recognize_plate_text(plate_regions)
            
            current_time = datetime.now()
            
            for plate_info in recognized_plates:
                plate_text = plate_info['text']
                coords = plate_info['coords']
                confidence = plate_info['confidence']
                
                # Bu plaka yakın zamanda tespit edilmiş mi kontrol et
                if plate_text in self.detected_plates:
                    last_detection = self.detected_plates[plate_text]
                    # Bekleme süresi içindeyse atla
                    if (current_time - last_detection).seconds < SAME_PLATE_COOLDOWN:
                        # Plaka metnini sarı renkte çiz (zaten işlenmiş)
                        cv2.putText(frame, plate_text, (coords[0], coords[3] + 20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        continue
                
                # Tespit zaman damgası ve sayacını güncelle
                self.detected_plates[plate_text] = current_time
                self.plate_counts[plate_text] += 1
                
                # Plaka metnini yeşil renkte çiz (yeni tespit)
                cv2.putText(frame, plate_text, (coords[0], coords[3] + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR, 2)
                
                # Bu plaka sık tespit edilmişse ve henüz kaydedilmemişse CSV'ye kaydet
                if self.plate_counts[plate_text] > 1 and plate_text not in self.already_saved_plates:
                    try:
                        with open(CSV_PATH, "a", newline='', encoding='utf-8') as csv_file:
                            csv_writer = csv.writer(csv_file)
                            csv_writer.writerow([current_time.strftime("%Y-%m-%d %H:%M:%S"), 
                                                plate_text, f"{confidence:.2f}"])
                        self.already_saved_plates.add(plate_text)
                        print(f"Plaka CSV'ye kaydedildi: {plate_text}")
                    except Exception as e:
                        print(f"CSV'ye yazarken hata: {e}")
        else:
            # OCR işlemi yapılmadığı karelerde, önceki tespitleri kullan
            current_time = datetime.now()
            for plate_info in plate_regions:
                coords = plate_info['coords']
                
                # Bu koordinatların yakınında önceden tespit edilen bir plaka var mı kontrol et
                for plate_text, timestamp in self.detected_plates.items():
                    # Son tespitten bu yana geçen süre bekleme süresinden az ise plakayı göster
                    if (current_time - timestamp).seconds < SAME_PLATE_COOLDOWN:
                        cv2.putText(frame, plate_text, (coords[0], coords[3] + 20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        break
        
        # İşleme modu bilgisini göster
        cv2.putText(frame, f"YOLO: {self.frame_count % YOLO_PROCESS_EVERY_N_FRAMES}/{YOLO_PROCESS_EVERY_N_FRAMES}", (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if self.frame_count % PROCESS_EVERY_N_FRAMES == 0:
            cv2.putText(frame, "OCR Aktif", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(frame, f"OCR Pasif ({PROCESS_EVERY_N_FRAMES - (self.frame_count % PROCESS_EVERY_N_FRAMES)})", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Toplam işlem süresini hesapla
        total_time = detection_time + recognition_time
        return frame, detection_time, recognition_time

def main():
    try:
        # Dedektörü başlat
        detector = PlateDetector(MODEL_PATH, use_gpu=False)  # GPU kullanımı varsayılan olarak etkin
        
        # Önce harici kameraya (ID=1) bağlanmayı dene
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Harici kamera bulunamadı, dahili kamera deneniyor...")
            cap = cv2.VideoCapture(1)  # Dahili kamera için 0 kullan
        
        if not cap.isOpened():
            print("Hata: Hiçbir kamera açılamadı.")
            return
            
        # Video çözünürlüğünü ayarla
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)
        # Kamera ayarları

        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # Otomatik pozlama
        cap.set(cv2.CAP_PROP_AUTO_WB, 1)          # Otomatik beyaz dengesi
        # Buffer boyutunu azalt (daha az bellek kullanımı için)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        print(f"Kamera başarıyla açıldı ({VIDEO_WIDTH}x{VIDEO_HEIGHT}). Çıkmak için 'q' tuşuna basın.")
        
        # FPS hesaplama için değişkenler
        fps_start_time = time.time()
        fps_frame_count = 0
        fps = 0
        
        try:
            while True:
                # Kameradan bir kare oku
                ret, frame = cap.read()
                if not ret:
                    print("Kare alınamadı. Yeniden bağlanmaya çalışılıyor...")
                    # Kamera bağlantısını yeniden dene
                    cap.release()
                    time.sleep(2)
                    cap = cv2.VideoCapture(1 if detector.frame_count > 0 else 0)
                    if not cap.isOpened():
                        print("Kamera bağlantısı kurulamadı. Çıkılıyor...")
                        break
                    continue
                
                # FPS hesapla
                fps_frame_count += 1
                fps_current_time = time.time()
                if fps_current_time - fps_start_time >= 1:
                    fps = fps_frame_count
                    fps_frame_count = 0
                    fps_start_time = fps_current_time
                
                # Kareyi işle
                processed_frame, detection_time, recognition_time = detector.process_frame(frame)
                
                # Performans metriklerini göster
                cv2.putText(processed_frame, f"FPS: {fps}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(processed_frame, f"Tespit: {detection_time:.1f}ms", (10, 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(processed_frame, f"Tanima: {recognition_time:.1f}ms", (10, 180), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # İşlenmiş kareyi göster
                cv2.imshow("Turk Plaka Tanima Sistemi", processed_frame)
                
                # 'q' tuşuna basılırsa döngüyü kır
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except KeyboardInterrupt:
            print("Klavye kesintisi algılandı. Çıkılıyor...")
        except Exception as e:
            print(f"Beklenmeyen bir hata oluştu: {e}")
        finally:
            # Kaynakları serbest bırak
            cap.release()
            cv2.destroyAllWindows()
            print("Tespit sistemi durduruldu.")
    except Exception as e:
        print(f"Program başlatılırken hata: {e}")

if __name__ == "__main__":
    main()