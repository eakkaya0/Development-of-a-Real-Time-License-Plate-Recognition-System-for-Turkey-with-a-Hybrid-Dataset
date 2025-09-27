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
import sys
from pathlib import Path
import threading
import tempfile

# E-posta sistemi - isteğe bağlı import
try:
    from email_notification_system import email_system
    EMAIL_AVAILABLE = True
except ImportError:
    print("E-posta sistemi bulunamadı, e-posta bildirimleri devre dışı.")
    EMAIL_AVAILABLE = False
    class DummyEmailSystem:
        def is_configured(self): return False
        def send_plate_notification(self, **kwargs): pass
    email_system = DummyEmailSystem()

def get_csv_path():
    """
    CSV dosyası için doğru yolu döndürür.
    - Exe modunda: CSV, her zaman exe'nin bulunduğu klasöre yazılır.
    - Normal çalışmada: CSV, script'in bulunduğu klasöre yazılır.
    """
    try:
        if getattr(sys, 'frozen', False):
            # PyInstaller exe
            base_dir = os.path.dirname(sys.executable)
        else:
            # Normal Python
            base_dir = os.path.dirname(os.path.abspath(__file__))
    except Exception:
        # Son çare olarak mevcut çalışma dizini
        base_dir = os.getcwd()

    # Hedef CSV yolu
    csv_path = os.path.join(base_dir, "detected_plates.csv")

    # Klasör yoksa oluştur
    try:
        os.makedirs(base_dir, exist_ok=True)
        # CSV yoksa header ile oluştur
        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline='', encoding='utf-8') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(["timestamp", "plate_number", "confidence"])
            print(f"Yeni CSV dosyası oluşturuldu: {csv_path}")
    except Exception as e:
        print(f"⚠ CSV dosyası oluşturma hatası: {e}")
        raise # Hata durumunda programdan çık

    return csv_path

def get_resource_path(relative_path):
    """PyInstaller ile paketlenmiş dosyalar için doğru yolu bulur (OKUMA amaçlı)."""
    try:
        # PyInstaller ile paketlendiğinde, geçici klasörde çalışır
        base_path = sys._MEIPASS
    except Exception:
        # Normal Python çalıştırma durumu
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def get_model_path():
    """Model dosyasının yolunu bulur - PyInstaller uyumlu"""
    model_files = ["bestnano3.pt", "best.pt", "yolo.pt", "model.pt"]
    
    # PyInstaller resource path'lerini kontrol et
    for model_file in model_files:
        try:
            # PyInstaller resource path'te ara (okuma amaçlı)
            resource_path = get_resource_path(f"models/{model_file}")
            if os.path.exists(resource_path):
                return resource_path
                
            # Ana dizinde de ara
            resource_path = get_resource_path(model_file)
            if os.path.exists(resource_path):
                return resource_path
        except:
            pass
    
    # Normal dosya sistemi kontrolü
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    
    for model_file in model_files:
        # models klasöründe ara
        model_path = os.path.join(models_dir, model_file)
        if os.path.exists(model_path):
            return model_path
        
        # Ana dizinde ara
        model_path = os.path.join(base_dir, model_file)
        if os.path.exists(model_path):
            return model_path
    
    # Hiçbir model bulunamadıysa hata ver
    raise FileNotFoundError(
        "Model dosyası bulunamadı! Lütfen aşağıdaki konumlardan birinde model dosyasını bulundurun:\n"
        f"- {models_dir}/bestnano3.pt\n"
        f"- {base_dir}/bestnano3.pt\n"
        "Desteklenen model dosyaları: bestnano3.pt, best.pt, yolo.pt, model.pt"
    )

def get_writable_images_dir():
    """Yazılabilir plaka görüntüleri klasörü yolu döndürür"""
    if getattr(sys, 'frozen', False):
        # PyInstaller ile çalışırken - exe'nin yanına
        exe_dir = os.path.dirname(sys.executable)
        images_dir = os.path.join(exe_dir, "plate_images")
    else:
        # Normal Python çalışırken
        script_dir = os.path.dirname(os.path.abspath(__file__))
        images_dir = os.path.join(script_dir, "plate_images")
    
    # Klasör yoksa oluştur
    try:
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
            print(f"Plaka görüntüleri klasörü oluşturuldu: {images_dir}")
    except Exception as e:
        print(f"❌ Plaka görüntüleri klasörü oluşturulamadı: {e}")
        raise # Hata durumunda programdan çık
    
    return images_dir

# Yapılandırma
CONFIDENCE_THRESHOLD = 0.6  # YOLO modeli güven eşiği
OCR_CONFIDENCE_THRESHOLD = 0.6  # OCR güven eşiği
COLOR = (0, 255, 0)
TURKISH_PLATE_PATTERN = r"^\d{2}[A-Z]{1,3}\d{2,4}$"
MIN_PLATE_LENGTH = 7
MAX_PLATE_LENGTH = 8
SAME_PLATE_COOLDOWN = 10  # aynı plaka tekrar işlenmeden önceki saniye
PROCESS_EVERY_N_FRAMES = 20  # Her 20 karede bir OCR işlemi yapılacak
YOLO_PROCESS_EVERY_N_FRAMES = 20  # Her 20 karede bir YOLO çalıştır
VIDEO_WIDTH = 640  # Video genişliği
VIDEO_HEIGHT = 480  # Video yüksekliği
MEMORY_CLEANUP_INTERVAL = 3600  # Her 1 saatte bir bellek temizliği (saniye cinsinden)
MAX_PLATE_HISTORY = 1000  # Maksimum saklanacak plaka sayısı
LOG_SYSTEM_RESOURCES = True  # Sistem kaynaklarını logla
SAVE_PLATE_IMAGES = True  # Plaka görüntülerini kaydet

class ThreadSafeCSVWriter:
    """Thread-safe CSV yazıcı"""
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.lock = threading.Lock()
        
    def write_plate(self, timestamp_str, plate_text, confidence):
        """Thread-safe CSV yazma - GELIŞTIRILMIŞ"""
        try:
            with self.lock:
                # Dosyanın var olduğundan emin ol
                if not os.path.exists(self.csv_path):
                    # Dizin yoksa oluştur
                    csv_dir = os.path.dirname(self.csv_path)
                    if csv_dir and not os.path.exists(csv_dir):
                        os.makedirs(csv_dir, exist_ok=True)
                    
                    with open(self.csv_path, "w", newline='', encoding='utf-8') as csv_file:
                        csv_writer = csv.writer(csv_file)
                        csv_writer.writerow(["timestamp", "plate_number", "confidence"])
                
                # BUFFER KAPATMA İLE ANLIK YAZMA
                with open(self.csv_path, "a", newline='', encoding='utf-8', buffering=1) as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow([timestamp_str, plate_text, f"{confidence:.2f}"])
                    
                    # Derhal diske yazılmasını sağla
                    csv_file.flush()
                    os.fsync(csv_file.fileno()) 
                
                print(f"✓ CSV'ye BAŞARIYLA yazıldı: {plate_text} - {timestamp_str}")
                
                # DOSYA BOYUTUNU KONTROL ET
                try:
                    file_size = os.path.getsize(self.csv_path)
                    print(f"✓ CSV dosya boyutu: {file_size} bytes")
                except:
                    pass
                
                return True
                
        except PermissionError as e:
            print(f"✗ CSV yazma yetkisi hatası: {e}")
            print(f"✗ Dosya yolu: {self.csv_path}")
            return False
        except Exception as e:
            print(f"✗ CSV yazma genel hatası: {e}")
            print(f"✗ Dosya yolu: {self.csv_path}")
            return False

class PlateDetector:
    def __init__(self, use_gpu=False):
        print("Plaka tanıma sistemi başlatılıyor...")
        
        # CSV yolunu belirle - yazılabilir klasörde (exe dizini veya script dizini)
        self.csv_path = get_csv_path()
        print(f"CSV yolu: {self.csv_path}")
        
        # Thread-safe CSV writer oluştur
        self.csv_writer = ThreadSafeCSVWriter(self.csv_path)
        
        # Model yolunu dinamik olarak bul
        try:
            self.model_path = get_model_path()
            print(f"Model dosyası bulundu: {self.model_path}")
        except FileNotFoundError as e:
            print(f"HATA: {e}")
            raise
        
        # Plaka görüntüleri klasörünü oluştur
        if SAVE_PLATE_IMAGES:
            self.plate_images_dir = get_writable_images_dir()
            print(f"Plaka görüntüleri klasörü: {self.plate_images_dir}")
        
        # Sistem kaynakları izleme için başlangıç zamanı
        self.start_time = time.time()
        self.last_cleanup_time = self.start_time
        self.last_resources_log_time = self.start_time
        
        try:
            self.model = YOLO(self.model_path)
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
        
        # Log dosyası oluştur - yazılabilir klasörde
        if getattr(sys, 'frozen', False):
            exe_dir = os.path.dirname(sys.executable)
            log_path = os.path.join(exe_dir, "system_resources.log")
        else:
            log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "system_resources.log")
            
        try:
            self.log_file = open(log_path, "a", encoding='utf-8')
            self.log_file.write(f"\n--- Sistem başlatıldı: {datetime.now()} ---\n")
            self.log_file.write(f"Model yolu: {self.model_path}\n")
            self.log_file.write(f"CSV yolu: {self.csv_path}\n")
            self.log_file.write(f"GPU kullanımı: {'Etkin' if use_gpu else 'Devre dışı'}\n")
        except Exception as e:
            print(f"Log dosyası açılamadı: {e}")
            self.log_file = None
        
        # E-posta sistemi durumunu kontrol et
        if EMAIL_AVAILABLE and email_system.is_configured():
            print("E-posta bildirim sistemi aktif.")
            if self.log_file:
                self.log_file.write("E-posta bildirim sistemi: Aktif\n")
        else:
            print("E-posta bildirim sistemi yapılandırılmamış veya mevcut değil.")
            if self.log_file:
                self.log_file.write("E-posta bildirim sistemi: Yapılandırılmamış/Mevcut değil\n")
        
        print(f"Başlatma tamamlandı. Tespit başlıyor... (GPU: {'Etkin' if use_gpu else 'Devre dışı'})")

    def __del__(self):
        """Sınıf yok edildiğinde kaynakları temizle"""
        if hasattr(self, 'log_file') and self.log_file:
            try:
                self.log_file.close()
            except:
                pass

    def save_plate_image(self, plate_roi, plate_text, timestamp):
        """Tespit edilen plaka görüntüsünü kaydet"""
        if not SAVE_PLATE_IMAGES:
            return None
            
        try:
            # Dosya adı oluştur (tarih_saat_plaka.jpg formatında)
            safe_timestamp = timestamp.replace(":", "-").replace(" ", "_")
            filename = f"{safe_timestamp}_{plate_text}.jpg"
            filepath = os.path.join(self.plate_images_dir, filename)
            
            # Görüntüyü kaydet
            success = cv2.imwrite(filepath, plate_roi)
            if success:
                print(f"Plaka görüntüsü kaydedildi: {filepath}")
                return filepath
            else:
                print(f"Plaka görüntüsü kaydedilemedi: {filepath}")
                return None
        except Exception as e:
            print(f"Plaka görüntüsü kaydedilirken hata: {e}")
            return None

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
                    # OCR parametrelerini değiştirerek dene
                    detections = self.reader.readtext(img, detail=1, 
                                                     paragraph=False,
                                                     decoder='greedy',
                                                     beamWidth=5,
                                                     batch_size=8,
                                                     width_ths=0.7,
                                                     height_ths=0.7)
                    
                    for detection in detections:
                        bbox, text, ocr_confidence = detection
                        
                        # Plaka metnini temizle ve doğrula
                        cleaned_plate = text.upper().replace(" ", "")
                    
                        # OCR güven değeri eşiğini biraz düşür
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
                        'confidence': best_confidence,
                        'roi': roi  # Görüntü kaydetmek için ROI'yi ekle
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
        if not LOG_SYSTEM_RESOURCES or not self.log_file:
            return
            
        current_time = time.time()
        
        # Her 5 dakikada bir kaynak kullanımını logla
        if current_time - self.last_resources_log_time >= 300:  # 5 dakika = 300 saniye
            try:
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
                    f"CSV dosyası: {self.csv_path}\n"
                    f"---------------------------\n"
                )
                
                print(log_message)
                if self.log_file and not self.log_file.closed:
                    self.log_file.write(log_message)
                    self.log_file.flush()  # Dosyaya hemen yaz
                    
                self.last_resources_log_time = current_time
            except Exception as e:
                print(f"Kaynak loglama hatası: {e}")

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
                roi = plate_info['roi']
                
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
                        # Zaman damgası formatı
                        timestamp_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
                        
                        # Plaka görüntüsünü kaydet
                        saved_image_path = self.save_plate_image(roi, plate_text, timestamp_str)
                        
                        # Thread-safe CSV yazma - GELIŞTIRILMIŞ
                        print(f"🔄 CSV yazma işlemi başlıyor: {plate_text}")
                        csv_success = self.csv_writer.write_plate(timestamp_str, plate_text, confidence)
                        
                        if csv_success:
                            self.already_saved_plates.add(plate_text)
                            print(f"✓ Plaka başarıyla kaydedildi: {plate_text} - {timestamp_str}")
                            
                            # *** E-POSTA BİLDİRİMİ GÖNDER - GELİŞTİRİLMİŞ ***
                            if EMAIL_AVAILABLE and email_system.is_configured():
                                try:
                                    print(f"📧 E-posta bildirimi gönderiliyor: {plate_text}")
                                    
                                    # E-posta gönderme işlemini ayrı thread'de yap (ana işlemi engellemez)
                                    def send_email_async():
                                        try:
                                            email_system.send_plate_notification(
                                                plate_number=plate_text,
                                                timestamp=timestamp_str,
                                                confidence=confidence,
                                                photo_path=saved_image_path
                                            )
                                            print(f"✓ E-posta başarıyla gönderildi: {plate_text}")
                                        except Exception as async_email_error:
                                            print(f"✗ Async e-posta hatası: {async_email_error}")
                                    
                                    # E-posta gönderimini background thread'de başlat
                                    email_thread = threading.Thread(target=send_email_async, daemon=True)
                                    email_thread.start()
                                    
                                    # Log dosyasına bildirim kaydı
                                    if self.log_file and not self.log_file.closed:
                                        self.log_file.write(f"E-posta bildirimi başlatıldı: {plate_text} - {timestamp_str}\n")
                                        self.log_file.flush()
                                        
                                except Exception as email_error:
                                    print(f"✗ E-posta gönderme hatası: {email_error}")
                            else:
                                print(f"ℹ E-posta sistemi: {'Mevcut değil' if not EMAIL_AVAILABLE else 'Yapılandırılmamış'}")
                        else:
                            print(f"✗ CSV yazma başarısız: {plate_text}")
                        
                    except Exception as e:
                        print(f"✗ Plaka kaydetme genel hatası: {e}")
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
        
        # Durum bilgilerini göster - GELIŞTIRİLMİŞ
        cv2.putText(frame, f"YOLO: {self.frame_count % YOLO_PROCESS_EVERY_N_FRAMES}/{YOLO_PROCESS_EVERY_N_FRAMES}", (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # CSV dosyası durumu - GELIŞTIRİLMİŞ
        csv_exists = os.path.exists(self.csv_path)
        # Dosya varsa bulunduğu dizin yazılabilir mi?
        csv_dir = os.path.dirname(self.csv_path) or "."
        csv_writable = os.access(csv_dir, os.W_OK)
        
        if csv_exists and csv_writable:
            csv_status = "✓ Hazır"
            csv_color = (0, 255, 0)
            # Dosya boyutunu da göster
            try:
                file_size = os.path.getsize(self.csv_path)
                csv_status = f"✓ Hazır ({file_size}b)"
            except:
                pass
        else:
            csv_status = "✗ Sorunlu"
            csv_color = (0, 0, 255)
            
        cv2.putText(frame, f"CSV: {csv_status}", (10, 140), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, csv_color, 2)
        
        # E-posta durumu göster
        if EMAIL_AVAILABLE:
            email_status = "✓ Aktif" if email_system.is_configured() else "⚠ Pasif"
            email_color = (0, 255, 0) if email_system.is_configured() else (0, 165, 255)
        else:
            email_status = "✗ Devre dışı"
            email_color = (128, 128, 128)
        cv2.putText(frame, f"E-posta: {email_status}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, email_color, 2)
        
        # OCR durumu
        if self.frame_count % PROCESS_EVERY_N_FRAMES == 0:
            cv2.putText(frame, "OCR: ✓ Aktif", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            remaining = PROCESS_EVERY_N_FRAMES - (self.frame_count % PROCESS_EVERY_N_FRAMES)
            cv2.putText(frame, f"OCR: Bekliyor ({remaining})", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
        return frame, detection_time, recognition_time

def main():
    try:
        # Dedektörü başlat (model yolunu otomatik bulacak)
        detector = PlateDetector(use_gpu=False)
        # Önce harici kameraya (ID=1) bağlanmayı dene
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("Harici kamera bulunamadı, dahili kamera deneniyor...")
            cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
            if not cap.isOpened():
                print("DSHOW backend ile kamera açılamadı, varsayılan backend deneniyor...")
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    print("Hata: Hiçbir kamera açılamadı.")
                    input("Devam etmek için Enter'a basın...")
                    return

        # Video çözünürlüğünü ayarla
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
        cap.set(cv2.CAP_PROP_AUTO_WB, 1)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print(f"Kamera başarıyla açıldı ({VIDEO_WIDTH}x{VIDEO_HEIGHT}). Çıkmak için 'q' tuşuna basın.")
        print(f"CSV dosyası yolu: {detector.csv_path}")
        print(f"CSV dosyası mevcut: {'✓' if os.path.exists(detector.csv_path) else '✗'}")
        
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
                cv2.putText(processed_frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(processed_frame, f"Tespit: {detection_time:.1f}ms", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(processed_frame, f"Tanima: {recognition_time:.1f}ms", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Kaydedilen plaka sayısını göster
                cv2.putText(processed_frame, f"Kaydedilen: {len(detector.already_saved_plates)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # İşlenmiş kareyi göster
                cv2.imshow("Turk Plaka Tanima Sistemi - v2.0", processed_frame)
                
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
            # CSV durumunu son bir kez kontrol et
            if os.path.exists(detector.csv_path):
                file_size = os.path.getsize(detector.csv_path)
                print(f"✓ CSV dosyası son boyutu: {file_size} bytes")
            else:
                print("✗ CSV dosyası bulunamadı!")

    except Exception as e:
        print(f"Program başlatılırken hata: {e}")
        input("Devam etmek için Enter'a basın...")

if __name__ == "__main__":
    main()