import smtplib
from email.message import EmailMessage
import json
import os
import threading
from datetime import datetime
import logging

# E-posta yapılandırma dosyası
CONFIG_FILE = "email_config.json"

class EmailNotificationSystem:
    def __init__(self):
        self.config = self.load_config()
        self.setup_logging()
        
    def setup_logging(self):
        """E-posta gönderim loglarını ayarla"""
        logging.basicConfig(
            filename='email_notifications.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            encoding='utf-8'
        )
        self.logger = logging.getLogger(__name__)
    
    def load_config(self):
        """E-posta ayarlarını yükle"""
        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            else:
                return None
        except Exception as e:
            print(f"E-posta ayarları yüklenirken hata: {e}")
            return None
    
    def is_configured(self):
        """E-posta sisteminin yapılandırılıp yapılandırılmadığını kontrol et"""
        if not self.config:
            return False
        
        required_fields = ["recipient_email", "sender_email", "app_password", 
                          "smtp_host", "smtp_port", "notifications_enabled"]
        
        return all(field in self.config and self.config[field] for field in required_fields)
    
    def send_plate_notification(self, plate_number, timestamp, confidence=None, photo_path=None):
        """Plaka tespit bildirimi gönder (arka planda)"""
        if not self.is_configured():
            print("E-posta ayarları yapılandırılmamış, bildirim gönderilmiyor.")
            return False
        
        if not self.config.get("notifications_enabled", False):
            print("E-posta bildirimleri devre dışı.")
            return False
        
        # E-postayı arka planda gönder (ana döngüyü bloklamaz)
        thread = threading.Thread(
            target=self._send_email_thread,
            args=(plate_number, timestamp, confidence, photo_path),
            daemon=True
        )
        thread.start()
        return True
    
    def _send_email_thread(self, plate_number, timestamp, confidence, photo_path):
        """E-posta gönderme thread fonksiyonu"""
        try:
            # E-posta içeriği oluştur
            msg = EmailMessage()
            msg["From"] = self.config["sender_email"]
            msg["To"] = self.config["recipient_email"]
            msg["Subject"] = f"🚗 Araç Girişi Tespit Edildi - {plate_number}"
            
            # E-posta gövdesi
            body = f"""
Plaka Tanıma Sistemi - Yeni Araç Girişi

📋 DETAYLAR:
Plaka Numarası: {plate_number}
Tespit Zamanı: {timestamp}
"""
            
            if confidence:
                body += f"Güven Oranı: %{confidence * 100:.2f}\n"
            
            body += f"""
📍 Konum: Otopark Giriş
🔄 Sistem: Otomatik Plaka Tanıma

Bu e-posta otomatik olarak gönderilmiştir.
"""
            
            msg.set_content(body)
            
            # Eğer fotoğraf varsa ekle
            if photo_path and os.path.exists(photo_path):
                try:
                    with open(photo_path, "rb") as f:
                        img_data = f.read()
                    
                    import imghdr
                    img_type = imghdr.what(None, img_data)
                    if img_type:
                        msg.add_attachment(img_data, maintype="image", 
                                         subtype=img_type, 
                                         filename=f"plaka_{plate_number}_{timestamp}.{img_type}")
                except Exception as e:
                    self.logger.warning(f"Fotoğraf eklenemedi: {e}")
            
            # E-postayı gönder
            with smtplib.SMTP(self.config["smtp_host"], int(self.config["smtp_port"])) as smtp:
                smtp.ehlo()
                smtp.starttls()
                smtp.login(self.config["sender_email"], self.config["app_password"])
                smtp.send_message(msg)
            
            self.logger.info(f"Bildirim gönderildi: {plate_number} - {timestamp}")
            print(f"E-posta bildirimi gönderildi: {plate_number}")
            
        except Exception as e:
            self.logger.error(f"E-posta gönderim hatası: {e}")
            print(f"E-posta gönderimi başarısız: {e}")
    
    def send_test_email(self):
        """Test e-postası gönder"""
        if not self.is_configured():
            return False, "E-posta ayarları yapılandırılmamış."
        
        try:
            msg = EmailMessage()
            msg["From"] = self.config["sender_email"]
            msg["To"] = self.config["recipient_email"]
            msg["Subject"] = "Test - Plaka Tanıma Sistemi"
            msg.set_content("""
Bu bir test e-postasıdır.

E-posta ayarlarınız başarıyla çalışıyor!

Plaka tanıma sisteminiz tespit ettiği her plaka için bu tür bildirimler gönderecektir.

Test zamanı: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            
            with smtplib.SMTP(self.config["smtp_host"], int(self.config["smtp_port"])) as smtp:
                smtp.ehlo()
                smtp.starttls()
                smtp.login(self.config["sender_email"], self.config["app_password"])
                smtp.send_message(msg)
            
            self.logger.info("Test e-postası başarıyla gönderildi")
            return True, "Test e-postası başarıyla gönderildi!"
            
        except Exception as e:
            self.logger.error(f"Test e-postası hatası: {e}")
            return False, f"Test e-postası gönderim hatası: {e}"
    
    def update_config(self, new_config):
        """Ayarları güncelle"""
        self.config = new_config
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(new_config, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Ayarlar kaydedilirken hata: {e}")
            return False

# Singleton pattern için global instance
email_system = EmailNotificationSystem()