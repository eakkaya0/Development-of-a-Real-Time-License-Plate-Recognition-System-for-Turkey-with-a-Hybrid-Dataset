🚗 Türkiye Plakaları için Akıllı Tanıma ve E-posta Uyarı Sistemi

Bu proje, düşük donanım gereksinimleriyle çalışabilen, Türkiye’ye özgü araç plakalarını gerçek zamanlı olarak tanıyabilen bir sistemdir.
Özellikle otopark giriş-çıkış kontrolü ve güvenlik uygulamaları için ekonomik ve erişilebilir bir çözüm sunmayı hedeflemektedir.

✨ Özellikler

📸 Kamera üzerinden plaka tespiti ve karakter ayrıştırması

🗄️ Tanınan plakaların veritabanına kaydedilmesi

⚡ Gerçek zamanlı plaka tanıma ve performans optimizasyonu

🇹🇷 Türkiye plaka formatına özel kurallar

🖥️ CPU ve GPU uyumlu çalışma desteği

📧 E-mail bildirim sistemi → Plaka tanındığında otopark sahibine araç bilgisi otomatik olarak gönderilir

🛠️ Kullanılan Teknolojiler

Python

Veri Etiketleme: Label Studio

Ön İşleme: Scikit-learn

Nesne Tespiti: YOLOv8 (YOLOv8s & YOLOv8nano)

Görüntü İşleme: OpenCV

Plaka Okuma: EasyOCR

Arayüz Geliştirme: Tkinter

Veritabanı: SQLite

📊 Veri Seti ve Eğitim

Başlangıçta 2500 araç görüntüsünden oluşan hibrit veri seti oluşturuldu.

YOLOv8s modeli Google Colab üzerinde eğitildi.

Performans sorunları nedeniyle YOLOv8nano modeline geçildi.

Data augmentation ile veri seti 3800 örneğe çıkarıldı.

⚡ Optimizasyon Teknikleri

🔄 Kare atlama mekanizması

🧠 Dinamik bellek yönetimi

🌌 Adaptif histogram eşitleme

🇹🇷 Türkiye plaka formatına özel kurallar




📧 E-mail Bildirimi Ayarları

E-mail bildirimleri için SMTP ayarlarını config.py dosyasında düzenleyebilirsiniz.

Plaka tanındığında, sistem otomatik olarak araç bilgilerini e-posta ile otopark sahibine gönderir.

📄 Lisans

Bu proje MIT Lisansı ile lisanslanmıştır.


![photo_2025-06-28_17-58-09](https://github.com/user-attachments/assets/a3d2d2bc-b8fb-496b-8ce8-718f70ab87b0)

<img width="1906" height="981" alt="Ekran görüntüsü 2025-09-27 135359" src="https://github.com/user-attachments/assets/1534c1db-9161-4ec5-9eba-dfbc8b3c81f1" />

![WhatsApp Görsel 2025-09-27 saat 14 06 09_3c340923](https://github.com/user-attachments/assets/2753bbcc-d3ec-40d1-bb40-5a5032c6f1fb)


