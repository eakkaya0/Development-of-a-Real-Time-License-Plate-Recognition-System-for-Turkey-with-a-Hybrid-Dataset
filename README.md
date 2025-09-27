 Türkiye Plakaları için Akıllı Tanıma ve E-posta Uyarı Sistemi

Bu proje, düşük donanım gereksinimleriyle çalışabilen ve Türkiye’ye özgü araç plakalarını tanıyabilen gerçek zamanlı araç plaka tanıma sistemidir. Sistem, özellikle otopark giriş-çıkış kontrolü ve güvenlik alanlarında ekonomik ve erişilebilir bir çözüm sunmayı hedeflemektedir.

Özellikler

Kamera aracılığıyla alınan araç görüntülerinden plaka tespiti ve karakter ayrıştırması

Plakaların okunarak veritabanına kaydedilmesi

Gerçek zamanlı plaka tanıma ve performans optimizasyonu

Türkiye plaka formatına özgü özel kurallar

CPU ve GPU uyumlu çalışabilme

E-mail bildirim sistemi: Plaka tanındığında otopark sahibine araç bilgisi e-posta olarak gönderilir

Kullanılan Teknolojiler

Python

Veri etiketleme: Label Studio

Ön işleme: Scikit-learn

Nesne tespiti: YOLOv8 (YOLOv8s ve YOLOv8nano)

Görüntü işleme: OpenCV

Plaka okuma: EasyOCR

Arayüz geliştirme: Tkinter

Veritabanı: SQLite

Veri Seti ve Eğitim

Başlangıçta 2500 araç görüntüsünden oluşan hibrit veri seti oluşturulmuştur.

YOLOv8s modeli Google Colab üzerinde eğitilmiştir.

Performans ve okuma sorunları nedeniyle YOLOv8nano modeline geçilmiş, data augmentation ile veri seti 3800 örneğe çıkarılmıştır.

Optimizasyon Teknikleri

Kare atlama mekanizması

Dinamik bellek yönetimi

Adaptif histogram eşitleme

Türkiye plaka formatına özel kurallar

Kurulum ve Çalıştırma

Python 3.x sürümü yüklü olmalıdır.

Gerekli kütüphaneler kurulmalıdır:

pip install opencv-python easyocr scikit-learn ultralytics tkinter sqlite3


Sistemi başlatmak için:

python main.py

E-mail Bildirimi Ayarları

E-mail bildirimleri için SMTP ayarlarını config.py dosyasında yapabilirsiniz.

Plaka tanındığında sistem otomatik olarak tanınan araç bilgilerini otopark sahibine gönderir.

Lisans

Bu proje MIT Lisansı ile lisanslanmıştır.



![photo_2025-06-28_17-58-09](https://github.com/user-attachments/assets/a3d2d2bc-b8fb-496b-8ce8-718f70ab87b0)
