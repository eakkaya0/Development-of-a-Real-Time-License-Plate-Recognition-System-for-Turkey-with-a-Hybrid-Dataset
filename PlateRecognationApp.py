import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import pandas as pd
import os
import sqlite3
import subprocess
import time
import signal
import sys
import psutil

class PlateRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Araç Plaka Tanıma Sistemi")
        self.root.geometry("1200x700")
        
        # Plaka tanıma süreci değişkeni
        self.detection_process = None
        
        # Ana frame
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Sol taraf - Durum ve Kontrol
        self.left_frame = tk.Frame(self.main_frame, width=400)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Durum etiketi
        self.status_var = tk.StringVar(value="Durum: Beklemede")
        self.status_label = tk.Label(self.left_frame, textvariable=self.status_var,
                                   font=("Arial", 14, "bold"), pady=20)
        self.status_label.pack(pady=20)
        
        # Butonlar için frame
        self.button_frame = tk.Frame(self.left_frame)
        self.button_frame.pack(fill=tk.X, pady=20)
        
        self.start_button = tk.Button(self.button_frame, text="Plaka Tanıma Başlat", 
                                     command=self.start_detection, bg="#4CAF50", fg="white",
                                     font=("Arial", 14, "bold"), padx=20, pady=10,
                                     width=20, height=2)
        self.start_button.pack(pady=10)
        
        self.stop_button = tk.Button(self.button_frame, text="Plaka Tanıma Durdur", 
                                    command=self.stop_detection, bg="#F44336", fg="white",
                                    font=("Arial", 14, "bold"), padx=20, pady=10,
                                    width=20, height=2, state=tk.DISABLED)
        self.stop_button.pack(pady=10)
        
        # Sağ taraf - Plaka listesi
        self.right_frame = tk.Frame(self.main_frame, width=800)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Arama özelliği için frame
        self.search_frame = tk.Frame(self.right_frame)
        self.search_frame.pack(fill=tk.X, pady=(10, 5))
        
        # Arama etiketi
        self.search_label = tk.Label(self.search_frame, text="Plaka Ara:", font=("Arial", 12))
        self.search_label.pack(side=tk.LEFT, padx=(0, 10))
        
        # Arama giriş kutusu
        self.search_var = tk.StringVar()
        self.search_entry = tk.Entry(self.search_frame, textvariable=self.search_var, 
                                    font=("Arial", 12), width=15)
        self.search_entry.pack(side=tk.LEFT, padx=(0, 10))
        
        # Arama butonu
        self.search_button = tk.Button(self.search_frame, text="Ara", 
                                      command=self.search_plates, bg="#2196F3", fg="white",
                                      font=("Arial", 10), padx=10, pady=2)
        self.search_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Filtreyi temizle butonu
        self.clear_filter_button = tk.Button(self.search_frame, text="Filtreyi Temizle", 
                                           command=self.clear_filter, bg="#FF9800", fg="white",
                                           font=("Arial", 10), padx=10, pady=2)
        self.clear_filter_button.pack(side=tk.LEFT)
        
        # Liste başlığı
        self.header_label = tk.Label(self.right_frame, text="Tespit Edilen Plakalar", 
                                    font=("Arial", 16, "bold"))
        self.header_label.pack(pady=(10, 10))
        
        # Treeview oluşturma
        self.columns = ("tarih_saat", "plaka", "dogruluk")
        self.plate_tree = ttk.Treeview(self.right_frame, columns=self.columns, show="headings")
        
        # Kolonları tanımla
        self.plate_tree.heading("tarih_saat", text="Tarih ve Saat")
        self.plate_tree.heading("plaka", text="Plaka No")
        self.plate_tree.heading("dogruluk", text="Doğruluk Oranı")
        
        # Kolon genişlikleri
        self.plate_tree.column("tarih_saat", width=200)
        self.plate_tree.column("plaka", width=150)
        self.plate_tree.column("dogruluk", width=150)
        
        # Scrollbar ekle
        self.tree_scroll = ttk.Scrollbar(self.right_frame, orient="vertical", command=self.plate_tree.yview)
        self.plate_tree.configure(yscrollcommand=self.tree_scroll.set)
        
        self.plate_tree.pack(fill=tk.BOTH, expand=True)
        self.tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Alt butonlar için frame
        self.bottom_button_frame = tk.Frame(self.right_frame)
        self.bottom_button_frame.pack(fill=tk.X, pady=10)
        
        # Kaydet butonu
        self.save_button = tk.Button(self.bottom_button_frame, text="Veritabanına Kaydet", 
                                    command=self.save_to_database, bg="#2196F3", fg="white",
                                    font=("Arial", 12), padx=10, pady=5)
        self.save_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Yenile butonu
        self.refresh_button = tk.Button(self.bottom_button_frame, text="Listeyi Yenile", 
                                      command=self.load_csv_data, bg="#FF9800", fg="white",
                                      font=("Arial", 12), padx=10, pady=5)
        self.refresh_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Listeyi temizle butonu
        self.clear_button = tk.Button(self.bottom_button_frame, text="Listeyi Temizle", 
                                    command=self.clear_plate_list, bg="#F44336", fg="white",
                                    font=("Arial", 12), padx=10, pady=5)
        self.clear_button.pack(side=tk.LEFT)
        
        # CSV dosyası değişikliği izleme için değişken
        self.last_modified_time = None
        
        # Tüm verileri saklamak için değişken (arama için)
        self.all_plate_data = []
        
        # İlk başta CSV verilerini yükle
        self.load_csv_data()
        
        # Periyodik CSV kontrolü başlat
        self.check_csv_updates()
        
        # Uygulama kapatıldığında process'i durdur
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def load_csv_data(self):
        """CSV dosyasından plaka verilerini yükler"""
        try:
            csv_path = "detected_plates.csv"
            if os.path.exists(csv_path):
                # CSV dosyasının son değiştirilme zamanını kaydet
                self.last_modified_time = os.path.getmtime(csv_path)
                
                # CSV dosyasını header olmadan oku ve manuel sütun isimleri ver
                df = pd.read_csv(csv_path, header=None, names=["tarih_saat", "plaka", "dogruluk"])
                
                # Tüm verileri sakla
                self.all_plate_data = []
                for _, row in df.iterrows():
                    try:
                        # Doğruluk değerini yüzde formatına çevir
                        dogruluk = f"%{float(row['dogruluk']) * 100:.2f}"
                        
                        self.all_plate_data.append({
                            "tarih_saat": row['tarih_saat'],
                            "plaka": row['plaka'],
                            "dogruluk": dogruluk
                        })
                    except Exception as e:
                        print(f"Veri işlenirken hata: {e}")
                
                # Verileri listele
                self.display_plates(self.all_plate_data)
            else:
                # CSV dosyası yoksa boş liste oluştur
                self.all_plate_data = []
                self.display_plates([])
                
        except Exception as e:
            print(f"CSV dosyası yüklenirken hata oluştu: {e}")
    
    def display_plates(self, plate_data):
        """Belirtilen plaka verilerini görüntüler"""
        # Önce listeyi temizle
        for item in self.plate_tree.get_children():
            self.plate_tree.delete(item)
        
        # Verileri ekle
        for plate in plate_data:
            self.plate_tree.insert("", 0, values=(  # 0 ile en başa ekle (yeni kayıtlar üstte)
                plate["tarih_saat"],
                plate["plaka"],
                plate["dogruluk"]
            ))
    
    def search_plates(self):
        """Arama kutusundaki metne göre plakaları filtreler"""
        search_text = self.search_var.get().strip().upper()
        if not search_text:
            # Arama metni boşsa tüm listeyi göster
            self.display_plates(self.all_plate_data)
            return
        
        # Arama metni varsa filtreleme yap
        filtered_data = [plate for plate in self.all_plate_data 
                        if search_text in plate["plaka"].upper()]
        
        # Filtrelenmiş verileri göster
        self.display_plates(filtered_data)
        
        # Sonuç sayısını bildir
        if not filtered_data:
            messagebox.showinfo("Arama Sonucu", "Aramanızla eşleşen plaka bulunamadı.")
        else:
            messagebox.showinfo("Arama Sonucu", f"{len(filtered_data)} plaka bulundu.")
    
    def clear_filter(self):
        """Filtreyi temizler ve tüm plakaları gösterir"""
        self.search_var.set("")  # Arama kutusunu temizle
        self.display_plates(self.all_plate_data)  # Tüm verileri göster
    
    def clear_plate_list(self):
        """Plaka listesini temizler (CSV ve arayüz)"""
        confirm = messagebox.askyesno("Onay", "Plaka listesi tamamen temizlenecek. Devam etmek istiyor musunuz?")
        
        if confirm:
            try:
                # CSV dosyasını temizle
                csv_path = "detected_plates.csv"
                if os.path.exists(csv_path):
                    # Dosyanın var olduğunu kontrol et
                    with open(csv_path, 'w') as file:
                        # Dosyayı boşalt
                        pass
                    
                # Arayüzdeki listeyi temizle
                for item in self.plate_tree.get_children():
                    self.plate_tree.delete(item)
                
                # Veri listesini temizle
                self.all_plate_data = []
                
                messagebox.showinfo("Bilgi", "Plaka listesi başarıyla temizlendi.")
            except Exception as e:
                messagebox.showerror("Hata", f"Liste temizlenirken hata oluştu: {e}")
    
    def check_csv_updates(self):
        """CSV dosyası değişikliklerini periyodik olarak kontrol eder"""
        try:
            csv_path = "detected_plates.csv"
            if os.path.exists(csv_path):
                current_modified_time = os.path.getmtime(csv_path)
                if self.last_modified_time is None or current_modified_time > self.last_modified_time:
                    self.load_csv_data()
        except Exception as e:
            print(f"CSV güncelleme kontrolü hatası: {e}")
        
        # Her 1 saniyede bir kontrol et
        self.root.after(1000, self.check_csv_updates)
    
    def start_detection(self):
        """Plaka tanıma sistemini başlatır"""
        if self.detection_process is None:
            try:
                # detect_and_recognize.py dosyasını alt işlem olarak çalıştır
                self.detection_process = subprocess.Popen(["python", "detect_and_recognize.py"])
                
                # Butonların durumunu güncelle
                self.start_button.config(state=tk.DISABLED)
                self.stop_button.config(state=tk.NORMAL)
                
                # Durum etiketini güncelle
                self.status_var.set("Durum: Plaka Tanıma Aktif")
                
                messagebox.showinfo("Bilgi", "Plaka tanıma sistemi başlatıldı.")
            except Exception as e:
                messagebox.showerror("Hata", f"Plaka tanıma sistemi başlatılırken hata: {e}")
    
    def stop_detection(self):
        """Plaka tanıma sistemini durdurur"""
        if self.detection_process is not None:
            try:
                # Prosesi nazikçe kapatmayı dene
                self.terminate_process()
                
                # Butonların durumunu güncelle
                self.start_button.config(state=tk.NORMAL)
                self.stop_button.config(state=tk.DISABLED)
                
                # Durum etiketini güncelle
                self.status_var.set("Durum: Beklemede")
                
                messagebox.showinfo("Bilgi", "Plaka tanıma sistemi durduruldu.")
            except Exception as e:
                messagebox.showerror("Hata", f"Plaka tanıma sistemi durdurulurken hata: {e}")
    
    def terminate_process(self):
        """İşlemi ve tüm alt işlemlerini güvenli bir şekilde sonlandırır"""
        if self.detection_process is not None:
            try:
                # İşlem hala çalışıyor mu kontrol et
                if self.detection_process.poll() is None:
                    process = psutil.Process(self.detection_process.pid)
                    
                    # Alt işlemleri de sonlandır
                    for child in process.children(recursive=True):
                        try:
                            child.terminate()
                        except:
                            pass
                    
                    # Ana işlemi sonlandır
                    self.detection_process.terminate()
                    
                    # İşlemin kapanmasını en fazla 3 saniye bekle
                    self.detection_process.wait(timeout=3)
                    
                    # Hala kapanmadıysa zorla kapat
                    if self.detection_process.poll() is None:
                        for child in process.children(recursive=True):
                            try:
                                child.kill()
                            except:
                                pass
                        self.detection_process.kill()
            except Exception as e:
                print(f"İşlem sonlandırma hatası: {e}")
            
            # İşlem değişkenini sıfırla
            self.detection_process = None
    
    def save_to_database(self):
        """CSV dosyasındaki verileri veritabanına kaydeder"""
        try:
            # Veritabanı bağlantısı
            conn = sqlite3.connect("plates_database.db")
            cursor = conn.cursor()
            
            # Tablo oluştur (eğer yoksa)
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS plates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tarih_saat TEXT,
                plaka TEXT,
                dogruluk REAL
            )
            ''')
            
            # CSV dosyasını oku
            csv_path = "detected_plates.csv"
            if not os.path.exists(csv_path):
                messagebox.showinfo("Bilgi", "CSV dosyası bulunamadı.")
                return
                
            df = pd.read_csv(csv_path, header=None, names=["tarih_saat", "plaka", "dogruluk"])
            
            # Her kaydı kontrol et ve ekle
            kayit_sayisi = 0
            for _, row in df.iterrows():
                # Önce bu kaydın veritabanında olup olmadığını kontrol et
                cursor.execute('''
                SELECT COUNT(*) FROM plates 
                WHERE tarih_saat = ? AND plaka = ?
                ''', (row['tarih_saat'], row['plaka']))
                
                if cursor.fetchone()[0] == 0:  # Eğer kayıt yoksa
                    cursor.execute('''
                    INSERT INTO plates (tarih_saat, plaka, dogruluk)
                    VALUES (?, ?, ?)
                    ''', (row['tarih_saat'], row['plaka'], float(row['dogruluk'])))
                    kayit_sayisi += 1
            
            # Değişiklikleri kaydet ve bağlantıyı kapat
            conn.commit()
            conn.close()
            
            if kayit_sayisi > 0:
                messagebox.showinfo("Başarılı", f"{kayit_sayisi} yeni kayıt veritabanına eklendi!")
            else:
                messagebox.showinfo("Bilgi", "Tüm kayıtlar zaten veritabanında mevcut.")
        except Exception as e:
            messagebox.showerror("Hata", f"Veritabanına kaydetme hatası: {e}")
    
    def on_closing(self):
        """Uygulama kapatıldığında çalışan işlemi sonlandır"""
        self.terminate_process()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = PlateRecognitionApp(root)
    root.mainloop()