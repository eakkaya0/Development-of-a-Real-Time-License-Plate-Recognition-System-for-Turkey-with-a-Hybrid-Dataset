import tkinter as tk
from tkinter import messagebox, ttk
import json
import os
import re
import smtplib
from email.message import EmailMessage

CONFIG_FILE = "email_config.json"

class EmailSettingsWindow:
    def __init__(self, parent=None):
        self.window = tk.Toplevel(parent) if parent else tk.Tk()
        self.window.title("E-posta Bildirim Ayarları")
        self.window.geometry("520x450")  # Biraz daha büyük yaptık
        self.window.resizable(True, True)  # Yeniden boyutlandırmayı etkinleştirdik
        
        # Pencereyi merkeze al
        if parent:
            self.window.transient(parent)
            self.window.grab_set()
        
        self.setup_ui()
        self.load_settings()
        
    def setup_ui(self):
        # Canvas ve Scrollbar için ana frame
        canvas = tk.Canvas(self.window)
        scrollbar = ttk.Scrollbar(self.window, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Ana frame (artık scrollable_frame içinde)
        main_frame = tk.Frame(scrollable_frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Başlık
        title_label = tk.Label(main_frame, text="E-posta Bildirim Ayarları", 
                              font=("Arial", 16, "bold"), fg="#2196F3")
        title_label.pack(pady=(0, 20))
        
        # E-posta bildirimleri checkbox (en üstte)
        self.notification_enabled_var = tk.BooleanVar(value=True)
        self.notification_checkbox = tk.Checkbutton(main_frame, 
                                                   text="E-posta bildirimlerini etkinleştir",
                                                   variable=self.notification_enabled_var,
                                                   font=("Arial", 12), fg="#4CAF50")
        self.notification_checkbox.pack(pady=(0, 15))
        
        # Bildirim alacak kişi
        notification_frame = tk.LabelFrame(main_frame, text="Bildirim Alacak Kişi", 
                                          font=("Arial", 12, "bold"), padx=10, pady=10)
        notification_frame.pack(fill=tk.X, pady=(0, 15))
        
        tk.Label(notification_frame, text="E-posta Adresi:", font=("Arial", 10)).pack(anchor=tk.W, pady=(0, 5))
        self.email_var = tk.StringVar()
        self.email_entry = tk.Entry(notification_frame, textvariable=self.email_var, 
                                   font=("Arial", 12), width=40)
        self.email_entry.pack(fill=tk.X, pady=(0, 5))
        
        # Örnek göster
        example_label = tk.Label(notification_frame, text="Örnek: kullanici@gmail.com", 
                               font=("Arial", 9), fg="gray")
        example_label.pack(anchor=tk.W)
        
        # SMTP Ayarları Frame
        smtp_frame = tk.LabelFrame(main_frame, text="SMTP Sunucu Ayarları", 
                                  font=("Arial", 12, "bold"), padx=10, pady=10)
        smtp_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Grid yapılandırması
        smtp_frame.columnconfigure(1, weight=1)
        
        # SMTP Sunucu
        tk.Label(smtp_frame, text="SMTP Sunucu:", font=("Arial", 10)).grid(
            row=0, column=0, sticky=tk.W, pady=5)
        self.smtp_host_var = tk.StringVar(value="smtp.gmail.com")
        self.smtp_host_entry = tk.Entry(smtp_frame, textvariable=self.smtp_host_var, 
                                       font=("Arial", 10))
        self.smtp_host_entry.grid(row=0, column=1, sticky=tk.EW, padx=(10, 0), pady=5)
        
        # Port
        tk.Label(smtp_frame, text="Port:", font=("Arial", 10)).grid(
            row=1, column=0, sticky=tk.W, pady=5)
        self.smtp_port_var = tk.StringVar(value="587")
        self.smtp_port_entry = tk.Entry(smtp_frame, textvariable=self.smtp_port_var, 
                                       font=("Arial", 10), width=10)
        self.smtp_port_entry.grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        
        # Gönderici E-posta
        tk.Label(smtp_frame, text="Gönderici E-posta:", font=("Arial", 10)).grid(
            row=2, column=0, sticky=tk.W, pady=5)
        self.sender_email_var = tk.StringVar()
        self.sender_email_entry = tk.Entry(smtp_frame, textvariable=self.sender_email_var, 
                                          font=("Arial", 10))
        self.sender_email_entry.grid(row=2, column=1, sticky=tk.EW, padx=(10, 0), pady=5)
        
        # Uygulama Şifresi
        tk.Label(smtp_frame, text="Uygulama Şifresi:", font=("Arial", 10)).grid(
            row=3, column=0, sticky=tk.W, pady=5)
        self.app_password_var = tk.StringVar()
        self.app_password_entry = tk.Entry(smtp_frame, textvariable=self.app_password_var, 
                                          font=("Arial", 10), show="*")
        self.app_password_entry.grid(row=3, column=1, sticky=tk.EW, padx=(10, 0), pady=5)
        
        # Yardım metni
        help_frame = tk.LabelFrame(main_frame, text="Gmail Ayarları İçin Yardım", 
                                  font=("Arial", 12, "bold"), padx=10, pady=10)
        help_frame.pack(fill=tk.X, pady=(0, 20))
        
        help_text = tk.Label(help_frame, 
                           text="1. Google hesabınızda 2 adımlı doğrulamayı açın\n"
                                "2. Google Hesabı → Güvenlik → Uygulama şifreleri\n"
                                "3. Yeni uygulama şifresi oluşturun\n"
                                "4. 16 haneli şifreyi yukarıdaki alana yapıştırın",
                           font=("Arial", 9), fg="blue", justify=tk.LEFT)
        help_text.pack(anchor=tk.W)
        
        # Butonlar frame - SABIT KONUM
        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(30, 20))
        
        # Test e-postası butonu
        self.test_button = tk.Button(button_frame, text="📧 Test E-postası Gönder", 
                                    command=self.test_email, bg="#FF9800", fg="white",
                                    font=("Arial", 11, "bold"), padx=15, pady=8, relief="raised")
        self.test_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Kaydet butonu - BÜYÜK VE BELIRGIN
        self.save_button = tk.Button(button_frame, text="✅ KAYDET", 
                                    command=self.save_settings, bg="#4CAF50", fg="white",
                                    font=("Arial", 12, "bold"), padx=30, pady=10, relief="raised")
        self.save_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # İptal butonu
        self.cancel_button = tk.Button(button_frame, text="❌ İptal", 
                                      command=self.window.destroy, bg="#F44336", fg="white",
                                      font=("Arial", 11, "bold"), padx=20, pady=8, relief="raised")
        self.cancel_button.pack(side=tk.LEFT)
        
        # Canvas ve scrollbar'ı pack et
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
    def load_settings(self):
        """Kaydedilmiş ayarları yükle"""
        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    config = json.load(f)
                
                self.email_var.set(config.get("recipient_email", ""))
                self.smtp_host_var.set(config.get("smtp_host", "smtp.gmail.com"))
                self.smtp_port_var.set(config.get("smtp_port", "587"))
                self.sender_email_var.set(config.get("sender_email", ""))
                self.app_password_var.set(config.get("app_password", ""))
                self.notification_enabled_var.set(config.get("notifications_enabled", True))
        except Exception as e:
            print(f"Ayarlar yüklenirken hata: {e}")
    
    def save_settings(self):
        """E-posta ayarlarını kaydet"""
        # E-posta doğrulama
        recipient_email = self.email_var.get().strip()
        sender_email = self.sender_email_var.get().strip()
        
        if not recipient_email:
            messagebox.showerror("Hata", "Bildirim alacak e-posta adresini girin!")
            self.email_entry.focus()
            return
            
        if not re.match(r"[^@]+@[^@]+\.[^@]+", recipient_email):
            messagebox.showerror("Hata", "Geçerli bir alıcı e-posta adresi girin!")
            self.email_entry.focus()
            return
            
        if not sender_email:
            messagebox.showerror("Hata", "Gönderici e-posta adresini girin!")
            self.sender_email_entry.focus()
            return
            
        if not re.match(r"[^@]+@[^@]+\.[^@]+", sender_email):
            messagebox.showerror("Hata", "Geçerli bir gönderici e-posta adresi girin!")
            self.sender_email_entry.focus()
            return
        
        if not self.app_password_var.get().strip():
            messagebox.showerror("Hata", "Uygulama şifresini girin!")
            self.app_password_entry.focus()
            return
        
        # Port doğrulama
        try:
            port = int(self.smtp_port_var.get())
            if port <= 0 or port > 65535:
                raise ValueError
        except ValueError:
            messagebox.showerror("Hata", "Geçerli bir port numarası girin (1-65535)!")
            self.smtp_port_entry.focus()
            return
        
        # Ayarları kaydet
        config = {
            "recipient_email": recipient_email,
            "smtp_host": self.smtp_host_var.get().strip(),
            "smtp_port": self.smtp_port_var.get().strip(),
            "sender_email": sender_email,
            "app_password": self.app_password_var.get().strip(),
            "notifications_enabled": self.notification_enabled_var.get()
        }
        
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            messagebox.showinfo("Başarılı", "✅ E-posta ayarları başarıyla kaydedildi!\n\n"
                              "Artık sistem plaka tanıdığında size bildirim gönderecek.")
            self.window.destroy()
        except Exception as e:
            messagebox.showerror("Hata", f"Ayarlar kaydedilirken hata: {e}")
    
    def test_email(self):
        """Test e-postası gönder"""
        try:
            recipient_email = self.email_var.get().strip()
            sender_email = self.sender_email_var.get().strip()
            smtp_host = self.smtp_host_var.get().strip()
            smtp_port = int(self.smtp_port_var.get())
            app_password = self.app_password_var.get().strip()
            
            if not all([recipient_email, sender_email, smtp_host, app_password]):
                messagebox.showerror("Hata", "Tüm alanları doldurun!")
                return
            
            # Test e-postası gönder
            msg = EmailMessage()
            msg["From"] = sender_email
            msg["To"] = recipient_email
            msg["Subject"] = "✅ Test - Plaka Tanıma Sistemi"
            msg.set_content("""Merhaba!

Bu bir test e-postasıdır. 

✅ E-posta ayarlarınız başarıyla çalışıyor!
📧 Sistem artık plaka tanıdığında size bildirim gönderebilir.

Sistem Bilgileri:
- SMTP Sunucu: """ + smtp_host + """
- Port: """ + str(smtp_port) + """
- Gönderici: """ + sender_email + """

İyi çalışmalar!
Plaka Tanıma Sistemi""")
            
            with smtplib.SMTP(smtp_host, smtp_port) as smtp:
                smtp.ehlo()
                smtp.starttls()
                smtp.login(sender_email, app_password)
                smtp.send_message(msg)
            
            messagebox.showinfo("Başarılı", "✅ Test e-postası başarıyla gönderildi!\n\n"
                              f"📧 {recipient_email} adresini kontrol edin.")
            
        except smtplib.SMTPAuthenticationError:
            messagebox.showerror("Hata", "❌ Kimlik doğrulama hatası!\n\n"
                               "• Uygulama şifresini kontrol edin\n"
                               "• 2 adımlı doğrulamanın açık olduğundan emin olun")
        except Exception as e:
            messagebox.showerror("Hata", f"❌ Test e-postası gönderilirken hata:\n\n{e}")

if __name__ == "__main__":
    app = EmailSettingsWindow()
    app.window.mainloop()