from sklearn.model_selection import train_test_split
import cv2
import os
import yaml

#################################################################
### Ayar: Dizin Yolu ve Geçerli Uzantılar
#################################################################
root_dir = r"C:\Users\eakkaya\Desktop\Source_Code\dataset\plaka_tanima_sistemi"
image_dir = os.path.join(root_dir, "images")
label_dir = os.path.join(root_dir, "labels")
valid_image_formats = [".jpg", ".jpeg", ".png"]

#################################################################
### Resim ve Etiketleri Eşleştir
#################################################################
image_paths = []
label_paths = []
pairs = []

for file in os.listdir(image_dir):
    ext = os.path.splitext(file)[1].lower()
    if ext in valid_image_formats:
        img_path = os.path.join(image_dir, file)
        img_name = os.path.splitext(file)[0]
        label_path = os.path.join(label_dir, f"{img_name}.txt")
        if os.path.exists(label_path):
            image_paths.append(img_path)
            label_paths.append(label_path)
            pairs.append((img_path, label_path))

print(f"Eşleşen veri sayısı: {len(pairs)}")

if len(pairs) == 0:
    print("HATA: Hiç eşleşen resim-etiket çifti bulunamadı. Klasörleri ve isimleri kontrol et.")
    exit()

#################################################################
### Veriyi Karıştır ve Böl (Train/Valid/Test)
#################################################################
train_pairs, val_test_pairs = train_test_split(pairs, test_size=0.3, random_state=42)
val_pairs, test_pairs = train_test_split(val_test_pairs, test_size=0.7, random_state=42)

#################################################################
### Belirtilen Klasöre Kopyala
#################################################################
def save_pairs(pairs, image_out_dir, label_out_dir):
    os.makedirs(image_out_dir, exist_ok=True)
    os.makedirs(label_out_dir, exist_ok=True)

    for img_path, label_path in pairs:
        img_name = os.path.basename(img_path)
        lbl_name = os.path.basename(label_path)

        img = cv2.imread(img_path)
        if img is None:
            print(f"Hatalı resim okunamadı: {img_path}")
            continue

        cv2.imwrite(os.path.join(image_out_dir, img_name), img)

        with open(label_path, "r") as lf, open(os.path.join(label_out_dir, lbl_name), "w") as f:
            f.write(lf.read())

save_pairs(train_pairs, "datasets/images/train", "datasets/labels/train")
save_pairs(val_pairs, "datasets/images/valid", "datasets/labels/valid")
save_pairs(test_pairs, "datasets/images/test", "datasets/labels/test")

#################################################################
### YAML Konfigürasyon Dosyası Yaz
#################################################################
data = {
    "path": "../datasets",
    "train": "images/train",
    "val": "images/valid",
    "test": "images/test",
    "names": ["number plate"]
}

with open("number-plate.yaml", "w") as f:
    yaml.dump(data, f)

print("✅ Veri başarıyla bölündü ve number-plate.yaml oluşturuldu.")
