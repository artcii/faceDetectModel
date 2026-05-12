#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 18:54:31 2026

@author: mac
"""

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms



"BÖLÜM 1: Veri Hazırlığı, Ön İşleme ve Yükleme (Data Loading & Preprocessing) "


# 1. ÖZEL VERİ SETİ SINIFI (Custom Dataset)
class FaceDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        csv_file: metadata.csv dosyasının yolu.
        root_dir: 'Fake faces' ve 'Real faces' klasörlerini içeren ana dizin.
        transform: Uygulanacak görüntü dönüşümleri (augmentation & normalization).
        """
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
        # Yapay zeka metin anlamaz, o yüzden 'real' -> 0, 'fake' -> 1 olarak haritalıyoruz.
        self.label_map = {'real': 0, 'fake': 1}

    def __len__(self):
        # Toplam veri sayısını döndürür (20.000)
        return len(self.annotations)

    def __getitem__(self, index):
        # CSV'den ilgili satırdaki dosya yolunu al (Örn: 'Fake faces/fake_9650.png')
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        
        # Resmi PIL kütüphanesi ile aç ve RGB formatına getir (Eğer siyah-beyaz varsa hata vermesin diye)
        image = Image.open(img_path).convert("RGB")
        
        # CSV'den etiketi al ve rakama çevir (0 veya 1)
        label_str = self.annotations.iloc[index, 1]
        y_label = torch.tensor(self.label_map[label_str], dtype=torch.float32) # BCE Loss float bekler
        
        # Eğer dönüşüm tanımlandıysa uygula (Tensöre çevirme vb.)
        if self.transform:
            image = self.transform(image)
            
        return image, y_label

# 2. DÖNÜŞÜMLER (Bir önceki adımdaki ile aynı)
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. VERİ SETİNİ BAŞLATMA VE BÖLME
# DİKKAT: Ana klasörünüzün yolunu buraya yapıştırın. metadata.csv bu klasörün içinde olmalı.
ANA_KLASOR_YOLU = "/Users/mac/Desktop/archive-2" # <--- BURAYI DEĞİŞTİRİN
CSV_YOLU = os.path.join(ANA_KLASOR_YOLU, "metadata.csv")

# Veri setini oluşturuyoruz
tum_veri_seti = FaceDataset(csv_file=CSV_YOLU, root_dir=ANA_KLASOR_YOLU, transform=train_transforms)

toplam_boyut = len(tum_veri_seti)
egitim_boyutu = int(0.8 * toplam_boyut)
dogrulama_boyutu = int(0.1 * toplam_boyut)
test_boyutu = toplam_boyut - egitim_boyutu - dogrulama_boyutu

egitim_seti, dogrulama_seti, test_seti = random_split(
    tum_veri_seti, [egitim_boyutu, dogrulama_boyutu, test_boyutu]
)

# Doğrulama ve test setleri için veri artırmayı (augmentation) kapatıyoruz
dogrulama_seti.dataset.transform = test_val_transforms
test_seti.dataset.transform = test_val_transforms

# 4. DATALOADER OLUŞTURMA
BATCH_SIZE = 32

train_loader = DataLoader(egitim_seti, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dogrulama_seti, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_seti, batch_size=BATCH_SIZE, shuffle=False)

print("Veri yükleme işlemi başarıyla tamamlandı!")




" BÖLÜM 2: Modelin Kurulumu ve Donanım Optimizasyonu "


import torch.nn as nn
from torchvision import models

# ---------------------------------------------------------
# ADIM 2: MODEL KURULUMU VE DONANIM OPTİMİZASYONU
# ---------------------------------------------------------

# 1. Donanım (Device) Seçimi
# Apple M serisi çipler için 'mps', yoksa 'cpu' kullan
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Kullanılan donanım: {device}")

# 2. Önceden Eğitilmiş ResNet18 Modelini Yükleme
# Önceden öğrenilmiş özellikleri kullanmak için ağırlıkları yüklüyoruz
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# 3. İnce Ayar (Fine-Tuning) için Son Katmanı Değiştirme
# Modelin orijinal son katmanı 1000 çıktı verir, biz bunu 1'e düşürüyoruz.
num_ftrs = model.fc.in_features # Son katmana giren özellik sayısı (ResNet18 için 512'dir)
model.fc = nn.Linear(num_ftrs, 1) # Tek bir nöron, 0 (gerçek) ile 1 (sahte) arası değer üretecek

# 4. Modeli Seçilen Donanıma (M4 GPU'suna) Gönderme
model = model.to(device)

print("ResNet18 modeli başarıyla kuruldu ve M4 çipine aktarıldı!")




" BÖLÜM 3 VE 4: KAYIP FONKSİYONU VE OPTİMİZASYON, EĞİTİM DÖNÜSÜ"


import torch.optim as optim
import time

# ---------------------------------------------------------
# ADIM 3: KAYIP FONKSİYONU VE OPTİMİZASYON
# ---------------------------------------------------------

# İkili sınıflandırma (Binary Classification) için en iyi kayıp fonksiyonu
criterion = nn.BCEWithLogitsLoss()

# Adam optimizasyon algoritması (Öğrenme oranı 0.0001 ile hassas adımlar atıyoruz)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# ---------------------------------------------------------
# ADIM 4: EĞİTİM DÖNGÜSÜ (TRAINING LOOP)
# ---------------------------------------------------------

EPOCHS = 5 # Başlangıç için modelimizi tüm verilerle 5 tur eğiteceğiz

print("Eğitim süreci başlıyor... Bu işlem M4 çipinizin hızına bağlı olarak biraz vakit alabilir.")

# Eğitimin sonunda elimizde kalacak olan "En İyi" modeli kaydetmek için hazırlık
best_val_loss = float('inf')
best_model_path = "en_iyi_sahte_yuz_modeli.pth"

for epoch in range(EPOCHS):
    baslangic_zamani = time.time()

    # --- 1. EĞİTİM FAZI ---
    model.train() # Modeli eğitim (öğrenme) moduna al
    train_loss = 0.0
    train_dogru = 0
    train_toplam = 0

    # DataLoader içinden 32'şerli resim ve etiket paketlerini alıyoruz
    for inputs, labels in train_loader:
        # Verileri işlemciden (CPU), M4'ün grafik çipine (MPS) gönder
        inputs = inputs.to(device)
        labels = labels.to(device).unsqueeze(1) # Etiketleri model çıktısıyla aynı boyuta [32, 1] getir

        # Önceki adımdan kalan hataları sıfırla (Çok önemli!)
        optimizer.zero_grad()

        # İleri Yayılım: Resimleri modele ver ve tahmin al
        outputs = model(inputs)

        # Hata payını (Loss) hesapla
        loss = criterion(outputs, labels)

        # Geri Yayılım: Hatanın kaynağını bulmak için geriye doğru türev al
        loss.backward()

        # Öğrenme: Ağırlıkları hatayı azaltacak şekilde güncelle
        optimizer.step()

        # Bu turun istatistiklerini hesapla
        train_loss += loss.item() * inputs.size(0)
        
        # Modelin ürettiği ham çıktıları olasılığa (0-1 arası) çevir
        # 0.5'ten büyükse 1 (Sahte), küçükse 0 (Gerçek) tahmini yapmıştır
        tahminler = torch.sigmoid(outputs) >= 0.5
        train_dogru += (tahminler == labels).sum().item()
        train_toplam += labels.size(0)

    # 1 tur bittiğinde ortalama eğitim kaybı ve doğruluğunu bul
    epoch_train_loss = train_loss / len(train_loader.dataset)
    epoch_train_acc = train_dogru / train_toplam

    # --- 2. DOĞRULAMA (VALIDATION) FAZI ---
    model.eval() # Modeli değerlendirme moduna al (Öğrenmeyi durdur)
    val_loss = 0.0
    val_dogru = 0
    val_toplam = 0

    # Doğrulama sırasında gradyan hesaplamaya (türev almaya) gerek yoktur
    # Bu kısmı 'no_grad' içine almak cihazın hafızasını ve hızını korur
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            tahminler = torch.sigmoid(outputs) >= 0.5
            val_dogru += (tahminler == labels).sum().item()
            val_toplam += labels.size(0)

    epoch_val_loss = val_loss / len(val_loader.dataset)
    epoch_val_acc = val_dogru / val_toplam
    
    gecen_sure = time.time() - baslangic_zamani

    # Ekrana bu turun sonuçlarını yazdır
    print(f"\nEpoch {epoch+1}/{EPOCHS} | Süre: {gecen_sure:.0f} saniye")
    print(f"Eğitim    -> Kayıp: {epoch_train_loss:.4f}, Doğruluk: %{epoch_train_acc*100:.2f}")
    print(f"Doğrulama -> Kayıp: {epoch_val_loss:.4f}, Doğruluk: %{epoch_val_acc*100:.2f}")

    # Eğer model doğrulama setinde şimdiye kadarki en düşük hatayı yaptıysa, ağırlıklarını kaydet
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"[*] Yeni en iyi model diske kaydedildi! (Kayıp: {best_val_loss:.4f})")
    
    print("-" * 50)

print(f"\nEğitim tamamlandı! En iyi sonuçları veren model '{best_model_path}' dosyasına kaydedildi.")






































































































































































































































































































































































































