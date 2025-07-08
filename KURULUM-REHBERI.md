# GitHub ile Proje Kurulum Rehberi

## 1. Adım: Mevcut Dosyaları Sil
```bash
# Eski dosyaları sil
rm -rf vucut-analiz-ui
rm -rf vucut-analiz-backend
```

## 2. Adım: Bu Projeyi GitHub'a Push Et
```bash
# Git başlat (eğer başlatılmamışsa)
git init

# Tüm dosyaları ekle
git add .

# Commit yap
git commit -m "Vücut analizi projesi - ilk commit"

# GitHub repository'nize push edin
git branch -M main
git remote add origin https://github.com/KULLANICI_ADINIZ/REPO_ADINIZ.git
git push -u origin main
```

## 3. Adım: Bilgisayarınızda Clone Et
```bash
# İstediğiniz klasöre gidin
cd ~/Desktop  # veya istediğiniz konum

# Projeyi clone edin
git clone https://github.com/KULLANICI_ADINIZ/REPO_ADINIZ.git

# Proje klasörüne girin
cd REPO_ADINIZ
```

## 4. Adım: Bağımlılıkları Yükle
```bash
# Frontend bağımlılıkları
npm install

# Backend bağımlılıkları
cd backend
pip install -r requirements.txt
cd ..
```

## 5. Adım: Projeyi Çalıştır
```bash
# Terminal 1: Backend
cd backend
python app.py

# Terminal 2: Frontend (yeni terminal açın)
npm run dev
```

## Alternatif Kolay Yol: Tek Komutla Kurulum

Aşağıdaki script'i `kurulum.sh` olarak kaydedin:

```bash
#!/bin/bash
echo "🚀 Proje kuruluyor..."

# Bağımlılıkları yükle
npm install
cd backend && pip install -r requirements.txt && cd ..

echo "✅ Kurulum tamamlandı!"
echo "Backend: cd backend && python app.py"
echo "Frontend: npm run dev"
```

Sonra çalıştırın:
```bash
chmod +x kurulum.sh
./kurulum.sh
```

## Sorun Çözme

### Python bağımlılık sorunu:
```bash
pip install --upgrade pip
pip install -r backend/requirements.txt
```

### Node.js sorunu:
```bash
npm cache clean --force
npm install
```

### Port sorunu:
- Backend: 5000 portu
- Frontend: 3000 portu
- Portlar boş olmalı

## Proje Yapısı
```
proje/
├── backend/          # Python Flask server
├── src/             # React frontend
├── package.json     # Node.js bağımlılıkları
└── README.md        # Proje açıklaması
```