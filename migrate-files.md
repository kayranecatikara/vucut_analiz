# Dosya Taşıma Rehberi

## Mevcut Durumunuz:
```
vucut-analiz-backend/
├── app.py
├── requirements.txt
└── README.md

vucut-analiz-ui/
├── src/
│   └── App.js
├── package.json
├── package-lock.json
└── public/
```

## Hedef Yapı:
```
proje/
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   └── README.md
├── src/
│   ├── App.tsx
│   ├── main.tsx
│   └── index.css
├── package.json
├── index.html
└── vite.config.ts
```

## Taşıma Komutları:

### 1. Backend dosyalarını taşıma:
```bash
# Backend klasörü oluştur
mkdir -p backend

# Backend dosyalarını kopyala
cp vucut-analiz-backend/app.py backend/
cp vucut-analiz-backend/requirements.txt backend/
cp vucut-analiz-backend/README.md backend/
```

### 2. Frontend dosyalarını güncelleme:
```bash
# Mevcut App.js dosyanızı App.tsx olarak kopyala
cp vucut-analiz-ui/src/App.js src/App.tsx

# Package.json'dan gerekli bağımlılıkları kontrol et
```

### 3. Otomatik taşıma scripti:
```bash
#!/bin/bash
# migrate.sh

echo "🚀 Dosyalar taşınıyor..."

# Backend dosyalarını taşı
mkdir -p backend
cp vucut-analiz-backend/app.py backend/
cp vucut-analiz-backend/requirements.txt backend/
cp vucut-analiz-backend/README.md backend/

# Frontend bağımlılıklarını yükle
npm install socket.io-client

echo "✅ Taşıma tamamlandı!"
echo "📋 Sonraki adımlar:"
echo "   1. Backend: cd backend && python app.py"
echo "   2. Frontend: npm run dev"
```

## Manuel Adımlar:

### 1. Backend'i çalıştırma:
```bash
cd backend
pip install -r requirements.txt
python app.py
```

### 2. Frontend'i çalıştırma:
```bash
npm install
npm run dev
```

## Önemli Notlar:

- Mevcut `App.js` dosyanızı `App.tsx` olarak yeniden adlandırmanız gerekiyor
- Socket.io-client bağımlılığı zaten yüklü
- Backend kodu zaten güncellenmiş durumda
- Vite konfigürasyonu hazır

## Dosya İçeriklerini Güncelleme:

Mevcut `App.js` dosyanızı şu şekilde güncelleyin:
- `.js` uzantısını `.tsx` olarak değiştirin
- TypeScript tiplerini ekleyin
- Modern React hooks kullanın