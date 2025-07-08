# 🎯 Canlı Vücut Analizi Sistemi

Intel RealSense kamera ve TensorFlow MoveNet kullanarak gerçek zamanlı vücut analizi yapan modern web uygulaması.

## ✨ Özellikler

- 🎥 **Canlı Kamera Görüntüsü**: Intel RealSense D435i desteği
- 🤖 **AI Pose Detection**: TensorFlow MoveNet Lightning modeli
- 📏 **3D Ölçümler**: Derinlik verisi ile gerçek ölçümler
- 🏃 **Vücut Tipi Analizi**: Ektomorf, Mezomorf, Endomorf sınıflandırması
- ⚡ **Gerçek Zamanlı**: WebSocket ile anlık veri aktarımı
- 🎨 **Modern Arayüz**: React + TypeScript + Tailwind CSS

## 🚀 Hızlı Kurulum

### Otomatik Kurulum
```bash
chmod +x kurulum.sh
./kurulum.sh
```

### Manuel Kurulum
```bash
# 1. Bağımlılıkları yükle
npm install
cd backend && pip install -r requirements.txt && cd ..

# 2. Backend'i başlat (Terminal 1)
cd backend
python app.py

# 3. Frontend'i başlat (Terminal 2)
npm run dev
```

## 📋 Gereksinimler

### Donanım
- Intel RealSense D435i kamera
- USB 3.0 portu
- Minimum 8GB RAM

### Yazılım
- Python 3.8+
- Node.js 16+
- Intel RealSense SDK 2.0

## 🛠️ Teknolojiler

### Backend
- **Flask**: Web server
- **SocketIO**: Gerçek zamanlı iletişim
- **OpenCV**: Görüntü işleme
- **TensorFlow**: AI model
- **pyrealsense2**: Kamera kontrolü

### Frontend
- **React 18**: UI framework
- **TypeScript**: Tip güvenliği
- **Tailwind CSS**: Styling
- **Socket.IO Client**: WebSocket
- **Lucide React**: İkonlar

## 📊 API Referansı

### WebSocket Events

#### Client → Server
- `start_video`: Kamera başlat
- `stop_video`: Kamera durdur

#### Server → Client
- `video_frame`: Video karesi (base64)
- `analyze_result`: Analiz sonuçları
- `stream_started`: Yayın başladı
- `stream_stopped`: Yayın durdu

### Analiz Verisi
```json
{
  "omuz_genisligi": 45.2,
  "bel_genisligi": 32.1,
  "omuz_bel_orani": 1.41,
  "vucut_tipi": "Mezomorf",
  "mesafe": 1.8,
  "confidence": 0.85
}
```

## 🔧 Konfigürasyon

### RealSense Ayarları
- **Çözünürlük**: 640x480 @ 30fps
- **Preset**: High Accuracy
- **Filtreler**: Spatial, Temporal, Hole-filling

### MoveNet Ayarları
- **Model**: SinglePose Lightning
- **Giriş Boyutu**: 192x192
- **Güven Eşiği**: 0.3

## 🐛 Sorun Giderme

### Kamera Bulunamadı
```bash
# RealSense SDK kurulumunu kontrol edin
realsense-viewer
```

### Port Çakışması
```bash
# Portları kontrol edin
lsof -i :3000  # Frontend
lsof -i :5000  # Backend
```

### Python Bağımlılık Sorunu
```bash
pip install --upgrade pip
pip install -r backend/requirements.txt
```

## 📁 Proje Yapısı

```
├── backend/              # Python Flask server
│   ├── app.py           # Ana server dosyası
│   ├── requirements.txt # Python bağımlılıkları
│   └── README.md        # Backend dokümantasyonu
├── src/                 # React frontend
│   ├── App.tsx          # Ana React bileşeni
│   ├── main.tsx         # React entry point
│   └── index.css        # Tailwind CSS
├── package.json         # Node.js bağımlılıkları
├── vite.config.ts       # Vite konfigürasyonu
└── kurulum.sh          # Otomatik kurulum scripti
```

## 📈 Performans İpuçları

- Kameraya 1-3 metre mesafede durun
- İyi aydınlatma sağlayın
- Kamera lensini temiz tutun
- Hızlı hareketlerden kaçının

## 📄 Lisans

MIT License - Detaylar için LICENSE dosyasına bakın.

## 🤝 Katkıda Bulunma

1. Fork edin
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit edin (`git commit -m 'Add amazing feature'`)
4. Push edin (`git push origin feature/amazing-feature`)
5. Pull Request açın

## 📞 İletişim

Sorularınız için issue açabilir veya pull request gönderebilirsiniz.