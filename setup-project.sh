#!/bin/bash

echo "🚀 Vücut Analizi Projesi Kurulumu Başlıyor..."

# Mevcut klasörleri sil
echo "🗑️ Eski dosyalar siliniyor..."
rm -rf vucut-analiz-ui
rm -rf vucut-analiz-backend

echo "📦 Gerekli bağımlılıklar yükleniyor..."
npm install

echo "🐍 Python backend bağımlılıkları yükleniyor..."
cd backend
pip install -r requirements.txt
cd ..

echo "✅ Kurulum tamamlandı!"
echo ""
echo "🚀 Projeyi çalıştırmak için:"
echo "   Terminal 1: cd backend && python app.py"
echo "   Terminal 2: npm run dev"
echo ""
echo "📋 Proje yapısı:"
echo "   ├── backend/          (Python Flask server)"
echo "   ├── src/              (React frontend)"
echo "   ├── package.json      (Node.js bağımlılıkları)"
echo "   └── vite.config.ts    (Vite konfigürasyonu)"