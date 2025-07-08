#!/bin/bash

echo "🚀 Vücut Analizi Projesi Kuruluyor..."
echo ""

# Renk kodları
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Node.js kontrolü
if ! command -v node &> /dev/null; then
    echo -e "${RED}❌ Node.js bulunamadı. Lütfen Node.js yükleyin.${NC}"
    exit 1
fi

# Python kontrolü
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo -e "${RED}❌ Python bulunamadı. Lütfen Python yükleyin.${NC}"
    exit 1
fi

echo -e "${BLUE}📦 Frontend bağımlılıkları yükleniyor...${NC}"
npm install

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Frontend bağımlılıkları yüklendi${NC}"
else
    echo -e "${RED}❌ Frontend bağımlılık yüklemesi başarısız${NC}"
    exit 1
fi

echo -e "${BLUE}🐍 Backend bağımlılıkları yükleniyor...${NC}"
cd backend

# Python komutunu belirle
PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

$PYTHON_CMD -m pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Backend bağımlılıkları yüklendi${NC}"
else
    echo -e "${RED}❌ Backend bağımlılık yüklemesi başarısız${NC}"
    exit 1
fi

cd ..

echo ""
echo -e "${GREEN}🎉 Kurulum başarıyla tamamlandı!${NC}"
echo ""
echo -e "${BLUE}🚀 Projeyi çalıştırmak için:${NC}"
echo ""
echo -e "${BLUE}Terminal 1 (Backend):${NC}"
echo "cd backend"
echo "$PYTHON_CMD app.py"
echo ""
echo -e "${BLUE}Terminal 2 (Frontend):${NC}"
echo "npm run dev"
echo ""
echo -e "${BLUE}📋 Tarayıcıda açın:${NC}"
echo "http://localhost:3000"
echo ""