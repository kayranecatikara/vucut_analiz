#!/usr/bin/env python3
"""
LogMeal API key test scripti
"""

import requests
import base64
from PIL import Image
import io

def test_api_key():
    api_key = "920c5f81c0264c2ca92a1d916e604a7694c560e9"
    
    print(f"🔑 API Key test ediliyor: {api_key[:20]}...")
    
    # Test görüntüsü oluştur (kırmızı elma)
    test_image = Image.new('RGB', (200, 200), color=(200, 50, 50))
    img_bytes = io.BytesIO()
    test_image.save(img_bytes, format='JPEG')
    img_data = img_bytes.getvalue()
    
    # Base64'e çevir
    image_base64 = base64.b64encode(img_data).decode('utf-8')
    
    # API'ye istek gönder
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "image": image_base64
    }
    
    try:
        print("📡 LogMeal API'ye test isteği gönderiliyor...")
        response = requests.post(
            "https://api.logmeal.com/v2/image/segmentation/complete",
            headers=headers,
            json=data,
            timeout=30
        )
        
        print(f"📊 Yanıt Kodu: {response.status_code}")
        print(f"📄 Yanıt: {response.text[:500]}...")
        
        if response.status_code == 200:
            print("✅ API Key çalışıyor!")
            return True
        elif response.status_code == 401:
            print("❌ API Key geçersiz!")
            return False
        elif response.status_code == 429:
            print("⚠️ API limiti aşıldı!")
            return False
        else:
            print(f"❌ API Hatası: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Bağlantı hatası: {e}")
        return False

if __name__ == "__main__":
    test_api_key()