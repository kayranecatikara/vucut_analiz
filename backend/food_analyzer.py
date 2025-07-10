#!/usr/bin/env python3
"""
LogMeal API ile yemek analizi ve kalori hesaplama modülü
"""

import requests
import base64
import json
import time
from typing import Dict, Any, Optional, List
from PIL import Image
import io

class FoodAnalyzer:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.logmeal.com/v2"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Türkçe yemek kalori veritabanı (100g için kalori değerleri)
        self.food_calories = {
            # Türk mutfağı temel yemekler
            "pilav": 130,
            "bulgur pilavı": 83,
            "makarna": 131,
            "ekmek": 265,
            "beyaz ekmek": 265,
            "tam buğday ekmek": 247,
            "tavuk": 165,
            "tavuk göğsü": 165,
            "dana eti": 250,
            "kuzu eti": 294,
            "balık": 206,
            "somon": 208,
            "ton balığı": 144,
            "yumurta": 155,
            "haşlanmış yumurta": 155,
            "omlet": 154,
            "peynir": 113,
            "beyaz peynir": 113,
            "kaşar peyniri": 330,
            "yoğurt": 59,
            "süt": 42,
            "tam yağlı süt": 61,
            
            # Sebzeler
            "domates": 18,
            "salatalık": 16,
            "marul": 15,
            "soğan": 40,
            "patates": 77,
            "kızarmış patates": 365,
            "havuç": 41,
            "brokoli": 34,
            "karnabahar": 25,
            "patlıcan": 25,
            "biber": 31,
            "kabak": 17,
            
            # Meyveler
            "elma": 52,
            "muz": 89,
            "portakal": 47,
            "üzüm": 62,
            "çilek": 32,
            "karpuz": 30,
            "kavun": 34,
            "armut": 57,
            "şeftali": 39,
            
            # İçecekler
            "çay": 1,
            "kahve": 2,
            "türk kahvesi": 2,
            "su": 0,
            "meyve suyu": 45,
            "kola": 41,
            "ayran": 36,
            
            # Tatlılar ve atıştırmalıklar
            "baklava": 517,
            "künefe": 223,
            "sütlaç": 122,
            "muhallebi": 130,
            "lokum": 322,
            "çikolata": 546,
            "dondurma": 207,
            "kek": 257,
            "kurabiye": 502,
            "cips": 536,
            "fındık": 628,
            "ceviz": 654,
            "badem": 579,
            
            # Yağlar ve soslar
            "zeytinyağı": 884,
            "tereyağı": 717,
            "margarin": 717,
            "mayonez": 680,
            "ketçap": 112,
            
            # Türk yemekleri
            "döner": 280,
            "kebap": 250,
            "köfte": 250,
            "lahmacun": 159,
            "pide": 245,
            "börek": 312,
            "menemen": 154,
            "çorba": 50,
            "mercimek çorbası": 50,
            "yayla çorbası": 45,
            "dolma": 180,
            "sarma": 180,
            "meze": 150,
            "salata": 20,
            "çoban salatası": 20,
            
            # Uluslararası yemekler
            "pizza": 266,
            "hamburger": 295,
            "sandviç": 250,
            "spagetti": 131,
            "sushi": 200,
            "salad": 20,
            "soup": 50
        }
    
    def analyze_food_image(self, image_data: bytes) -> Dict[str, Any]:
        """
        Yemek fotoğrafını analiz et ve kalori hesapla
        """
        try:
            # Görüntüyü base64'e çevir
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            # LogMeal API'ya gönderilecek veri
            data = {
                "image": image_base64
            }
            
            print(f"🔍 LogMeal API'ya istek gönderiliyor...")
            
            # Food recognition endpoint'ine istek gönder
            response = requests.post(
                f"{self.base_url}/image/segmentation/complete",
                headers=self.headers,
                json=data,
                timeout=30
            )
            
            print(f"📡 API Yanıtı: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ API başarılı: {result}")
                return self._process_logmeal_results(result, image_base64)
            else:
                print(f"❌ API Hatası: {response.status_code} - {response.text}")
                return self._create_smart_fallback_result(image_base64)
                
        except Exception as e:
            print(f"❌ Yemek analizi hatası: {e}")
            return self._create_smart_fallback_result(image_base64)
    
    def _process_logmeal_results(self, api_result: Dict, image_base64: str) -> Dict[str, Any]:
        """
        LogMeal API sonuçlarını işle ve kalori hesapla
        """
        try:
            detected_foods = []
            total_calories = 0
            confidence_scores = []
            
            print(f"🔍 LogMeal API sonucu: {api_result}")
            
            # LogMeal API sonuçlarını kontrol et
            if 'segmentation_results' in api_result:
                segmentation_results = api_result['segmentation_results']
                
                for result in segmentation_results:
                    if 'recognition_results' in result:
                        recognition_results = result['recognition_results']
                        
                        for food_item in recognition_results:
                            food_name = food_item.get('name', '').lower()
                            confidence = food_item.get('prob', 0)
                            
                            print(f"🍽️ Tespit edilen: {food_name} (güven: {confidence})")
                            
                            if confidence > 0.3:  # %30'dan yüksek güvenilirlik
                                # Kalori hesapla
                                calories = self._calculate_calories(food_name)
                                
                                detected_foods.append({
                                    'name': self._translate_food_name(food_name),
                                    'confidence': confidence,
                                    'calories': calories
                                })
                                
                                total_calories += calories
                                confidence_scores.append(confidence)
            
            # Alternatif API yapısını kontrol et
            elif 'foodFamily' in api_result or 'food_family' in api_result:
                food_family = api_result.get('foodFamily') or api_result.get('food_family', [])
                for food_item in food_family:
                    food_name = food_item.get('name', '').lower()
                    confidence = food_item.get('confidence', 0.5)
                    
                    print(f"🍽️ Food family tespit: {food_name} (güven: {confidence})")
                    
                    if confidence > 0.3:
                        calories = self._calculate_calories(food_name)
                        detected_foods.append({
                            'name': self._translate_food_name(food_name),
                            'confidence': confidence,
                            'calories': calories
                        })
                        total_calories += calories
                        confidence_scores.append(confidence)
            
            # Eğer hiç yemek tespit edilmediyse akıllı varsayılan
            if not detected_foods:
                print("⚠️ Hiç yemek tespit edilemedi, fallback kullanılıyor")
                return self._create_smart_fallback_result(image_base64)
            
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
            
            print(f"✅ Toplam {len(detected_foods)} yemek tespit edildi, toplam kalori: {total_calories}")
            
            return {
                'success': True,
                'detected_foods': detected_foods,
                'total_calories': int(total_calories),
                'confidence': avg_confidence,
                'image': image_base64,
                'analysis_time': time.time(),
                'api_used': 'LogMeal'
            }
            
        except Exception as e:
            print(f"❌ LogMeal sonuç işleme hatası: {e}")
            return self._create_smart_fallback_result(image_base64)
    
    def _calculate_calories(self, food_name: str) -> int:
        """
        Yemek adına göre kalori hesapla (ortalama porsiyon için)
        """
        print(f"🔢 Kalori hesaplanıyor: {food_name}")
        
        # Önce tam eşleşme ara
        if food_name in self.food_calories:
            calories = int(self.food_calories[food_name] * 1.5)  # Ortalama porsiyon (150g)
            print(f"✅ Tam eşleşme bulundu: {calories} kalori")
            return calories
        
        # Kısmi eşleşme ara
        for key, calories in self.food_calories.items():
            if key in food_name or food_name in key:
                result_calories = int(calories * 1.5)
                print(f"✅ Kısmi eşleşme bulundu ({key}): {result_calories} kalori")
                return result_calories
        
        # İngilizce-Türkçe eşleştirme
        english_turkish = {
            'rice': 'pilav',
            'bread': 'ekmek',
            'chicken': 'tavuk',
            'beef': 'dana eti',
            'fish': 'balık',
            'egg': 'yumurta',
            'cheese': 'peynir',
            'milk': 'süt',
            'apple': 'elma',
            'banana': 'muz',
            'orange': 'portakal',
            'potato': 'patates',
            'tomato': 'domates',
            'pasta': 'makarna',
            'pizza': 'pizza',
            'burger': 'hamburger',
            'sandwich': 'sandviç',
            'soup': 'çorba',
            'salad': 'salata'
        }
        
        for eng, tur in english_turkish.items():
            if eng in food_name and tur in self.food_calories:
                result_calories = int(self.food_calories[tur] * 1.5)
                print(f"✅ İngilizce eşleşme bulundu ({eng}->{tur}): {result_calories} kalori")
                return result_calories
        
        # Yemek kategorisine göre varsayılan değerler
        if any(word in food_name for word in ['meat', 'beef', 'chicken', 'et', 'tavuk', 'kebap']):
            print(f"🥩 Et kategorisi tespit edildi: 300 kalori")
            return 300  # Et yemekleri
        elif any(word in food_name for word in ['bread', 'rice', 'pasta', 'ekmek', 'pilav', 'makarna']):
            print(f"🍞 Karbonhidrat kategorisi tespit edildi: 200 kalori")
            return 200  # Karbonhidrat
        elif any(word in food_name for word in ['vegetable', 'salad', 'sebze', 'salata']):
            print(f"🥗 Sebze kategorisi tespit edildi: 50 kalori")
            return 50   # Sebze
        elif any(word in food_name for word in ['fruit', 'apple', 'banana', 'meyve']):
            print(f"🍎 Meyve kategorisi tespit edildi: 80 kalori")
            return 80   # Meyve
        elif any(word in food_name for word in ['dessert', 'cake', 'chocolate', 'tatlı', 'baklava']):
            print(f"🍰 Tatlı kategorisi tespit edildi: 400 kalori")
            return 400  # Tatlı
        elif any(word in food_name for word in ['drink', 'juice', 'içecek', 'suyu']):
            print(f"🥤 İçecek kategorisi tespit edildi: 60 kalori")
            return 60   # İçecek
        else:
            print(f"❓ Kategori bulunamadı, varsayılan: 150 kalori")
            return 150  # Genel varsayılan
    
    def _translate_food_name(self, food_name: str) -> str:
        """
        İngilizce yemek adlarını Türkçe'ye çevir
        """
        translations = {
            'rice': 'Pilav',
            'bread': 'Ekmek',
            'chicken': 'Tavuk',
            'beef': 'Dana Eti',
            'fish': 'Balık',
            'salmon': 'Somon',
            'egg': 'Yumurta',
            'cheese': 'Peynir',
            'milk': 'Süt',
            'yogurt': 'Yoğurt',
            'apple': 'Elma',
            'banana': 'Muz',
            'orange': 'Portakal',
            'potato': 'Patates',
            'tomato': 'Domates',
            'cucumber': 'Salatalık',
            'onion': 'Soğan',
            'carrot': 'Havuç',
            'salad': 'Salata',
            'pasta': 'Makarna',
            'pizza': 'Pizza',
            'burger': 'Hamburger',
            'sandwich': 'Sandviç',
            'soup': 'Çorba',
            'cake': 'Kek',
            'cookie': 'Kurabiye',
            'chocolate': 'Çikolata',
            'ice cream': 'Dondurma',
            'water': 'Su',
            'juice': 'Meyve Suyu',
            'coffee': 'Kahve',
            'tea': 'Çay',
            'kebab': 'Kebap',
            'doner': 'Döner',
            'baklava': 'Baklava',
            'turkish delight': 'Lokum'
        }
        
        # Önce direkt çeviri ara
        translated = translations.get(food_name.lower())
        if translated:
            return translated
        
        # Kısmi eşleşme ara
        for eng, tur in translations.items():
            if eng in food_name.lower():
                return tur
        
        # Bulunamazsa başlık formatında döndür
        return food_name.title()
    
    def _create_smart_fallback_result(self, image_base64: str) -> Dict[str, Any]:
        """
        API başarısız olduğunda akıllı varsayılan sonuç oluştur
        """
        # Basit görüntü analizi ile daha akıllı tahmin
        try:
            # Base64'ü decode et ve basit renk analizi yap
            import base64
            from PIL import Image
            import io
            
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))
            
            # Görüntüyü küçült ve renk analizi yap
            image = image.resize((100, 100))
            pixels = list(image.getdata())
            
            # Ortalama renk hesapla
            avg_r = sum(p[0] for p in pixels) / len(pixels)
            avg_g = sum(p[1] for p in pixels) / len(pixels)
            avg_b = sum(p[2] for p in pixels) / len(pixels)
            
            # Renk bazlı tahmin
            if avg_r > 150 and avg_g < 100 and avg_b < 100:
                # Kırmızı tonları - elma, domates, kırmızı biber
                selected_food = {'name': 'Elma', 'confidence': 0.6, 'calories': 80}
            elif avg_g > 120 and avg_r < 100:
                # Yeşil tonları - sebze
                selected_food = {'name': 'Yeşil Sebze', 'confidence': 0.5, 'calories': 50}
            elif avg_r > 200 and avg_g > 150 and avg_b < 100:
                # Sarı/turuncu tonları - muz, portakal
                selected_food = {'name': 'Meyve', 'confidence': 0.5, 'calories': 70}
            elif avg_r > 100 and avg_g > 80 and avg_b > 60:
                # Kahverengi tonları - et, ekmek
                selected_food = {'name': 'Ana Yemek', 'confidence': 0.4, 'calories': 250}
            else:
                # Varsayılan
                selected_food = {'name': 'Bilinmeyen Yemek', 'confidence': 0.3, 'calories': 150}
                
        except Exception as e:
            print(f"Renk analizi hatası: {e}")
            # Hata durumunda güvenli varsayılan
            selected_food = {'name': 'Bilinmeyen Yemek', 'confidence': 0.3, 'calories': 150}
        
        return {
            'success': False,
            'detected_foods': [selected_food],
            'total_calories': selected_food['calories'],
            'confidence': selected_food['confidence'],
            'image': image_base64,
            'analysis_time': time.time(),
            'api_used': 'Fallback',
            'note': 'API analizi başarısız, renk bazlı tahmin kullanıldı'
        }

# Test fonksiyonu
def test_food_analyzer():
    """
    Food analyzer'ı test et
    """
    api_key = "920c5f81c0264c2ca92a1d916e604a7694c560e9"
    analyzer = FoodAnalyzer(api_key)
    
    # Test için basit bir görüntü oluştur
    test_image = Image.new('RGB', (100, 100), color='red')
    img_bytes = io.BytesIO()
    test_image.save(img_bytes, format='JPEG')
    img_bytes = img_bytes.getvalue()
    
    result = analyzer.analyze_food_image(img_bytes)
    print("Test sonucu:", json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    test_food_analyzer()