#!/usr/bin/env python3
"""
Clarify AI ile yemek analizi ve kalori hesaplama modülü
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
        self.base_url = "https://api.clarifai.com/v2"
        self.headers = {
            "Authorization": f"Key {api_key}",
            "Content-Type": "application/json"
        }
        
        # Yemek kalori veritabanı (yaklaşık değerler)
        self.food_calories = {
            # Türk mutfağı
            "pilav": 130,
            "bulgur": 83,
            "makarna": 131,
            "ekmek": 265,
            "tavuk": 165,
            "et": 250,
            "balık": 206,
            "yumurta": 155,
            "peynir": 113,
            "yoğurt": 59,
            "süt": 42,
            "domates": 18,
            "salatalık": 16,
            "marul": 15,
            "soğan": 40,
            "patates": 77,
            "havuç": 41,
            "elma": 52,
            "muz": 89,
            "portakal": 47,
            "çay": 1,
            "kahve": 2,
            "şeker": 387,
            "bal": 304,
            "zeytin": 115,
            "zeytinyağı": 884,
            "tereyağı": 717,
            
            # Genel yemekler
            "rice": 130,
            "bread": 265,
            "chicken": 165,
            "beef": 250,
            "fish": 206,
            "egg": 155,
            "cheese": 113,
            "milk": 42,
            "apple": 52,
            "banana": 89,
            "orange": 47,
            "potato": 77,
            "tomato": 18,
            "salad": 20,
            "pasta": 131,
            "pizza": 266,
            "burger": 295,
            "sandwich": 250,
            "soup": 50,
            "cake": 257,
            "cookie": 502,
            "chocolate": 546,
            "ice cream": 207,
            "yogurt": 59,
            "water": 0,
            "juice": 45,
            "soda": 41
        }
    
    def analyze_food_image(self, image_data: bytes) -> Dict[str, Any]:
        """
        Yemek fotoğrafını analiz et ve kalori hesapla
        """
        try:
            # Görüntüyü base64'e çevir
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            # Clarify AI'ya gönderilecek veri
            data = {
                "inputs": [
                    {
                        "data": {
                            "image": {
                                "base64": image_base64
                            }
                        }
                    }
                ]
            }
            
            # Food model ile analiz et
            response = requests.post(
                f"{self.base_url}/models/food-item-recognition/outputs",
                headers=self.headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return self._process_food_results(result, image_base64)
            else:
                print(f"API Hatası: {response.status_code} - {response.text}")
                return self._create_fallback_result(image_base64)
                
        except Exception as e:
            print(f"Yemek analizi hatası: {e}")
            return self._create_fallback_result(image_base64)
    
    def _process_food_results(self, api_result: Dict, image_base64: str) -> Dict[str, Any]:
        """
        API sonuçlarını işle ve kalori hesapla
        """
        try:
            detected_foods = []
            total_calories = 0
            confidence_scores = []
            
            # API sonuçlarını kontrol et
            if 'outputs' in api_result and len(api_result['outputs']) > 0:
                output = api_result['outputs'][0]
                
                if 'data' in output and 'concepts' in output['data']:
                    concepts = output['data']['concepts']
                    
                    # En yüksek güvenilirlik skoruna sahip 5 yemeği al
                    top_foods = sorted(concepts, key=lambda x: x.get('value', 0), reverse=True)[:5]
                    
                    for concept in top_foods:
                        food_name = concept.get('name', '').lower()
                        confidence = concept.get('value', 0)
                        
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
            
            # Eğer hiç yemek tespit edilmediyse varsayılan değer
            if not detected_foods:
                detected_foods = [{'name': 'Genel Yemek', 'confidence': 0.5, 'calories': 200}]
                total_calories = 200
                confidence_scores = [0.5]
            
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
            
            return {
                'success': True,
                'detected_foods': detected_foods,
                'total_calories': int(total_calories),
                'confidence': avg_confidence,
                'image': image_base64,
                'analysis_time': time.time()
            }
            
        except Exception as e:
            print(f"Sonuç işleme hatası: {e}")
            return self._create_fallback_result(image_base64)
    
    def _calculate_calories(self, food_name: str) -> int:
        """
        Yemek adına göre kalori hesapla (100g için)
        """
        # Önce tam eşleşme ara
        if food_name in self.food_calories:
            return self.food_calories[food_name]
        
        # Kısmi eşleşme ara
        for key, calories in self.food_calories.items():
            if key in food_name or food_name in key:
                return calories
        
        # Yemek türüne göre varsayılan değerler
        if any(word in food_name for word in ['meat', 'beef', 'chicken', 'et', 'tavuk']):
            return 200
        elif any(word in food_name for word in ['bread', 'rice', 'pasta', 'ekmek', 'pilav']):
            return 150
        elif any(word in food_name for word in ['vegetable', 'salad', 'sebze', 'salata']):
            return 30
        elif any(word in food_name for word in ['fruit', 'apple', 'banana', 'meyve']):
            return 60
        elif any(word in food_name for word in ['dessert', 'cake', 'chocolate', 'tatlı']):
            return 300
        else:
            return 100  # Varsayılan değer
    
    def _translate_food_name(self, food_name: str) -> str:
        """
        İngilizce yemek adlarını Türkçe'ye çevir
        """
        translations = {
            'rice': 'Pilav',
            'bread': 'Ekmek',
            'chicken': 'Tavuk',
            'beef': 'Et',
            'fish': 'Balık',
            'egg': 'Yumurta',
            'cheese': 'Peynir',
            'milk': 'Süt',
            'apple': 'Elma',
            'banana': 'Muz',
            'orange': 'Portakal',
            'potato': 'Patates',
            'tomato': 'Domates',
            'salad': 'Salata',
            'pasta': 'Makarna',
            'pizza': 'Pizza',
            'burger': 'Hamburger',
            'sandwich': 'Sandviç',
            'soup': 'Çorba',
            'cake': 'Kek',
            'cookie': 'Kurabiye',
            'chocolate': 'Çikolata',
            'yogurt': 'Yoğurt',
            'water': 'Su',
            'juice': 'Meyve Suyu'
        }
        
        return translations.get(food_name.lower(), food_name.title())
    
    def _create_fallback_result(self, image_base64: str) -> Dict[str, Any]:
        """
        API başarısız olduğunda varsayılan sonuç oluştur
        """
        return {
            'success': False,
            'detected_foods': [
                {'name': 'Tespit Edilemeyen Yemek', 'confidence': 0.3, 'calories': 150}
            ],
            'total_calories': 150,
            'confidence': 0.3,
            'image': image_base64,
            'analysis_time': time.time(),
            'error': 'API analizi başarısız, varsayılan değer kullanıldı'
        }

# Test fonksiyonu
def test_food_analyzer():
    """
    Food analyzer'ı test et
    """
    api_key = "29b4f47bf7184373bbe0c8eb1d102529"
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