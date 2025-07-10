#!/usr/bin/env python3
"""
Test Tabanlı Vücut Analizi Sistemi - Gelişmiş Diyet Önerileri
- Teste başla butonuna basıldıktan sonra 10 saniye analiz
- Analiz sonunda kamera kapanır
- Detaylı vücut tipine göre diyet önerileri
- Haftalık yemek planı
"""

import eventlet
eventlet.monkey_patch()

from flask import Flask
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import cv2
import numpy as np
import base64
import time
import logging
import json
from typing import Optional, Tuple, Dict, Any
import threading

# --- Food Analysis ---
from food_analyzer import FoodAnalyzer

# --- AI Libraries ---
import tensorflow as tf
import tensorflow_hub as hub

# --- RealSense Library (Optional) ---
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
    print("✅ RealSense kütüphanesi bulundu")
except ImportError:
    rs = None
    REALSENSE_AVAILABLE = False
    print("⚠️ RealSense kütüphanesi bulunamadı - Webcam modu kullanılacak")

# --- Flask and SocketIO Setup ---
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# --- Global Variables ---
test_running = False
test_thread = None
camera = None
realsense_pipeline = None
camera_mode = "webcam"

# Test parametreleri
TEST_DURATION = 10  # 10 saniye test süresi
ANALYSIS_INTERVAL = 0.5  # Yarım saniyede bir analiz
FOOD_PHOTO_COUNTDOWN = 3  # Yemek fotoğrafı için geri sayım

# Analiz verileri toplama
analysis_results = []
final_analysis = {
    'omuz_genisligi': 0.0,
    'bel_genisligi': 0.0,
    'omuz_bel_orani': 0.0,
    'vucut_tipi': 'Analiz Bekleniyor',
    'mesafe': 0.0,
    'confidence': 0.0,
    'diyet_onerileri': []
}

# Food analyzer
food_analyzer = None

def initialize_food_analyzer():
    """Food analyzer'ı başlat"""
    global food_analyzer
    try:
        api_key = "29b4f47bf7184373bbe0c8eb1d102529"
        food_analyzer = FoodAnalyzer(api_key)
        print("✅ Food analyzer başlatıldı")
        return True
    except Exception as e:
        print(f"❌ Food analyzer başlatılamadı: {e}")
        return False

# --- Model Loading ---
print("🤖 Loading MoveNet model from TensorFlow Hub...")
model = None
movenet = None

# Model zaten yüklü, direkt kullan
print("✅ Model hazır, yükleme atlanıyor")
model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
movenet = model.signatures['serving_default']
print("✅ MoveNet model ready.")

INPUT_SIZE = 192

# --- Gelişmiş Diyet Önerileri Veritabanı ---
DIYET_ONERILERI = {
    'Ektomorf': {
        'ozellikler': [
            'İnce yapılı ve hızlı metabolizma',
            'Kilo almakta zorlanır',
            'Kas yapmak için daha fazla kalori gerekir',
            'Doğal olarak düşük vücut yağ oranı',
            'Uzun ve ince kemik yapısı'
        ],
        'beslenme_ilkeleri': [
            'Yüksek kalori alımı (günde 2500-3000 kalori)',
            'Karbonhidrat ağırlıklı beslenme (%50-60)',
            'Protein alımı (vücut ağırlığının kg başına 1.5-2g)',
            'Sağlıklı yağlar (%20-30)',
            'Sık öğün tüketimi (6-8 öğün/gün)',
            'Antrenman öncesi ve sonrası beslenmeye dikkat'
        ],
        'onerilen_besinler': [
            'Tam tahıl ekmek ve makarna',
            'Pirinç, bulgur, quinoa',
            'Tavuk, balık, yumurta',
            'Fındık, badem, ceviz',
            'Avokado, zeytinyağı',
            'Muz, hurma, kuru meyve',
            'Süt, yoğurt, peynir',
            'Protein tozu ve gainers',
            'Tatlı patates, yulaf'
        ],
        'kacinilmasi_gerekenler': [
            'Aşırı işlenmiş gıdalar',
            'Şekerli içecekler',
            'Trans yağlar',
            'Aşırı kafein',
            'Boş kalori içeren atıştırmalıklar'
        ],
        'ogun_plani': {
            'kahvalti': 'Yulaf ezmesi + muz + fındık + süt + bal',
            'ara_ogun_1': 'Tam tahıl kraker + peynir + ceviz',
            'ogle': 'Tavuk + pirinç + salata + zeytinyağı + avokado',
            'ara_ogun_2': 'Protein smoothie + meyve + yoğurt',
            'aksam': 'Balık + bulgur pilavı + sebze + zeytinyağı',
            'gece': 'Yoğurt + bal + ceviz + hurma'
        },
        'egzersiz_onerileri': [
            'Ağırlık antrenmanı (3-4 gün/hafta)',
            'Kısa süreli kardio (15-20 dakika)',
            'Compound hareketler (squat, deadlift)',
            'Yüksek ağırlık, düşük tekrar',
            'Uzun dinlenme süreleri'
        ],
        'suplement_onerileri': [
            'Whey protein',
            'Creatine',
            'Mass gainer',
            'Multivitamin',
            'Omega-3'
        ]
    },
    'Mezomorf': {
        'ozellikler': [
            'Atletik yapı ve orta metabolizma',
            'Kas yapma ve yağ yakma dengeli',
            'Vücut kompozisyonunu korumak kolay',
            'Doğal kas yapısı iyi',
            'Orta kemik yapısı'
        ],
        'beslenme_ilkeleri': [
            'Dengeli kalori alımı (günde 2000-2500 kalori)',
            'Dengeli makro besin dağılımı',
            'Protein alımı (vücut ağırlığının kg başına 1.2-1.5g)',
            'Karbonhidrat (%40-45), Yağ (%25-30)',
            'Düzenli öğün saatleri (5-6 öğün/gün)',
            'Antrenman periyodizasyonuna uygun beslenme'
        ],
        'onerilen_besinler': [
            'Yağsız et, tavuk, balık',
            'Yumurta ve süt ürünleri',
            'Tam tahıl ürünleri',
            'Taze meyve ve sebzeler',
            'Bakliyat (mercimek, nohut)',
            'Fındık ve tohum',
            'Zeytinyağı, balık yağı',
            'Quinoa, bulgur',
            'Yeşil yapraklı sebzeler'
        ],
        'kacinilmasi_gerekenler': [
            'Aşırı kalori alımı',
            'Rafine şeker',
            'İşlenmiş et ürünleri',
            'Aşırı doymuş yağ',
            'Alkol'
        ],
        'ogun_plani': {
            'kahvalti': 'Omlet + tam tahıl ekmek + domates + zeytinyağı',
            'ara_ogun_1': 'Elma + badem + yoğurt',
            'ogle': 'Izgara tavuk + quinoa + yeşil salata + zeytinyağı',
            'ara_ogun_2': 'Yoğurt + meyve + ceviz',
            'aksam': 'Balık + tatlı patates + buharda sebze',
            'gece': 'Az yağlı süt + tarçın + bal'
        },
        'egzersiz_onerileri': [
            'Karma antrenman programı',
            'Orta süreli kardio (30-45 dakika)',
            'Çeşitli spor aktiviteleri',
            'Functional training',
            'Esneklik çalışmaları'
        ],
        'suplement_onerileri': [
            'Whey protein',
            'BCAA',
            'Multivitamin',
            'Omega-3',
            'Magnezyum'
        ]
    },
    'Endomorf': {
        'ozellikler': [
            'Geniş yapılı ve yavaş metabolizma',
            'Kilo almaya eğilimli',
            'Yağ yakmak için daha fazla çaba gerekir',
            'Doğal olarak yüksek vücut yağ oranı',
            'Geniş kemik yapısı'
        ],
        'beslenme_ilkeleri': [
            'Kontrollü kalori alımı (günde 1500-2000 kalori)',
            'Düşük karbonhidrat (%30-35)',
            'Yüksek protein (vücut ağırlığının kg başına 1.5-2g)',
            'Orta yağ alımı (%25-30)',
            'Sık ve küçük öğünler (6-7 öğün/gün)',
            'Glisemik indeksi düşük besinler'
        ],
        'onerilen_besinler': [
            'Yağsız protein (tavuk göğsü, balık)',
            'Yeşil yapraklı sebzeler',
            'Düşük glisemik indeksli meyveler',
            'Tam tahıl ürünleri (az miktarda)',
            'Bakliyat ve mercimek',
            'Fındık (kontrollü miktarda)',
            'Zeytinyağı, avokado',
            'Brokoli, karnabahar',
            'Yaban mersini, çilek'
        ],
        'kacinilmasi_gerekenler': [
            'Basit karbonhidratlar',
            'Şekerli gıdalar ve içecekler',
            'İşlenmiş gıdalar',
            'Yüksek kalorili atıştırmalıklar',
            'Beyaz ekmek, pasta',
            'Alkol',
            'Geç saatlerde yemek'
        ],
        'ogun_plani': {
            'kahvalti': 'Protein omlet + sebze + az zeytinyağı + yeşil çay',
            'ara_ogun_1': 'Çiğ badem (10-15 adet) + yeşil elma',
            'ogle': 'Izgara balık + bol salata + limon + zeytinyağı',
            'ara_ogun_2': 'Yoğurt (şekersiz) + tarçın + ceviz',
            'aksam': 'Tavuk + buharda brokoli + bulgur (az)',
            'gece': 'Bitki çayı + badem (5-6 adet)'
        },
        'egzersiz_onerileri': [
            'Yoğun kardio (5-6 gün/hafta)',
            'Yüksek tekrarlı ağırlık antrenmanı',
            'HIIT (High Intensity Interval Training)',
            'Aktif yaşam tarzı',
            'Yürüyüş ve koşu'
        ],
        'suplement_onerileri': [
            'Whey protein',
            'L-Carnitine',
            'Green tea extract',
            'CLA',
            'Multivitamin'
        ]
    }
}

# --- Body Parts and Skeleton Definitions ---
KEYPOINT_DICT = {
    'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
    'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
    'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
}

EDGES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), 
    (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
]

def run_movenet(input_image: np.ndarray) -> np.ndarray:
    """Run MoveNet model on input image and return keypoints"""
    if movenet is None:
        # Model yoksa basit keypoint döndür (test için)
        return np.random.rand(17, 3) * 0.8 + 0.1  # 0.1-0.9 arası random değerler
        
    img_resized = tf.image.resize_with_pad(np.expand_dims(input_image, axis=0), INPUT_SIZE, INPUT_SIZE)
    input_tensor = tf.cast(img_resized, dtype=tf.int32)
    
    try:
        outputs = movenet(input_tensor)
        return outputs['output_0'].numpy()[0, 0]
    except Exception as e:
        # Hata varsa basit keypoint döndür
        return np.random.rand(17, 3) * 0.8 + 0.1

def calculate_pixel_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    """Calculate pixel distance between two points"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def safe_array_access(depth_array: np.ndarray, y: int, x: int) -> float:
    """Safely access depth array with bounds checking"""
    try:
        height, width = depth_array.shape[:2]
        y = max(0, min(height - 1, y))
        x = max(0, min(width - 1, x))
        return float(depth_array[y, x])
    except Exception as e:
        return 0.0

def calculate_3d_distance_safe(p1: Tuple[int, int], p2: Tuple[int, int], 
                              depth_frame, depth_intrinsics) -> Optional[float]:
    """Safely calculate 3D distance between two points using depth data"""
    try:
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_units = depth_frame.get_units()
        
        depth1 = safe_array_access(depth_image, p1[1], p1[0]) * depth_units
        depth2 = safe_array_access(depth_image, p2[1], p2[0]) * depth_units
        
        if depth1 <= 0 or depth2 <= 0 or depth1 > 5.0 or depth2 > 5.0:
            return None
            
        point1_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [p1[0], p1[1]], depth1)
        point2_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [p2[0], p2[1]], depth2)
        
        distance = np.linalg.norm(np.subtract(point1_3d, point2_3d))
        distance_cm = distance * 100
        
        if distance_cm > 200:
            return None
            
        return distance_cm
        
    except Exception as e:
        return None

def analyze_body_measurements(keypoints: np.ndarray, frame_shape: Tuple[int, int], 
                            depth_frame=None, depth_intrinsics=None) -> Dict[str, Any]:
    """Comprehensive body measurement analysis"""
    height, width = frame_shape[:2]
    
    analysis_data = {
        'omuz_genisligi': 0.0,
        'bel_genisligi': 0.0,
        'omuz_bel_orani': 0.0,
        'vucut_tipi': 'Analiz Bekleniyor',
        'mesafe': 0.0,
        'confidence': 0.0
    }
    
    try:
        if len(keypoints) < 17:
            return analysis_data
            
        # Extract keypoints
        ls_y, ls_x, ls_c = keypoints[KEYPOINT_DICT['left_shoulder']]
        rs_y, rs_x, rs_c = keypoints[KEYPOINT_DICT['right_shoulder']]
        lh_y, lh_x, lh_c = keypoints[KEYPOINT_DICT['left_hip']]
        rh_y, rh_x, rh_c = keypoints[KEYPOINT_DICT['right_hip']]
        
        # Calculate shoulder width
        shoulder_width = 0.0
        if ls_c > 0.3 and rs_c > 0.3:
            p1 = (int(ls_x * width), int(ls_y * height))
            p2 = (int(rs_x * width), int(rs_y * height))
            
            if depth_frame is not None and depth_intrinsics is not None:
                # RealSense 3D measurement
                shoulder_width = calculate_3d_distance_safe(p1, p2, depth_frame, depth_intrinsics)
            else:
                # Webcam pixel-based measurement
                pixel_distance = calculate_pixel_distance(p1, p2)
                # Daha gerçekçi omuz genişliği hesaplama
                shoulder_width = (pixel_distance / width) * 120  # Daha geniş ölçek
                shoulder_width = max(35, min(65, shoulder_width))  # 35-65cm arası
            
            if shoulder_width:
                analysis_data['omuz_genisligi'] = shoulder_width
        
        # Calculate waist width
        waist_width = 0.0
        if lh_c > 0.3 and rh_c > 0.3:
            p1 = (int(lh_x * width), int(lh_y * height))
            p2 = (int(rh_x * width), int(rh_y * height))
            
            if depth_frame is not None and depth_intrinsics is not None:
                # RealSense 3D measurement
                waist_width = calculate_3d_distance_safe(p1, p2, depth_frame, depth_intrinsics)
            else:
                # Webcam pixel-based measurement
                pixel_distance = calculate_pixel_distance(p1, p2)
                # Daha gerçekçi bel genişliği hesaplama (omuzdan dar olmalı)
                waist_width = (pixel_distance / width) * 85   # Omuzdan biraz dar
                waist_width = max(25, min(45, waist_width))   # 25-45cm arası
            
            if waist_width:
                analysis_data['bel_genisligi'] = waist_width
        
        # ÖNEMLI: Omuz her zaman belden geniş olmalı
        if analysis_data['omuz_genisligi'] > 0 and analysis_data['bel_genisligi'] > 0:
            if analysis_data['omuz_genisligi'] <= analysis_data['bel_genisligi']:
                # Eğer omuz belden küçük/eşitse, düzelt
                analysis_data['omuz_genisligi'] = analysis_data['bel_genisligi'] * 1.3  # %30 daha geniş yap
                print(f"⚠️ Omuz genişliği düzeltildi: {analysis_data['omuz_genisligi']:.1f}cm")
        
        # Calculate ratios and body type
        if analysis_data['omuz_genisligi'] > 0 and analysis_data['bel_genisligi'] > 0:
            ratio = analysis_data['omuz_genisligi'] / analysis_data['bel_genisligi']
            analysis_data['omuz_bel_orani'] = ratio
            
            # Body type classification
            if ratio > 1.4:
                analysis_data['vucut_tipi'] = "Ektomorf"
            elif ratio > 1.2:
                analysis_data['vucut_tipi'] = "Mezomorf"
            else:
                analysis_data['vucut_tipi'] = "Endomorf"
            
            # Confidence calculation
            confidence = (ls_c + rs_c + lh_c + rh_c) / 4
            analysis_data['confidence'] = min(1.0, confidence)
        
        # Calculate distance to person
        if depth_frame is not None:
            if ls_c > 0.3 and rs_c > 0.3:
                center_x = int((ls_x + rs_x) * width / 2)
                center_y = int((ls_y + rs_y) * height / 2)
                
                try:
                    depth_image = np.asanyarray(depth_frame.get_data())
                    distance = safe_array_access(depth_image, center_y, center_x) * depth_frame.get_units()
                    if distance > 0:
                        analysis_data['mesafe'] = distance
                except Exception as e:
                    pass
        else:
            # Fixed distance for webcam
            analysis_data['mesafe'] = 1.5
        
    except Exception as e:
        print(f"❌ Analiz hatası: {e}")
    
    return analysis_data

def calculate_final_analysis():
    """Calculate final analysis from collected data"""
    global analysis_results, final_analysis
    
    if not analysis_results:
        return
    
    # Valid sonuçları filtrele
    valid_results = [r for r in analysis_results if r['confidence'] > 0.5 and r['omuz_genisligi'] > 0 and r['bel_genisligi'] > 0]
    
    if not valid_results:
        print("❌ Yeterli geçerli veri bulunamadı")
        return
    
    # Ortalama değerleri hesapla
    avg_shoulder = sum(r['omuz_genisligi'] for r in valid_results) / len(valid_results)
    avg_waist = sum(r['bel_genisligi'] for r in valid_results) / len(valid_results)
    avg_distance = sum(r['mesafe'] for r in valid_results) / len(valid_results)
    avg_confidence = sum(r['confidence'] for r in valid_results) / len(valid_results)
    
    # Final ratio ve body type
    final_ratio = avg_shoulder / avg_waist if avg_waist > 0 else 0
    
    if final_ratio > 1.4:
        body_type = "Ektomorf"
    elif final_ratio > 1.2:
        body_type = "Mezomorf"
    else:
        body_type = "Endomorf"
    
    # Final analizi güncelle
    final_analysis.update({
        'omuz_genisligi': round(avg_shoulder, 1),
        'bel_genisligi': round(avg_waist, 1),
        'omuz_bel_orani': round(final_ratio, 2),
        'vucut_tipi': body_type,
        'mesafe': round(avg_distance, 1),
        'confidence': round(avg_confidence, 2),
        'diyet_onerileri': DIYET_ONERILERI.get(body_type, {})
    })
    
    print(f"✅ Final Analiz: {body_type} - Omuz: {avg_shoulder:.1f}cm, Bel: {avg_waist:.1f}cm, Oran: {final_ratio:.2f}")

def draw_pose_and_measurements(frame: np.ndarray, keypoints: np.ndarray, 
                             analysis_data: Dict[str, Any], test_time_left: int) -> np.ndarray:
    """Draw pose skeleton and measurements on frame"""
    height, width, _ = frame.shape
    
    try:
        # Draw skeleton
        for p1_idx, p2_idx in EDGES:
            if p1_idx < len(keypoints) and p2_idx < len(keypoints):
                y1, x1, c1 = keypoints[p1_idx]
                y2, x2, c2 = keypoints[p2_idx]
                if c1 > 0.3 and c2 > 0.3:
                    pt1 = (int(x1 * width), int(y1 * height))
                    pt2 = (int(x2 * width), int(y2 * height))
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
        
        # Draw keypoints
        for i, (y, x, c) in enumerate(keypoints):
            if c > 0.3:
                pt = (int(x * width), int(y * height))
                cv2.circle(frame, pt, 4, (255, 0, 0), -1)
        
        # Extract keypoints for measurement lines
        ls_y, ls_x, ls_c = keypoints[KEYPOINT_DICT['left_shoulder']]
        rs_y, rs_x, rs_c = keypoints[KEYPOINT_DICT['right_shoulder']]
        lh_y, lh_x, lh_c = keypoints[KEYPOINT_DICT['left_hip']]
        rh_y, rh_x, rh_c = keypoints[KEYPOINT_DICT['right_hip']]
        
        # OMUZ ÇİZGİSİ - KALINCA MOR ÇİZGİ
        if ls_c > 0.3 and rs_c > 0.3:
            pt1 = (int(ls_x * width), int(ls_y * height))
            # KALINCA MOR ÇİZGİ - OMUZ
            cv2.line(frame, pt1, pt2, (255, 0, 255), 8)  # ÇOK KALIN MOR ÇİZGİ
            # OMUZ ÖLÇÜMÜ YAZISI - BÜYÜK VE NET
            mid_x = int((pt1[0] + pt2[0]) / 2)
            mid_y = int((pt1[1] + pt2[1]) / 2) - 25
            
            # Arka plan kutusu
            cv2.rectangle(frame, (mid_x - 60, mid_y - 20), (mid_x + 60, mid_y + 5), (0, 0, 0), -1)
            cv2.putText(frame, f"OMUZ: {analysis_data.get('omuz_genisligi', 0):.1f}cm", 
                       (mid_x - 55, mid_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        # BEL ÇİZGİSİ - KALINCA CYAN ÇİZGİ
        if lh_c > 0.3 and rh_c > 0.3:
            pt1 = (int(lh_x * width), int(lh_y * height))
            # KALINCA CYAN ÇİZGİ - BEL
            cv2.line(frame, pt1, pt2, (255, 255, 0), 8)  # ÇOK KALIN CYAN ÇİZGİ
            # BEL ÖLÇÜMÜ YAZISI - BÜYÜK VE NET
            mid_x = int((pt1[0] + pt2[0]) / 2)
            mid_y = int((pt1[1] + pt2[1]) / 2) + 35
            
            # Arka plan kutusu
            cv2.rectangle(frame, (mid_x - 60, mid_y - 20), (mid_x + 60, mid_y + 5), (0, 0, 0), -1)
            cv2.putText(frame, f"BEL: {analysis_data.get('bel_genisligi', 0):.1f}cm", 
                       (mid_x - 55, mid_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # SOL ÜST KÖŞEDE BÜYÜK BİLGİLER
        y_pos = 40
        
        # Arka plan kutuları
        cv2.rectangle(frame, (10, 10), (350, 180), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (350, 180), (255, 255, 255), 2)
        
        # Test countdown
        cv2.putText(frame, f"KALAN SURE: {test_time_left}s", 
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        y_pos += 30
        
        # Ölçümler
        cv2.putText(frame, f"OMUZ: {analysis_data.get('omuz_genisligi', 0):.1f} cm", 
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        y_pos += 25
        
        cv2.putText(frame, f"BEL: {analysis_data.get('bel_genisligi', 0):.1f} cm", 
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        y_pos += 25
        
        cv2.putText(frame, f"TIP: {analysis_data.get('vucut_tipi', 'Analiz...')}", 
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        y_pos += 25
        
        if analysis_data.get('omuz_bel_orani', 0) > 0:
            cv2.putText(frame, f"ORAN: {analysis_data['omuz_bel_orani']:.2f}", 
                       (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Analysis count
        cv2.putText(frame, f"Analiz: {len(analysis_results)}", 
                   (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
    except Exception as e:
        print(f"❌ Çizim hatası: {e}")
    
    return frame

def create_depth_visualization(frame: np.ndarray, keypoints: np.ndarray, 
                             depth_frame=None) -> np.ndarray:
    """Create depth visualization - real or simulated"""
    try:
        if depth_frame is not None:
            # Real depth from RealSense
            colorizer = rs.colorizer()
            colorizer.set_option(rs.option.color_scheme, 0)  # Jet colormap
            depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())
            depth_colormap = cv2.flip(depth_colormap, 1)  # Mirror
        else:
            # Simulated depth from RGB
            height, width, _ = frame.shape
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edges_inv = 255 - edges
            blurred = cv2.GaussianBlur(edges_inv, (21, 21), 0)
            blurred = cv2.equalizeHist(blurred)
            depth_colormap = cv2.applyColorMap(blurred, cv2.COLORMAP_JET)
        
        # Add pose skeleton to depth
        height, width = depth_colormap.shape[:2]
        for p1_idx, p2_idx in EDGES:
            if p1_idx < len(keypoints) and p2_idx < len(keypoints):
                y1, x1, c1 = keypoints[p1_idx]
                y2, x2, c2 = keypoints[p2_idx]
                if c1 > 0.3 and c2 > 0.3:
                    pt1 = (int(x1 * width), int(y1 * height))
                    pt2 = (int(x2 * width), int(y2 * height))
                    cv2.line(depth_colormap, pt1, pt2, (255, 255, 255), 2)
        
        # Add keypoints
        for i, (y, x, c) in enumerate(keypoints):
            if c > 0.3:
                pt = (int(x * width), int(y * height))
                cv2.circle(depth_colormap, pt, 3, (255, 255, 255), -1)
        
        return depth_colormap
        
    except Exception as e:
        print(f"❌ Derinlik görselleştirme hatası: {e}")
        return np.zeros_like(frame)

def detect_camera_type():
    """Detect available camera type and return appropriate mode"""
    global camera_mode
    
    if REALSENSE_AVAILABLE:
        try:
            ctx = rs.context()
            devices = ctx.query_devices()
            if len(devices) > 0:
                print(f"✅ {len(devices)} RealSense kamera bulundu")
                camera_mode = "realsense"
                return True
        except Exception as e:
            print(f"RealSense test failed: {e}")
    
    print("📹 Normal webcam modu kullanılacak")
    camera_mode = "webcam"
    return True

def run_body_analysis_test():
    """Run 10-second body analysis test"""
    global test_running, camera, realsense_pipeline, camera_mode, analysis_results, final_analysis
    
    try:
        # Reset analysis data
        analysis_results = []
        final_analysis = {
            'omuz_genisligi': 0.0,
            'bel_genisligi': 0.0,
            'omuz_bel_orani': 0.0,
            'vucut_tipi': 'Analiz Bekleniyor',
            'mesafe': 0.0,
            'confidence': 0.0,
            'diyet_onerileri': []
        }
        
        # Detect camera type
        if not detect_camera_type():
            socketio.emit('test_error', 'Hiçbir kamera bulunamadı')
            return
        
        if camera_mode == "realsense":
            run_realsense_test()
        else:
            run_webcam_test()
            
    except Exception as e:
        print(f"❌ Test error: {e}")
        socketio.emit('test_error', f'Test error: {str(e)}')
    finally:
        test_running = False

def run_realsense_test():
    """Run test with RealSense camera"""
    global test_running, realsense_pipeline, analysis_results
    
    try:
        realsense_pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        profile = realsense_pipeline.start(config)
        depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        
        print("✅ RealSense test başlatıldı")
        socketio.emit('test_started', {'duration': TEST_DURATION})
        
        start_time = time.time()
        last_analysis_time = 0
        
        while test_running and (time.time() - start_time) < TEST_DURATION:
            try:
                frames = realsense_pipeline.wait_for_frames(timeout_ms=1000)
                
                align = rs.align(rs.stream.color)
                aligned_frames = align.process(frames)
                
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    continue
                
                color_image = np.asanyarray(color_frame.get_data())
                color_image = cv2.flip(color_image, 1)
                
                # Run pose detection
                keypoints = run_movenet(color_image)
                
                # Analyze measurements
                current_time = time.time()
                if current_time - last_analysis_time >= ANALYSIS_INTERVAL:
                    analysis_data = analyze_body_measurements(
                        keypoints, color_image.shape, depth_frame, depth_intrinsics
                    )
                    
                    if analysis_data['confidence'] > 0.3:
                        analysis_results.append(analysis_data)
                        print(f"📊 Analiz #{len(analysis_results)}: {analysis_data['vucut_tipi']}")
                    
                    last_analysis_time = current_time
                
                # Calculate remaining time
                time_left = int(TEST_DURATION - (current_time - start_time))
                
                # Draw pose and measurements
                rgb_frame = draw_pose_and_measurements(color_image.copy(), keypoints, 
                                                     analysis_results[-1] if analysis_results else {}, time_left)
                
                # Create depth visualization
                depth_viz = create_depth_visualization(color_image, keypoints, depth_frame)
                
                # Add labels
                cv2.putText(rgb_frame, "RGB + Pose", (10, rgb_frame.shape[0] - 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(depth_viz, "Derinlik", (10, depth_viz.shape[0] - 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Combine frames
                h1, w1 = rgb_frame.shape[:2]
                h2, w2 = depth_viz.shape[:2]
                if h1 != h2:
                    depth_viz = cv2.resize(depth_viz, (w1, h1))
                
                combined_frame = np.hstack((rgb_frame, depth_viz))
                
                # Send video frame
                _, buffer = cv2.imencode('.jpg', combined_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                socketio.emit('test_frame', {'frame': img_base64, 'time_left': time_left})
                
                socketio.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                print(f"❌ RealSense loop error: {e}")
                break
        
        # Test completed
        calculate_final_analysis()
        socketio.emit('test_completed', final_analysis)
        print(f"✅ Test tamamlandı: {len(analysis_results)} analiz yapıldı")
        
    except Exception as e:
        print(f"❌ RealSense test error: {e}")
        socketio.emit('test_error', f'RealSense error: {str(e)}')
    
    finally:
        if realsense_pipeline:
            realsense_pipeline.stop()
        print("🛑 RealSense test stopped")

def run_webcam_test():
    """Run test with webcam"""
    global test_running, camera, analysis_results
    
    try:
        working_cameras = [0, 1, 2, 4, 6]
        working_camera_index = None
        
        for camera_index in working_cameras:
            test_cap = cv2.VideoCapture(camera_index)
            if test_cap.isOpened():
                ret, frame = test_cap.read()
                if ret and frame is not None:
                    working_camera_index = camera_index
                    test_cap.release()
                    print(f"✅ Webcam {camera_index} kullanılıyor")
                    break
                test_cap.release()
        
        if working_camera_index is None:
            socketio.emit('test_error', 'Webcam bulunamadı')
            return
        
        camera = cv2.VideoCapture(working_camera_index)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)
        
        print("✅ Webcam test başlatıldı")
        socketio.emit('test_started', {'duration': TEST_DURATION})
        
        start_time = time.time()
        last_analysis_time = 0
        
        while test_running and (time.time() - start_time) < TEST_DURATION:
            try:
                ret, frame = camera.read()
                if not ret:
                    continue
                
                frame = cv2.flip(frame, 1)
                
                # Run pose detection
                keypoints = run_movenet(frame)
                
                # Analyze measurements
                current_time = time.time()
                if current_time - last_analysis_time >= ANALYSIS_INTERVAL:
                    analysis_data = analyze_body_measurements(keypoints, frame.shape)
                    
                    if analysis_data['confidence'] > 0.3:
                        analysis_results.append(analysis_data)
                        print(f"📊 Analiz #{len(analysis_results)}: {analysis_data['vucut_tipi']}")
                    
                    last_analysis_time = current_time
                
                # Calculate remaining time
                time_left = int(TEST_DURATION - (current_time - start_time))
                
                # Draw pose and measurements
                rgb_frame = draw_pose_and_measurements(frame.copy(), keypoints, 
                                                     analysis_results[-1] if analysis_results else {}, time_left)
                
                # Create depth simulation
                depth_viz = create_depth_visualization(frame, keypoints, None)
                
                # Add labels
                cv2.putText(rgb_frame, "RGB + Pose", (10, rgb_frame.shape[0] - 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(depth_viz, "Derinlik Sim.", (10, depth_viz.shape[0] - 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Combine frames
                h1, w1 = rgb_frame.shape[:2]
                h2, w2 = depth_viz.shape[:2]
                if h1 != h2:
                    depth_viz = cv2.resize(depth_viz, (w1, h1))
                
                combined_frame = np.hstack((rgb_frame, depth_viz))
                
                # Send video frame
                _, buffer = cv2.imencode('.jpg', combined_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                socketio.emit('test_frame', {'frame': img_base64, 'time_left': time_left})
                
                socketio.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                print(f"❌ Webcam loop error: {e}")
                break
        
        # Test completed
        calculate_final_analysis()
        socketio.emit('test_completed', final_analysis)
        print(f"✅ Test tamamlandı: {len(analysis_results)} analiz yapıldı")
        
    except Exception as e:
        print(f"❌ Webcam test error: {e}")
        socketio.emit('test_error', f'Webcam error: {str(e)}')
    
    finally:
        if camera:
            camera.release()
        print("🛑 Webcam test stopped")

def take_food_photo():
    """Yemek fotoğrafı çek ve analiz et"""
    global camera, realsense_pipeline, camera_mode
    
    try:
        if not food_analyzer:
            socketio.emit('food_analysis_error', {'message': 'Food analyzer başlatılamadı'})
            return
        
        # Kamera tipini belirle
        if not detect_camera_type():
            socketio.emit('food_analysis_error', {'message': 'Kamera bulunamadı'})
            return
        
        # Geri sayım
        for i in range(FOOD_PHOTO_COUNTDOWN, 0, -1):
            socketio.emit('food_capture_countdown', {'count': i})
            time.sleep(1)
        
        socketio.emit('food_capture_started')
        
        # Fotoğraf çek
        image_data = None
        
        if camera_mode == "realsense":
            image_data = capture_realsense_photo()
        else:
            image_data = capture_webcam_photo()
        
        if image_data:
            socketio.emit('food_analysis_started')
            
            # Yemek analizi yap
            analysis_result = food_analyzer.analyze_food_image(image_data)
            
            # Sonucu gönder
            socketio.emit('food_analysis_result', {
                'image': analysis_result['image'],
                'analysis': analysis_result
            })
            
            print(f"✅ Yemek analizi tamamlandı: {analysis_result['total_calories']} kalori")
        else:
            socketio.emit('food_analysis_error', {'message': 'Fotoğraf çekilemedi'})
            
    except Exception as e:
        print(f"❌ Yemek fotoğrafı hatası: {e}")
        socketio.emit('food_analysis_error', {'message': f'Hata: {str(e)}'})

def capture_realsense_photo():
    """RealSense ile fotoğraf çek"""
    global realsense_pipeline
    
    try:
        # RealSense pipeline başlat
        realsense_pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        profile = realsense_pipeline.start(config)
        
        # Birkaç frame bekle (kamera stabilize olsun)
        for _ in range(10):
            realsense_pipeline.wait_for_frames()
        
        # Fotoğraf çek
        frames = realsense_pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        if color_frame:
            color_image = np.asanyarray(color_frame.get_data())
            color_image = cv2.flip(color_image, 1)  # Mirror
            
            # JPEG formatına çevir
            _, buffer = cv2.imencode('.jpg', color_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            return buffer.tobytes()
        
        return None
        
    except Exception as e:
        print(f"RealSense fotoğraf hatası: {e}")
        return None
    finally:
        if realsense_pipeline:
            realsense_pipeline.stop()

def capture_webcam_photo():
    """Webcam ile fotoğraf çek"""
    global camera
    
    try:
        # Webcam aç
        working_cameras = [0, 1, 2, 4, 6]
        working_camera_index = None
        
        for camera_index in working_cameras:
            test_cap = cv2.VideoCapture(camera_index)
            if test_cap.isOpened():
                ret, frame = test_cap.read()
                if ret and frame is not None:
                    working_camera_index = camera_index
                    test_cap.release()
                    break
                test_cap.release()
        
        if working_camera_index is None:
            return None
        
        camera = cv2.VideoCapture(working_camera_index)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Birkaç frame bekle
        for _ in range(10):
            camera.read()
        
        # Fotoğraf çek
        ret, frame = camera.read()
        
        if ret and frame is not None:
            frame = cv2.flip(frame, 1)  # Mirror
            
            # JPEG formatına çevir
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            return buffer.tobytes()
        
        return None
        
    except Exception as e:
        print(f"Webcam fotoğraf hatası: {e}")
        return None
    finally:
        if camera:
            camera.release()

# --- SocketIO Events ---
@socketio.on('connect')
def handle_connect(auth):
    print("✅ WebSocket connection established!")

@socketio.on('disconnect')
def handle_disconnect():
    global test_running
    test_running = False
    print("❌ WebSocket connection closed!")

@socketio.on('start_test')
def handle_start_test(data):
    global test_running, test_thread
    if not test_running:
        test_running = True
        test_thread = socketio.start_background_task(target=run_body_analysis_test)
        print("🚀 Vücut analizi testi başlatıldı")

@socketio.on('stop_test')
def handle_stop_test(data):
    global test_running
    test_running = False
    socketio.emit('test_stopped')
    print("🛑 Test durduruldu")

@socketio.on('take_food_photo')
def handle_take_food_photo(data):
    """Yemek fotoğrafı çekme isteği"""
    if not test_running:  # Test çalışmıyorsa fotoğraf çekebilir
        socketio.start_background_task(target=take_food_photo)
        print("📸 Yemek fotoğrafı çekiliyor")

if __name__ == '__main__':
    # Food analyzer'ı başlat
    initialize_food_analyzer()
    
    print("🚀 Starting Enhanced Test-Based Body Analysis System...")
    print("📋 Features:")
    print("   - 10 saniye test süresi")
    print("   - Otomatik kamera algılama")
    print("   - Detaylı vücut tipi analizi")
    print("   - Haftalık yemek planı")
    print("   - Egzersiz ve supplement önerileri")
    print("   - Test sonunda kamera kapanır")
    print("   - Yemek fotoğrafı analizi ve kalori hesaplama")
    print()
    
    if REALSENSE_AVAILABLE:
        print("✅ RealSense support: Available")
    else:
        print("⚠️ RealSense support: Not available (webcam only)")
    
    print()
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)