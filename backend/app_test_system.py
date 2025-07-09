#!/usr/bin/env python3
"""
Test Tabanlı Vücut Analizi Sistemi - Timeout Sorunları Düzeltildi
- WebSocket bağlantı sorunları çözüldü
- RealSense kamera timeout yönetimi iyileştirildi
- Otomatik yeniden bağlanma sistemi
- Heartbeat ve connection monitoring
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

# WebSocket ayarları - timeout sorunları için optimize edildi
socketio = SocketIO(app, 
                   cors_allowed_origins="*", 
                   async_mode='eventlet',
                   ping_timeout=120,  # 2 dakika timeout
                   ping_interval=30,  # 30 saniyede bir ping
                   logger=False, 
                   engineio_logger=False,
                   transports=['websocket', 'polling'],
                   allow_upgrades=True)

# --- Global Variables ---
test_running = False
test_thread = None
camera = None
realsense_pipeline = None
camera_mode = "webcam"
connected_clients = set()
heartbeat_thread = None

# Test parametreleri
TEST_DURATION = 10  # 10 saniye test süresi
ANALYSIS_INTERVAL = 0.5  # Yarım saniyede bir analiz

# Analiz verileri toplama
analysis_results = []
current_analysis = {
    'omuz_genisligi': 0.0,
    'bel_genisligi': 0.0,
    'omuz_bel_orani': 0.0,
    'vucut_tipi': 'Analiz Bekleniyor',
    'mesafe': 0.0,
    'confidence': 0.0
}

final_analysis = {
    'omuz_genisligi': 0.0,
    'bel_genisligi': 0.0,
    'omuz_bel_orani': 0.0,
    'vucut_tipi': 'Analiz Bekleniyor',
    'mesafe': 0.0,
    'confidence': 0.0,
    'diyet_onerileri': []
}

# --- Model Loading ---
print("🤖 Loading MoveNet model from TensorFlow Hub...")
model = None
movenet = None

def load_movenet_model():
    """Load MoveNet model with retry mechanism"""
    global model, movenet
    
    max_retries = 3
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            print(f"📥 Model yükleme denemesi {attempt + 1}/{max_retries}...")
            
            # Timeout ile model yükleme
            import socket
            socket.setdefaulttimeout(60)  # 60 saniye timeout
            
            model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
            movenet = model.signatures['serving_default']
            print("✅ MoveNet model loaded successfully.")
            return True
            
        except Exception as e:
            print(f"❌ Deneme {attempt + 1} başarısız: {e}")
            if attempt < max_retries - 1:
                print(f"⏳ {retry_delay} saniye bekleyip tekrar denenecek...")
                time.sleep(retry_delay)
            else:
                print("❌ Model yüklenemedi. Lütfen internet bağlantınızı kontrol edin.")
                return False
    
    return False

# Model yüklemeyi dene
if not load_movenet_model():
    print("🛑 Sistem model olmadan çalışamaz. Çıkılıyor...")
    exit(1)

INPUT_SIZE = 192

# --- Diyet Önerileri Veritabanı ---
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
            'pazartesi': {
                'kahvalti': 'Yulaf ezmesi + muz + fındık + süt + bal',
                'ara_ogun_1': 'Tam tahıl kraker + peynir + ceviz',
                'ogle': 'Tavuk + pirinç + salata + zeytinyağı + avokado',
                'ara_ogun_2': 'Protein smoothie + meyve + yoğurt',
                'aksam': 'Balık + bulgur pilavı + sebze + zeytinyağı',
                'gece': 'Yoğurt + bal + ceviz + hurma'
            },
            'sali': {
                'kahvalti': 'Omlet (3 yumurta) + tam tahıl ekmek + domates + peynir',
                'ara_ogun_1': 'Muz + badem + süt',
                'ogle': 'Dana eti + makarna + salata + parmesan',
                'ara_ogun_2': 'Protein bar + elma',
                'aksam': 'Somon + quinoa + buharda sebze',
                'gece': 'Süt + tarçın + bal + fındık'
            },
            'carsamba': {
                'kahvalti': 'Müsli + yoğurt + meyve + fındık',
                'ara_ogun_1': 'Tam tahıl sandviç + hindi + peynir',
                'ogle': 'Köfte + bulgur + cacık + salata',
                'ara_ogun_2': 'Smoothie (muz + protein + süt)',
                'aksam': 'Tavuk + tatlı patates + yeşil fasulye',
                'gece': 'Yoğurt + granola + bal'
            },
            'persembe': {
                'kahvalti': 'Pancake (yulaf unu) + meyve + akçaağaç şurubu',
                'ara_ogun_1': 'Kuruyemiş karışımı + kuru meyve',
                'ogle': 'Balık + pirinç + sebze sote',
                'ara_ogun_2': 'Yoğurt + meyve + granola',
                'aksam': 'Tavuk + makarna + brokoli',
                'gece': 'Süt + bisküvi + fındık ezmesi'
            },
            'cuma': {
                'kahvalti': 'Menemen + peynir + tam tahıl ekmek',
                'ara_ogun_1': 'Protein shake + muz',
                'ogle': 'Tavuk döner + bulgur + salata',
                'ara_ogun_2': 'Elma + fındık ezmesi',
                'aksam': 'Balık + pirinç + karışık sebze',
                'gece': 'Yoğurt + bal + ceviz'
            },
            'cumartesi': {
                'kahvalti': 'French toast + meyve + süt',
                'ara_ogun_1': 'Smoothie bowl + granola',
                'ogle': 'Köri tavuk + pirinç + naan',
                'ara_ogun_2': 'Protein bar + kuruyemiş',
                'aksam': 'Biftek + patates + salata',
                'gece': 'Süt + tarçın + bal'
            },
            'pazar': {
                'kahvalti': 'Kahvaltı tabağı (yumurta + peynir + zeytin + ekmek)',
                'ara_ogun_1': 'Meyve salatası + yoğurt',
                'ogle': 'Kuzu eti + bulgur + sebze',
                'ara_ogun_2': 'Protein smoothie + hurma',
                'aksam': 'Balık + quinoa + asparagus',
                'gece': 'Yoğurt + granola + meyve'
            }
        }
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
            'pazartesi': {
                'kahvalti': 'Omlet + tam tahıl ekmek + domates + zeytinyağı',
                'ara_ogun_1': 'Elma + badem + yoğurt',
                'ogle': 'Izgara tavuk + quinoa + yeşil salata + zeytinyağı',
                'ara_ogun_2': 'Yoğurt + meyve + ceviz',
                'aksam': 'Balık + tatlı patates + buharda sebze',
                'gece': 'Az yağlı süt + tarçın + bal'
            },
            'sali': {
                'kahvalti': 'Yulaf ezmesi + meyve + fındık + süt',
                'ara_ogun_1': 'Tam tahıl kraker + peynir',
                'ogle': 'Dana eti + bulgur + salata',
                'ara_ogun_2': 'Smoothie (meyve + yoğurt)',
                'aksam': 'Tavuk + pirinç + sebze sote',
                'gece': 'Yoğurt + bal + ceviz'
            },
            'carsamba': {
                'kahvalti': 'Peynirli omlet + tam tahıl ekmek + salatalık',
                'ara_ogun_1': 'Muz + fındık ezmesi',
                'ogle': 'Balık + quinoa + yeşil fasulye',
                'ara_ogun_2': 'Yoğurt + granola',
                'aksam': 'Tavuk + bulgur + karışık salata',
                'gece': 'Süt + tarçın'
            },
            'persembe': {
                'kahvalti': 'Müsli + yoğurt + meyve',
                'ara_ogun_1': 'Elma + badem',
                'ogle': 'Köfte + pirinç + cacık',
                'ara_ogun_2': 'Protein smoothie',
                'aksam': 'Somon + tatlı patates + brokoli',
                'gece': 'Yoğurt + bal'
            },
            'cuma': {
                'kahvalti': 'Menemen + peynir + ekmek',
                'ara_ogun_1': 'Kuruyemiş karışımı',
                'ogle': 'Tavuk + makarna + salata',
                'ara_ogun_2': 'Yoğurt + meyve',
                'aksam': 'Balık + bulgur + sebze',
                'gece': 'Süt + bisküvi'
            },
            'cumartesi': {
                'kahvalti': 'Pancake + meyve + bal',
                'ara_ogun_1': 'Smoothie bowl',
                'ogle': 'Izgara et + quinoa + salata',
                'ara_ogun_2': 'Yoğurt + granola',
                'aksam': 'Tavuk + pirinç + sebze',
                'gece': 'Süt + tarçın + bal'
            },
            'pazar': {
                'kahvalti': 'Kahvaltı tabağı (dengeli)',
                'ara_ogun_1': 'Meyve + yoğurt',
                'ogle': 'Balık + bulgur + salata',
                'ara_ogun_2': 'Fındık + kuru meyve',
                'aksam': 'Tavuk + quinoa + sebze',
                'gece': 'Yoğurt + bal + ceviz'
            }
        }
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
            'pazartesi': {
                'kahvalti': 'Protein omlet + sebze + az zeytinyağı + yeşil çay',
                'ara_ogun_1': 'Çiğ badem (10-15 adet) + yeşil elma',
                'ogle': 'Izgara balık + bol salata + limon + zeytinyağı',
                'ara_ogun_2': 'Yoğurt (şekersiz) + tarçın + ceviz',
                'aksam': 'Tavuk + buharda brokoli + bulgur (az)',
                'gece': 'Bitki çayı + badem (5-6 adet)'
            },
            'sali': {
                'kahvalti': 'Sebzeli omlet + domates + salatalık',
                'ara_ogun_1': 'Yoğurt (şekersiz) + çilek',
                'ogle': 'Tavuk salatası + yeşil yapraklar + zeytinyağı',
                'ara_ogun_2': 'Fındık (10 adet) + yeşil çay',
                'aksam': 'Balık + karnabahar + az bulgur',
                'gece': 'Bitki çayı'
            },
            'carsamba': {
                'kahvalti': 'Protein shake + sebze + avokado',
                'ara_ogun_1': 'Elma + badem ezmesi (az)',
                'ogle': 'Dana eti + bol salata + limon',
                'ara_ogun_2': 'Yoğurt + yaban mersini',
                'aksam': 'Tavuk + buharda sebze + quinoa (az)',
                'gece': 'Yeşil çay + ceviz (3-4 adet)'
            },
            'persembe': {
                'kahvalti': 'Omlet + ıspanak + mantar',
                'ara_ogun_1': 'Yoğurt + tarçın',
                'ogle': 'Balık + yeşil salata + avokado',
                'ara_ogun_2': 'Badem (8-10 adet) + çay',
                'aksam': 'Tavuk + brokoli + tatlı patates (az)',
                'gece': 'Bitki çayı'
            },
            'cuma': {
                'kahvalti': 'Protein omlet + sebze karışımı',
                'ara_ogun_1': 'Çilek + yoğurt (şekersiz)',
                'ogle': 'Izgara tavuk + bol yeşillik + zeytinyağı',
                'ara_ogun_2': 'Fındık + yeşil çay',
                'aksam': 'Balık + asparagus + bulgur (az)',
                'gece': 'Bitki çayı + badem (5 adet)'
            },
            'cumartesi': {
                'kahvalti': 'Sebzeli scrambled egg + domates',
                'ara_ogun_1': 'Yoğurt + yaban mersini',
                'ogle': 'Balık salatası + yeşil yapraklar',
                'ara_ogun_2': 'Elma + badem (8 adet)',
                'aksam': 'Tavuk + karışık sebze + quinoa (az)',
                'gece': 'Yeşil çay'
            },
            'pazar': {
                'kahvalti': 'Protein omlet + avokado + domates',
                'ara_ogun_1': 'Yoğurt + tarçın + ceviz (3 adet)',
                'ogle': 'Izgara et + büyük salata + limon',
                'ara_ogun_2': 'Çilek + badem (6 adet)',
                'aksam': 'Balık + buharda sebze + bulgur (az)',
                'gece': 'Bitki çayı'
            }
        }
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
        print("❌ Model yüklenmemiş!")
        return np.zeros((17, 3))
        
    img_resized = tf.image.resize_with_pad(np.expand_dims(input_image, axis=0), INPUT_SIZE, INPUT_SIZE)
    input_tensor = tf.cast(img_resized, dtype=tf.int32)
    
    try:
        outputs = movenet(input_tensor)
        return outputs['output_0'].numpy()[0, 0]
    except Exception as e:
        print(f"❌ Model çalıştırma hatası: {e}")
        return np.zeros((17, 3))

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
        
        # Daha geniş alan ortalaması al (5x5 piksel)
        depth1_values = []
        depth2_values = []
        
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                d1 = safe_array_access(depth_image, p1[1] + dy, p1[0] + dx) * depth_units
                d2 = safe_array_access(depth_image, p2[1] + dy, p2[0] + dx) * depth_units
                if d1 > 0: depth1_values.append(d1)
                if d2 > 0: depth2_values.append(d2)
        
        if not depth1_values or not depth2_values:
            return None
            
        depth1 = np.median(depth1_values)  # Median daha kararlı
        depth2 = np.median(depth2_values)
        
        if depth1 <= 0.3 or depth2 <= 0.3 or depth1 > 3.0 or depth2 > 3.0:
            return None
            
        # Derinlik farkı çok fazlaysa güvenilmez
        if abs(depth1 - depth2) > 0.5:
            return None
            
        point1_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [p1[0], p1[1]], depth1)
        point2_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [p2[0], p2[1]], depth2)
        
        distance = np.linalg.norm(np.subtract(point1_3d, point2_3d))
        distance_cm = distance * 100
        
        # Daha gerçekçi sınırlar
        if distance_cm < 15 or distance_cm > 80:
            return None
            
        return distance_cm
        
    except Exception as e:
        return None

def analyze_body_measurements(keypoints: np.ndarray, frame_shape: Tuple[int, int], 
                            depth_frame=None, depth_intrinsics=None) -> Dict[str, Any]:
    """Comprehensive body measurement analysis"""
    global current_analysis
    
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
                
                # 3D başarısız olursa 2D'ye geç
                if shoulder_width is None:
                    pixel_distance = calculate_pixel_distance(p1, p2)
                    # Mesafeye göre kalibre et
                    if analysis_data.get('mesafe', 0) > 0:
                        distance_factor = analysis_data['mesafe']
                        shoulder_width = (pixel_distance / width) * (45 * distance_factor)
                    else:
                        shoulder_width = (pixel_distance / width) * 90
                    shoulder_width = max(25, min(75, shoulder_width))
            else:
                # Webcam pixel-based measurement
                pixel_distance = calculate_pixel_distance(p1, p2)
                shoulder_width = (pixel_distance / width) * 90
                shoulder_width = max(25, min(75, shoulder_width))
            
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
                
                # 3D başarısız olursa 2D'ye geç
                if waist_width is None:
                    pixel_distance = calculate_pixel_distance(p1, p2)
                    if analysis_data.get('mesafe', 0) > 0:
                        distance_factor = analysis_data['mesafe']
                        waist_width = (pixel_distance / width) * (35 * distance_factor)
                    else:
                        waist_width = (pixel_distance / width) * 70
                    waist_width = max(20, min(55, waist_width))
            else:
                # Webcam pixel-based measurement
                pixel_distance = calculate_pixel_distance(p1, p2)
                waist_width = (pixel_distance / width) * 70
                waist_width = max(20, min(55, waist_width))
            
            if waist_width:
                analysis_data['bel_genisligi'] = waist_width
        
        # Calculate ratios and body type - HER ZAMAN HESAPLA
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
        
        # Update current analysis
        current_analysis = analysis_data
        
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
        # Draw skeleton - HER ZAMAN ÇİZ
        for p1_idx, p2_idx in EDGES:
            if p1_idx < len(keypoints) and p2_idx < len(keypoints):
                y1, x1, c1 = keypoints[p1_idx]
                y2, x2, c2 = keypoints[p2_idx]
                if c1 > 0.3 and c2 > 0.3:
                    pt1 = (int(x1 * width), int(y1 * height))
                    pt2 = (int(x2 * width), int(y2 * height))
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
        
        # Draw keypoints - HER ZAMAN ÇİZ
        for i, (y, x, c) in enumerate(keypoints):
            if c > 0.3:
                pt = (int(x * width), int(y * height))
                cv2.circle(frame, pt, 4, (255, 0, 0), -1)
        
        # Extract keypoints for measurement lines
        ls_y, ls_x, ls_c = keypoints[KEYPOINT_DICT['left_shoulder']]
        rs_y, rs_x, rs_c = keypoints[KEYPOINT_DICT['right_shoulder']]
        lh_y, lh_x, lh_c = keypoints[KEYPOINT_DICT['left_hip']]
        rh_y, rh_x, rh_c = keypoints[KEYPOINT_DICT['right_hip']]
        
        # Shoulder measurement line - DAIMA ÇİZ (confidence yeterli ise)
        if ls_c > 0.3 and rs_c > 0.3:
            pt1 = (int(ls_x * width), int(ls_y * height))
            pt2 = (int(rs_x * width), int(rs_y * height))
            
            # MOR ÇİZGİ - HER ZAMAN GÖRÜNÜR OLSUN
            cv2.line(frame, pt1, pt2, (255, 0, 255), 4)  # Kalın mor çizgi
            
            # Measurement text
            if analysis_data.get('omuz_genisligi', 0) > 0:
                mid_x = int((pt1[0] + pt2[0]) / 2)
                mid_y = int((pt1[1] + pt2[1]) / 2) - 15
                cv2.putText(frame, f"{analysis_data['omuz_genisligi']:.1f}cm", 
                           (mid_x - 40, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        # Hip measurement line - DAIMA ÇİZ (confidence yeterli ise)
        if lh_c > 0.3 and rh_c > 0.3:
            pt1 = (int(lh_x * width), int(lh_y * height))
            pt2 = (int(rh_x * width), int(rh_y * height))
            
            # MAVİ ÇİZGİ - HER ZAMAN GÖRÜNÜR OLSUN
            cv2.line(frame, pt1, pt2, (255, 255, 0), 4)  # Kalın cyan çizgi
            
            # Measurement text
            if analysis_data.get('bel_genisligi', 0) > 0:
                mid_x = int((pt1[0] + pt2[0]) / 2)
                mid_y = int((pt1[1] + pt2[1]) / 2) + 25
                cv2.putText(frame, f"{analysis_data['bel_genisligi']:.1f}cm", 
                           (mid_x - 40, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Add measurement text overlay - BÜYÜK VE NET
        y_offset = 30
        cv2.putText(frame, f"Omuz: {analysis_data.get('omuz_genisligi', 0):.1f}cm", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        y_offset += 35
        cv2.putText(frame, f"Bel: {analysis_data.get('bel_genisligi', 0):.1f}cm", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        y_offset += 35
        cv2.putText(frame, f"Tip: {analysis_data.get('vucut_tipi', 'Analiz Bekleniyor')}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        if analysis_data.get('omuz_bel_orani', 0) > 0:
            y_offset += 35
            cv2.putText(frame, f"Oran: {analysis_data['omuz_bel_orani']:.2f}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        if analysis_data.get('mesafe', 0) > 0:
            y_offset += 35
            cv2.putText(frame, f"Mesafe: {analysis_data['mesafe']:.1f}m", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Test countdown
        cv2.putText(frame, f"Test Suresi: {test_time_left}s", 
                   (10, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        
        # Analysis count
        cv2.putText(frame, f"Analiz Sayisi: {len(analysis_results)}", 
                   (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
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
            safe_emit('test_error', 'Hiçbir kamera bulunamadı')
            return
        
        if camera_mode == "realsense":
            run_realsense_test()
        else:
            run_webcam_test()
            
    except Exception as e:
        print(f"❌ Test error: {e}")
        safe_emit('test_error', f'Test error: {str(e)}')
    finally:
        test_running = False

def safe_emit(event, data):
    """Safely emit WebSocket message with error handling"""
    try:
        if len(connected_clients) > 0:
            socketio.emit(event, data)
    except Exception as e:
        print(f"❌ Emit hatası ({event}): {e}")

def run_realsense_test():
    """Run test with RealSense camera - improved timeout handling"""
    global test_running, realsense_pipeline, analysis_results
    
    try:
        realsense_pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        profile = realsense_pipeline.start(config)
        
        # Depth sensor ayarları
        depth_sensor = profile.get_device().first_depth_sensor()
        try:
            # Daha iyi derinlik için ayarlar
            depth_sensor.set_option(rs.option.visual_preset, 3)  # High Accuracy
            depth_sensor.set_option(rs.option.laser_power, 300)  # Yüksek laser gücü
            depth_sensor.set_option(rs.option.confidence_threshold, 1)
            depth_sensor.set_option(rs.option.min_distance, 0)
            depth_sensor.set_option(rs.option.enable_auto_exposure, 1)
            print("✅ RealSense depth sensor optimize edildi")
        except Exception as e:
            print(f"⚠️ Depth sensor ayarları uygulanamadı: {e}")
        
        depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        
        print("✅ RealSense test başlatıldı")
        safe_emit('test_started', {'duration': TEST_DURATION})
        
        start_time = time.time()
        last_analysis_time = 0
        frame_timeout_count = 0
        max_timeout_count = 10  # 10 timeout sonrası çık
        
        while test_running and (time.time() - start_time) < TEST_DURATION:
            try:
                # Daha kısa timeout ile frame bekle
                frames = realsense_pipeline.wait_for_frames(timeout_ms=1000)
                frame_timeout_count = 0  # Reset timeout counter
                
                align = rs.align(rs.stream.color)
                aligned_frames = align.process(frames)
                
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    continue
                
                # Depth filtering
                try:
                    depth_frame = rs.decimation_filter(2).process(depth_frame)
                    depth_frame = rs.spatial_filter().process(depth_frame)
                    depth_frame = rs.temporal_filter().process(depth_frame)
                    depth_frame = rs.hole_filling_filter().process(depth_frame)
                except Exception as filter_error:
                    print(f"⚠️ Depth filtering error: {filter_error}")
                
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
                    
                    # Daha düşük confidence threshold
                    if analysis_data['confidence'] > 0.2:
                        analysis_results.append(analysis_data)
                        print(f"📊 Analiz #{len(analysis_results)}: {analysis_data['vucut_tipi']}")
                    
                    last_analysis_time = current_time
                
                # Calculate remaining time
                time_left = int(TEST_DURATION - (current_time - start_time))
                
                # Draw pose and measurements
                rgb_frame = draw_pose_and_measurements(color_image.copy(), keypoints, 
                                                     current_analysis, time_left)
                
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
                try:
                    _, buffer = cv2.imencode('.jpg', combined_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    img_base64 = base64.b64encode(buffer).decode('utf-8')
                    safe_emit('test_frame', {'frame': img_base64, 'time_left': time_left})
                except Exception as emit_error:
                    print(f"⚠️ Frame gönderme hatası: {emit_error}")
                
                socketio.sleep(0.033)  # ~30 FPS
                
            except RuntimeError as timeout_error:
                frame_timeout_count += 1
                print(f"⚠️ RealSense frame timeout #{frame_timeout_count}: {timeout_error}")
                
                if frame_timeout_count >= max_timeout_count:
                    print("❌ Çok fazla timeout, test durduruluyor")
                    break
                    
                socketio.sleep(0.1)
                continue
                
            except Exception as e:
                print(f"❌ RealSense loop error: {e}")
                socketio.sleep(0.1)
                continue
        
        # Test completed
        calculate_final_analysis()
        safe_emit('test_completed', final_analysis)
        print(f"✅ Test tamamlandı: {len(analysis_results)} analiz yapıldı")
        
    except Exception as e:
        print(f"❌ RealSense test error: {e}")
        safe_emit('test_error', f'RealSense error: {str(e)}')
    
    finally:
        if realsense_pipeline:
            try:
                realsense_pipeline.stop()
            except:
                pass
        print("🛑 RealSense test stopped")

def run_webcam_test():
    """Run test with webcam - improved timeout handling"""
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
            safe_emit('test_error', 'Webcam bulunamadı')
            return
        
        camera = cv2.VideoCapture(working_camera_index)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)
        
        print("✅ Webcam test başlatıldı")
        safe_emit('test_started', {'duration': TEST_DURATION})
        
        start_time = time.time()
        last_analysis_time = 0
        failed_frame_count = 0
        max_failed_frames = 30  # 30 başarısız frame sonrası çık
        
        while test_running and (time.time() - start_time) < TEST_DURATION:
            try:
                ret, frame = camera.read()
                if not ret:
                    failed_frame_count += 1
                    if failed_frame_count >= max_failed_frames:
                        print("❌ Çok fazla başarısız frame, test durduruluyor")
                        break
                    continue
                
                failed_frame_count = 0  # Reset failed frame counter
                frame = cv2.flip(frame, 1)
                
                # Run pose detection
                keypoints = run_movenet(frame)
                
                # Analyze measurements
                current_time = time.time()
                if current_time - last_analysis_time >= ANALYSIS_INTERVAL:
                    analysis_data = analyze_body_measurements(keypoints, frame.shape)
                    
                    if analysis_data['confidence'] > 0.2:
                        analysis_results.append(analysis_data)
                        print(f"📊 Analiz #{len(analysis_results)}: {analysis_data['vucut_tipi']}")
                    
                    last_analysis_time = current_time
                
                # Calculate remaining time
                time_left = int(TEST_DURATION - (current_time - start_time))
                
                # Draw pose and measurements
                rgb_frame = draw_pose_and_measurements(frame.copy(), keypoints, 
                                                     current_analysis, time_left)
                
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
                try:
                    _, buffer = cv2.imencode('.jpg', combined_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    img_base64 = base64.b64encode(buffer).decode('utf-8')
                    safe_emit('test_frame', {'frame': img_base64, 'time_left': time_left})
                except Exception as emit_error:
                    print(f"⚠️ Frame gönderme hatası: {emit_error}")
                
                socketio.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                print(f"❌ Webcam loop error: {e}")
                socketio.sleep(0.1)
                continue
        
        # Test completed
        calculate_final_analysis()
        safe_emit('test_completed', final_analysis)
        print(f"✅ Test tamamlandı: {len(analysis_results)} analiz yapıldı")
        
    except Exception as e:
        print(f"❌ Webcam test error: {e}")
        safe_emit('test_error', f'Webcam error: {str(e)}')
    
    finally:
        if camera:
            camera.release()
        print("🛑 Webcam test stopped")

def heartbeat_monitor():
    """Background heartbeat to keep connections alive"""
    while True:
        try:
            if len(connected_clients) > 0:
                safe_emit('heartbeat', {'timestamp': time.time()})
            socketio.sleep(30)  # Her 30 saniyede bir heartbeat
        except Exception as e:
            print(f"❌ Heartbeat hatası: {e}")
            socketio.sleep(5)

# --- SocketIO Events ---
@socketio.on('connect')
def handle_connect(auth):
    global connected_clients, heartbeat_thread
    
    client_id = request.sid if 'request' in globals() else 'unknown'
    connected_clients.add(client_id)
    print(f"✅ WebSocket connection established! Client: {client_id}")
    
    # Start heartbeat thread if not running
    if heartbeat_thread is None or not heartbeat_thread.is_alive():
        heartbeat_thread = socketio.start_background_task(target=heartbeat_monitor)
    
    # Send connection confirmation
    safe_emit('connection_ok', {'status': 'connected', 'timestamp': time.time()})

@socketio.on('disconnect')
def handle_disconnect():
    global test_running, connected_clients
    
    client_id = request.sid if 'request' in globals() else 'unknown'
    connected_clients.discard(client_id)
    
    # Stop test if no clients connected
    if len(connected_clients) == 0:
        test_running = False
    
    print(f"❌ WebSocket connection closed! Client: {client_id}, Remaining: {len(connected_clients)}")

@socketio.on('start_test')
def handle_start_test(data):
    global test_running, test_thread
    try:
        if not test_running:
            test_running = True
            test_thread = socketio.start_background_task(target=run_body_analysis_test)
            safe_emit('stream_started', {'type': 'stream_started'})
            print("🚀 Vücut analizi testi başlatıldı")
        else:
            print("⚠️ Test zaten çalışıyor")
    except Exception as e:
        print(f"❌ Test başlatma hatası: {e}")
        safe_emit('test_error', f'Test başlatma hatası: {str(e)}')

@socketio.on('stop_test')
def handle_stop_test(data):
    global test_running
    try:
        test_running = False
        safe_emit('test_stopped')
        print("🛑 Test durduruldu")
    except Exception as e:
        print(f"❌ Test durdurma hatası: {e}")

# Heartbeat sistemi
@socketio.on('ping')
def handle_ping(data):
    try:
        safe_emit('pong', {'timestamp': time.time()})
    except Exception as e:
        print(f"❌ Ping hatası: {e}")

@socketio.on('check_connection')
def handle_check_connection(data):
    try:
        safe_emit('connection_ok', {'status': 'ok', 'timestamp': time.time()})
    except Exception as e:
        print(f"❌ Connection check hatası: {e}")

if __name__ == '__main__':
    print("🚀 Starting Test-Based Body Analysis System...")
    print("📋 Features:")
    print("   - 10 saniye test süresi")
    print("   - Otomatik kamera algılama")
    print("   - Vücut tipi analizi")
    print("   - Sol ekranda ölçüm verileri")
    print("   - Sağ tarafta detaylı sonuçlar")
    print("   - Kişiselleştirilmiş diyet önerileri")
    print("   - Test sonunda kamera kapanır")
    print("   - Gelişmiş omuz algılama")
    print("   - Kararlı WebSocket bağlantısı")
    print("   - Otomatik bağlantı koruma")
    print("   - Timeout sorunları düzeltildi")
    print()
    
    if REALSENSE_AVAILABLE:
        print("✅ RealSense support: Available")
    else:
        print("⚠️ RealSense support: Not available (webcam only)")
    
    print()
    try:
        socketio.run(app, host='0.0.0.0', port=5000, debug=False, 
                    use_reloader=False, log_output=False, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print("\n🛑 Sistem kapatılıyor...")
        test_running = False
    except Exception as e:
        print(f"❌ Server hatası: {e}")