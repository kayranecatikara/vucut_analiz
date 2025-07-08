#!/usr/bin/env python3
"""
Basit RealSense kamera test scripti
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import time

def test_camera():
    print("🔍 RealSense kamera test ediliyor...")
    
    # Context oluştur
    ctx = rs.context()
    devices = ctx.query_devices()
    
    if len(devices) == 0:
        print("❌ Kamera bulunamadı!")
        return False
    
    print(f"✅ {len(devices)} kamera bulundu:")
    for i, device in enumerate(devices):
        print(f"   {i}: {device.get_info(rs.camera_info.name)}")
    
    # Pipeline oluştur
    pipeline = rs.pipeline()
    config = rs.config()
    
    # En basit ayarlar
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    try:
        print("📹 Kamerayı başlatıyor...")
        pipeline.start(config)
        print("✅ Kamera başlatıldı!")
        
        frame_count = 0
        start_time = time.time()
        
        for i in range(100):  # 100 frame test et
            try:
                # Frame bekle (5 saniye timeout)
                frames = pipeline.wait_for_frames(timeout_ms=5000)
                color_frame = frames.get_color_frame()
                
                if color_frame:
                    frame_count += 1
                    if frame_count % 10 == 0:
                        print(f"📊 {frame_count} frame alındı")
                else:
                    print("⚠️ Boş frame")
                    
            except Exception as e:
                print(f"❌ Frame hatası: {e}")
                break
        
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        
        print(f"\n📈 Test Sonuçları:")
        print(f"   Toplam frame: {frame_count}/100")
        print(f"   Süre: {elapsed:.1f} saniye")
        print(f"   FPS: {fps:.1f}")
        
        if frame_count > 50:
            print("✅ Kamera çalışıyor!")
            return True
        else:
            print("❌ Kamera düzgün çalışmıyor")
            return False
            
    except Exception as e:
        print(f"❌ Kamera başlatılamadı: {e}")
        return False
    
    finally:
        try:
            pipeline.stop()
            print("🛑 Kamera durduruldu")
        except:
            pass

if __name__ == "__main__":
    success = test_camera()
    
    if success:
        print("\n✅ Kamera testi başarılı!")
        print("Ana uygulamayı çalıştırabilirsiniz: python app.py")
    else:
        print("\n❌ Kamera testi başarısız!")
        print("Çözüm önerileri:")
        print("1. Kamerayı çıkarıp tekrar takın")
        print("2. USB 3.0 porta takın")
        print("3. Bilgisayarı yeniden başlatın")
        print("4. RealSense SDK'yı yeniden yükleyin")