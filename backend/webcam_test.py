#!/usr/bin/env python3
"""
Webcam test scripti - hangi kameraların mevcut olduğunu kontrol eder
"""

import cv2
import os

def test_webcam():
    print("🔍 Mevcut kameraları test ediliyor...")
    
    # Video cihazlarını kontrol et
    video_devices = []
    for i in range(10):  # 0-9 arası test et
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    video_devices.append(i)
                    print(f"✅ Kamera {i}: Çalışıyor")
                else:
                    print(f"⚠️ Kamera {i}: Açılıyor ama frame gelmiyor")
                cap.release()
            else:
                print(f"❌ Kamera {i}: Açılamıyor")
        except Exception as e:
            print(f"❌ Kamera {i}: Hata - {e}")
    
    if not video_devices:
        print("\n❌ Hiç webcam bulunamadı!")
        print("\n🔧 Çözüm önerileri:")
        print("1. Webcam'in bağlı olduğundan emin olun")
        print("2. Başka uygulamaların kamerayı kullanmadığından emin olun")
        print("3. Terminal'de şu komutu çalıştırın: ls /dev/video*")
        print("4. Kamera izinlerini kontrol edin")
        return None
    
    print(f"\n✅ {len(video_devices)} webcam bulundu: {video_devices}")
    return video_devices[0]  # İlk çalışan kamerayı döndür

def test_camera_detailed(camera_index):
    print(f"\n📹 Kamera {camera_index} detaylı test ediliyor...")
    
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print("❌ Kamera açılamadı")
        return False
    
    # Kamera özelliklerini al
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"📊 Kamera özellikleri:")
    print(f"   Çözünürlük: {int(width)}x{int(height)}")
    print(f"   FPS: {fps}")
    
    # 10 frame test et
    success_count = 0
    for i in range(10):
        ret, frame = cap.read()
        if ret and frame is not None:
            success_count += 1
        else:
            print(f"⚠️ Frame {i+1}: Başarısız")
    
    cap.release()
    
    print(f"📈 Test sonucu: {success_count}/10 frame başarılı")
    
    if success_count >= 8:
        print("✅ Kamera çalışıyor!")
        return True
    else:
        print("❌ Kamera düzgün çalışmıyor")
        return False

if __name__ == "__main__":
    # Sistem bilgilerini göster
    print("🖥️ Sistem bilgileri:")
    os.system("lsb_release -a 2>/dev/null | grep Description")
    
    # Video cihazlarını listele
    print("\n📱 Video cihazları:")
    os.system("ls -la /dev/video* 2>/dev/null || echo 'Video cihazı bulunamadı'")
    
    # Kameraları test et
    working_camera = test_webcam()
    
    if working_camera is not None:
        test_camera_detailed(working_camera)
        print(f"\n✅ Kullanılacak kamera: {working_camera}")
        print(f"app_webcam.py dosyasında kamera index'ini {working_camera} olarak ayarlayın")
    else:
        print("\n❌ Webcam testi başarısız!")