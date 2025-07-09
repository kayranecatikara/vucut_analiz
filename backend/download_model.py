#!/usr/bin/env python3
"""
MoveNet modelini önceden indirip kaydetme scripti
Bu script modeli yerel olarak kaydeder, böylece her seferinde internetten indirmek gerekmez
"""

import tensorflow as tf
import tensorflow_hub as hub
import os
import time

def download_and_save_model():
    """MoveNet modelini indir ve kaydet"""
    
    model_url = "https://tfhub.dev/google/movenet/singlepose/lightning/4"
    model_dir = "./movenet_model"
    
    print("🤖 MoveNet modelini indiriliyor...")
    print(f"📍 URL: {model_url}")
    print(f"📁 Kayıt yeri: {model_dir}")
    
    try:
        # Timeout ayarla
        import socket
        socket.setdefaulttimeout(60)  # 60 saniye timeout
        
        # Modeli indir
        print("📥 Model indiriliyor... (Bu işlem birkaç dakika sürebilir)")
        model = hub.load(model_url)
        
        # Modeli kaydet
        print("💾 Model kaydediliyor...")
        tf.saved_model.save(model, model_dir)
        
        print("✅ Model başarıyla indirildi ve kaydedildi!")
        print(f"📁 Konum: {os.path.abspath(model_dir)}")
        
        # Test et
        print("🧪 Model test ediliyor...")
        movenet = model.signatures['serving_default']
        
        # Dummy input ile test
        dummy_input = tf.zeros((1, 192, 192, 3), dtype=tf.int32)
        output = movenet(dummy_input)
        
        print("✅ Model testi başarılı!")
        print(f"📊 Output shape: {output['output_0'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model indirme hatası: {e}")
        print("\n💡 Çözüm önerileri:")
        print("   1. İnternet bağlantınızı kontrol edin")
        print("   2. VPN kullanıyorsanız kapatmayı deneyin")
        print("   3. Firewall ayarlarını kontrol edin")
        print("   4. Birkaç dakika sonra tekrar deneyin")
        return False

def load_saved_model():
    """Kaydedilmiş modeli yükle"""
    model_dir = "./movenet_model"
    
    if not os.path.exists(model_dir):
        print(f"❌ Model bulunamadı: {model_dir}")
        return None
    
    try:
        print("📂 Kaydedilmiş model yükleniyor...")
        model = tf.saved_model.load(model_dir)
        print("✅ Model başarıyla yüklendi!")
        return model
    except Exception as e:
        print(f"❌ Model yükleme hatası: {e}")
        return None

if __name__ == "__main__":
    print("🚀 MoveNet Model İndirici")
    print("=" * 50)
    
    # Önce kaydedilmiş model var mı kontrol et
    if os.path.exists("./movenet_model"):
        print("📁 Kaydedilmiş model bulundu!")
        model = load_saved_model()
        if model:
            print("✅ Mevcut model kullanılabilir!")
        else:
            print("❌ Mevcut model bozuk, yeniden indiriliyor...")
            download_and_save_model()
    else:
        print("📥 Model bulunamadı, indiriliyor...")
        download_and_save_model()
    
    print("\n🎯 Kullanım:")
    print("   Model indirildikten sonra app_test_system.py'yi çalıştırabilirsiniz")
    print("   python app_test_system.py")