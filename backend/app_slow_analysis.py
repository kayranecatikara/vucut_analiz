#!/usr/bin/env python3
"""
Yavaşlatılmış Vücut Analizi Sistemi
- Saniyede bir veri güncellemesi
- Daha kararlı ölçümler
- Sürekli vücut tipi analizi
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
streaming = False
analysis_thread = None
camera = None
realsense_pipeline = None
camera_mode = "webcam"

# --- Timing Control Variables ---
last_analysis_time = 0
ANALYSIS_INTERVAL = 1.0  # Saniyede bir analiz
current_analysis_data = {
    'omuz_genisligi': 0.0,
    'bel_genisligi': 0.0,
    'omuz_bel_orani': 0.0,
    'vucut_tipi': 'Analiz Bekleniyor',
    'mesafe': 0.0,
    'confidence': 0.0
}

# --- Model Loading ---
print("🤖 Loading MoveNet model from TensorFlow Hub...")
try:
    model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
    movenet = model.signatures['serving_default']
    print("✅ MoveNet model loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load MoveNet model: {e}")
    exit()

INPUT_SIZE = 192

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
    img_resized = tf.image.resize_with_pad(np.expand_dims(input_image, axis=0), INPUT_SIZE, INPUT_SIZE)
    input_tensor = tf.cast(img_resized, dtype=tf.int32)
    outputs = movenet(input_tensor)
    return outputs['output_0'].numpy()[0, 0]

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
    """Comprehensive body measurement analysis with timing control"""
    global last_analysis_time, current_analysis_data
    
    current_time = time.time()
    
    # Sadece saniyede bir analiz yap
    if current_time - last_analysis_time < ANALYSIS_INTERVAL:
        return current_analysis_data
    
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
        
        print(f"🔍 Keypoint confidence - LS: {ls_c:.2f}, RS: {rs_c:.2f}, LH: {lh_c:.2f}, RH: {rh_c:.2f}")
        
        # Calculate shoulder width
        shoulder_width = 0.0
        if ls_c > 0.3 and rs_c > 0.3:
            p1 = (int(ls_x * width), int(ls_y * height))
            p2 = (int(rs_x * width), int(rs_y * height))
            
            if depth_frame is not None and depth_intrinsics is not None:
                # RealSense 3D measurement
                shoulder_width = calculate_3d_distance_safe(p1, p2, depth_frame, depth_intrinsics)
                print(f"📏 3D Omuz genişliği: {shoulder_width}")
            else:
                # Webcam pixel-based measurement
                pixel_distance = calculate_pixel_distance(p1, p2)
                shoulder_width = (pixel_distance / width) * 90
                shoulder_width = max(25, min(75, shoulder_width))
                print(f"📏 2D Omuz genişliği: {shoulder_width}")
            
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
                print(f"📏 3D Bel genişliği: {waist_width}")
            else:
                # Webcam pixel-based measurement
                pixel_distance = calculate_pixel_distance(p1, p2)
                waist_width = (pixel_distance / width) * 70
                waist_width = max(20, min(55, waist_width))
                print(f"📏 2D Bel genişliği: {waist_width}")
            
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
            
            print(f"🎯 Vücut tipi: {analysis_data['vucut_tipi']} (Oran: {ratio:.2f}, Güven: {confidence:.2f})")
        
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
                        print(f"📐 Mesafe: {distance:.1f}m")
                except Exception as e:
                    print(f"Mesafe hesaplama hatası: {e}")
        else:
            # Fixed distance for webcam
            analysis_data['mesafe'] = 1.5
        
        # Update global data and timestamp
        current_analysis_data = analysis_data
        last_analysis_time = current_time
        
        print(f"✅ Analiz güncellendi: {analysis_data}")
        
    except Exception as e:
        print(f"❌ Analiz hatası: {e}")
    
    return analysis_data

def draw_pose_and_measurements(frame: np.ndarray, keypoints: np.ndarray, 
                             analysis_data: Dict[str, Any]) -> np.ndarray:
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
            if analysis_data['omuz_genisligi'] > 0:
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
            if analysis_data['bel_genisligi'] > 0:
                mid_x = int((pt1[0] + pt2[0]) / 2)
                mid_y = int((pt1[1] + pt2[1]) / 2) + 25
                cv2.putText(frame, f"{analysis_data['bel_genisligi']:.1f}cm", 
                           (mid_x - 40, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Add measurement text overlay - BÜYÜK VE NET
        y_offset = 30
        cv2.putText(frame, f"Omuz: {analysis_data['omuz_genisligi']:.1f}cm", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        y_offset += 35
        cv2.putText(frame, f"Bel: {analysis_data['bel_genisligi']:.1f}cm", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        y_offset += 35
        cv2.putText(frame, f"Tip: {analysis_data['vucut_tipi']}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        if analysis_data['omuz_bel_orani'] > 0:
            y_offset += 35
            cv2.putText(frame, f"Oran: {analysis_data['omuz_bel_orani']:.2f}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        if analysis_data['mesafe'] > 0:
            y_offset += 35
            cv2.putText(frame, f"Mesafe: {analysis_data['mesafe']:.1f}m", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
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

def stream_frames():
    """Stream frames with automatic camera detection"""
    global streaming, camera, realsense_pipeline, camera_mode
    
    try:
        if not detect_camera_type():
            socketio.emit('error', 'Hiçbir kamera bulunamadı')
            return
        
        if camera_mode == "realsense":
            return stream_realsense()
        else:
            return stream_webcam()
            
    except Exception as e:
        print(f"❌ Stream error: {e}")
        socketio.emit('error', f'Stream error: {str(e)}')

def stream_realsense():
    """Stream from RealSense camera with slow analysis"""
    global streaming, realsense_pipeline
    
    try:
        realsense_pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        profile = realsense_pipeline.start(config)
        depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        
        print("✅ RealSense kamera başlatıldı - Yavaş analiz modu")
        
        frame_count = 0
        last_time = time.time()
        last_data_send = 0
        
        while streaming:
            try:
                frames = realsense_pipeline.wait_for_frames(timeout_ms=5000)
                
                align = rs.align(rs.stream.color)
                aligned_frames = align.process(frames)
                
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    continue
                
                color_image = np.asanyarray(color_frame.get_data())
                color_image = cv2.flip(color_image, 1)
                
                # Run pose detection every frame for smooth tracking
                keypoints = run_movenet(color_image)
                
                # Analyze measurements (with timing control inside)
                analysis_data = analyze_body_measurements(
                    keypoints, color_image.shape, depth_frame, depth_intrinsics
                )
                
                # Draw pose and measurements
                rgb_frame = draw_pose_and_measurements(color_image.copy(), keypoints, analysis_data)
                
                # Create depth visualization
                depth_viz = create_depth_visualization(color_image, keypoints, depth_frame)
                
                # Add labels
                cv2.putText(rgb_frame, "RGB + Pose", (10, rgb_frame.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(depth_viz, "Derinlik", (10, depth_viz.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Combine frames
                h1, w1 = rgb_frame.shape[:2]
                h2, w2 = depth_viz.shape[:2]
                if h1 != h2:
                    depth_viz = cv2.resize(depth_viz, (w1, h1))
                
                combined_frame = np.hstack((rgb_frame, depth_viz))
                
                # Send video frame every frame (for smooth video)
                _, buffer = cv2.imencode('.jpg', combined_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                socketio.emit('video_frame', {'type': 'video_frame', 'frame': img_base64})
                
                # Send analysis data only once per second
                current_time = time.time()
                if current_time - last_data_send >= 1.0:
                    socketio.emit('analyze_result', {'type': 'analyze_result', 'data': analysis_data})
                    last_data_send = current_time
                    print(f"📊 Veri gönderildi: {analysis_data}")
                
                # FPS control
                frame_count += 1
                if current_time - last_time >= 1.0:
                    print(f"📊 RealSense FPS: {frame_count}")
                    frame_count = 0
                    last_time = current_time
                
                socketio.sleep(0.033)  # ~30 FPS for video
                
            except Exception as e:
                print(f"❌ RealSense loop error: {e}")
                break
                
    except Exception as e:
        print(f"❌ RealSense start error: {e}")
        socketio.emit('error', f'RealSense error: {str(e)}')
    
    finally:
        if realsense_pipeline:
            realsense_pipeline.stop()
        print("🛑 RealSense stopped")

def stream_webcam():
    """Stream from regular webcam with slow analysis"""
    global streaming, camera
    
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
            socketio.emit('error', 'Webcam bulunamadı')
            return
        
        camera = cv2.VideoCapture(working_camera_index)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)
        
        print("✅ Webcam başlatıldı - Yavaş analiz modu")
        
        frame_count = 0
        last_time = time.time()
        last_data_send = 0
        
        while streaming:
            try:
                ret, frame = camera.read()
                if not ret:
                    continue
                
                frame = cv2.flip(frame, 1)
                
                # Run pose detection every frame for smooth tracking
                keypoints = run_movenet(frame)
                
                # Analyze measurements (with timing control inside)
                analysis_data = analyze_body_measurements(keypoints, frame.shape)
                
                # Draw pose and measurements
                rgb_frame = draw_pose_and_measurements(frame.copy(), keypoints, analysis_data)
                
                # Create depth simulation
                depth_viz = create_depth_visualization(frame, keypoints, None)
                
                # Add labels
                cv2.putText(rgb_frame, "RGB + Pose", (10, rgb_frame.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(depth_viz, "Derinlik Sim.", (10, depth_viz.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Combine frames
                h1, w1 = rgb_frame.shape[:2]
                h2, w2 = depth_viz.shape[:2]
                if h1 != h2:
                    depth_viz = cv2.resize(depth_viz, (w1, h1))
                
                combined_frame = np.hstack((rgb_frame, depth_viz))
                
                # Send video frame every frame (for smooth video)
                _, buffer = cv2.imencode('.jpg', combined_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                socketio.emit('video_frame', {'type': 'video_frame', 'frame': img_base64})
                
                # Send analysis data only once per second
                current_time = time.time()
                if current_time - last_data_send >= 1.0:
                    socketio.emit('analyze_result', {'type': 'analyze_result', 'data': analysis_data})
                    last_data_send = current_time
                    print(f"📊 Veri gönderildi: {analysis_data}")
                
                # FPS control
                frame_count += 1
                if current_time - last_time >= 1.0:
                    print(f"📊 Webcam FPS: {frame_count}")
                    frame_count = 0
                    last_time = current_time
                
                socketio.sleep(0.033)  # ~30 FPS for video
                
            except Exception as e:
                print(f"❌ Webcam loop error: {e}")
                break
                
    except Exception as e:
        print(f"❌ Webcam start error: {e}")
        socketio.emit('error', f'Webcam error: {str(e)}')
    
    finally:
        if camera:
            camera.release()
        print("🛑 Webcam stopped")

# --- SocketIO Events ---
@socketio.on('connect')
def handle_connect(auth):
    print("✅ WebSocket connection established!")

@socketio.on('disconnect')
def handle_disconnect():
    global streaming
    streaming = False
    print("❌ WebSocket connection closed!")

@socketio.on('start_video')
def handle_start_video(data):
    global streaming, analysis_thread
    if not streaming:
        streaming = True
        analysis_thread = socketio.start_background_task(target=stream_frames)
        socketio.emit('stream_started', {'type': 'stream_started'})

@socketio.on('stop_video')
def handle_stop_video(data):
    global streaming
    streaming = False
    socketio.emit('stream_stopped', {'type': 'stream_stopped'})

if __name__ == '__main__':
    print("🚀 Starting Slow Analysis Body System...")
    print("📋 Features:")
    print("   - Saniyede bir veri güncellemesi")
    print("   - Kararlı ölçüm sistemi")
    print("   - Sürekli vücut tipi analizi")
    print("   - Görünür mor/cyan ölçüm çizgileri")
    print("   - Otomatik kamera algılama")
    print()
    
    if REALSENSE_AVAILABLE:
        print("✅ RealSense support: Available")
    else:
        print("⚠️ RealSense support: Not available (webcam only)")
    
    print()
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)