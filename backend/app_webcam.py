#!/usr/bin/env python3
"""
Normal webcam ile çalışan vücut analizi
(RealSense olmadan, sadece pose detection)
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

def estimate_body_measurements(keypoints: np.ndarray, frame_shape: Tuple[int, int]) -> Dict[str, Any]:
    """Estimate body measurements from keypoints (pixel-based approximation)"""
    height, width = frame_shape[:2]
    
    analysis_data = {
        'omuz_genisligi': 0.0,
        'bel_genisligi': 0.0,
        'omuz_bel_orani': 0.0,
        'vucut_tipi': 'Analiz Bekleniyor',
        'mesafe': 1.5,  # Sabit mesafe (webcam için)
        'confidence': 0.0
    }
    
    try:
        # Extract keypoints
        ls_y, ls_x, ls_c = keypoints[KEYPOINT_DICT['left_shoulder']]
        rs_y, rs_x, rs_c = keypoints[KEYPOINT_DICT['right_shoulder']]
        lh_y, lh_x, lh_c = keypoints[KEYPOINT_DICT['left_hip']]
        rh_y, rh_x, rh_c = keypoints[KEYPOINT_DICT['right_hip']]
        
        # Calculate shoulder width (pixel to cm approximation)
        if ls_c > 0.3 and rs_c > 0.3:
            p1 = (int(ls_x * width), int(ls_y * height))
            p2 = (int(rs_x * width), int(rs_y * height))
            
            pixel_distance = calculate_pixel_distance(p1, p2)
            # Rough approximation: assume person is ~1.5m away, shoulder width ~40-50cm
            shoulder_width = (pixel_distance / width) * 100  # Basit oran
            analysis_data['omuz_genisligi'] = max(20, min(80, shoulder_width))  # 20-80cm arası sınırla
        
        # Calculate waist width
        if lh_c > 0.3 and rh_c > 0.3:
            p1 = (int(lh_x * width), int(lh_y * height))
            p2 = (int(rh_x * width), int(rh_y * height))
            
            pixel_distance = calculate_pixel_distance(p1, p2)
            waist_width = (pixel_distance / width) * 80  # Biraz daha dar
            analysis_data['bel_genisligi'] = max(15, min(60, waist_width))
        
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
            
            # Simple confidence based on keypoint visibility
            confidence = (ls_c + rs_c + lh_c + rh_c) / 4
            analysis_data['confidence'] = min(1.0, confidence)
        
    except Exception as e:
        print(f"Error in measurements: {e}")
    
    return analysis_data

def draw_pose(frame: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
    """Draw pose skeleton on frame"""
    height, width, _ = frame.shape
    
    # Draw skeleton
    for p1_idx, p2_idx in EDGES:
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
    
    # Draw measurement lines
    ls_y, ls_x, ls_c = keypoints[KEYPOINT_DICT['left_shoulder']]
    rs_y, rs_x, rs_c = keypoints[KEYPOINT_DICT['right_shoulder']]
    lh_y, lh_x, lh_c = keypoints[KEYPOINT_DICT['left_hip']]
    rh_y, rh_x, rh_c = keypoints[KEYPOINT_DICT['right_hip']]
    
    # Shoulder measurement line
    if ls_c > 0.3 and rs_c > 0.3:
        pt1 = (int(ls_x * width), int(ls_y * height))
        pt2 = (int(rs_x * width), int(rs_y * height))
        cv2.line(frame, pt1, pt2, (255, 0, 0), 3)
        
        # Add measurement text on line
        mid_x = int((pt1[0] + pt2[0]) / 2)
        mid_y = int((pt1[1] + pt2[1]) / 2) - 10
        pixel_distance = calculate_pixel_distance(pt1, pt2)
        shoulder_width = (pixel_distance / width) * 100
        shoulder_width = max(20, min(80, shoulder_width))
        cv2.putText(frame, f"{shoulder_width:.1f}cm", (mid_x - 30, mid_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Hip measurement line
    if lh_c > 0.3 and rh_c > 0.3:
        pt1 = (int(lh_x * width), int(lh_y * height))
        pt2 = (int(rh_x * width), int(rh_y * height))
        cv2.line(frame, pt1, pt2, (255, 0, 255), 3)
        
        # Add measurement text on line
        mid_x = int((pt1[0] + pt2[0]) / 2)
        mid_y = int((pt1[1] + pt2[1]) / 2) + 20
        pixel_distance = calculate_pixel_distance(pt1, pt2)
        waist_width = (pixel_distance / width) * 80
        waist_width = max(15, min(60, waist_width))
        cv2.putText(frame, f"{waist_width:.1f}cm", (mid_x - 30, mid_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    
    return frame

def create_depth_simulation(frame: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
    """Create a simple depth-like visualization for webcam"""
    height, width, _ = frame.shape
    
    # Convert to grayscale first for speed
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply colormap for depth effect (much faster)
    depth_sim = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    
    # Highlight person with keypoints
    for i, (y, x, c) in enumerate(keypoints):
        if c > 0.3:
            pt = (int(x * width), int(y * height))
            cv2.circle(depth_sim, pt, 8, (255, 255, 255), -1)
    
    return depth_sim
def stream_frames():
    """Stream frames from webcam with analysis"""
    global streaming, camera
    
    try:
        # Test sonucuna göre çalışan kamera index'i
        working_cameras = [2, 4, 6, 0, 1]  # Test sonucundan + ekstra
        working_camera_index = None
        
        for camera_index in working_cameras:
            test_cap = cv2.VideoCapture(camera_index)
            if test_cap.isOpened():
                ret, frame = test_cap.read()
                if ret and frame is not None:
                    working_camera_index = camera_index
                    test_cap.release()
                    print(f"✅ Kamera {camera_index} kullanılıyor")
                    break
                test_cap.release()
        
        if working_camera_index is None:
            print("❌ Test edilen kameralar açılamadı!")
            socketio.emit('error', 'Webcam açılamadı. Kameranın başka bir uygulama tarafından kullanılmadığından emin olun.')
            return
        
        # Open webcam
        camera = cv2.VideoCapture(working_camera_index)
        
        if not camera.isOpened():
            print(f"❌ Kamera {working_camera_index} açılamadı!")
            socketio.emit('error', 'Webcam açılamadı. Kameranın bağlı olduğundan emin olun.')
            return
        
        # Set camera properties
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)
        
        print("✅ Webcam başlatıldı!")
        
        frame_count = 0
        last_time = time.time()
        
        while streaming:
            try:
                ret, frame = camera.read()
                
                if not ret:
                    print("❌ Frame okunamadı")
                    continue
                
                # Mirror the frame
                frame = cv2.flip(frame, 1)
                
                # Run pose detection (every 2nd frame for performance)
                if frame_count % 2 == 0:
                    keypoints = run_movenet(frame)
                
                # Create clean RGB copy
                rgb_frame = frame.copy()
                
                # Draw pose on RGB frame
                rgb_frame = draw_pose(rgb_frame, keypoints)
                
                # Estimate measurements
                analysis_data = estimate_body_measurements(keypoints, rgb_frame.shape)
                
                # Create depth simulation (only every 3rd frame for performance)
                if frame_count % 3 == 0:
                    depth_sim = create_depth_simulation(frame, keypoints)
                
                # Add measurement text to RGB frame
                if analysis_data['omuz_genisligi'] > 0:
                    cv2.putText(rgb_frame, f"Omuz: {analysis_data['omuz_genisligi']:.1f}cm", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                if analysis_data['bel_genisligi'] > 0:
                    cv2.putText(rgb_frame, f"Bel: {analysis_data['bel_genisligi']:.1f}cm", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.putText(rgb_frame, f"Tip: {analysis_data['vucut_tipi']}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                if analysis_data['omuz_bel_orani'] > 0:
                    cv2.putText(rgb_frame, f"Oran: {analysis_data['omuz_bel_orani']:.2f}", 
                               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Add labels
                cv2.putText(rgb_frame, "RGB + Pose", (10, rgb_frame.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(depth_sim, "Depth Simulation", (10, depth_sim.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Combine images side by side
                h1, w1 = rgb_frame.shape[:2]
                h2, w2 = depth_sim.shape[:2]
                
                if h1 != h2:
                    depth_sim = cv2.resize(depth_sim, (w1, h1))
                
                combined_frame = np.hstack((rgb_frame, depth_sim))
                
                # Encode frame
                _, buffer = cv2.imencode('.jpg', combined_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # Send data to client
                socketio.emit('video_frame', {
                    'type': 'video_frame',
                    'frame': img_base64
                })
                
                socketio.emit('analyze_result', {
                    'type': 'analyze_result',
                    'data': analysis_data
                })
                
                # Control frame rate
                frame_count += 1
                current_time = time.time()
                if current_time - last_time >= 1.0:
                    print(f"📊 FPS: {frame_count}")
                    frame_count = 0
                    last_time = current_time
                
                socketio.sleep(0.05)  # ~20 FPS (daha stabil)
                
            except Exception as e:
                print(f"❌ Error in stream loop: {e}")
                break
                
    except Exception as e:
        print(f"❌ Failed to start webcam: {e}")
        socketio.emit('error', f'Failed to start camera: {str(e)}')
        return
    
    finally:
        if camera:
            camera.release()
        print("🛑 Webcam stopped.")

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
    print("🚀 Starting Webcam Body Analysis Server...")
    print("📋 Features:")
    print("   - Normal webcam support")
    print("   - MoveNet Lightning pose detection")
    print("   - Pixel-based measurement estimation")
    print("   - Real-time body type classification")
    print("   - WebSocket communication")
    print()
    print("⚠️  Not: Bu versiyon normal webcam kullanır, 3D ölçümler yaklaşık değerlerdir.")
    print()
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)