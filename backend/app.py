# app.py - Enhanced Real-time Body Analysis Backend

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

# --- AI and Camera Libraries ---
import tensorflow as tf
import tensorflow_hub as hub
try:
    import pyrealsense2 as rs
except ImportError:
    print("⚠️ pyrealsense2 library not found. This application requires Intel RealSense camera.")
    rs = None

# --- Flask and SocketIO Setup ---
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# --- Global Variables ---
streaming = False
analysis_thread = None
realsense_pipeline = None
depth_filters = {}

# --- Model Loading ---
print("🤖 Loading MoveNet model from TensorFlow Hub...")
try:
    model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
    movenet = model.signatures['serving_default']
    print("✅ MoveNet model loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load MoveNet model. Check your internet connection and TensorFlow installation. Error: {e}")
    exit()

INPUT_SIZE = 192  # Lightning model expects 192x192 input

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

# --- Helper Functions ---

def setup_realsense_filters(pipeline_profile):
    """Setup RealSense post-processing filters for better depth quality"""
    global depth_filters
    
    try:
        # Spatial filter - reduces noise
        depth_filters['spatial'] = rs.spatial_filter()
        try:
            if hasattr(rs.option, 'filter_magnitude'):
                depth_filters['spatial'].set_option(rs.option.filter_magnitude, 2)
            if hasattr(rs.option, 'filter_smooth_alpha'):
                depth_filters['spatial'].set_option(rs.option.filter_smooth_alpha, 0.5)
            if hasattr(rs.option, 'filter_smooth_delta'):
                depth_filters['spatial'].set_option(rs.option.filter_smooth_delta, 20)
        except Exception as e:
            print(f"⚠️ Spatial filter options not available: {e}")
        
        # Temporal filter - reduces flickering
        depth_filters['temporal'] = rs.temporal_filter()
        try:
            if hasattr(rs.option, 'filter_smooth_alpha'):
                depth_filters['temporal'].set_option(rs.option.filter_smooth_alpha, 0.4)
            if hasattr(rs.option, 'filter_smooth_delta'):
                depth_filters['temporal'].set_option(rs.option.filter_smooth_delta, 20)
        except Exception as e:
            print(f"⚠️ Temporal filter options not available: {e}")
        
        # Hole filling filter - fills holes in depth data
        depth_filters['hole_filling'] = rs.hole_filling_filter()
        try:
            if hasattr(rs.option, 'holes_fill'):
                depth_filters['hole_filling'].set_option(rs.option.holes_fill, 1)  # Fill from farthest
        except Exception as e:
            print(f"⚠️ Hole filling filter options not available: {e}")
        
        # Decimation filter - reduces resolution but improves performance
        depth_filters['decimation'] = rs.decimation_filter()
        try:
            if hasattr(rs.option, 'filter_magnitude'):
                depth_filters['decimation'].set_option(rs.option.filter_magnitude, 2)
        except Exception as e:
            print(f"⚠️ Decimation filter options not available: {e}")
            
        print("✅ RealSense depth filters configured (with available options).")
        
    except Exception as e:
        print(f"⚠️ Could not setup depth filters: {e}")
        depth_filters = {}  # Boş bırak, filtresiz çalışsın
    
    print("✅ RealSense depth filters configured successfully.")

def apply_depth_filters(depth_frame):
    """Apply all depth filters to improve depth quality"""
    if not depth_filters:
        return depth_frame
    
    # Apply filters in sequence
    filtered_frame = depth_frame
    
    # Decimation filter (first to reduce data)
    if 'decimation' in depth_filters:
        filtered_frame = depth_filters['decimation'].process(filtered_frame)
    
    # Spatial filter
    if 'spatial' in depth_filters:
        filtered_frame = depth_filters['spatial'].process(filtered_frame)
    
    # Temporal filter
    if 'temporal' in depth_filters:
        filtered_frame = depth_filters['temporal'].process(filtered_frame)
    
    # Hole filling filter (last to fill remaining holes)
    if 'hole_filling' in depth_filters:
        filtered_frame = depth_filters['hole_filling'].process(filtered_frame)
    
    return filtered_frame

def run_movenet(input_image: np.ndarray) -> np.ndarray:
    """Run MoveNet model on input image and return keypoints"""
    img_resized = tf.image.resize_with_pad(np.expand_dims(input_image, axis=0), INPUT_SIZE, INPUT_SIZE)
    input_tensor = tf.cast(img_resized, dtype=tf.int32)
    outputs = movenet(input_tensor)
    return outputs['output_0'].numpy()[0, 0]

def calculate_3d_distance(p1: Tuple[int, int], p2: Tuple[int, int], 
                         depth_frame, depth_intrinsics) -> Optional[float]:
    """Calculate 3D distance between two points using depth data"""
    try:
        # Get depth values at both points
        depth1 = depth_frame.get_distance(p1[0], p1[1])
        depth2 = depth_frame.get_distance(p2[0], p2[1])
        
        # Validate depth values
        if depth1 <= 0 or depth2 <= 0 or depth1 > 5.0 or depth2 > 5.0:
            return None
            
        # Convert pixel coordinates to 3D world coordinates
        point1_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [p1[0], p1[1]], depth1)
        point2_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [p2[0], p2[1]], depth2)
        
        # Calculate 3D distance
        distance = np.linalg.norm(np.subtract(point1_3d, point2_3d))
        
        # Convert to centimeters and validate
        distance_cm = distance * 100
        if distance_cm > 200:  # Sanity check: no body part should be > 200cm
            return None
            
        return distance_cm
        
    except Exception as e:
        print(f"Error calculating 3D distance: {e}")
        return None

def classify_body_type(shoulder_width: float, waist_width: float, 
                      shoulder_waist_ratio: float) -> Tuple[str, float]:
    """Classify body type based on measurements"""
    if shoulder_width <= 0 or waist_width <= 0:
        return "Analiz Bekleniyor", 0.0
    
    # Calculate confidence based on measurement quality
    confidence = min(1.0, max(0.3, 1.0 - abs(shoulder_width - 45) / 100))
    
    if shoulder_waist_ratio > 1.4:
        return "Ektomorf", confidence
    elif shoulder_waist_ratio > 1.2:
        return "Mezomorf", confidence
    else:
        return "Endomorf", confidence

def draw_and_analyze(frame: np.ndarray, keypoints: np.ndarray, 
                    depth_frame, depth_intrinsics) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Draw keypoints and analyze body measurements"""
    height, width, _ = frame.shape
    analysis_data = {
        'omuz_genisligi': 0.0,
        'bel_genisligi': 0.0,
        'omuz_bel_orani': 0.0,
        'vucut_tipi': 'Analiz Bekleniyor',
        'mesafe': 0.0,
        'confidence': 0.0
    }
    
    try:
        # Draw skeleton
        for p1_idx, p2_idx in EDGES:
            y1, x1, c1 = keypoints[p1_idx]
            y2, x2, c2 = keypoints[p2_idx]
            if c1 > 0.3 and c2 > 0.3:
                pt1 = (int(x1 * width), int(y1 * height))
                pt2 = (int(x2 * width), int(y2 * height))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
        
        # Extract keypoints
        ls_y, ls_x, ls_c = keypoints[KEYPOINT_DICT['left_shoulder']]
        rs_y, rs_x, rs_c = keypoints[KEYPOINT_DICT['right_shoulder']]
        lh_y, lh_x, lh_c = keypoints[KEYPOINT_DICT['left_hip']]
        rh_y, rh_x, rh_c = keypoints[KEYPOINT_DICT['right_hip']]
        
        # Calculate shoulder width
        if ls_c > 0.3 and rs_c > 0.3:
            p1 = (int(ls_x * width), int(ls_y * height))
            p2 = (int(rs_x * width), int(rs_y * height))
            
            shoulder_width = calculate_3d_distance(p1, p2, depth_frame, depth_intrinsics)
            if shoulder_width:
                analysis_data['omuz_genisligi'] = shoulder_width
                
                # Draw measurement line
                cv2.line(frame, p1, p2, (255, 0, 0), 3)
                cv2.putText(frame, f"{shoulder_width:.1f} cm", 
                           (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (255, 255, 255), 2)
        
        # Calculate waist width
        if lh_c > 0.3 and rh_c > 0.3:
            p1 = (int(lh_x * width), int(lh_y * height))
            p2 = (int(rh_x * width), int(rh_y * height))
            
            waist_width = calculate_3d_distance(p1, p2, depth_frame, depth_intrinsics)
            if waist_width:
                analysis_data['bel_genisligi'] = waist_width
                
                # Draw measurement line
                cv2.line(frame, p1, p2, (255, 0, 255), 3)
                cv2.putText(frame, f"{waist_width:.1f} cm", 
                           (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (255, 255, 255), 2)
        
        # Calculate ratios and body type
        if analysis_data['omuz_genisligi'] > 0 and analysis_data['bel_genisligi'] > 0:
            ratio = analysis_data['omuz_genisligi'] / analysis_data['bel_genisligi']
            analysis_data['omuz_bel_orani'] = ratio
            
            body_type, confidence = classify_body_type(
                analysis_data['omuz_genisligi'], 
                analysis_data['bel_genisligi'], 
                ratio
            )
            analysis_data['vucut_tipi'] = body_type
            analysis_data['confidence'] = confidence
        
        # Calculate average distance to person
        if ls_c > 0.3 and rs_c > 0.3:
            center_x = int((ls_x + rs_x) * width / 2)
            center_y = int((ls_y + rs_y) * height / 2)
            distance = depth_frame.get_distance(center_x, center_y)
            if distance > 0:
                analysis_data['mesafe'] = distance
                
    except Exception as e:
        print(f"Error in draw_and_analyze: {e}")
    
    return frame, analysis_data

def stream_frames():
    """Stream frames from camera with analysis"""
    global streaming, realsense_pipeline
    
    if not rs:
        socketio.emit('error', 'RealSense library not available.')
        return

    try:
        # Önce mevcut cihazları kontrol et
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            print("❌ Intel RealSense kamera bulunamadı!")
            socketio.emit('error', 'Intel RealSense kamera bulunamadı. Kameranın bağlı olduğundan emin olun.')
            return
        
        print(f"✅ {len(devices)} Intel RealSense kamera bulundu")
        for i, device in enumerate(devices):
            print(f"   Kamera {i}: {device.get_info(rs.camera_info.name)}")

        # Configure RealSense pipeline
        realsense_pipeline = rs.pipeline()
        config = rs.config()
        
        # Optimize edilmiş ayarlar
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # Start pipeline
        profile = realsense_pipeline.start(config)
        
        # Get depth sensor and set high accuracy preset
        depth_sensor = profile.get_device().first_depth_sensor()
        
        # Depth sensor ayarları
        try:
            # High Accuracy preset
            depth_sensor.set_option(rs.option.visual_preset, 3)
            # Laser power ayarı (daha iyi derinlik için)
            depth_sensor.set_option(rs.option.laser_power, 240)  # Max: 360
            # Confidence threshold
            depth_sensor.set_option(rs.option.confidence_threshold, 1)
            print("✅ Depth sensor optimize edildi")
        except Exception as preset_error:
            print(f"⚠️ Could not set visual preset: {preset_error}")
        
        # Get depth intrinsics
        depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        
        # Setup filters
        setup_realsense_filters(profile)
        
        # Colorizer for depth visualization
        colorizer = rs.colorizer()
        colorizer.set_option(rs.option.color_scheme, 0)  # Jet colormap
        
        print("✅ Intel RealSense camera started successfully.")
        
        frame_count = 0
        last_time = time.time()
        
        while streaming:
            try:
                # Wait for frames with timeout
                frames = realsense_pipeline.wait_for_frames(timeout_ms=5000)
                
                # Get aligned frames
                align = rs.align(rs.stream.color)
                aligned_frames = align.process(frames)
                
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    continue
                
                # Apply depth filters
                filtered_depth_frame = apply_depth_filters(depth_frame)
                
                # Convert color frame to numpy array
                color_image = np.asanyarray(color_frame.get_data())
                color_image = cv2.flip(color_image, 1)  # Mirror image
                
                # Convert depth frame to colorized image
                depth_colormap = np.asanyarray(colorizer.colorize(filtered_depth_frame).get_data())
                depth_colormap = cv2.flip(depth_colormap, 1)  # Mirror image
                
                # Run pose detection
                keypoints = run_movenet(color_image)
                
                # Analyze and draw
                processed_frame, analysis_data = draw_and_analyze(
                    color_image, keypoints, filtered_depth_frame, depth_intrinsics
                )
                
                # Create side-by-side view
                # Resize images to same height if needed
                h1, w1 = processed_frame.shape[:2]
                h2, w2 = depth_colormap.shape[:2]
                
                if h1 != h2:
                    depth_colormap = cv2.resize(depth_colormap, (w1, h1))
                
                # Add labels
                cv2.putText(processed_frame, "RGB + Pose", (10, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(depth_colormap, "Depth", (10, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Combine images side by side
                combined_frame = np.hstack((processed_frame, depth_colormap))
                
                # Encode combined frame
                _, buffer = cv2.imencode('.jpg', combined_frame, 
                                       [cv2.IMWRITE_JPEG_QUALITY, 85])
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
                
                socketio.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                print(f"❌ Error in stream loop: {e}")
                break
                
    except Exception as e:
        print(f"❌ Failed to start Intel RealSense camera: {e}")
        print("💡 Çözüm önerileri:")
        print("   1. Kameranın USB 3.0 porta bağlı olduğundan emin olun")
        print("   2. Başka uygulamaların kamerayı kullanmadığından emin olun")
        print("   3. 'realsense-viewer' ile kamerayı test edin")
        print("   4. Kamerayı çıkarıp tekrar takın")
        socketio.emit('error', f'Failed to start camera: {str(e)}')
        return
    
    finally:
        if realsense_pipeline:
            realsense_pipeline.stop()
        print("🛑 Camera and stream stopped.")

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
    print("🚀 Starting Enhanced Body Analysis Server...")
    print("📋 Features:")
    print("   - Intel RealSense D435i camera support")
    print("   - MoveNet Lightning pose detection")
    print("   - 3D depth-based measurements")
    print("   - Advanced depth filtering")
    print("   - Real-time body type classification")
    print("   - WebSocket communication")
    print()
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)