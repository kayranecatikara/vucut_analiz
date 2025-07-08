#!/usr/bin/env python3
"""
Hibrit Vücut Analizi Sistemi
- RealSense kamera varsa: Gerçek 3D ölçümler
- Normal webcam varsa: Pixel tabanlı yaklaşık ölçümler
- Her iki durumda da temiz RGB görüntü
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
camera_mode = "webcam"  # "webcam" or "realsense"

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
        # Bounds checking
        y = max(0, min(height - 1, y))
        x = max(0, min(width - 1, x))
        return float(depth_array[y, x])
    except Exception as e:
        print(f"Array access error: {e}")
        return 0.0

def calculate_3d_distance_safe(p1: Tuple[int, int], p2: Tuple[int, int], 
                              depth_frame, depth_intrinsics) -> Optional[float]:
    """Safely calculate 3D distance between two points using depth data"""
    try:
        # Convert depth frame to numpy array
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_units = depth_frame.get_units()
        
        # Get depth values with bounds checking
        depth1 = safe_array_access(depth_image, p1[1], p1[0]) * depth_units
        depth2 = safe_array_access(depth_image, p2[1], p2[0]) * depth_units
        
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
        if distance_cm > 200:  # Sanity check
            return None
            
        return distance_cm
        
    except Exception as e:
        print(f"3D distance calculation error: {e}")
        return None

def estimate_body_measurements_webcam(keypoints: np.ndarray, frame_shape: Tuple[int, int]) -> Dict[str, Any]:
    """Estimate body measurements from keypoints (webcam mode - pixel-based)"""
    height, width = frame_shape[:2]
    
    analysis_data = {
        'omuz_genisligi': 0.0,
        'bel_genisligi': 0.0,
        'omuz_bel_orani': 0.0,
        'vucut_tipi': 'Analiz Bekleniyor',
        'mesafe': 1.5,  # Fixed distance for webcam
        'confidence': 0.0
    }
    
    try:
        # Extract keypoints with bounds checking
        if len(keypoints) < 17:
            return analysis_data
            
        ls_y, ls_x, ls_c = keypoints[KEYPOINT_DICT['left_shoulder']]
        rs_y, rs_x, rs_c = keypoints[KEYPOINT_DICT['right_shoulder']]
        lh_y, lh_x, lh_c = keypoints[KEYPOINT_DICT['left_hip']]
        rh_y, rh_x, rh_c = keypoints[KEYPOINT_DICT['right_hip']]
        
        # Calculate shoulder width (pixel to cm approximation)
        if ls_c > 0.3 and rs_c > 0.3:
            p1 = (int(ls_x * width), int(ls_y * height))
            p2 = (int(rs_x * width), int(rs_y * height))
            
            pixel_distance = calculate_pixel_distance(p1, p2)
            # Improved approximation based on typical shoulder width
            shoulder_width = (pixel_distance / width) * 90  # Better scaling
            analysis_data['omuz_genisligi'] = max(25, min(75, shoulder_width))
        
        # Calculate waist width
        if lh_c > 0.3 and rh_c > 0.3:
            p1 = (int(lh_x * width), int(lh_y * height))
            p2 = (int(rh_x * width), int(rh_y * height))
            
            pixel_distance = calculate_pixel_distance(p1, p2)
            waist_width = (pixel_distance / width) * 70  # Narrower than shoulders
            analysis_data['bel_genisligi'] = max(20, min(55, waist_width))
        
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
            
            # Confidence based on keypoint visibility
            confidence = (ls_c + rs_c + lh_c + rh_c) / 4
            analysis_data['confidence'] = min(1.0, confidence)
        
    except Exception as e:
        print(f"Webcam measurement error: {e}")
    
    return analysis_data

def estimate_body_measurements_realsense(keypoints: np.ndarray, frame_shape: Tuple[int, int], 
                                       depth_frame, depth_intrinsics) -> Dict[str, Any]:
    """Estimate body measurements using RealSense depth data"""
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
        # Extract keypoints with bounds checking
        if len(keypoints) < 17:
            return analysis_data
            
        ls_y, ls_x, ls_c = keypoints[KEYPOINT_DICT['left_shoulder']]
        rs_y, rs_x, rs_c = keypoints[KEYPOINT_DICT['right_shoulder']]
        lh_y, lh_x, lh_c = keypoints[KEYPOINT_DICT['left_hip']]
        rh_y, rh_x, rh_c = keypoints[KEYPOINT_DICT['right_hip']]
        
        # Calculate shoulder width using 3D depth
        if ls_c > 0.3 and rs_c > 0.3:
            p1 = (int(ls_x * width), int(ls_y * height))
            p2 = (int(rs_x * width), int(rs_y * height))
            
            shoulder_width = calculate_3d_distance_safe(p1, p2, depth_frame, depth_intrinsics)
            if shoulder_width:
                analysis_data['omuz_genisligi'] = shoulder_width
        
        # Calculate waist width using 3D depth
        if lh_c > 0.3 and rh_c > 0.3:
            p1 = (int(lh_x * width), int(lh_y * height))
            p2 = (int(rh_x * width), int(rh_y * height))
            
            waist_width = calculate_3d_distance_safe(p1, p2, depth_frame, depth_intrinsics)
            if waist_width:
                analysis_data['bel_genisligi'] = waist_width
        
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
            
            # Confidence based on keypoint visibility
            confidence = (ls_c + rs_c + lh_c + rh_c) / 4
            analysis_data['confidence'] = min(1.0, confidence)
        
        # Calculate distance to person
        if ls_c > 0.3 and rs_c > 0.3:
            center_x = int((ls_x + rs_x) * width / 2)
            center_y = int((ls_y + rs_y) * height / 2)
            
            try:
                depth_image = np.asanyarray(depth_frame.get_data())
                distance = safe_array_access(depth_image, center_y, center_x) * depth_frame.get_units()
                if distance > 0:
                    analysis_data['mesafe'] = distance
            except Exception as e:
                print(f"Distance calculation error: {e}")
        
    except Exception as e:
        print(f"RealSense measurement error: {e}")
    
    return analysis_data

def draw_pose_and_measurements(frame: np.ndarray, keypoints: np.ndarray, 
                             analysis_data: Dict[str, Any]) -> np.ndarray:
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
        
        # Draw measurement lines
        ls_y, ls_x, ls_c = keypoints[KEYPOINT_DICT['left_shoulder']]
        rs_y, rs_x, rs_c = keypoints[KEYPOINT_DICT['right_shoulder']]
        lh_y, lh_x, lh_c = keypoints[KEYPOINT_DICT['left_hip']]
        rh_y, rh_x, rh_c = keypoints[KEYPOINT_DICT['right_hip']]
        
        # Shoulder measurement line
        if ls_c > 0.3 and rs_c > 0.3 and analysis_data['omuz_genisligi'] > 0:
            pt1 = (int(ls_x * width), int(ls_y * height))
            pt2 = (int(rs_x * width), int(rs_y * height))
            cv2.line(frame, pt1, pt2, (255, 0, 0), 3)
            
            mid_x = int((pt1[0] + pt2[0]) / 2)
            mid_y = int((pt1[1] + pt2[1]) / 2) - 10
            cv2.putText(frame, f"{analysis_data['omuz_genisligi']:.1f}cm", 
                       (mid_x - 30, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Hip measurement line
        if lh_c > 0.3 and rh_c > 0.3 and analysis_data['bel_genisligi'] > 0:
            pt1 = (int(lh_x * width), int(lh_y * height))
            pt2 = (int(rh_x * width), int(rh_y * height))
            cv2.line(frame, pt1, pt2, (255, 0, 255), 3)
            
            mid_x = int((pt1[0] + pt2[0]) / 2)
            mid_y = int((pt1[1] + pt2[1]) / 2) + 20
            cv2.putText(frame, f"{analysis_data['bel_genisligi']:.1f}cm", 
                       (mid_x - 30, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        # Add measurement text
        cv2.putText(frame, f"Omuz: {analysis_data['omuz_genisligi']:.1f}cm", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Bel: {analysis_data['bel_genisligi']:.1f}cm", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Tip: {analysis_data['vucut_tipi']}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if analysis_data['omuz_bel_orani'] > 0:
            cv2.putText(frame, f"Oran: {analysis_data['omuz_bel_orani']:.2f}", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if analysis_data['mesafe'] > 0:
            cv2.putText(frame, f"Mesafe: {analysis_data['mesafe']:.1f}m", 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    except Exception as e:
        print(f"Drawing error: {e}")
    
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
        print(f"Depth visualization error: {e}")
        # Return black frame on error
        return np.zeros_like(frame)

def detect_camera_type():
    """Detect available camera type and return appropriate mode"""
    global camera_mode
    
    # First try RealSense
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
    
    # Fallback to webcam
    print("📹 Normal webcam modu kullanılacak")
    camera_mode = "webcam"
    return True

def stream_frames():
    """Stream frames with automatic camera detection"""
    global streaming, camera, realsense_pipeline, camera_mode
    
    try:
        # Detect camera type
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
    """Stream from RealSense camera"""
    global streaming, realsense_pipeline
    
    try:
        # Configure RealSense
        realsense_pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # Start pipeline
        profile = realsense_pipeline.start(config)
        depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        
        print("✅ RealSense kamera başlatıldı")
        
        frame_count = 0
        last_time = time.time()
        
        while streaming:
            try:
                frames = realsense_pipeline.wait_for_frames(timeout_ms=5000)
                
                align = rs.align(rs.stream.color)
                aligned_frames = align.process(frames)
                
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    continue
                
                # Convert to numpy
                color_image = np.asanyarray(color_frame.get_data())
                color_image = cv2.flip(color_image, 1)
                
                # Run pose detection
                keypoints = run_movenet(color_image)
                
                # Analyze measurements
                analysis_data = estimate_body_measurements_realsense(
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
                
                # Encode and send
                _, buffer = cv2.imencode('.jpg', combined_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                
                socketio.emit('video_frame', {'type': 'video_frame', 'frame': img_base64})
                socketio.emit('analyze_result', {'type': 'analyze_result', 'data': analysis_data})
                
                # FPS control
                frame_count += 1
                current_time = time.time()
                if current_time - last_time >= 1.0:
                    print(f"📊 RealSense FPS: {frame_count}")
                    frame_count = 0
                    last_time = current_time
                
                socketio.sleep(0.033)
                
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
    """Stream from regular webcam"""
    global streaming, camera
    
    try:
        # Find working camera
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
        
        # Open camera
        camera = cv2.VideoCapture(working_camera_index)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)
        
        print("✅ Webcam başlatıldı")
        
        frame_count = 0
        last_time = time.time()
        
        while streaming:
            try:
                ret, frame = camera.read()
                if not ret:
                    continue
                
                # Mirror frame
                frame = cv2.flip(frame, 1)
                
                # Run pose detection
                keypoints = run_movenet(frame)
                
                # Analyze measurements
                analysis_data = estimate_body_measurements_webcam(keypoints, frame.shape)
                
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
                
                # Encode and send
                _, buffer = cv2.imencode('.jpg', combined_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                
                socketio.emit('video_frame', {'type': 'video_frame', 'frame': img_base64})
                socketio.emit('analyze_result', {'type': 'analyze_result', 'data': analysis_data})
                
                # FPS control
                frame_count += 1
                current_time = time.time()
                if current_time - last_time >= 1.0:
                    print(f"📊 Webcam FPS: {frame_count}")
                    frame_count = 0
                    last_time = current_time
                
                socketio.sleep(0.033)
                
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
    print("🚀 Starting Hybrid Body Analysis Server...")
    print("📋 Features:")
    print("   - Automatic camera detection (RealSense/Webcam)")
    print("   - Clean RGB video display")
    print("   - Accurate distance measurements")
    print("   - MoveNet pose detection")
    print("   - Real-time body analysis")
    print("   - WebSocket communication")
    print()
    
    if REALSENSE_AVAILABLE:
        print("✅ RealSense support: Available")
    else:
        print("⚠️ RealSense support: Not available (webcam only)")
    
    print()
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)