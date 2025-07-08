# Real-time Body Analysis System

A comprehensive real-time body analysis system using Intel RealSense cameras and TensorFlow's MoveNet model.

## Features

### Core Functionality
- **Real-time Pose Detection**: Uses MoveNet Lightning model for accurate pose estimation
- **3D Depth Measurements**: Calculates actual shoulder and waist widths using depth data
- **Body Type Classification**: Automatically classifies body type (Ectomorph, Mesomorph, Endomorph)
- **Live Video Streaming**: WebSocket-based real-time video feed with analysis overlays

### Technical Improvements
- **Advanced Depth Filtering**: Spatial, temporal, and hole-filling filters for accurate measurements
- **High Accuracy Mode**: RealSense preset optimized for precise measurements
- **Error Handling**: Comprehensive error handling and validation
- **Performance Optimization**: Efficient frame processing and controlled frame rates

## Installation

### Prerequisites
- Python 3.8+
- Intel RealSense Camera (D435i or similar)
- Intel RealSense SDK 2.0

### Setup
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Connect Intel RealSense camera

3. Run the server:
```bash
python app.py
```

## Configuration

### RealSense Settings
- **Visual Preset**: High Accuracy (preset 3)
- **Resolution**: 640x480 @ 30fps
- **Depth Filters**: Spatial, Temporal, Hole-filling, Decimation

### MoveNet Model
- **Model**: SinglePose Lightning
- **Input Size**: 192x192
- **Confidence Threshold**: 0.3

## API Reference

### WebSocket Events

#### Client to Server
- `start_video`: Start camera stream and analysis
- `stop_video`: Stop camera stream

#### Server to Client
- `video_frame`: Base64 encoded video frame
- `analyze_result`: Analysis data with measurements
- `stream_started`: Confirmation of stream start
- `stream_stopped`: Confirmation of stream stop

### Analysis Data Format
```json
{
  "omuz_genisligi": 45.2,
  "bel_genisligi": 32.1,
  "omuz_bel_orani": 1.41,
  "vucut_tipi": "Mezomorf",
  "mesafe": 1.8,
  "confidence": 0.85
}
```

## Troubleshooting

### Common Issues
1. **Camera not detected**: Ensure RealSense camera is properly connected and drivers are installed
2. **Inaccurate measurements**: Check lighting conditions and ensure person is within 1-3 meters
3. **Poor depth quality**: Verify camera lens is clean and person is facing the camera

### Performance Tips
- Ensure good lighting for better pose detection
- Maintain 1-3 meter distance from camera
- Keep person centered in frame
- Avoid rapid movements for stable measurements

## Technical Details

### Depth Processing Pipeline
1. **Frame Alignment**: Align depth and color frames
2. **Decimation Filter**: Reduce resolution for performance
3. **Spatial Filter**: Remove noise
4. **Temporal Filter**: Reduce flickering
5. **Hole Filling**: Fill missing depth data

### Body Type Classification
- **Ectomorph**: Shoulder/Waist ratio > 1.4
- **Mesomorph**: Shoulder/Waist ratio 1.2-1.4
- **Endomorph**: Shoulder/Waist ratio < 1.2

## License
MIT License