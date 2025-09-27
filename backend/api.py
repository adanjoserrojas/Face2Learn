# Face2Learn Backend API
# This Flask API provides emotion detection services for the Face2Learn Chrome extension
# It uses a pre-trained CNN model to detect emotions in facial images

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import base64
import io
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import os

# Initialize Flask app with CORS enabled for Chrome extension
app = Flask(__name__)
CORS(app)  # Enable CORS for Chrome extension communication

def create_model():
    """
    Creates and loads the emotion detection CNN model.
    
    Model Architecture:
    - Input: 48x48x1 grayscale images
    - 4 Convolutional layers with ReLU activation
    - 3 MaxPooling layers for downsampling
    - 2 Dropout layers for regularization
    - 2 Dense layers for classification
    - Output: 7 emotion classes (softmax)
    
    Returns:
        Sequential: Loaded Keras model with pre-trained weights
    """
    model = Sequential()
    
    # First convolutional block
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))  # Prevent overfitting

    # Second convolutional block
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))  # Prevent overfitting

    # Classification layers
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))  # High dropout for regularization
    model.add(Dense(7, activation='softmax'))  # 7 emotion classes
    
    # Load pre-trained weights
    model.load_weights('model.h5')
    return model

# Initialize model and face detection cascade
model = create_model()
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # Haar cascade for face detection
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

def process_image(image_data):
    """
    Process a base64 encoded image and return emotion predictions with bounding boxes.
    
    Args:
        image_data (str): Base64 encoded image data (format: "data:image/jpeg;base64,...")
    
    Returns:
        list: List of dictionaries containing emotion detection results for each face found
              Each result includes face_id, emotion, confidence, and bounding box coordinates
    """
    try:
        # Decode base64 image data
        # Split on comma to remove data URL prefix and get just the base64 data
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_bytes))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale for face detection (Haar cascade works better on grayscale)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces using Haar cascade classifier with more conservative settings
        # scaleFactor: How much the image size is reduced at each scale (1.1 = 10% reduction)
        # minNeighbors: How many neighbors each candidate rectangle should have to retain it
        # minSize/maxSize: Filter out unreasonably small or large detections
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1,     # More conservative scaling
            minNeighbors=8,      # Higher neighbor requirement (was 6)
            minSize=(30, 30),    # Minimum face size
            maxSize=(300, 300),  # Maximum face size
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        results = []
        for i, (x, y, w, h) in enumerate(faces):
            # Extract face region of interest (ROI) from grayscale image
            roi_gray = gray[y:y + h, x:x + w]
            
            # Resize face to 48x48 pixels (model input size) and add batch dimension
            # Model expects shape: (batch_size, 48, 48, 1)
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            
            # Predict emotion using the trained model
            prediction = model.predict(cropped_img, verbose=0)
            maxindex = int(np.argmax(prediction))  # Get index of highest probability
            emotion = emotion_dict[maxindex]       # Convert index to emotion name
            confidence = float(prediction[0][maxindex])  # Get confidence score
            
            # Create result object with all necessary data for the frontend
            result = {
                'face_id': i,  # Unique identifier for this face
                'emotion': emotion,  # Detected emotion name
                'confidence': confidence,  # Confidence score (0-1)
                'bounding_box': {
                    'x': int(x),      # Top-left x coordinate
                    'y': int(y),      # Top-left y coordinate
                    'width': int(w),  # Width of bounding box
                    'height': int(h)  # Height of bounding box
                },
                # Additional coordinates for UI positioning
                'label_position': {
                    'x': int(x + 20),  # X position for emotion label
                    'y': int(y - 10)   # Y position for emotion label
                },
                'rectangle_coords': {
                    'top_left': {'x': int(x), 'y': int(y - 50)},
                    'bottom_right': {'x': int(x + w), 'y': int(y + h + 10)}
                }
            }
            results.append(result)
        
        return results
    
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        return []
    
@app.route('/detect_emotions', methods=['POST'])
def detect_emotions():
    """
    Main API endpoint for emotion detection in images.
    
    This endpoint receives a base64 encoded image from the Chrome extension,
    processes it to detect faces and emotions, and returns the results.
    
    Expected JSON payload:
        {
            "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."
        }
    
    Returns:
        JSON response with emotion detection results:
        {
            "success": true,
            "faces_detected": 2,
            "results": [
                {
                    "face_id": 0,
                    "emotion": "Happy",
                    "confidence": 0.85,
                    "bounding_box": {"x": 100, "y": 150, "width": 120, "height": 120},
                    ...
                }
            ]
        }
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Validate that image data is provided
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Process the image and detect emotions
        results = process_image(data['image'])
        
        # Prepare response
        response = {
            'success': True,
            'faces_detected': len(results),
            'results': results
        }
        
        return jsonify(response)
    
    except Exception as e:
        # Return error response with details
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify API is running.
    
    Returns:
        JSON response indicating API status
    """
    return jsonify({'status': 'healthy', 'message': 'Emotion detection API is running'})


@app.route('/test_emotions', methods=['GET'])
def test_emotions():
    """
    Test endpoint that returns varied sample emotion data for testing functionality.
    
    This endpoint simulates different emotion detection scenarios for testing
    the Chrome extension's UI and animation capabilities. It cycles through
    different test cases every 10 seconds to provide varied test data.
    
    Test Cases:
        1. moving_face: Face moves in a circular pattern
        2. size_changing: Face size changes over time (zoom effect)
        3. multiple_faces: Multiple faces with different emotions
        4. emotion_cycle: Cycles through all 7 emotions
        5. pulsing: Face pulses in and out rhythmically
        6. wandering: Face wanders randomly with changing emotions
    
    Returns:
        JSON response with simulated emotion detection results
    """
    import random
    import time
    import math
    
    # Get current time for animation patterns
    current_time = time.time()
    
    # Different test cases that cycle every 10 seconds
    test_cases = [
        "moving_face",      # Face moves around the screen in circular pattern
        "size_changing",    # Face size changes (zoom in/out effect)
        "multiple_faces",   # Multiple faces with different emotions
        "emotion_cycle",    # Cycles through different emotions
        "pulsing",          # Face pulses in size
        "wandering"         # Face wanders around randomly
    ]
    
    # Select test case based on time (changes every 10 seconds)
    case_index = int(current_time / 10) % len(test_cases)
    test_case = test_cases[case_index]
    
    # Available emotions for testing
    emotions = ["Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad", "Surprised"]
    
    results = []
    
    if test_case == "moving_face":
        # Test case 1: Face moves in a circular pattern around the screen
        center_x, center_y = 320, 240  # Center of the screen
        radius = 100  # Radius of circular motion
        angle = current_time * 0.5  # Slow rotation speed
        
        # Calculate circular position using trigonometry
        x = int(center_x + radius * math.cos(angle))
        y = int(center_y + radius * math.sin(angle))
        width, height = 120, 120
        emotion = "Happy"
        confidence = 0.85
        
        results.append({
            'face_id': 0,
            'emotion': emotion,
            'confidence': confidence,
            'bounding_box': {'x': x, 'y': y, 'width': width, 'height': height}
        })
        
    elif test_case == "size_changing":
        # Test case 2: Face size changes over time (zoom in/out effect)
        base_size = 100
        size_variation = int(50 * math.sin(current_time * 2))  # Size oscillates
        width = height = base_size + size_variation
        
        # Center the face as it changes size
        x = 300 - width // 2
        y = 200 - height // 2
        emotion = "Surprised"
        confidence = 0.78
        
        results.append({
            'face_id': 0,
            'emotion': emotion,
            'confidence': confidence,
            'bounding_box': {'x': x, 'y': y, 'width': width, 'height': height}
        })
        
    elif test_case == "multiple_faces":
        # Test case 3: Multiple faces with different emotions
        face_positions = [
            (150, 150, "Happy"),    # Top-left face
            (400, 200, "Sad"),      # Top-right face
            (300, 350, "Angry"),    # Bottom-center face
            (500, 100, "Neutral")   # Far right face
        ]
        
        for i, (x, y, emotion) in enumerate(face_positions):
            # Add subtle movement to each face using different phase offsets
            x += int(20 * math.sin(current_time + i))
            y += int(15 * math.cos(current_time + i))
            width = height = 100 + random.randint(-20, 20)  # Random size variation
            confidence = round(random.uniform(0.7, 0.9), 2)
            
            results.append({
                'face_id': i,
                'emotion': emotion,
                'confidence': confidence,
                'bounding_box': {'x': x, 'y': y, 'width': width, 'height': height}
            })
            
    elif test_case == "emotion_cycle":
        # Test case 4: Cycles through all 7 emotions
        emotion_index = int(current_time * 0.5) % len(emotions)  # Change emotion every 2 seconds
        emotion = emotions[emotion_index]
        confidence = 0.8
        
        # Face moves slightly while changing emotions
        x = 300 + int(30 * math.sin(current_time))
        y = 200 + int(20 * math.cos(current_time))
        width = height = 110
        
        results.append({
            'face_id': 0,
            'emotion': emotion,
            'confidence': confidence,
            'bounding_box': {'x': x, 'y': y, 'width': width, 'height': height}
        })
        
    elif test_case == "pulsing":
        # Test case 5: Face pulses in and out rhythmically
        pulse_factor = 0.5 + 0.5 * math.sin(current_time * 3)  # Pulse between 0.5 and 1.0
        base_size = 100
        width = height = int(base_size * pulse_factor)
        
        # Center the pulsing face
        x = 300 - width // 2
        y = 200 - height // 2
        emotion = "Fearful"
        confidence = 0.75
        
        results.append({
            'face_id': 0,
            'emotion': emotion,
            'confidence': confidence,
            'bounding_box': {'x': x, 'y': y, 'width': width, 'height': height}
        })
        
    elif test_case == "wandering":
        # Test case 6: Face wanders around randomly with changing emotions
        # Complex movement pattern using multiple sine waves
        x = int(200 + 200 * math.sin(current_time * 0.3) + 100 * math.cos(current_time * 0.7))
        y = int(150 + 150 * math.cos(current_time * 0.4) + 80 * math.sin(current_time * 0.6))
        width = height = 90 + int(30 * math.sin(current_time * 1.5))  # Size also varies
        emotion = random.choice(emotions)  # Random emotion each time
        confidence = round(random.uniform(0.6, 0.9), 2)
        
        results.append({
            'face_id': 0,
            'emotion': emotion,
            'confidence': confidence,
            'bounding_box': {'x': x, 'y': y, 'width': width, 'height': height}
        })
    
    # Log the current test case for debugging
    print(f"API called at {time.strftime('%H:%M:%S')} - Test case: {test_case} - {len(results)} faces")
    
    # Return the test data in the same format as the real emotion detection
    sample_data = {
        'success': True,
        'faces_detected': len(results),
        'results': results
    }
    return jsonify(sample_data)


if __name__ == '__main__':
    """
    Main entry point for the Flask application.
    
    Starts the emotion detection API server on all interfaces (0.0.0.0)
    on port 5001 with debug mode enabled for development.
    """
    print("Starting Emotion Detection API...")
    print("Model loaded successfully!")
    print("API endpoints available:")
    print("  - POST /detect_emotions - Main emotion detection endpoint")
    print("  - GET  /test_emotions   - Test endpoint with simulated data")
    print("  - GET  /health          - Health check endpoint")
    print("Server starting on http://0.0.0.0:5001")
    
    # Start Flask development server
    app.run(host='0.0.0.0', port=5001, debug=True)