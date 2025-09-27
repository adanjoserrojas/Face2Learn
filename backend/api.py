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

app = Flask(__name__)
CORS(app) # Enable CORS for Chrome 

# Load model

def create_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    
    model.load_weights('model.h5')
    return model

# initialize model and face cascade
model = create_model()
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # removes background noise
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

def process_image(image_data):
    """
    Process a frame and return emotion predictions with bounding boxes
    """
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_bytes))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces | change scaleFactor and minNeighbors for performance tuning
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        results = []
        for i, (x, y, w, h) in enumerate(faces):
            # Extract face region
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            
            # Predict emotion
            prediction = model.predict(cropped_img, verbose=0)
            maxindex = int(np.argmax(prediction))
            emotion = emotion_dict[maxindex]
            confidence = float(prediction[0][maxindex])
            
            # Create result object with bounding box coordinates
            result = {
                'face_id': i,
                'emotion': emotion,
                'confidence': confidence,
                'bounding_box': {
                    'x': int(x),
                    'y': int(y),
                    'width': int(w),
                    'height': int(h)
                },
                # Additional coordinates for drawing
                'label_position': {
                    'x': int(x + 20),
                    'y': int(y - 10)
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
    API endpoint to detect emotions in an image
    Expects: JSON with 'image' field containing base64 encoded image
    Returns: JSON with emotion predictions and bounding box coordinates
    """
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Process the image
        results = process_image(data['image'])
        
        response = {
            'success': True,
            'faces_detected': len(results),
            'results': results
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'Emotion detection API is running'})

if __name__ == '__main__':
    print("Starting Emotion Detection API...")
    print("Model loaded successfully!")
    app.run(host='0.0.0.0', port=5001, debug=True)