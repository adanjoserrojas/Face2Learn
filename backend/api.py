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
import google.generativeai as genai
from config import GEMINI_API_KEY, CONFIDENCE_THRESHOLD, MAX_FACES

# Disable MTCNN for now to focus on DNN
USE_MTCNN = False
print("MTCNN disabled - focusing on OpenCV DNN face detection")

# Check for OpenCV DNN face detection model files
import urllib.request
DNN_MODEL_FILES = {
    'pbtxt': 'opencv_face_detector.pbtxt',
    'pb': 'opencv_face_detector_fp16.pb'
}

# URLs for downloading DNN models if not present  
DNN_MODEL_URLS = {
    'pbtxt': 'https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/face_detector/opencv_face_detector.pbtxt',
    'pb': 'https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20180205_fp16/opencv_face_detector_fp16.pb'
}

# Check if DNN models are available
USE_DNN = True
for file_type, filename in DNN_MODEL_FILES.items():
    if not os.path.exists(filename):
        print(f"DNN model file {filename} not found. Downloading...")
        try:
            print(f"Downloading from: {DNN_MODEL_URLS[file_type]}")
            urllib.request.urlretrieve(DNN_MODEL_URLS[file_type], filename)
            print(f"Successfully downloaded {filename}")
            # Verify file size
            file_size = os.path.getsize(filename)
            print(f"Downloaded file size: {file_size} bytes")
            if file_size < 1000:  # Very small file indicates download failure
                print(f"Warning: {filename} seems too small, might be corrupted")
        except Exception as e:
            print(f"Failed to download {filename}: {e}")
            USE_DNN = False
            break

if USE_DNN:
    print("OpenCV DNN face detection available")
else:
    print("OpenCV DNN face detection not available - using Haar cascade")

# For now, let's disable DNN due to compatibility issues and use optimized Haar cascade
USE_DNN = False
print("DNN temporarily disabled - using optimized Haar cascade detection")

# Initialize Flask app with CORS enabled for Chrome extension
app = Flask(__name__)
CORS(app)  # Enable CORS for Chrome extension communication

# Configure Gemini AI
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-pro')

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

# MTCNN disabled - focusing on DNN
mtcnn_detector = None

# Initialize OpenCV DNN detector if available
if USE_DNN:
    try:
        dnn_net = cv2.dnn.readNetFromTensorflow(DNN_MODEL_FILES['pb'], DNN_MODEL_FILES['pbtxt'])
        print("OpenCV DNN detector initialized successfully")
    except Exception as e:
        print(f"Failed to initialize DNN detector: {e}")
        USE_DNN = False
        dnn_net = None
else:
    dnn_net = None

# OpenCV DNN face detection function
def detect_faces_dnn(image):
    """
    Detect faces using OpenCV DNN face detection
    
    Args:
        image: BGR image (numpy array)
        
    Returns:
        list: List of (x, y, w, h) face rectangles with high confidence
    """
    try:
        h, w = image.shape[:2]
        
        # Create blob from image
        # Input size for the DNN model is 300x300
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123])
        
        # Set input to the network
        dnn_net.setInput(blob)
        
        # Run forward pass to get detections
        detections = dnn_net.forward()
        
        face_boxes = []
        
        # Process detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            # Filter out weak detections
            if confidence > 0.7:  # Confidence threshold
                # Get bounding box coordinates
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                
                # Convert to (x, y, width, height) format
                x = max(0, x1)
                y = max(0, y1)
                width = min(x2 - x1, w - x)
                height = min(y2 - y1, h - y)
                
                if width > 0 and height > 0:
                    face_boxes.append((x, y, width, height))
                    print(f"DNN face detected with confidence: {confidence:.2f} at ({x}, {y}, {width}, {height})")
        
        print(f"OpenCV DNN detected {len(face_boxes)} faces (confidence > 0.7)")
        return face_boxes
        
    except Exception as e:
        print(f"DNN detection error: {e}")
        return []

# MTCNN face detection function
def detect_faces_mtcnn(rgb_image):
    """
    Detect faces using MTCNN neural network
    
    Args:
        rgb_image: RGB image (numpy array)
        
    Returns:
        list: List of (x, y, w, h) face rectangles with high confidence
    """
    try:
        faces = mtcnn_detector.detect_faces(rgb_image)
        face_boxes = []
        
        for face in faces:
            x, y, w, h = face['box']
            confidence = face['confidence']
            
            # Lower confidence threshold for better detection
            print(f"MTCNN face detected with confidence: {confidence}")
            if confidence > 0.7:  # Lowered from 0.8 to 0.7
                # Ensure coordinates are positive and within image bounds
                x = max(0, x)
                y = max(0, y)
                w = min(w, rgb_image.shape[1] - x)
                h = min(h, rgb_image.shape[0] - y)
                
                if w > 0 and h > 0:  # Valid face box
                    face_boxes.append((x, y, w, h))
                    print(f"Added face: ({x}, {y}, {w}, {h})")
        
        print(f"MTCNN detected {len(face_boxes)} faces (confidence > 0.7)")
        return face_boxes
        
    except Exception as e:
        print(f"MTCNN detection error: {e}")
        return []

# Enhanced Haar cascade face detection function
def detect_faces_optimized(gray_image):
    """
    Enhanced Haar cascade face detection with multiple approaches
    
    Args:
        gray_image: Grayscale image for face detection
        
    Returns:
        list: List of (x, y, w, h) face rectangles
    """
    
    # Try multiple detection parameters for better results
    detection_params = [
        # Standard parameters
        {'scaleFactor': 1.1, 'minNeighbors': 10, 'minSize': (50, 30), 'maxSize': (300, 300)},
        # More sensitive parameters
        {'scaleFactor': 1.1, 'minNeighbors': 8, 'minSize': (50, 30), 'maxSize': (400, 400)},
        # Very sensitive parameters
        {'scaleFactor': 1.05, 'minNeighbors': 6, 'minSize': (50, 30), 'maxSize': (500, 500)},
    ]
    
    all_faces = []
    
    for params in detection_params:
        faces = face_cascade.detectMultiScale(
            gray_image,
            scaleFactor=params['scaleFactor'],
            minNeighbors=params['minNeighbors'],
            minSize=params['minSize'],
            maxSize=params['maxSize'],
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) > 0:
            print(f"Haar cascade detected {len(faces)} faces with params: {params}")
            all_faces.extend(faces)
            break  # Use first successful detection
    
    # Remove duplicate detections
    if len(all_faces) > 1:
        # Simple duplicate removal based on overlap
        unique_faces = []
        for face in all_faces:
            x, y, w, h = face
            is_duplicate = False
            for unique_face in unique_faces:
                ux, uy, uw, uh = unique_face
                # Check if faces overlap significantly
                overlap_x = max(0, min(x + w, ux + uw) - max(x, ux))
                overlap_y = max(0, min(y + h, uy + uh) - max(y, uy))
                overlap_area = overlap_x * overlap_y
                face_area = w * h
                if overlap_area > 0.5 * face_area:  # 50% overlap threshold
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_faces.append(face)
        all_faces = unique_faces
    
    return list(all_faces)

def get_fallback_educational_content(emotion, confidence):
    """
    Generate fallback educational content when Gemini API is not available.
    """
    fallback_content = {
        "Happy": f"I can see you're feeling positive about learning (confidence: {confidence:.1%}). Your expression suggests you're engaged and enjoying the learning process. This is an excellent state for absorbing new information. Try tackling more challenging concepts while you're in this positive mood, or help others learn by explaining what you know! Consider your current learning environment - is it comfortable and well-lit? Make sure you're taking advantage of this positive energy!",
        
        "Sad": f"I notice you might be feeling a bit down (confidence: {confidence:.1%}). Learning can be challenging sometimes, and it's okay to feel this way. Your expression suggests you might be struggling with the material or feeling overwhelmed. Consider taking a short break, reviewing easier material to build confidence, or reaching out to a study partner for support. Check your learning environment - sometimes a change of scenery or better lighting can help improve your mood.",
        
        "Angry": f"It looks like you might be feeling frustrated (confidence: {confidence:.1%}). This is completely normal when learning something difficult. Your expression suggests you're hitting a learning obstacle. Try breaking the material into smaller chunks, taking deep breaths, or switching to a different topic for a while. Remember, every expert was once a beginner! Consider your study setup - are you comfortable? Sometimes adjusting your posture or taking a short walk can help reset your mindset.",
        
        "Fearful": f"I can see some uncertainty in your expression (confidence: {confidence:.1%}). It's natural to feel apprehensive about new concepts. Your expression suggests you might be worried about not understanding something or making mistakes. Start with the basics, ask questions, and remember that making mistakes is part of the learning process. You've got this! Make sure you're in a comfortable, distraction-free environment that helps you feel safe to learn.",
        
        "Surprised": f"Wow! You look surprised (confidence: {confidence:.1%}). This suggests you've just discovered something unexpected or new! This is one of the best moments in learning. Your expression shows genuine curiosity and engagement. Take time to explore this discovery further and see what other connections you can make. This is a perfect time to take notes or discuss what you've learned with someone else!",
        
        "Disgusted": f"It seems like this content might not be resonating with you (confidence: {confidence:.1%}). That's okay! Your expression suggests you're not connecting with the current material or approach. Try a different approach, find alternative resources, or take a break and come back with fresh eyes. Learning should feel engaging, not forced. Consider whether your learning environment is optimal - sometimes a change of setting can make a big difference.",
        
        "Neutral": f"You appear focused and attentive (confidence: {confidence:.1%}). This is an ideal learning state! Your expression shows you're ready to absorb new information. Try to maintain this concentration and consider taking notes to reinforce what you're learning. Your current learning environment seems to be working well for you - keep up the good work!"
    }
    
    return fallback_content.get(emotion, f"You're showing a {emotion.lower()} expression (confidence: {confidence:.1%}). This is a normal part of the learning process. Take a moment to reflect on how you're feeling and adjust your learning approach accordingly. Consider your current environment and whether it's supporting your learning goals.")

def generate_educational_prompt_from_image(image_data, emotion, confidence, context=""):
    """
    Generate educational prompts based on the actual captured image using Gemini AI.
    
    Args:
        image_data (str): Base64 encoded image data
        emotion (str): Detected emotion name
        confidence (float): Confidence score of the emotion detection
        context (str): Additional context about the learning situation
    
    Returns:
        dict: Educational prompt with title, content, and suggestions
    """
    try:
        # Convert base64 to PIL Image for Gemini
        if ',' in image_data:
            image_bytes = base64.b64decode(image_data.split(',')[1])
        else:
            image_bytes = base64.b64decode(image_data)
        
        # Try to open the image with PIL
        try:
            image = Image.open(io.BytesIO(image_bytes))
            # Convert to RGB if necessary (handles RGBA, P, etc.)
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as pil_error:
            print(f"PIL error in Gemini function: {pil_error}")
            # Fallback: try to decode directly with OpenCV and convert back to PIL
            nparr = np.frombuffer(image_bytes, np.uint8)
            cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if cv_image is None:
                raise ValueError("Could not decode image for Gemini analysis")
            # Convert OpenCV BGR to RGB for PIL
            cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(cv_image_rgb)
        
        # Create a comprehensive prompt for Gemini with the image
        prompt = f"""
        You are an educational AI assistant helping learners understand and manage their emotions during learning.
        
        Look at this image carefully. I can see their facial expression shows "{emotion}" emotion with {confidence:.1%} confidence.
        Additional context: {context if context else "General learning environment"}
        
        Please analyze this image and provide:
        1. A detailed description of what you see in the scene - describe the setting, lighting, what the person is doing, their posture, and any other relevant details
        2. An explanation of what this emotion and expression might indicate about their learning state, considering the scene context
        3. 2-3 specific learning strategies or activities they could try based on this emotional state and scene
        4. Encouragement or motivation appropriate for this emotional state and learning situation
        5. A short, actionable tip they can implement right now based on what you observe
        
        Be specific about what you observe in the scene and how it relates to their learning experience. Consider the environment, their setup, and how it might be affecting their learning.
        Format your response as a helpful, supportive educational guide. Keep it concise but meaningful (under 250 words).
        """
        
        # Use Gemini Pro Vision model for image analysis
        try:
            vision_model = genai.GenerativeModel('gemini-pro-vision')
            response = vision_model.generate_content([prompt, image])
            
            # Parse the response and structure it
            content = response.text.strip()
        except Exception as gemini_error:
            print(f"Gemini API error: {gemini_error}")
            # Use fallback content based on emotion
            content = get_fallback_educational_content(emotion, confidence)
        
        # Create a structured response
        educational_prompt = {
            "emotion": emotion,
            "confidence": confidence,
            "title": f"Learning Guidance - {emotion} Expression",
            "content": content,
            "timestamp": __import__('time').time(),
            "analysis_type": "image_based"
        }
        
        return educational_prompt
        
    except Exception as e:
        print(f"Error generating educational prompt from image: {str(e)}")
        # Use fallback content when Gemini fails
        content = get_fallback_educational_content(emotion, confidence)
        
        return {
            "emotion": emotion,
            "confidence": confidence,
            "title": f"Learning Guidance - {emotion} Expression",
            "content": content,
            "timestamp": __import__('time').time(),
            "analysis_type": "fallback"
        }

def generate_educational_prompt_text_only(emotion, confidence, context=""):
    """
    Fallback function to generate educational prompts based on emotion text only.
    
    Args:
        emotion (str): Detected emotion name
        confidence (float): Confidence score of the emotion detection
        context (str): Additional context about the learning situation
    
    Returns:
        dict: Educational prompt with title, content, and suggestions
    """
    try:
        # Create a comprehensive prompt for Gemini
        prompt = f"""
        You are an educational AI assistant helping learners understand and manage their emotions during learning.
        
        Context: A learner is watching educational content and their facial expression shows "{emotion}" emotion with {confidence:.1%} confidence.
        Additional context: {context if context else "General learning environment"}
        
        Please provide:
        1. A brief explanation of what this emotion might indicate about their learning state
        2. 2-3 specific learning strategies or activities they could try based on this emotion
        3. Encouragement or motivation appropriate for this emotional state
        4. A short, actionable tip they can implement right now
        
        Format your response as a helpful, supportive educational guide. Keep it concise but meaningful (under 200 words).
        """
        
        response = model.generate_content(prompt)
        
        # Parse the response and structure it
        content = response.text.strip()
        
        # Create a structured response
        educational_prompt = {
            "emotion": emotion,
            "confidence": confidence,
            "title": f"Learning Guidance - {emotion} Expression",
            "content": content,
            "timestamp": __import__('time').time(),
            "analysis_type": "text_based"
        }
        
        return educational_prompt
        
    except Exception as e:
        print(f"Error generating educational prompt: {str(e)}")
        # Use fallback content when Gemini fails
        content = get_fallback_educational_content(emotion, confidence)
        return {
            "emotion": emotion,
            "confidence": confidence,
            "title": f"Learning Guidance - {emotion} Expression",
            "content": content,
            "timestamp": __import__('time').time(),
            "error": "AI prompt generation failed, showing fallback message",
            "analysis_type": "fallback"
        }

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
        
        # Enhance image preprocessing for better face detection
        # Apply histogram equalization to improve contrast
        gray = cv2.equalizeHist(gray)
        
        # Use different detection methods based on availability (in order of preference)
        faces = []
        detection_method = ""
        
        # Try OpenCV DNN first if available
        if USE_DNN and dnn_net is not None:
            try:
                faces = detect_faces_dnn(frame)
                detection_method = "OpenCV DNN"
                
                if len(faces) == 0:
                    print("OpenCV DNN found no faces, falling back to Haar cascade")
                    faces = detect_faces_optimized(gray)
                    detection_method = "Haar cascade (DNN fallback)"
                    
            except Exception as e:
                print(f"OpenCV DNN failed: {e}, falling back to Haar cascade")
                faces = detect_faces_optimized(gray)
                detection_method = "Haar cascade (DNN failed)"
        
        # Final fallback to Haar cascade
        else:
            faces = detect_faces_optimized(gray)
            detection_method = "Haar cascade"

        
        print(f"Detected {len(faces)} faces using {detection_method}")
        
        if len(faces) == 0:
            print("No faces detected - trying fallback with lower thresholds")
            
            # Try MTCNN with lower threshold
            if USE_MTCNN and mtcnn_detector is None:
                try:
                    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    fallback_faces = mtcnn_detector.detect_faces(rgb_image)
                    print(f"MTCNN fallback found {len(fallback_faces)} faces with any confidence")
                    for face in fallback_faces:
                        conf = face['confidence']
                        print(f"  MTCNN Face confidence: {conf}")
                        if conf > 0.5:  # Very low threshold
                            x, y, w, h = face['box']
                            faces.append((x, y, w, h))
                    print(f"MTCNN fallback added {len(faces)} faces with confidence > 0.5")
                except Exception as e:
                    print(f"MTCNN fallback detection failed: {e}")
            
            # If still no faces, try DNN with lower threshold
            if len(faces) == 0 and USE_DNN and dnn_net is not None:
                try:
                    print("Trying DNN fallback with lower confidence threshold")
                    h, w = frame.shape[:2]
                    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123])
                    dnn_net.setInput(blob)
                    detections = dnn_net.forward()
                    
                    for i in range(detections.shape[2]):
                        confidence = detections[0, 0, i, 2]
                        print(f"  DNN Face confidence: {confidence}")
                        
                        if confidence > 0.5:  # Lower threshold for fallback
                            x1 = int(detections[0, 0, i, 3] * w)
                            y1 = int(detections[0, 0, i, 4] * h)
                            x2 = int(detections[0, 0, i, 5] * w)
                            y2 = int(detections[0, 0, i, 6] * h)
                            
                            x = max(0, x1)
                            y = max(0, y1)
                            width = min(x2 - x1, w - x)
                            height = min(y2 - y1, h - y)
                            
                            if width > 0 and height > 0:
                                faces.append((x, y, width, height))
                    
                    print(f"DNN fallback added {len(faces)} faces with confidence > 0.5")
                except Exception as e:
                    print(f"DNN fallback detection failed: {e}")
        
        results = []
        for i, (x, y, w, h) in enumerate(faces):
            # Extract face region of interest (ROI) from grayscale image
            roi_gray = gray[y:y + h, x:x + w]
            
            # Resize face to 48x48 pixels (model input size) and add batch dimension
            # Model expects shape: (batch_size, 48, 48, 1)
            resized_roi = cv2.resize(roi_gray, (48, 48))
            cropped_img = np.expand_dims(np.expand_dims(resized_roi, -1), 0)
            
            # Ensure correct data type and normalization
            cropped_img = cropped_img.astype('float32') / 255.0
            
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
                },
                # Border display timing
                'timestamp': __import__('time').time(),  # Current timestamp
                'display_duration': 3000,  # Show border for 3 seconds (3000ms)
                'fade_duration': 500  # Fade out animation duration (500ms)
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
        
        # Note: Educational prompts are now only generated when user clicks the button
        # This saves API calls and improves performance
        
        # Prepare response
        response = {
            'success': True,
            'faces_detected': len(results),
            'results': results,
            'educational_prompts': []  # Empty array - prompts generated on-demand
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

@app.route('/generate_prompt', methods=['POST'])
def generate_prompt():
    """
    Generate educational prompt based on emotion data.
    
    Expected JSON payload:
        {
            "emotion": "Happy",
            "confidence": 0.85,
            "context": "Learning math concepts"
        }
    
    Returns:
        JSON response with educational prompt
    """
    try:
        data = request.get_json()
        
        if not data or 'emotion' not in data:
            return jsonify({'error': 'Emotion data is required'}), 400
        
        emotion = data.get('emotion')
        confidence = data.get('confidence', 0.5)
        context = data.get('context', '')
        
        prompt = generate_educational_prompt_text_only(emotion, confidence, context)
        
        return jsonify({
            'success': True,
            'prompt': prompt
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate_prompt_from_image', methods=['POST'])
def generate_prompt_from_image():
    """
    Generate educational prompt based on captured image using Gemini Vision.
    
    Expected JSON payload:
        {
            "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ...",
            "emotion": "Happy",
            "confidence": 0.85,
            "context": "Learning math concepts"
        }
    
    Returns:
        JSON response with educational prompt based on image analysis
    """
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'error': 'Image data is required'}), 400
        
        image = data.get('image')
        emotion = data.get('emotion', 'Neutral')
        confidence = data.get('confidence', 0.5)
        context = data.get('context', '')
        
        prompt = generate_educational_prompt_from_image(image, emotion, confidence, context)
        
        return jsonify({
            'success': True,
            'prompt': prompt
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


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
    print("  - POST /detect_emotions           - Main emotion detection endpoint with image-based prompts")
    print("  - POST /generate_prompt_from_image - Generate educational prompts from images using Gemini Vision")
    print("  - POST /generate_prompt           - Generate educational prompts from emotion text only")
    print("  - GET  /test_emotions             - Test endpoint with simulated data")
    print("  - GET  /health                    - Health check endpoint")
    print("Server starting on http://0.0.0.0:5001")
    
    # Start Flask development server
    app.run(host='0.0.0.0', port=5001, debug=True)