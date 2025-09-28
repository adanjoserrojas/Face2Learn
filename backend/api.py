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
import time
import google.generativeai as genai
from config import GEMINI_API_KEY, CONFIDENCE_THRESHOLD, MAX_FACES
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


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

#Selector
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

# Load model + tokenizer once
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# If GPU is available, use it; else CPU
device = 0 if torch.cuda.is_available() else -1

classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    return_all_scores=True,
    device=device
)

@app.route("/", methods=["POST"])
def analyze():
    try:
        data = request.get_json()
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "No text provided"}), 400

        results = classifier(text)[0]  
        # results is a list of dicts: e.g.
        # [
        #   {"label": "anger", "score": 0.01},
        #   {"label": "joy",   "score": 0.80},
        #   ...
        # ]

        # Choose top-1
        top = max(results, key=lambda x: x["score"])

        return jsonify({
            "top_label": top["label"],
            "top_score": float(top["score"]),
            "all_scores": { r["label"]: float(r["score"]) for r in results }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500






# Configure Gemini AI
genai.configure(api_key=GEMINI_API_KEY)
genAIModel = genai.GenerativeModel('gemini-2.5-pro')

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

# Validate that the cascade classifier is properly loaded
if face_cascade.empty():
    print("ERROR: Failed to load Haar cascade classifier!")
    print("Please ensure 'haarcascade_frontalface_default.xml' exists in the backend directory")
    # Try alternative cascade file
    face_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')
    if face_cascade.empty():
        print("ERROR: Alternative cascade also failed to load!")
        print("Face detection will be severely limited")
    else:
        print("Loaded alternative profile face cascade")
else:
    print("Haar cascade classifier loaded successfully")

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# MTCNN disabled - focusing on DNN
mtcnn_detector = None


# OpenCV DNN face detection function


# MTCNN face detection function


# Enhanced Haar cascade face detection function
def detect_faces_optimized(gray_image):
    """
    Enhanced Haar cascade face detection with segmentation fault prevention
    
    Args:
        gray_image: Grayscale image for face detection
        
    Returns:
        list: List of (x, y, w, h) face rectangles
    """
    
    # Validate input image thoroughly
    if gray_image is None:
        print("Input image is None")
        return []
    
    if not isinstance(gray_image, np.ndarray):
        print(f"Input is not numpy array, got: {type(gray_image)}")
        return []
    
    if gray_image.size == 0:
        print("Input image has zero size")
        return []
    
    if len(gray_image.shape) != 2:
        print(f"Input image must be grayscale (2D), got shape: {gray_image.shape}")
        return []
    
    height, width = gray_image.shape
    if height <= 0 or width <= 0:
        print(f"Invalid image dimensions: {width}x{height}")
        return []
    
    # Check minimum image size
    if width < 50 or height < 50:
        print(f"Image too small for face detection: {width}x{height}")
        return []
    
    # Check if cascade classifier is loaded
    if face_cascade.empty():
        print("Haar cascade classifier not loaded, cannot detect faces")
        return []
    
    print(f"Processing image: {width}x{height}")
    
    # Ensure image data is valid and contiguous
    try:
        # Check if image data is contiguous in memory
        if not gray_image.flags.c_contiguous:
            print("Converting image to contiguous array")
            gray_image = np.ascontiguousarray(gray_image, dtype=np.uint8)
        
        # Ensure correct data type
        if gray_image.dtype != np.uint8:
            print(f"Converting image from {gray_image.dtype} to uint8")
            # Normalize to 0-255 range if needed
            if gray_image.max() <= 1.0:
                gray_image = (gray_image * 255).astype(np.uint8)
            else:
                gray_image = gray_image.astype(np.uint8)
        
        # Validate final image
        if gray_image is None or gray_image.size == 0:
            print("Image became invalid after processing")
            return []
            
    except Exception as e:
        print(f"Error preparing image data: {e}")
        return []
    
    # Try multiple detection approaches quickly
    all_faces = []
    
    # Approach 1: Standard detection
    try:
        print("Trying standard detection")
        faces1 = face_cascade.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(50, 50),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        if len(faces1) > 0:
            print(f"Standard detection found {len(faces1)} faces")
            all_faces.extend(faces1)
    except Exception as e:
        print(f"Standard detection failed: {e}")
    
    # Approach 2: More sensitive detection if no faces found
    if len(all_faces) == 0:
        try:
            print("Trying sensitive detection")
            faces2 = face_cascade.detectMultiScale(
                gray_image,
                scaleFactor=1.1,
                minNeighbors=2,
                minSize=(40, 40),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            if len(faces2) > 0:
                print(f"Sensitive detection found {len(faces2)} faces")
                all_faces.extend(faces2)
        except Exception as e:
            print(f"Sensitive detection failed: {e}")
    
    # Approach 3: Very sensitive detection as last resort
    if len(all_faces) == 0:
        try:
            print("Trying very sensitive detection")
            faces3 = face_cascade.detectMultiScale(
                gray_image,
                scaleFactor=1.05,
                minNeighbors=1,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            if len(faces3) > 0:
                print(f"Very sensitive detection found {len(faces3)} faces")
                all_faces.extend(faces3)
        except Exception as e:
            print(f"Very sensitive detection failed: {e}")
    
    if len(all_faces) == 0:
        print("All detection approaches failed to find faces")
        return []
    
    # Remove duplicate detections if any
    if len(all_faces) > 1:
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
                if overlap_area > 0.5 * face_area:  # 30% overlap threshold - reduced for better detection
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
        "Happy": f"You seem to be feeling positive (confidence: {confidence:.1%}). Your expression shows energy and engagement, which is an excellent state for learning. Use this momentum to explore challenging material or share your knowledge with others. Take note of your study environment too—if it feels comfortable and supportive, lean into that while your motivation is high.",
        
        "Sad": f"You may be feeling a bit down (confidence: {confidence:.1%}). This often happens when learning feels overwhelming or progress feels slow. Be kind to yourself—take a short break, revisit simpler concepts, or talk through the material with someone supportive. Small adjustments to your environment, like better lighting or a change of scenery, can also lift your mood.",
        
        "Angry": f"You look frustrated (confidence: {confidence:.1%}). Frustration is natural when learning is difficult. Try breaking the problem into smaller steps, switching topics briefly, or pausing for a reset. Remember, persistence matters more than perfection. Also check your setup—sometimes posture, noise, or distractions add to stress more than the material itself.",
        
        "Fearful": f"You might be feeling unsure or nervous (confidence: {confidence:.1%}). This suggests some worry about mistakes or not understanding fully. Focus on the basics, ask clear questions, and remember that mistakes are essential steps in learning. Make sure your environment feels safe and distraction-free so you can approach new ideas with confidence.",
        
        "Surprised": f"You look surprised (confidence: {confidence:.1%}). This usually means you’ve encountered something new or unexpected, which is a great spark for curiosity. Take a moment to explore the idea further, connect it to what you already know, or share your discovery. Capture this energy in notes or discussions while your attention is sharp.",
        
        "Disgusted": f"You seem uncomfortable with the material (confidence: {confidence:.1%}). This might mean the approach doesn’t suit your style or the content feels off-putting. Try switching resources, rephrasing the material, or stepping away briefly. Learning should feel engaging—sometimes adjusting your environment or study method makes a big difference.",
        
        "Neutral": f"You appear calm and attentive (confidence: {confidence:.1%}). This is a strong state for learning since your focus is steady and clear. Take advantage by practicing note-taking, summarizing key points, or applying ideas actively. Keep your environment consistent and distraction-free so you can maintain this balance."
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
        You are an educational AI assistant helping socially disabled understand and recognize the emotions of others.
        
        My very innacurrate and lightweight model as provided the following information: The main target's face shows "{emotion}" emotion.
        Additional context: {context if context else "General learning environment"}
        
        Please analyze this image and provide:
        1. A detailed description of what you see in the scene - describe the setting, lighting, what the person is doing, their posture, and any other relevant details that relate to helping teach about social cues and emotion.
        2. An explanation of what this emotion and expression might indicate about this person's mental state, considering the scene context to help the user understand key signals.
        3. A short, actionable tip they can implement right now based on what you observe and how to interact with this emotion.
        
        Be specific about what you observe in the scene and how it relates to their learning experience. Consider the environment, their setup, and how it might be affecting their learning.
        Format your response as a helpful, supportive educational guide. Keep it concise but meaningful (under 250 words) and within three non-complex sentences.
        """
        
        # Use Gemini Pro Vision model for image analysis
        try:
            vision_model = genai.GenerativeModel('gemini-2.5-pro')
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
        You are an educational AI assistant helping socially disabled understand and recognize the emotions of others.
        
        My very innacurrate and lightweight model as provided the following information: The main target's face shows "{emotion}" emotion.
        Additional context: {context if context else "General learning environment"}
        
        Please analyze this image and provide:
        1. A detailed description of what you see in the scene - describe the setting, lighting, what the person is doing, their posture, and any other relevant details that relate to helping teach about social cues and emotion.
        2. An explanation of what this emotion and expression might indicate about this person's mental state, considering the scene context to help the user understand key signals.
        3. A short, actionable tip they can implement right now based on what you observe and how to interact with this emotion.
        
        Be specific about what you observe in the scene and how it relates to their learning experience. Consider the environment, their setup, and how it might be affecting their learning.
        Format your response as a helpful, supportive educational guide. Keep it concise but meaningful (under 250 words) and within three non-complex sentences.
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
        
        # Validate frame dimensions
        if frame is None or frame.size == 0:
            print("Invalid frame data")
            return []
            
        height, width = frame.shape[:2]
        if height <= 0 or width <= 0:
            print(f"Invalid frame dimensions: {width}x{height}")
            return []
        
        print(f"Processing frame: {width}x{height}")
        
        # Convert to grayscale for face detection (Haar cascade works better on grayscale)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Validate grayscale image
        if gray is None or gray.size == 0:
            print("Failed to convert to grayscale")
            return []
        
        # Enhance image preprocessing for better face detection
        # Apply histogram equalization to improve contrast
        try:
            gray = cv2.equalizeHist(gray)
        except Exception as e:
            print(f"Histogram equalization failed: {e}")
            # Continue with original grayscale image
        
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
            try:
                faces = detect_faces_optimized(gray)
                detection_method = "Haar cascade"
            except Exception as e:
                print(f"All face detection methods failed: {e}")
                faces = []
                detection_method = "No detection (all methods failed)"

        
        print(f"Detected {len(faces)} faces using {detection_method}")
        
        if len(faces) == 0:
            print("No faces detected - trying fallback with lower thresholds")
      
        
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
    import signal
    import sys
    import atexit
    
    def cleanup_handler(signum=None, frame=None):
        """Clean up resources on shutdown"""
        print("Cleaning up resources...")
        try:
            # Clean up OpenCV resources
            cv2.destroyAllWindows()
            # Force garbage collection
            import gc
            gc.collect()
        except Exception as e:
            print(f"Cleanup error: {e}")
        print("Cleanup completed")
        if signum is not None:
            sys.exit(0)
    
    # Register cleanup handlers
    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)
    atexit.register(cleanup_handler)
    
    print("Starting Emotion Detection API...")
    print("Model loaded successfully!")
    print("API endpoints available:")
    print("  - POST /detect_emotions           - Main emotion detection endpoint with image-based prompts")
    print("  - POST /generate_prompt_from_image - Generate educational prompts from images using Gemini Vision")
    print("  - POST /generate_prompt           - Generate educational prompts from emotion text only")
    print("  - GET  /test_emotions             - Test endpoint with simulated data")
    print("  - GET  /health                    - Health check endpoint")
    print("Server starting on http://0.0.0.0:5001")
    print("Press Ctrl+C to stop the server")
    
    try:
        # Start Flask development server with reduced multiprocessing to avoid semaphore leaks
        app.run(host='0.0.0.0', port=5001, debug=False, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print("Server stopped by user")
        cleanup_handler()
    except Exception as e:
        print(f"Server error: {e}")
        cleanup_handler()