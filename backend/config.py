# Face2Learn Backend Configuration
import os

# Gemini AI Configuration
# Replace 'your-actual-api-key-here' with your real Gemini API key
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'AIzaSyA9srElpO2kNju0w8sicntl8ci_UrXJRYY')

# Flask Configuration
FLASK_ENV = os.getenv('FLASK_ENV', 'development')
FLASK_DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'

# API Configuration
API_HOST = os.getenv('API_HOST', '0.0.0.0')
API_PORT = int(os.getenv('API_PORT', '5001'))

# Emotion Detection Configuration
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.7'))
MAX_FACES = int(os.getenv('MAX_FACES', '5'))

# Educational Prompt Configuration
# Note: Prompt templates are now handled by the fallback system in api.py
# This provides richer, more personalized educational content
