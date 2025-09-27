# Face2Learn with Gemini AI Integration

This project now includes Gemini AI integration to generate personalized educational prompts based on detected emotions.

## 🚀 Quick Setup

### 1. Get Your Gemini API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy your API key

### 2. Configure the Backend

1. **Set your Gemini API key as an environment variable:**
   ```bash
   # Windows (Command Prompt)
   set GEMINI_API_KEY=your-actual-api-key-here
   
   # Windows (PowerShell)
   $env:GEMINI_API_KEY="your-actual-api-key-here"
   
   # Linux/Mac
   export GEMINI_API_KEY="your-actual-api-key-here"
   ```

2. **Or edit the config file directly:**
   - Open `backend/config.py`
   - Replace `'your-gemini-api-key-here'` with your actual API key

### 3. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 4. Run the Backend

```bash
python api.py
```

### 5. Install the Chrome Extension

1. Open Chrome and go to `chrome://extensions/`
2. Enable Developer Mode
3. Click "Load unpacked" and select the `app/extension` folder

## 🎯 How It Works

1. **Emotion Detection**: The system detects emotions from facial expressions in real-time
2. **AI Analysis**: Gemini AI analyzes the detected emotion and generates personalized educational prompts
3. **Learning Support**: Users receive contextual learning advice, strategies, and encouragement

## 📝 Features

- **Real-time emotion detection** with confidence scores
- **Personalized educational prompts** based on detected emotions
- **Learning strategies** tailored to emotional state
- **Encouragement and motivation** appropriate for each emotion
- **Actionable tips** for immediate implementation

## 🎨 UI Components

- **Overlay prompts** appear on YouTube videos with detected emotions
- **Extension popup** shows captured snapshots with learning insights
- **Beautiful gradient design** with smooth animations
- **Auto-dismiss** prompts after 10 seconds

## 🔧 Configuration

The system uses configurable templates for different emotions:

- **Happy**: "Great Learning Energy!" - Encourages continued engagement
- **Sad**: "Learning Support" - Provides comfort and alternative approaches
- **Angry**: "Frustration Management" - Helps manage learning challenges
- **Fearful**: "Building Confidence" - Supports confidence building
- **Surprised**: "Discovery Moment" - Celebrates learning discoveries
- **Disgusted**: "Content Adjustment" - Suggests content alternatives
- **Neutral**: "Focused Learning" - Maintains learning momentum

## 🚨 Troubleshooting

### Gemini API Issues
- Ensure your API key is correctly set
- Check that you have sufficient API quota
- Verify internet connectivity

### Extension Issues
- Make sure the backend is running on `http://localhost:5001`
- Check browser console for error messages
- Reload the extension if needed

### Emotion Detection Issues
- Ensure good lighting for face detection
- Position face clearly in camera view
- Check that the video is playing

## 📊 API Endpoints

- `POST /detect_emotions` - Main emotion detection with educational prompts
- `POST /generate_prompt` - Generate prompt for specific emotion
- `GET /test_emotions` - Test endpoint with simulated data
- `GET /health` - Health check

## 🎓 Educational Benefits

This integration helps learners by:

1. **Self-awareness**: Understanding their emotional state during learning
2. **Personalized support**: Getting advice tailored to their current mood
3. **Learning optimization**: Adjusting study strategies based on emotions
4. **Motivation**: Receiving encouragement when needed
5. **Reflection**: Encouraging mindful learning practices

## 🔮 Future Enhancements

- Learning progress tracking
- Emotion-based content recommendations
- Study session analytics
- Integration with learning management systems
- Multi-language support
