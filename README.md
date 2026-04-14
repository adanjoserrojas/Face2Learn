# Face2Learn

A web application that detects facial emotions and generates educational prompts via Gemini AI to help socially disabled users understand emotional cues.

## Description

Face2Learn combines a CNN-based facial emotion detector with Google's Gemini AI to produce contextual, educational explanations of human expressions. The system is built as a Chrome extension backed by a Flask API.

**How it works:**

1. The Flask backend (`backend/api.py`) receives a base64-encoded image from the extension via `POST /detect_emotions`.
2. A pre-trained CNN model (weights loaded from `model.h5`) classifies the detected face into one of seven emotions: Angry, Disgusted, Fearful, Happy, Neutral, Sad, or Surprised.
3. Face detection uses an OpenCV Haar cascade (`haarcascade_frontalface_default.xml`) with multiple sensitivity fallbacks.
4. When the user requests a prompt, the image and detected emotion are sent to `gemini-2.5-pro` (via `google-generativeai`), which returns a scene-aware educational description of the emotion and how to interact with it.
5. If the Gemini API is unavailable, a built-in fallback dictionary provides pre-written descriptions for each of the seven emotions.
6. A secondary text-classification pipeline using `j-hartmann/emotion-english-distilroberta-base` (via `transformers` and `torch`) is also available at `POST /` for text-based emotion analysis.

The frontend is a Next.js application (TypeScript/JavaScript) served on `http://localhost:3000`.

## Tech Stack

- **Backend:** Python, Flask, TensorFlow / Keras, OpenCV, Pillow, NumPy, `google-generativeai`, `transformers`, PyTorch
- **Frontend:** Next.js (TypeScript, JavaScript, CSS, HTML)

## Installation

### Backend

Install Python dependencies:

```bash
pip install -r backend/requirements.txt
```

Set your Gemini API key (optional — a default key is present in `config.py`):

```bash
export GEMINI_API_KEY=your-api-key-here
```

Place the following files in the `backend/` directory before starting:
- `model.h5` — pre-trained CNN weights
- `haarcascade_frontalface_default.xml` — OpenCV Haar cascade

Start the Flask server:

```bash
python backend/api.py
```

The API will be available at `http://0.0.0.0:5001`.

### Frontend

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/detect_emotions` | Accepts a base64 image, returns face bounding boxes and detected emotions |
| `POST` | `/generate_prompt_from_image` | Accepts an image + emotion, returns a Gemini Vision educational prompt |
| `POST` | `/generate_prompt` | Accepts emotion text only, returns a text-based educational prompt |
| `POST` | `/` | Accepts plain text, returns emotion classification scores |
| `GET`  | `/test_emotions` | Returns simulated emotion data for UI testing |
| `GET`  | `/health` | Health check — returns `{"status": "healthy"}` |

## Configuration

`backend/config.py` exposes the following environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | *(hardcoded fallback)* | Google Gemini API key |
| `FLASK_ENV` | `development` | Flask environment |
| `API_HOST` | `0.0.0.0` | Host to bind the Flask server |
| `API_PORT` | `5001` | Port for the Flask server |
| `CONFIDENCE_THRESHOLD` | `0.7` | Minimum confidence for emotion results |
| `MAX_FACES` | `5` | Maximum number of faces to process per image |