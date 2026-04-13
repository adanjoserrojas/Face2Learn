# Face2Learn

A web application and Chrome extension backend that detects facial emotions in real time and generates educational prompts via Gemini AI to help socially disabled individuals understand emotional cues.

## Description

Face2Learn combines a CNN-based facial emotion classifier with Google's Gemini AI to produce contextual, educational explanations of detected emotions. The system has two main parts:

**Python / Flask backend (`backend/api.py`)**
- Accepts base64-encoded images over HTTP and detects faces using OpenCV's Haar cascade classifier (`haarcascade_frontalface_default.xml`).
- Classifies each detected face into one of seven emotions — Angry, Disgusted, Fearful, Happy, Neutral, Sad, Surprised — using a pre-trained Keras CNN model (`model.h5`) that takes 48×48 grayscale input.
- Sends the captured image and detected emotion to Gemini (`gemini-2.5-pro`) to generate a short, scene-aware educational description of the emotion and how to interact with it.
- Also exposes a text-classification endpoint (`POST /`) powered by the `j-hartmann/emotion-english-distilroberta-base` DistilRoBERTa model (via Hugging Face `transformers` and `torch`) for emotion detection from plain text.
- Falls back to hard-coded educational descriptions when the Gemini API is unavailable.

**Next.js frontend (`app/`)**
- Bootstrapped with `create-next-app`.
- Uses `next/font` to load the Geist font family.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/detect_emotions` | Detect faces and emotions in a base64 image |
| `POST` | `/generate_prompt_from_image` | Generate an educational prompt from an image + emotion using Gemini Vision |
| `POST` | `/generate_prompt` | Generate an educational prompt from emotion text only |
| `POST` | `/` | Classify emotion from plain text (DistilRoBERTa) |
| `GET` | `/test_emotions` | Return simulated emotion data for UI testing |
| `GET` | `/health` | Health check |

The backend listens on `0.0.0.0:5001`.

## Installation

### Backend

```bash
cd backend
pip install -r requirements.txt
```

Place the following files in the `backend/` directory before starting:
- `model.h5` — pre-trained CNN weights
- `haarcascade_frontalface_default.xml` — OpenCV Haar cascade

Set your Gemini API key:

```bash
export GEMINI_API_KEY=your_key_here
```

Start the Flask server:

```bash
python api.py
```

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

## Configuration

`backend/config.py` exposes the following environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | — | Google Gemini API key |
| `FLASK_ENV` | `development` | Flask environment |
| `API_HOST` | `0.0.0.0` | Server host |
| `API_PORT` | `5001` | Server port |
| `CONFIDENCE_THRESHOLD` | `0.7` | Minimum confidence for emotion results |
| `MAX_FACES` | `5` | Maximum faces to process per image |

## Tech Stack

- **Python** — Flask, TensorFlow/Keras, OpenCV, Pillow, NumPy, `google-generativeai`, `transformers`, PyTorch
- **TypeScript / JavaScript / HTML / CSS** — Next.js frontend