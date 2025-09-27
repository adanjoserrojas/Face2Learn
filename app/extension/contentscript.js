function captureFrame() {
    //Log the attempt to capture a frame
    console.log("Attempting to capture frame from video element.");
  const video = document.querySelector("video");
  if (!video) return null;

  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  return canvas.toDataURL("image/png"); // Base64
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.action === "capture") {
    console.log("Received capture request from extension.");
    sendResponse({ image: captureFrame() });
  }
});console.log("face2learn content script loaded");

// API Configuration
const API_BASE_URL = 'http://localhost:5001';

// Canvas setup
const canvas = document.createElement("canvas");
canvas.id = "face2learn-canvas";
canvas.style.position = "fixed";
canvas.style.top = "0";
canvas.style.left = "0";
canvas.style.pointerEvents = "none";
canvas.style.zIndex = "1000";
document.body.appendChild(canvas);

const ctx = canvas.getContext("2d");

function resizeCanvas() { 
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
}
resizeCanvas();
window.addEventListener("resize", resizeCanvas);

// Function to get emotion data from API
async function getEmotionData() {
    try {
        const response = await fetch(`${API_BASE_URL}/test_emotions`, {
            method: 'GET'
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        return data.results || [];
    } catch (error) {
        console.error('Error calling emotion detection API:', error);
        return [];
    }
}

// Function to process captured video frame for emotion detection
async function processVideoFrame() {
    try {
        // Capture frame from video
        const imageData = captureFrame();
        
        if (!imageData) {
            console.log('No video found or unable to capture frame');
            return [];
        }

        console.log('Captured frame, sending to emotion detection API');

        // Send captured frame to emotion detection API
        const response = await fetch(`${API_BASE_URL}/detect_emotions`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image: imageData
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        console.log('Emotion detection results:', data);
        return data.results || [];
    } catch (error) {
        console.error('Error processing video frame for emotion detection:', error);
        return [];
    }
}

// Function to draw emotion boxes
function drawEmotionBoxes(results) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    if (!results || results.length === 0) {
        return;
    }

    // Get video element to calculate scaling factors
    const video = document.querySelector("video");
    if (!video) {
        console.log('No video element found for coordinate scaling');
        return;
    }

    // Get video's position and size on the page
    const videoRect = video.getBoundingClientRect();
    
    // Calculate scaling factors between native video resolution and displayed size
    const scaleX = videoRect.width / video.videoWidth;
    const scaleY = videoRect.height / video.videoHeight;
    
    console.log(`Video native: ${video.videoWidth}x${video.videoHeight}`);
    console.log(`Video displayed: ${videoRect.width}x${videoRect.height}`);
    console.log(`Scale factors: ${scaleX}, ${scaleY}`);

    results.forEach(result => {
        const { bounding_box, emotion, confidence } = result;
        const { x, y, width, height } = bounding_box;
        
        // Scale coordinates from native video resolution to displayed size
        const scaledX = (x * scaleX) + videoRect.left;
        const scaledY = (y * scaleY) + videoRect.top;
        const scaledWidth = width * scaleX;
        const scaledHeight = height * scaleY;
        
        // Draw bounding box
        ctx.strokeStyle = "red";
        ctx.lineWidth = 2;
        ctx.strokeRect(scaledX, scaledY, scaledWidth, scaledHeight);
        
        // Draw emotion label
        const label = `${emotion} (${Math.round(confidence * 100)}%)`;
        ctx.fillStyle = "red";
        ctx.font = "16px Arial";
        ctx.fillText(label, scaledX, scaledY - 5);
    });
}

// Main processing function
async function processFrame() {
    const results = await getEmotionData();
    drawEmotionBoxes(results);
}

// Test processing function that uses real video capture
async function testProcessFrame() {
    const results = await processVideoFrame();
    drawEmotionBoxes(results);
}

// Add keyboard shortcut to trigger test capture
document.addEventListener('keydown', function(event) {
    // Press 'T' key to trigger test emotion detection on video
    if (event.key === 't' || event.key === 'T') {
        console.log('Test capture triggered by user');
        testProcessFrame();
    }
});

// Start processing every 500ms for smooth animation (using mock data)
// setInterval(processFrame, 500);

// Uncomment the line below and comment the line above to use real video capture instead
setInterval(testProcessFrame, 300);