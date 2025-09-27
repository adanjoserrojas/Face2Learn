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

// Function to draw emotion boxes
function drawEmotionBoxes(results) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    if (!results || results.length === 0) {
        return;
    }

    results.forEach(result => {
        const { bounding_box, emotion, confidence } = result;
        const { x, y, width, height } = bounding_box;
        
        // Draw bounding box
        ctx.strokeStyle = "red";
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, width, height);
        
        // Draw emotion label
        const label = `${emotion} (${Math.round(confidence * 100)}%)`;
        ctx.fillStyle = "red";
        ctx.font = "16px Arial";
        ctx.fillText(label, x, y - 5);
    });
}

// Main processing function
async function processFrame() {
    const results = await getEmotionData();
    drawEmotionBoxes(results);
}

// Start processing every 500ms for smooth animation
setInterval(processFrame, 500);