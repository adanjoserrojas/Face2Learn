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
canvas.style.pointerEvents = "auto"; // Enable pointer events for clickable rectangles
canvas.style.zIndex = "1000";
canvas.style.cursor = "pointer";
document.body.appendChild(canvas);

const ctx = canvas.getContext("2d");

// Store current emotion boxes for click detection
let currentEmotionBoxes = [];

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

// Function to draw emotion boxes with improved filtering
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

    // Filter results by confidence threshold and reasonable size
    const filteredResults = results.filter(result => {
        const { confidence, bounding_box } = result;
        const { width, height } = bounding_box;
        
        // Filter by confidence (only show results with >70% confidence)
        if (confidence < 0.7) return false;
        
        // Filter by reasonable face size (avoid tiny false positives)
        const minSize = 30; // minimum face size in pixels
        const maxSize = Math.min(video.videoWidth, video.videoHeight) * 0.8; // max 80% of video size
        
        return width >= minSize && height >= minSize && 
               width <= maxSize && height <= maxSize;
    });

    // Clear current boxes and store new ones for click detection
    currentEmotionBoxes = [];
    
    filteredResults.forEach((result, index) => {
        const { bounding_box, emotion, confidence } = result;
        const { x, y, width, height } = bounding_box;
        
        // Scale coordinates from native video resolution to displayed size
        const scaledX = (x * scaleX) + videoRect.left;
        const scaledY = (y * scaleY) + videoRect.top;
        const scaledWidth = width * scaleX;
        const scaledHeight = height * scaleY;
        
        // Store box info for click detection
        currentEmotionBoxes.push({
            x: scaledX,
            y: scaledY,
            width: scaledWidth,
            height: scaledHeight,
            emotion: emotion,
            confidence: confidence,
            face_id: result.face_id || index
        });
        
        // Use different colors for different confidence levels
        let boxColor = "red";
        if (confidence > 0.9) boxColor = "green";
        else if (confidence > 0.8) boxColor = "orange";
        
        // Draw bounding box
        ctx.strokeStyle = boxColor;
        ctx.lineWidth = 2;
        ctx.strokeRect(scaledX, scaledY, scaledWidth, scaledHeight);
        
        // Draw emotion label with background for better readability
        const label = `${emotion} (${Math.round(confidence * 100)}%)`;
        ctx.font = "14px Arial";
        
        // Measure text for background
        const textMetrics = ctx.measureText(label);
        const textWidth = textMetrics.width;
        const textHeight = 20;
        
        // Draw background for text
        ctx.fillStyle = boxColor;
        ctx.fillRect(scaledX, scaledY - textHeight - 5, textWidth + 10, textHeight);
        
        // Draw text
        ctx.fillStyle = "white";
        ctx.fillText(label, scaledX + 5, scaledY - 8);
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

// Function to check if a point is inside a rectangle
function isPointInRect(x, y, rect) {
    return x >= rect.x && x <= rect.x + rect.width &&
           y >= rect.y && y <= rect.y + rect.height;
}

// Function to handle emotion box clicks
function handleEmotionBoxClick(emotionData) {
    console.log('Emotion box clicked:', emotionData);
    
    // Create a detailed popup or alert with emotion information
    const details = `
Face ID: ${emotionData.face_id}
Emotion: ${emotionData.emotion}
Confidence: ${Math.round(emotionData.confidence * 100)}%
Coordinates: (${Math.round(emotionData.x)}, ${Math.round(emotionData.y)})
Size: ${Math.round(emotionData.width)} x ${Math.round(emotionData.height)}
    `;
    
    // You can customize this to show a nicer modal or send data somewhere
    alert(`Emotion Detection Details:\n${details}`);
    
    // Optional: You could also send this data to your backend for logging
    // logEmotionClick(emotionData);
}

// Add click event listener to canvas
canvas.addEventListener('click', function(event) {
    const rect = canvas.getBoundingClientRect();
    const clickX = event.clientX - rect.left;
    const clickY = event.clientY - rect.top;
    
    console.log(`Canvas clicked at: (${clickX}, ${clickY})`);
    
    // Check if click is inside any emotion box
    for (let i = 0; i < currentEmotionBoxes.length; i++) {
        const box = currentEmotionBoxes[i];
        if (isPointInRect(clickX, clickY, box)) {
            handleEmotionBoxClick(box);
            break; // Only handle the first matching box
        }
    }
});

// Add hover effect to show pointer cursor only over emotion boxes
canvas.addEventListener('mousemove', function(event) {
    const rect = canvas.getBoundingClientRect();
    const mouseX = event.clientX - rect.left;
    const mouseY = event.clientY - rect.top;
    
    // Check if mouse is over any emotion box
    let overBox = false;
    for (let i = 0; i < currentEmotionBoxes.length; i++) {
        const box = currentEmotionBoxes[i];
        if (isPointInRect(mouseX, mouseY, box)) {
            overBox = true;
            break;
        }
    }
    
    // Change cursor based on hover state
    canvas.style.cursor = overBox ? "pointer" : "default";
});

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
setInterval(testProcessFrame, 100);