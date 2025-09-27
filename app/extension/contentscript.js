function captureFrame() {
    //Log the attempt to capture frame from video element.
    console.log("Attempting to capture frame from video element.");
  const video = document.querySelector("video");
  if (!video) {
    console.log("No video element found");
    return null;
  }

  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  const dataURL = canvas.toDataURL("image/jpeg", 0.8); // Use JPEG with compression
  console.log("Captured frame data URL length:", dataURL.length);
  console.log("Data URL starts with:", dataURL.substring(0, 50));
  
  return dataURL;
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
canvas.style.pointerEvents = "none"; // Disable click events
canvas.style.zIndex = "1000";
document.body.appendChild(canvas);

const ctx = canvas.getContext("2d");

function resizeCanvas() { 
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
}
resizeCanvas();
window.addEventListener("resize", resizeCanvas);

// Note: Gemini analysis button moved to extension popup UI

// Note: Gemini analysis functionality moved to extension popup UI

// Test canvas functionality on load
setTimeout(() => {
    console.log('Testing canvas functionality...');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = "blue";
    ctx.lineWidth = 3;
    ctx.strokeRect(50, 50, 150, 100);
    ctx.fillStyle = "blue";
    ctx.font = "14px Arial";
    ctx.fillText("Canvas Test - Face2Learn Active", 60, 70);
    
    // Clear test rectangle after 3 seconds
    setTimeout(() => {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        console.log('Canvas test completed');
    }, 3000);
}, 1000);

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
        return {
            results: data.results || [],
            educational_prompts: data.educational_prompts || []
        };
    } catch (error) {
        console.error('Error processing video frame for emotion detection:', error);
        return { results: [], educational_prompts: [] };
    }
}

// Function to draw emotion boxes with improved filtering
function drawEmotionBoxes(results) {
    console.log('drawEmotionBoxes called with results:', results);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    if (!results || results.length === 0) {
        console.log('No results to draw, clearing canvas');
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

    filteredResults.forEach((result, index) => {
        const { bounding_box, emotion, confidence } = result;
        const { x, y, width, height } = bounding_box;
        
        // Scale coordinates from native video resolution to displayed size
        const scaledX = (x * scaleX) + videoRect.left;
        const scaledY = (y * scaleY) + videoRect.top;
        const scaledWidth = width * scaleX;
        const scaledHeight = height * scaleY;
        
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

// Function to display educational prompts
function displayEducationalPrompts(prompts) {
    // Remove existing prompt containers
    const existingPrompts = document.querySelectorAll('.face2learn-prompt');
    existingPrompts.forEach(prompt => prompt.remove());
    
    if (!prompts || prompts.length === 0) return;
    
    // Create prompt container
    const promptContainer = document.createElement('div');
    promptContainer.className = 'face2learn-prompt';
    promptContainer.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        width: 300px;
        max-height: 400px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        z-index: 1001;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-size: 14px;
        line-height: 1.4;
        overflow-y: auto;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
    `;
    
    // Add close button
    const closeBtn = document.createElement('button');
    closeBtn.innerHTML = '×';
    closeBtn.style.cssText = `
        position: absolute;
        top: 8px;
        right: 8px;
        background: none;
        border: none;
        color: white;
        font-size: 18px;
        cursor: pointer;
        width: 24px;
        height: 24px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
    `;
    closeBtn.onclick = () => promptContainer.remove();
    
    // Add content for each prompt
    prompts.forEach((prompt, index) => {
        const promptDiv = document.createElement('div');
        promptDiv.style.cssText = `
            margin-bottom: ${index < prompts.length - 1 ? '15px' : '0'};
            padding-bottom: ${index < prompts.length - 1 ? '15px' : '0'};
            border-bottom: ${index < prompts.length - 1 ? '1px solid rgba(255,255,255,0.2)' : 'none'};
        `;
        
        promptDiv.innerHTML = `
            <div style="font-weight: bold; margin-bottom: 8px; color: #ffd700;">
                ${prompt.title}
            </div>
            <div style="margin-bottom: 8px; font-size: 12px; opacity: 0.8;">
                Confidence: ${Math.round(prompt.confidence * 100)}%
            </div>
            <div style="white-space: pre-wrap;">
                ${prompt.content}
            </div>
        `;
        
        promptContainer.appendChild(promptDiv);
    });
    
    promptContainer.appendChild(closeBtn);
    document.body.appendChild(promptContainer);
    
    // Auto-remove after 10 seconds
    setTimeout(() => {
        if (promptContainer.parentNode) {
            promptContainer.remove();
        }
    }, 10000);
}

// Main processing function
async function processFrame() {
    const data = await getEmotionData();
    console.log('ProcessFrame data:', data);
    drawEmotionBoxes(data.results || data);
}

// Test processing function that uses real video capture
async function testProcessFrame() {
    console.log('testProcessFrame called');
    try {
        const data = await processVideoFrame();
        console.log('testProcessFrame data:', data);
        
        if (data && data.results && data.results.length > 0) {
            console.log('Found', data.results.length, 'emotion results');
            drawEmotionBoxes(data.results);
            window.currentEmotionResults = data.results;
        } else {
            console.log('No emotion results found, clearing canvas');
            drawEmotionBoxes([]);
            window.currentEmotionResults = [];
        }
    } catch (error) {
        console.error('Error in testProcessFrame:', error);
        drawEmotionBoxes([]);
        window.currentEmotionResults = [];
    }
}


// Add keyboard shortcut to trigger test capture
document.addEventListener('keydown', function(event) {
    // Press 'T' key to trigger test emotion detection on video
    if (event.key === 't' || event.key === 'T') {
        console.log('Test capture triggered by user');
        testProcessFrame();
    }
    
    // Press 'R' key to draw a test rectangle
    if (event.key === 'r' || event.key === 'R') {
        console.log('Drawing test rectangle');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = "red";
        ctx.lineWidth = 3;
        ctx.strokeRect(100, 100, 200, 150);
        ctx.fillStyle = "red";
        ctx.font = "16px Arial";
        ctx.fillText("Test Rectangle - Canvas Working!", 110, 120);
    }
    
    // Press 'C' key to clear loading indicator (emergency stop)
    if (event.key === 'c' || event.key === 'C') {
        console.log('Clearing loading indicator');
        hideLoadingIndicator();
        showNotification('Loading indicator cleared', 'info');
    }
});

// Start processing every 500ms for smooth animation (using mock data)
// setInterval(processFrame, 500);

// Uncomment the line below and comment the line above to use real video capture instead
setInterval(testProcessFrame, 100);