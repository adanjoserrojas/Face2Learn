let rectanglesVisible = true;

// Toggle states from popup
let autoplayEnabled = false; // When true: boxes clickable, screenshot hidden
let liveAnalysisEnabled = false; // When true: boxes always show, when false: only on hover

console.log("Content Script Initialized");

setInterval(testProcessFrame, 700);

/*
rectanglesVisible.addListener(() => {
    console.log("Rectangles visibility changed:", rectanglesVisible);
});
*/

function captureFrame() {
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
    } else if (msg.action === "updateToggleStates") {
        console.log("Received toggle states update:", msg);
        autoplayEnabled = msg.autoplayEnabled;
        liveAnalysisEnabled = msg.liveAnalysisEnabled;
        console.log(`Toggle states updated - Autoplay: ${autoplayEnabled}, Live Analysis: ${liveAnalysisEnabled}`);
        sendResponse({ success: true });
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
canvas.style.cursor = "default";
document.body.appendChild(canvas);

const ctx = canvas.getContext("2d");

// Store current emotion boxes for click detection
let currentEmotionBoxes = [];
// Track mouse position in client coordinates for hover detection
let currentMouseClient = { x: null, y: null };

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
        
        // Draw bounding box based on live analysis toggle and hover state
        const boxRect = { x: scaledX, y: scaledY, width: scaledWidth, height: scaledHeight };
        const hovered = currentMouseClient.x !== null && isPointInRect(currentMouseClient.x, currentMouseClient.y, boxRect);
        
        // Show boxes if: live analysis is enabled OR if hovered (when live analysis is disabled)
        const shouldShowBox = liveAnalysisEnabled || hovered;
        
        if (shouldShowBox) {
            ctx.strokeStyle = boxColor;
            ctx.lineWidth = 2;
            ctx.strokeRect(scaledX, scaledY, scaledWidth, scaledHeight);

                    
            // Draw emotion label with background for better readability
            const label = `${emotion}`;
            ctx.font = "14px Arial";
            
            // Measure text for background
            const textMetrics = ctx.measureText(label);
            const textWidth = textMetrics.width;
            const textHeight = 20;
            
            // Draw background for text
            ctx.fillStyle = boxColor;
            ctx.fillRect(scaledX + (scaledWidth / 2) - (textWidth / 2) + (ctx.lineWidth / 2) - 1, scaledY - textHeight, textWidth + 2, textHeight + 2);
            
            // Draw text
            ctx.fillStyle = "white";
            ctx.fillText(label, scaledX + (scaledWidth / 2) - (textWidth / 2) + (ctx.lineWidth / 2), scaledY + 2);
        }
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
    promptContainer.style.cssText =`
        position: fixed;
        margin-top: 56px;
        right: 2px;
        width: 300px;
        max-height: 400px;
        background: #fff6ec;
        color: white;
        padding: 12px 15px 15px 12px; /* slightly less left padding */
        border-radius: 12px;
        z-index: 50;
        font-family: 'Bitcount Prop Single Ink', system-ui;
        font-weight: 400;
        font-variation-setting: 'slnt' 0, 'CRSV' 0.5;
        font-size: 2.25rem;
        line-height: 1.4;
        overflow-y: auto;
        backdrop-filter: blur(10px);
    `;
    
    // Add close button
    const closeBtn = document.createElement('button');
    closeBtn.innerHTML = '×';
    closeBtn.style.cssText = `
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
        color: #3d348b
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
            <div style="font-weight: bold; margin-bottom: 8px; color: #3d348d;">
                ${prompt.title}
            </div>
            <div style="margin-bottom: 8px; font-size: 12px; opacity: 0.8; color: #3d348d;">
                Confidence: ${Math.round(prompt.confidence * 100)}%
            </div>
            <div style="margin-bottom: 8px; font-size: 12px; opacity: 0.8; color: #3d348d;">
                ${prompt.content}
            </div>
        `;
        
        promptContainer.appendChild(promptDiv);
    });
    
    promptContainer.appendChild(closeBtn);
    document.body.appendChild(promptContainer);
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

// Function to check if a point is inside a rectangle
function isPointInRect(x, y, rect) {
    return x >= rect.x && x <= rect.x + rect.width &&
        y >= rect.y && y <= rect.y + rect.height;
}

// Function to handle emotion box clicks
async function handleEmotionBoxClick(emotionData) {
    console.log('Emotion box clicked:', emotionData);
      // Pause all videos on the page when a box is clicked
    try {
        const videos = document.querySelectorAll('video');
        videos.forEach(v => {
            try { v.pause(); } catch (e) { /* ignore if pause fails */ }
        });
        console.log(`Paused ${videos.length} video(s) due to box click`);
    } catch (e) {
        console.warn('Error attempting to pause videos:', e);
    }
    // Send emotion data to popup for display
    handleEmoji(emotionData);
    
    // Create a detailed popup or alert with emotion information (optional - can remove if you prefer)
    const details = `
Face ID: ${emotionData.face_id}
Emotion: ${emotionData.emotion}
Confidence: ${Math.round(emotionData.confidence * 100)}%
Coordinates: (${Math.round(emotionData.x)}, ${Math.round(emotionData.y)})
Size: ${Math.round(emotionData.width)} x ${Math.round(emotionData.height)}
    `;

// Loading Container
    const promptContainer = document.createElement('div');
    promptContainer.className = 'face2learn-prompt';
    promptContainer.style.cssText = `
        position: fixed;
        margin-top: 56px;
        right: 2px;
        width: 300px;
        max-height: 400px;
        background: #fff6ec;
        padding: 12px 15px 15px 12px; /* slightly less left padding */
        border-radius: 12px;
        z-index: 50;
        font-family: 'Bitcount Prop Single Ink', system-ui;
        font-weight: 400;
        font-variation-setting: 'slnt' 0, 'CRSV' 0.5;
        font-size: 2.25rem;
        line-height: 1.4;
        overflow-y: auto;
        backdrop-filter: blur(10px);
    `;
    
    // Add close button
    const closeBtn = document.createElement('button');
    closeBtn.innerHTML = '×';
    closeBtn.style.cssText = `
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
        align-items: flex-end;
        justify-content: center;
        color: #3d348b
    `;
    closeBtn.onclick = () => promptContainer.remove();
    
        // Loading bar
        const promptDiv = document.createElement('div');
        promptDiv.style.cssText = `
            margin-bottom: ${0 < 15 - 1 ? '15px' : '0'};
            padding-bottom: ${0 < 15 - 1 ? '15px' : '0'};
            border-bottom: ${0 < 15 - 1 ? '1px solid rgba(255,255,255,0.2)' : 'none'};
        `;
            
        promptDiv.innerHTML = `
            <div style="white-space: pre-wrap;">
                Loading...
            </div>
            <div style="font-weight: bold; margin-bottom: 8px; color: #3d348b">
                ???
            </div>
            <div style="margin-bottom: 8px; font-size: 12px; opacity: 0.8;">
                Confidence: ?%
            </div>
        `;
            
        promptContainer.appendChild(promptDiv);
        
        promptContainer.appendChild(closeBtn);
        document.body.appendChild(promptContainer);

    // Capture frame (captureFrame may be sync or return a Promise)
    let image;
    try {
        image = await Promise.resolve(captureFrame());
    } catch (e) {
        console.warn('captureFrame failed:', e);
        image = null;
    }

    // Request prompt from backend (with image if available)
    try {
        let prompt = null;

        if (image) {
            prompt = await geminiAdviceWithImage(image, emotionData.emotion, emotionData.confidence);
        } else {
            prompt = await geminiAdviceWithoutImage(emotionData.emotion, emotionData.confidence);
        }

        if (prompt && (prompt.content || prompt.title)) {
            // Put gemini prompt into the educational advice UI and show a quick alert
            try {
                displayEducationalPrompts([prompt]);
            } catch (e) {
                console.warn('displayEducationalPrompts failed:', e);
            }

        } else {
            console.warn('No prompt returned from backend');
        }
    } catch (e) {
        console.error('Error fetching Gemini prompt:', e);
    }
}

async function geminiAdviceWithoutImage(emotion, confidence) {
    try {
        const response = await fetch(`${API_BASE_URL}/generate_prompt`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ emotion: emotion, confidence: confidence, context: 'User requested guidance' })
        });

        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

        const data = await response.json();
        return data.prompt || null;
    } catch (error) {
        console.error('Error calling emotion detection API:', error);
        return [];
    }
}

async function geminiAdviceWithImage(imageData, emotionData, confidenceData) {
    try {
        const response = await fetch(`${API_BASE_URL}/generate_prompt_from_image`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image: imageData,
                emotion: emotionData,
                confidence: confidenceData,
                context: " "
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        // API returns { success: true, prompt: {...} }
        return data.prompt || null;
    } catch (error) {
        console.error('Error calling emotion detection API:', error);
        return [];
    }
}

// Function to handle emoji display in popup based on emotion data
function handleEmoji(emotionData) {
    console.log('handleEmoji called with:', emotionData);
    
    // Send emotion data to popup (if it's open)
    try {
        chrome.runtime.sendMessage({
            action: 'updateEmotion',
            emotion: emotionData.emotion,
            confidence: emotionData.confidence,
            face_id: emotionData.face_id || 0
        }, (response) => {
            if (chrome.runtime.lastError) {
                console.log('Popup not open or error:', chrome.runtime.lastError.message);
            } else {
                console.log('Emotion data sent to popup successfully');
            }
        });
    } catch (error) {
        console.log('Error sending message to popup:', error);
    }
}

// Function to process getEmotionData output and update popup with most confident emotion
async function updateEmotionFromAPI() {
    try {
        const emotionResults = await getEmotionData();
        console.log('getEmotionData results:', emotionResults);
        
        if (emotionResults && emotionResults.length > 0) {
            // Find the emotion with highest confidence
            const mostConfidentEmotion = emotionResults.reduce((prev, current) => {
                return (prev.confidence > current.confidence) ? prev : current;
            });
            
            console.log('Most confident emotion:', mostConfidentEmotion);
            
            // Update popup with the most confident emotion
            handleEmoji({
                emotion: mostConfidentEmotion.emotion,
                confidence: mostConfidentEmotion.confidence,
                face_id: mostConfidentEmotion.face_id || 0
            });
            
            return mostConfidentEmotion;
        } else {
            console.log('No emotions detected, using neutral');
            // Default to neutral if no emotions detected
            handleEmoji({
                emotion: 'neutral',
                confidence: 1.0,
                face_id: 0
            });
            return null;
        }
    } catch (error) {
        console.error('Error in updateEmotionFromAPI:', error);
        return null;
    }
}

// Add click event listener to canvas
addEventListener('click', function(event) {
    // Only handle clicks if autoplay is enabled
    if (!autoplayEnabled) {
        console.log('Autoplay disabled - ignoring box clicks');
        return;
    }

    // Use client coordinates for click detection (box coords are in client space)
    const clickClientX = event.clientX;
    const clickClientY = event.clientY;

    console.log(`Canvas clicked at client coords: (${clickClientX}, ${clickClientY})`);

    // Check if click is inside any emotion box
    for (let i = 0; i < currentEmotionBoxes.length; i++) {
        const box = currentEmotionBoxes[i];
        if (isPointInRect(clickClientX, clickClientY, box)) {
            // Consume the event only when click lands inside a face box so underlying page
            // doesn't receive the click. When click is outside boxes we do nothing and allow
            // the event to propagate to the page.
            handleEmotionBoxClick(box);
            try {
                event.stopPropagation();
                event.preventDefault();
            } catch (e) {
                // ignore in case event isn't cancelable
            }
            break; // Only handle the first matching box
        }
    }
});

// Add hover effect to show pointer cursor only over emotion boxes
addEventListener('mousemove', function(event) {
    // Use client coords for hover detection so they match box coordinates
    const clientX = event.clientX;
    const clientY = event.clientY;

    // Update tracked mouse client position for draw-time checks
    currentMouseClient.x = clientX;
    currentMouseClient.y = clientY;

    // Check if mouse is over any emotion box (client space)
    let overBox = false;
    for (let i = 0; i < currentEmotionBoxes.length; i++) {
        const box = currentEmotionBoxes[i];
        if (isPointInRect(clientX, clientY, box)) {
            overBox = true;
            break;
        }
    }

    // If hovering a box and autoplay is enabled, enable pointer events on the canvas so clicks are caught by it
    // Otherwise disable pointer events so clicks pass through to the underlying page.
    if (overBox && autoplayEnabled) {
        canvas.style.pointerEvents = "auto";
        canvas.style.cursor = "pointer";
    } else {
        canvas.style.pointerEvents = "none";
        canvas.style.cursor = "default";
        // clear client mouse pos so draw logic knows there's no hover
        if (!overBox) {
            currentMouseClient.x = null;
            currentMouseClient.y = null;
        }
    }
});


