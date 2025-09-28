document.addEventListener('DOMContentLoaded', () => {
  const thumb = document.getElementById('thumb');
  const placeholder = document.getElementById('thumb-placeholder');
  
  // Toggle functionality
  const screenshotToggle = document.getElementById('screenshot-toggle');
  const screenshotButtonContainer = document.getElementById('screenshot-button-container');
  const screenshotBtn = document.getElementById('screenshot-btn');

  let currentDataUrl = null;

  // Emotion mapping for dynamic image display
  const emotionImageMap = {
    'sad': 'Images/1F62D_color.png',
    'surprised': 'Images/1F62F_color.png', 
    'angry': 'Images/1F92C_color.png',
    'disgusted': 'Images/1F92E_color.png',
    'happy': 'Images/1F601_color.png',
    'neutral': 'Images/1F610_color.png',
    'fearful': 'Images/1F628_color.png'
  };

  // Get current emotion display elements
  const currentEmotionImg = document.querySelector('.current-emotion-img');
  const currentEmotionText = document.querySelector('.current-emotion-text');

  // Set initial state - button should be visible when toggle is OFF (unchecked)
  if (!screenshotToggle.checked) {
    screenshotButtonContainer.classList.remove('opacity-0', 'scale-95');
    screenshotButtonContainer.classList.add('opacity-100', 'scale-100');
  }

  // Handle toggle switch (inverted logic)
  screenshotToggle.addEventListener('change', function() {
    if (!this.checked) {
      // Toggle is OFF - show screenshot button with smooth transition
      screenshotButtonContainer.classList.remove('opacity-0', 'scale-95');
      screenshotButtonContainer.classList.add('opacity-100', 'scale-100');
    } else {
      // Toggle is ON - hide screenshot button with smooth transition
      screenshotButtonContainer.classList.remove('opacity-100', 'scale-100');
      screenshotButtonContainer.classList.add('opacity-0', 'scale-95');
    }
  });

  // Handle screenshot button click
  screenshotBtn.addEventListener('click', async () => {
    console.log('Screenshot button clicked!');
    
    try {
      const tab = await getActiveTab();
      if (!tab) { 
        showNoImage('No active tab'); 
        return; 
      }

      // First try to send capture message to existing content script
      let res = await sendCaptureMessage(tab.id);
      
      if (res && res.image) {
        setThumbnail(res.image);
        console.log('Screenshot captured successfully');
      } else {
        // Try injecting the content script then request again
        try {
          await injectContentScript(tab.id);
          res = await sendCaptureMessage(tab.id);
          if (res && res.image) {
            setThumbnail(res.image);
            console.log('Screenshot captured successfully after injection');
          } else {
            showNoImage('No video found on page');
          }
        } catch (e) {
          console.warn('Injection or capture failed', e);
          showNoImage('Capture failed');
        }
      }
    } catch (error) {
      console.error('Screenshot error:', error);
      showNoImage('Screenshot failed');
    }
  });

  // Function to update emotion display
  function updateEmotionDisplay(emotion, confidence) {
    console.log('Updating emotion display:', emotion, confidence);
    
    // Get emotion image path
    const imagePath = emotionImageMap[emotion.toLowerCase()];
    
    if (imagePath && currentEmotionImg) {
      currentEmotionImg.src = imagePath;
      currentEmotionImg.alt = `${emotion} Face`;
      console.log('Updated image src to:', imagePath);
    }
    
    if (currentEmotionText) {
      currentEmotionText.textContent = `Current Emotion: ${emotion} (${Math.round(confidence * 100)}%)`;
    }
  }

  // Listen for emotion updates from content script
  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    console.log('Popup received message:', message);
    
    if (message.action === 'updateEmotion') {
      updateEmotionDisplay(message.emotion, message.confidence);
      sendResponse({ success: true });
    }
  });

  function setThumbnail(dataUrl) {
    currentDataUrl = dataUrl;
    placeholder.style.display = 'none';
    let img = thumb.querySelector('img');
    if (!img) {
      img = document.createElement('img');
      img.style.maxWidth = '100%';
      img.style.height = 'auto';
      img.style.display = 'block';
      thumb.appendChild(img);
    }
    img.src = dataUrl;
  }

  function showNoImage(message) {
    currentDataUrl = null;
    currentPrompts = null;
    currentEmotionResults = null;
    const img = thumb.querySelector('img');
    if (img) img.remove();
    placeholder.style.display = 'block';
    if (message) placeholder.textContent = message;
    else placeholder.textContent = 'No snapshot yet';
    hidePrompts();
  }



  function getActiveTab() {
    return new Promise((resolve) => {
      chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => resolve(tabs[0]));
    });
  }

  function sendCaptureMessage(tabId) {
    return new Promise((resolve) => {
      chrome.tabs.sendMessage(tabId, { action: 'capture' }, (response) => {
        if (chrome.runtime.lastError) return resolve({ error: chrome.runtime.lastError.message });
        resolve(response || {});
      });
    });
  }

  async function detectEmotions(imageData) {
    try {
      const response = await fetch('http://localhost:5001/detect_emotions', {
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
      return {
        results: data.results || [],
        educational_prompts: data.educational_prompts || []
      };
    } catch (error) {
      console.error('Error detecting emotions:', error);
      return { results: [], educational_prompts: [] };
    }
  }

  async function getGeminiAnalysis(imageData, emotion, confidence) {
    try {
      const response = await fetch('http://localhost:5001/generate_prompt_from_image', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: imageData,
          emotion: emotion,
          confidence: confidence,
          context: "YouTube educational content - user requested guidance"
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      return data.prompt || null;
    } catch (error) {
      console.error('Error getting Gemini analysis:', error);
      return null;
    }
  }

  function injectContentScript(tabId) {
    return new Promise((resolve, reject) => {
      if (!chrome.scripting || !chrome.scripting.executeScript) return reject(new Error('scripting API unavailable'));
      chrome.scripting.executeScript({
        target: { tabId },
        files: ['contentscript.js']
      }, () => {
        if (chrome.runtime.lastError) reject(chrome.runtime.lastError);
        else resolve();
      });
    });
  }

  showNoImage();
});