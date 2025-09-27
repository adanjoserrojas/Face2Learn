document.addEventListener('DOMContentLoaded', () => {
  const captureBtn = document.getElementById('capture');
  const analyzeBtn = document.getElementById('analyze');
  const openBtn = document.getElementById('open');
  const downloadBtn = document.getElementById('download');
  const thumb = document.getElementById('thumb');
  const placeholder = document.getElementById('thumb-placeholder');

  let currentDataUrl = null;
  let currentPrompts = null;
  let currentEmotionResults = null;

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
    
    // Reset analyze button
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = '🧠 Analyze';
  }

  function displayPrompts(prompts) {
    currentPrompts = prompts;
    const promptsSection = document.getElementById('prompts-section');
    const promptsContent = document.getElementById('prompts-content');
    
    if (!prompts || prompts.length === 0) {
      hidePrompts();
      return;
    }
    
    promptsContent.innerHTML = prompts.map(prompt => `
      <div class="mb-3 p-2 bg-white rounded border-l-4 border-blue-400">
        <div class="font-medium text-gray-800 text-xs mb-1">${prompt.title}</div>
        <div class="text-xs text-gray-500 mb-1">Confidence: ${Math.round(prompt.confidence * 100)}%</div>
        <div class="text-xs text-gray-700">${prompt.content}</div>
      </div>
    `).join('');
    
    promptsSection.classList.remove('hidden');
  }

  function hidePrompts() {
    const promptsSection = document.getElementById('prompts-section');
    promptsSection.classList.add('hidden');
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

  captureBtn.addEventListener('click', async () => {
    captureBtn.disabled = true;
    captureBtn.textContent = 'Capturing...';
    analyzeBtn.disabled = true;
    
    try {
      const tab = await getActiveTab();
      if (!tab) { showNoImage('No active tab'); return; }

      let res = await sendCaptureMessage(tab.id);
      if (res && res.image) {
        setThumbnail(res.image);
        // Detect emotions in the captured image
        const emotionData = await detectEmotions(res.image);
        currentEmotionResults = emotionData.results;
        
        if (emotionData.results.length > 0) {
          analyzeBtn.disabled = false;
          analyzeBtn.textContent = `🧠 Analyze (${emotionData.results.length} emotions)`;
        } else {
          analyzeBtn.disabled = true;
          analyzeBtn.textContent = '🧠 Analyze';
        }
        
        // Show basic emotion detection results
        displayPrompts(emotionData.educational_prompts);
      } else {
        // Try injecting the content script then request again
        try {
          await injectContentScript(tab.id);
          res = await sendCaptureMessage(tab.id);
          if (res && res.image) {
            setThumbnail(res.image);
            // Detect emotions in the captured image
            const emotionData = await detectEmotions(res.image);
            currentEmotionResults = emotionData.results;
            
            if (emotionData.results.length > 0) {
              analyzeBtn.disabled = false;
              analyzeBtn.textContent = `🧠 Analyze (${emotionData.results.length} emotions)`;
            } else {
              analyzeBtn.disabled = true;
              analyzeBtn.textContent = '🧠 Analyze';
            }
            
            // Show basic emotion detection results
            displayPrompts(emotionData.educational_prompts);
          } else {
            showNoImage('No video found on page');
          }
        } catch (e) {
          console.warn('Injection or capture failed', e);
          showNoImage('Capture failed');
        }
      }
    } finally {
      captureBtn.disabled = false;
      captureBtn.textContent = 'Capture';
    }
  });

  analyzeBtn.addEventListener('click', async () => {
    if (!currentDataUrl || !currentEmotionResults || currentEmotionResults.length === 0) {
      showNoImage('No emotions detected. Please capture first.');
      return;
    }

    analyzeBtn.disabled = true;
    analyzeBtn.textContent = 'Analyzing...';
    
    try {
      // Get Gemini analysis for the first detected emotion
      const firstEmotion = currentEmotionResults[0];
      const geminiPrompt = await getGeminiAnalysis(
        currentDataUrl, 
        firstEmotion.emotion, 
        firstEmotion.confidence
      );
      
      if (geminiPrompt) {
        displayPrompts([geminiPrompt]);
      } else {
        showNoImage('Analysis failed. Please try again.');
      }
    } catch (error) {
      console.error('Analysis error:', error);
      showNoImage('Analysis failed. Please try again.');
    } finally {
      analyzeBtn.disabled = false;
      analyzeBtn.textContent = `🧠 Analyze (${currentEmotionResults.length} emotions)`;
    }
  });

  openBtn.addEventListener('click', () => {
    if (!currentDataUrl) return showNoImage('No snapshot to open');
    // Open data URL in a new tab
    try {
      chrome.tabs.create({ url: currentDataUrl });
    } catch (e) {
      // Fallback
      window.open(currentDataUrl, '_blank');
    }
  });

  downloadBtn.addEventListener('click', () => {
    if (!currentDataUrl) return showNoImage('No snapshot to download');
    const a = document.createElement('a');
    a.href = currentDataUrl;
    a.download = 'face2learn-snapshot.png';
    document.body.appendChild(a);
    a.click();
    a.remove();
  });

  // Initial state
  showNoImage();
});