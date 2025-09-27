document.addEventListener('DOMContentLoaded', () => {
  const thumb = document.getElementById('thumb');
  const placeholder = document.getElementById('thumb-placeholder');
  
  // Toggle functionality
  const screenshotToggle = document.getElementById('screenshot-toggle');
  const screenshotButtonContainer = document.getElementById('screenshot-button-container');
  const screenshotBtn = document.getElementById('screenshot-btn');

  let currentDataUrl = null;

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
  screenshotBtn.addEventListener('click', () => {
    // You can implement screenshot functionality here
    console.log('Screenshot button clicked!');
    // Add your screenshot logic here
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
    const img = thumb.querySelector('img');
    if (img) img.remove();
    placeholder.style.display = 'block';
    if (message) placeholder.textContent = message;
    else placeholder.textContent = 'No snapshot yet';
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
    try {
      const tab = await getActiveTab();
      if (!tab) { showNoImage('No active tab'); return; }

      let res = await sendCaptureMessage(tab.id);
      if (res && res.image) {
        setThumbnail(res.image);
      } else {
        // Try injecting the content script then request again
        try {
          await injectContentScript(tab.id);
          res = await sendCaptureMessage(tab.id);
          if (res && res.image) setThumbnail(res.image);
          else showNoImage('No video found on page');
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