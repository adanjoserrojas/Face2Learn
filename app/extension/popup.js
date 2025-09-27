document.addEventListener('DOMContentLoaded', () => {
  const captureBtn = document.getElementById('capture');
  const openBtn = document.getElementById('open');
  const downloadBtn = document.getElementById('download');
  const thumb = document.getElementById('thumb');
  const placeholder = document.getElementById('thumb-placeholder');

  let currentDataUrl = null;

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