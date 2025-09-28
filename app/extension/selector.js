const EMOTION_COLORS = {
    joy: "#FFD700",      // gold
    sadness: "#1E90FF",  // blue
    anger: "#FF4500",    // orange-red
    disgust: "#556B2F",  // dark olive green
    fear: "#800080",     // purple
    surprise: "#FF69B4", // pink
    neutral: "#808080"   // gray
};


chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "analyzeEmotion",
    title: "Analyze Emotion",
    contexts: ["selection"] // Only show when text is selected
  });
});


chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "analyzeEmotion" && info.selectionText) {
    fetch("http://localhost:5001", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: info.selectionText })
    })
    .then(res => res.json())
    .then(data => {
      chrome.scripting.executeScript({
        target: { tabId: tab.id },
        func: (emotionData, selectedText) => {
          const EMOTION_STYLES = {
              joy: { color: "#FFD700", meaning: "A feeling of great happiness and delight." },
              sadness: { color: "#1E90FF", meaning: "A state of unhappiness or grief." },
              anger: { color: "#FF4500", meaning: "A strong feeling of annoyance or hostility." },
              disgust: { color: "#556B2F", meaning: "A feeling of revulsion or strong disapproval." },
              fear: { color: "#800080", meaning: "An unpleasant emotion caused by threat or danger." },
              surprise: { color: "#FF69B4", meaning: "A feeling of shock or amazement." },
              neutral: { color: "#808080", meaning: "A balanced state, neither positive nor negative." }
          };

          function highlightText(text, color) {
            const selection = window.getSelection();
            if (!selection.rangeCount) return;
            const range = selection.getRangeAt(0);
            const span = document.createElement("span");
            span.style.backgroundColor = color;
            span.style.color = "#000";
            span.style.borderRadius = "3px";
            span.style.padding = "2px";
            span.textContent = text;
            range.deleteContents();
            range.insertNode(span);
            selection.removeAllRanges();
          }

          function displayEmotionResult(emotionData, selectedText) {
            const { top_label } = emotionData;
            const style = EMOTION_STYLES[top_label] || { color: "#000000", meaning: "" };
            const existing = document.querySelector('.face2learn-emotion-overlay');
            if (existing) existing.remove();
            const container = document.createElement('div');
            container.className = 'face2learn-emotion-overlay';
            container.style.cssText = `
              position: fixed; top: 20px; right: 20px; width: 320px;
              background: rgba(255,246,236,0.95); padding: 20px; border-radius: 16px;
              z-index: 99999; font-family: Arial; box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            `;
            const closeBtn = document.createElement('button');
            closeBtn.innerHTML = '×';
            closeBtn.style.cssText = `
              position: absolute; top: 10px; right: 10px; background: none; border: none;
              font-size: 18px; cursor: pointer;
            `;
            closeBtn.onclick = () => container.remove();
            container.appendChild(closeBtn);
            const title = document.createElement('div');
            title.textContent = top_label.toUpperCase();
            title.style.cssText = `font-size: 2.5rem; font-weight: bold; color: ${style.color}; margin-bottom: 10px;`;
            container.appendChild(title);
            const meaning = document.createElement('div');
            meaning.textContent = style.meaning;
            meaning.style.cssText = `font-size: 1rem; color: ${style.color}; opacity: 0.85;`;
            container.appendChild(meaning);
            document.body.appendChild(container);
            highlightText(selectedText, style.color);
          }

          displayEmotionResult(emotionData, selectedText);
        },
        args: [data, info.selectionText]
      });
    });
  }
});




function displayEmotionResult(emotionData, selectedText) {
    const { top_label } = emotionData;
    const style = EMOTION_STYLES[top_label] || { color: "#000000", meaning: "" };

    // Remove existing overlay
    const existing = document.querySelector('.face2learn-emotion-overlay');
    if (existing) existing.remove();

    // Create overlay
    const container = document.createElement('div');
    container.className = 'face2learn-emotion-overlay';
    container.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        width: 320px;
        background: rgba(255, 246, 236, 0.95);
        padding: 20px;
        border-radius: 16px;
        z-index: 99999;
        font-family: Arial, sans-serif;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    `;

    // Close button
    const closeBtn = document.createElement('button');
    closeBtn.innerHTML = '×';
    closeBtn.style.cssText = `
        position: absolute;
        top: 10px;
        right: 10px;
        background: none;
        border: none;
        font-size: 18px;
        cursor: pointer;
    `;
    closeBtn.onclick = () => container.remove();
    container.appendChild(closeBtn);

    // Emotion title
    const title = document.createElement('div');
    title.textContent = top_label.toUpperCase();
    title.style.cssText = `
        font-size: 2.5rem;
        font-weight: bold;
        color: ${style.color};
        margin-bottom: 10px;
    `;
    container.appendChild(title);

    // Meaning text
    const meaning = document.createElement('div');
    meaning.textContent = style.meaning;
    meaning.style.cssText = `
        font-size: 1rem;
        color: ${style.color};
        opacity: 0.85;
    `;
    container.appendChild(meaning);

    document.body.appendChild(container);

    // Highlight selected text
    highlightText(selectedText, style.color);
}



function highlightText(text, bgColor) {
    const selection = window.getSelection();
    if (!selection.rangeCount) return;

    const range = selection.getRangeAt(0);

    // Utility to lighten a hex color
    function lightenColor(hex, amount = 60) {
        let col = hex.replace("#", "");
        if (col.length === 3) col = col.split("").map(c => c + c).join("");
        const num = parseInt(col, 16);

        let r = (num >> 16) + amount;
        let g = ((num >> 8) & 0x00FF) + amount;
        let b = (num & 0x0000FF) + amount;

        r = Math.min(255, r);
        g = Math.min(255, g);
        b = Math.min(255, b);

        return `rgb(${r}, ${g}, ${b})`;
    }

    const span = document.createElement("span");
    span.style.backgroundColor = bgColor;
    span.style.color = lightenColor(bgColor, 100); // lighter version of background
    span.style.borderRadius = "3px";
    span.style.padding = "2px";
    span.textContent = text;

    range.deleteContents();
    range.insertNode(span);

    selection.removeAllRanges();
}


