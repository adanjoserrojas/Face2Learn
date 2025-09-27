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
});