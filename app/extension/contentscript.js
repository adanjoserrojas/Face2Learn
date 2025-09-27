console.log("face2learn content script loaded");

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

function drawBox(x, y, width, height, label) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = "red";
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, width, height);

    if (label){
        ctx.fillStyle = "red";
        ctx.font = "16px Arial";
        ctx.fillText(label, x, y - 5);
    }
}

setInterval(() => {
    drawBox(200, 150, 120, 120, "Happy (82%)");
}, 1000);