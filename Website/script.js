let allPredictions  = [];
let allClassifications = [];
let loadedImage     = null;
let activeIndex     = null;
let confidenceThreshold = 0.50;

const canvas        = document.getElementById("viewer-canvas");
const ctx           = canvas.getContext("2d");
const dropZone      = document.getElementById("drop-zone");
const fileInput     = document.getElementById("file-input");
const spinner       = document.getElementById("spinner");
const uploadBtn     = document.getElementById("upload-btn");
const slider        = document.getElementById("confidence-slider");
const confValue     = document.getElementById("confidence-value");
const findingsList  = document.getElementById("findings-list");
const findingsCount = document.getElementById("findings-count");
const statusDot     = document.getElementById("status-dot");
const statusText    = document.getElementById("status-text");
const errorToast    = document.getElementById("error-toast");
const classificationList = document.getElementById("classification-list");
const classificationCount = document.getElementById("classification-count");
const clearBtn = document.getElementById("clear-btn");

const CLASS_COLOURS = [
"#00e5a0", "#f5a623", "#e84040", "#4fc3f7", "#ce93d8",
"#ffb74d", "#81c784", "#f48fb1", "#80cbc4", "#fff176",
"#bcaaa4", "#90caf9", "#a5d6a7", "#ffcc02", "#ff7043",
];

const LOCAL_API = "http://127.0.0.1:5000";

function colourForClass(classId) {
return CLASS_COLOURS[classId % CLASS_COLOURS.length];
}

async function checkServerHealth() {
const base = LOCAL_API;
try {
    const response = await fetch(`${base}/health`, { signal: AbortSignal.timeout(2000), cache: "no-store" });
    if (response.ok) {
    statusDot.classList.add("online");
    statusText.textContent = "local server online";
    return true;
    }
} catch (_) {}
statusDot.classList.remove("online");
statusText.textContent = "local server offline";
return false;
}

checkServerHealth();
setInterval(checkServerHealth, 5000);

function showError(message) {
errorToast.textContent = message;
errorToast.classList.add("visible");
setTimeout(() => errorToast.classList.remove("visible"), 4000);
}

uploadBtn.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("click",  () => fileInput.click());

fileInput.addEventListener("change", (event) => {
const file = event.target.files[0];
if (file) handleImageFile(file);
});

dropZone.addEventListener("dragover", (event) => {
event.preventDefault();
dropZone.classList.add("dragging");
});

dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragging"));

dropZone.addEventListener("drop", (event) => {
event.preventDefault();
dropZone.classList.remove("dragging");
const file = event.dataTransfer.files[0];
if (file && file.type.startsWith("image/")) handleImageFile(file);
});

async function handleImageFile(file) {
const imageUrl = URL.createObjectURL(file);
loadedImage    = await loadHTMLImage(imageUrl);

allClassifications = [];
dropZone.classList.add("hidden");
canvas.style.display = "block";
spinner.classList.add("visible");
allPredictions = [];
activeIndex    = null;
renderCanvas();
renderFindingsList();

const formData = new FormData();
formData.append("image", file);

const base = LOCAL_API;
try {
    const response = await fetch(`${base}/predict`, { method: "POST", body: formData, cache: "no-store" });
    if (!response.ok) {
    const errorBody = await response.json().catch(() => ({}));
    throw new Error(errorBody.detail || errorBody.error || `HTTP ${response.status}`);
    }
    const data     = await response.json();
    allPredictions = data.predictions || [];
    allClassifications = data.classifications || [];
} catch (error) {
    showError(`Inference failed: ${error.message}`);
} finally {
    spinner.classList.remove("visible");
}

renderClassifications();
renderCanvas();
renderFindingsList();
}

function clearViewer() {
    loadedImage = null;
    allPredictions = [];
    allClassifications = [];
    activeIndex = null;

    fileInput.value = "";

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    canvas.style.display = "none";

    dropZone.classList.remove("hidden");
    spinner.classList.remove("visible");

    renderClassifications();
    renderFindingsList();
}

function renderClassifications() {
classificationCount.textContent = allClassifications.length;

if (allClassifications.length === 0) {
    classificationList.innerHTML = `<div class="empty-state">No classification results.</div>`;
    return;
}

classificationList.innerHTML = "";

allClassifications.forEach((item) => {
    const div = document.createElement("div");
    div.className = "finding-item";

    div.innerHTML = `
    <div class="finding-row">
        <span class="finding-name">${item.class_name}</span>
        <span class="finding-conf">${(item.probability * 100).toFixed(1)}%</span>
    </div>
    <div class="conf-bar-track">
        <div class="conf-bar-fill" style="width:${item.probability * 100}%"></div>
    </div>
    `;

    classificationList.appendChild(div);
});
}

function loadHTMLImage(src) {
return new Promise((resolve, reject) => {
    const img  = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src    = src;
});
}

function visiblePredictions() {
return allPredictions.filter(p => p.confidence >= confidenceThreshold);
}

function renderCanvas() {
if (!loadedImage) return;

const container = canvas.parentElement;
const scale     = Math.min(
    container.clientWidth  / loadedImage.width,
    container.clientHeight / loadedImage.height,
);
canvas.width  = loadedImage.width  * scale;
canvas.height = loadedImage.height * scale;

ctx.drawImage(loadedImage, 0, 0, canvas.width, canvas.height);

const visible = visiblePredictions();
visible.forEach((prediction, index) => {
    const colour    = colourForClass(prediction.class_id);
    const isActive  = activeIndex === index;

    const scaleX    = canvas.width  / prediction.image_width;
    const scaleY    = canvas.height / prediction.image_height;

    const x         = prediction.x_min * scaleX;
    const y         = prediction.y_min * scaleY;
    const boxWidth  = (prediction.x_max - prediction.x_min) * scaleX;
    const boxHeight = (prediction.y_max - prediction.y_min) * scaleY;

    ctx.save();

    ctx.fillStyle = colour + (isActive ? "28" : "14");
    ctx.fillRect(x, y, boxWidth, boxHeight);

    ctx.strokeStyle = colour;
    ctx.lineWidth   = isActive ? 2.5 : 1.5;
    ctx.setLineDash(isActive ? [] : []);
    ctx.strokeRect(x, y, boxWidth, boxHeight);

    const cornerSize = Math.min(12, boxWidth * 0.2, boxHeight * 0.2);
    ctx.strokeStyle  = colour;
    ctx.lineWidth    = 2;
    [[x, y, 1, 1], [x + boxWidth, y, -1, 1],
    [x, y + boxHeight, 1, -1], [x + boxWidth, y + boxHeight, -1, -1]].forEach(([cx, cy, dx, dy]) => {
    ctx.beginPath();
    ctx.moveTo(cx + dx * cornerSize, cy);
    ctx.lineTo(cx, cy);
    ctx.lineTo(cx, cy + dy * cornerSize);
    ctx.stroke();
    });

    const labelText    = `${prediction.class_name}  ${(prediction.confidence * 100).toFixed(0)}%`;
    const fontSize     = Math.max(10, Math.min(13, boxWidth * 0.08));
    ctx.font           = `500 ${fontSize}px 'DM Sans', sans-serif`;
    const textWidth    = ctx.measureText(labelText).width;
    const labelPadX    = 7;
    const labelPadY    = 4;
    const labelHeight  = fontSize + labelPadY * 2;
    const labelY       = y > labelHeight + 2 ? y - labelHeight - 2 : y + 2;

    ctx.fillStyle = colour;
    ctx.beginPath();
    ctx.roundRect(x, labelY, textWidth + labelPadX * 2, labelHeight, 3);
    ctx.fill();

    ctx.fillStyle = "#0a0c0f";
    ctx.fillText(labelText, x + labelPadX, labelY + labelHeight - labelPadY - 1);

    ctx.restore();
});
}

function renderFindingsList() {
const visible = visiblePredictions();
findingsCount.textContent = visible.length;

if (visible.length === 0 && allPredictions.length === 0 && !loadedImage) {
    findingsList.innerHTML = `<div class="empty-state">No image loaded.<br>Upload a chest X-ray to begin analysis.</div>`;
    return;
}

if (visible.length === 0) {
    findingsList.innerHTML = `<div class="empty-state">No findings above<br>${(confidenceThreshold * 100).toFixed(0)}% confidence.</div>`;
    return;
}

findingsList.innerHTML = "";
visible.forEach((prediction, index) => {
    const colour   = colourForClass(prediction.class_id);
    const item     = document.createElement("div");
    item.className = "finding-item" + (activeIndex === index ? " active" : "");

    item.innerHTML = `
    <div class="finding-row">
        <span class="finding-name" style="color:${colour}">${prediction.class_name}</span>
        <span class="finding-conf">${(prediction.confidence * 100).toFixed(1)}%</span>
    </div>
    <div class="conf-bar-track">
        <div class="conf-bar-fill" style="width:${prediction.confidence * 100}%;background:${colour}"></div>
    </div>
    <div class="finding-coords">
        x₁ ${Math.round(prediction.x_min)} &nbsp; y₁ ${Math.round(prediction.y_min)} &nbsp;
        x₂ ${Math.round(prediction.x_max)} &nbsp; y₂ ${Math.round(prediction.y_max)}
    </div>
    `;

    item.addEventListener("click", () => {
    activeIndex = activeIndex === index ? null : index;
    renderCanvas();
    renderFindingsList();
    });

    findingsList.appendChild(item);
});
}

slider.addEventListener("input", () => {
confidenceThreshold = slider.value / 100;
confValue.textContent = confidenceThreshold.toFixed(2);
activeIndex = null;
renderCanvas();
renderFindingsList();
});

window.addEventListener("resize", renderCanvas);

clearBtn.addEventListener("click", clearViewer);
canvas.addEventListener("click", () => fileInput.click());