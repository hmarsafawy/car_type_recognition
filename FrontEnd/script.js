const imageInput = document.getElementById("imageInput");
const predictBtn = document.getElementById("predictBtn");
const clearBtn = document.getElementById("clearBtn");
const preview = document.getElementById("preview");
const resultDiv = document.getElementById("result");
const dropZone = document.getElementById("dropZone");

predictBtn.addEventListener("click", uploadImage);
clearBtn.addEventListener("click", clearResults);
imageInput.addEventListener("change", renderPreview);
dropZone.addEventListener("dragover", handleDragOver);
dropZone.addEventListener("dragleave", handleDragLeave);
dropZone.addEventListener("drop", handleDrop);

function renderPreview() {
    const file = imageInput.files[0];
    if (!file) {
        preview.innerHTML = `<div class="preview-placeholder"><span>No image selected</span></div>`;
        return;
    }

    preview.innerHTML = `<img src="${URL.createObjectURL(file)}" alt="Selected car image">`;
}

function handleDragOver(event) {
    event.preventDefault();
    event.stopPropagation();
    dropZone.classList.add("dragging");
}

function handleDragLeave(event) {
    event.preventDefault();
    event.stopPropagation();
    dropZone.classList.remove("dragging");
}

function handleDrop(event) {
    event.preventDefault();
    event.stopPropagation();
    dropZone.classList.remove("dragging");

    const files = event.dataTransfer.files;
    if (!files || files.length === 0) {
        return;
    }

    const file = files[0];
    const dataTransfer = new DataTransfer();
    dataTransfer.items.add(file);
    imageInput.files = dataTransfer.files;
    renderPreview();
}

async function uploadImage() {
    const file = imageInput.files[0];
    if (!file) {
        resultDiv.innerHTML = `<div class="result-empty">Please select an image before predicting.</div>`;
        return;
    }

    predictBtn.disabled = true;
    predictBtn.textContent = "Predicting...";
    resultDiv.innerHTML = `<div class="result-empty">Working on your prediction...</div>`;

    const formData = new FormData();
    formData.append("file", file);

    try {
        const baseUrl = window.location.protocol === "file:" ? "http://127.0.0.1:8000" : window.location.origin;
        const response = await fetch(`${baseUrl}/predict`, {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Backend connection failed (${response.status})`);
        }

        const data = await response.json();
        renderResult(data);
    } catch (error) {
        console.error(error);
        resultDiv.innerHTML = `<div class="result-empty"><strong>Error:</strong> ${error.message}</div>`;
    } finally {
        predictBtn.disabled = false;
        predictBtn.textContent = "Predict";
    }
}

function renderResult(data) {
    const confidence = (data.confidence * 100).toFixed(1);
    const scores = data.all_scores || {};

    const scoreRows = Object.entries(scores).map(([cls, score]) => {
        const percent = (score * 100).toFixed(1);
        return `
            <div class="score-row">
                <div>
                    <div class="score-label">${cls}</div>
                    <div class="progbar"><div class="progbar-fill" style="width:${percent}%"></div></div>
                </div>
                <div class="score-value">${percent}%</div>
            </div>
        `;
    }).join("");

    resultDiv.innerHTML = `
        <div class="result-heading">
            <div>
                <h3>Prediction</h3>
                <p class="score-label">${data.predicted_class}</p>
            </div>
            <div class="result-badge">Confidence ${confidence}%</div>
        </div>
        ${scoreRows}
    `;
}

function clearResults() {
    imageInput.value = "";
    preview.innerHTML = `<div class="preview-placeholder"><span>No image selected</span></div>`;
    resultDiv.innerHTML = `<div class="result-empty">Prediction results will appear here.</div>`;
    predictBtn.disabled = false;
    predictBtn.textContent = "Predict";
}

clearResults();
