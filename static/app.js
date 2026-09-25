const form = document.getElementById("upload-form");
const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const preview = document.getElementById("preview");
const fileName = document.getElementById("file-name");
const submitBtn = document.getElementById("submit-btn");
const statusEl = document.getElementById("status");
const resultEl = document.getElementById("result");
const classLabel = document.getElementById("class-label");
const confidenceText = document.getElementById("confidence-text");
const confidenceBar = document.getElementById("confidence-bar");
const overlayImg = document.getElementById("overlay-img");
const heatmapImg = document.getElementById("heatmap-img");

let selectedFile = null;

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.classList.toggle("is-error", isError);
}

function setFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    setStatus("Bitte eine Bilddatei wählen.", true);
    return;
  }

  selectedFile = file;
  fileName.textContent = file.name;
  submitBtn.disabled = false;
  setStatus("");
  resultEl.hidden = true;

  const reader = new FileReader();
  reader.onload = () => {
    preview.src = reader.result;
    preview.hidden = false;
    dropZone.classList.add("has-preview");
  };
  reader.readAsDataURL(file);
}

["dragenter", "dragover"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("is-dragover");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("is-dragover");
  });
});

dropZone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  setFile(file);
});

fileInput.addEventListener("change", () => {
  setFile(fileInput.files[0]);
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!selectedFile) {
    setStatus("Bitte zuerst ein Bild ablegen.", true);
    return;
  }

  const body = new FormData();
  body.append("image", selectedFile);

  submitBtn.disabled = true;
  setStatus("Klassifiziere …");

  try {
    const response = await fetch("/classify", { method: "POST", body });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Klassifikation fehlgeschlagen.");
    }

    const percent = Math.round(data.confidence * 1000) / 10;
    classLabel.textContent = data.class_label;
    confidenceText.textContent = `${percent.toLocaleString("de-DE")} % Sicherheit (Softmax) für ${data.class_name}`;
    confidenceBar.style.width = `${Math.max(4, percent)}%`;
    overlayImg.src = data.overlay;
    heatmapImg.src = data.heatmap;
    resultEl.hidden = false;
    setStatus("");
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    submitBtn.disabled = !selectedFile;
  }
});
