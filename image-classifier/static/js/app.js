/**
 * VisionLabel — Frontend JS
 * Handles drag-and-drop, file upload, API calls, and results rendering.
 */

(function () {
  "use strict";

  // ── DOM refs ───────────────────────────────────────────────────────────────
  const uploadZone   = document.getElementById("uploadZone");
  const uploadInner  = document.getElementById("uploadInner");
  const browseBtn    = document.getElementById("browseBtn");
  const fileInput    = document.getElementById("fileInput");
  const resultsPanel = document.getElementById("resultsPanel");
  const loadingState = document.getElementById("loadingState");
  const errorState   = document.getElementById("errorState");
  const errorMsg     = document.getElementById("errorMsg");
  const previewImg   = document.getElementById("previewImg");
  const previewMeta  = document.getElementById("previewMeta");
  const predList     = document.getElementById("predList");
  const resetBtn     = document.getElementById("resetBtn");
  const errorResetBtn= document.getElementById("errorResetBtn");

  // ── State helpers ──────────────────────────────────────────────────────────
  function showView(view) {
    uploadZone.style.display   = view === "upload"  ? "block"  : "none";
    resultsPanel.style.display = view === "results" ? "grid"   : "none";
    loadingState.style.display = view === "loading" ? "flex"   : "none";
    errorState.style.display   = view === "error"   ? "flex"   : "none";
  }

  function reset() {
    fileInput.value = "";
    predList.innerHTML = "";
    showView("upload");
  }

  // ── File handling ──────────────────────────────────────────────────────────
  function handleFile(file) {
    if (!file || !file.type.startsWith("image/")) {
      showError("Please upload an image file.");
      return;
    }

    showView("loading");
    const formData = new FormData();
    formData.append("image", file);

    fetch("/classify", { method: "POST", body: formData })
      .then((res) => res.json())
      .then((data) => {
        if (data.error) {
          showError(data.error);
        } else {
          renderResults(data);
        }
      })
      .catch(() => showError("Could not reach the server. Is it running?"));
  }

  function showError(msg) {
    errorMsg.textContent = msg;
    showView("error");
  }

  // ── Results rendering ──────────────────────────────────────────────────────
  function renderResults(data) {
    // Preview image
    previewImg.src = data.preview;
    previewImg.alt = data.filename;
    previewMeta.textContent = `${data.filename}  ·  ${data.size}`;

    // Predictions
    predList.innerHTML = "";
    const topConf = data.predictions[0]?.confidence || 1;

    data.predictions.forEach((pred, i) => {
      const barWidth = ((pred.confidence / Math.max(topConf, 1)) * 100).toFixed(1);
      const item = document.createElement("div");
      item.className = "pred-item";
      item.innerHTML = `
        <div class="pred-row">
          <span class="pred-label" title="${pred.label}">${pred.label}</span>
          <span class="pred-score">${pred.confidence.toFixed(1)}%</span>
        </div>
        <div class="pred-bar-track">
          <div class="pred-bar-fill" style="width:0%" data-width="${barWidth}%"></div>
        </div>
      `;
      predList.appendChild(item);
    });

    showView("results");

    // Animate bars after paint
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        document.querySelectorAll(".pred-bar-fill").forEach((bar) => {
          bar.style.width = bar.dataset.width;
        });
      });
    });
  }

  // ── Event listeners ────────────────────────────────────────────────────────

  // Browse button → open file picker
  browseBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    fileInput.click();
  });

  // Clicking the zone also opens picker
  uploadZone.addEventListener("click", () => fileInput.click());

  // File input change
  fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) handleFile(fileInput.files[0]);
  });

  // Drag and drop
  uploadZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadZone.classList.add("drag-over");
  });

  uploadZone.addEventListener("dragleave", (e) => {
    if (!uploadZone.contains(e.relatedTarget)) {
      uploadZone.classList.remove("drag-over");
    }
  });

  uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadZone.classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    handleFile(file);
  });

  // Reset buttons
  resetBtn.addEventListener("click", reset);
  errorResetBtn.addEventListener("click", reset);

  // Paste image from clipboard
  document.addEventListener("paste", (e) => {
    const items = e.clipboardData?.items;
    if (!items) return;
    for (const item of items) {
      if (item.type.startsWith("image/")) {
        handleFile(item.getAsFile());
        return;
      }
    }
  });

  // Init
  showView("upload");
})();
