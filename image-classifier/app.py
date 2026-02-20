"""
Image Classifier Web App
Uses PyTorch + MobileNetV2 (pretrained on ImageNet) for image classification.
"""

import io
import json
import base64
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from PIL import Image

# ── Try importing torch; graceful error if not installed ──────────────────────
try:
    import torch
    import torchvision.transforms as transforms
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB max upload

# ── Load model once at startup ────────────────────────────────────────────────
model = None
transform = None
labels = []

def load_model():
    global model, transform, labels

    if not TORCH_AVAILABLE:
        return

    # Load ImageNet class labels
    labels_path = Path(__file__).parent / "model" / "imagenet_classes.json"
    if labels_path.exists():
        with open(labels_path) as f:
            labels = json.load(f)

    # Load pretrained MobileNetV2
    weights = MobileNet_V2_Weights.IMAGENET1K_V1
    model = mobilenet_v2(weights=weights)
    model.eval()

    # Standard ImageNet preprocessing
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    print("✅ MobileNetV2 loaded successfully")


def classify_image(image: Image.Image) -> list[dict]:
    """Run inference and return top-5 predictions."""
    if not TORCH_AVAILABLE or model is None:
        return [{"label": "PyTorch not installed", "confidence": 0.0}]

    # Ensure RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    tensor = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    # Top 5 predictions
    top5_prob, top5_idx = torch.topk(probabilities, 5)
    results = []
    for prob, idx in zip(top5_prob.tolist(), top5_idx.tolist()):
        label = labels[idx] if labels and idx < len(labels) else f"Class {idx}"
        # Clean up label (ImageNet labels are often like "n01234567 golden_retriever")
        if " " in label:
            label = label.split(" ", 1)[1]
        label = label.replace("_", " ").title()
        results.append({
            "label": label,
            "confidence": round(prob * 100, 2)
        })

    return results


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", torch_available=TORCH_AVAILABLE)


@app.route("/classify", methods=["POST"])
def classify():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    allowed = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
    if Path(file.filename).suffix.lower() not in allowed:
        return jsonify({"error": "Unsupported file type. Use JPG, PNG, or WebP."}), 400

    try:
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes))
    except Exception:
        return jsonify({"error": "Could not open image. File may be corrupt."}), 400

    # Classify
    predictions = classify_image(image)

    # Return thumbnail as base64 for preview
    image.thumbnail((400, 400))
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=85)
    preview_b64 = base64.b64encode(buf.getvalue()).decode()

    return jsonify({
        "predictions": predictions,
        "preview": f"data:image/jpeg;base64,{preview_b64}",
        "filename": file.filename,
        "size": f"{image.width}×{image.height}px"
    })


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    load_model()
    print("🚀 Starting Image Classifier at http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
