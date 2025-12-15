# app.py
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import torch
from torchvision import transforms
from PIL import Image
import base64, io, os
import cv2
import numpy as np
from models import load_model
from gradcam import generate_gradcam
from feedback_pdf import save_feedback, generate_pdf_report

app = Flask(__name__)
app.secret_key = "your_super_secure_key"

# ==== Device ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Load Model ====
BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "best_model.pth")
model = None
if os.path.exists(model_path):
    try:
        model = load_model(model_path, map_location=device) if "map_location" in load_model.__code__.co_varnames else load_model(model_path)
        model.to(device)
        model.eval()
        print("✅ Model loaded successfully.")
    except Exception as e:
        print("❌ Error loading model:", e)

# ==== Threshold ====
threshold = 0.5
config_path = os.path.join(BASE_DIR, "config.pth")
if os.path.exists(config_path):
    try:
        cfg = torch.load(config_path, map_location=device)
        threshold = float(cfg.get("threshold", threshold))
        print(f"Loaded threshold: {threshold}")
    except:
        print("⚠️ Could not load config.pth. Using default threshold 0.5.")

classes = ["Benign", "Malignant"]

# ==== Image Transform ====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ==== Helpers ====
def encode_image_to_base64(img_array):
    if img_array.ndim == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    img_array = np.clip(img_array * 255.0, 0, 255).astype(np.uint8) if img_array.dtype != np.uint8 else img_array
    _, buf = cv2.imencode(".png", img_array)
    return base64.b64encode(buf).decode("utf-8")

def overlay_heatmap_on_image(orig_pil, heatmap_np):
    orig = np.array(orig_pil.convert("RGB"))
    orig_bgr = orig[:, :, ::-1]
    h, w = orig_bgr.shape[:2]
    if heatmap_np is None:
        return orig_bgr
    heat = heatmap_np.copy()
    if heat.dtype in [np.float32, np.float64]:
        heat = np.clip(heat * 255.0, 0, 255).astype(np.uint8)
    if heat.ndim == 2:
        heat_resized = cv2.resize(heat, (w, h))
        heat_color = cv2.applyColorMap(heat_resized, cv2.COLORMAP_JET)
    else:
        heat_resized = cv2.resize(heat, (w, h))
        heat_color = cv2.cvtColor(heat_resized, cv2.COLOR_GRAY2BGR) if heat_resized.shape[2] != 3 else heat_resized
    overlay = cv2.addWeighted(orig_bgr, 0.6, heat_color, 0.4, 0)
    return overlay

# ==== Routes ====
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/login", methods=["POST"])
def login():
    hospital_id = request.form.get("hospitalId", "").strip()
    authorized_ids = ["SUSEE", "HSTGROUP"]
    if hospital_id in authorized_ids:
        session['hospital_id'] = hospital_id
        return jsonify({"success": True})
    return jsonify({"success": False, "message": "Unauthorized hospital ID"}), 401

@app.route("/predict", methods=["POST"])
def predict():
    if 'hospital_id' not in session:
        return jsonify({"error": "Unauthorized - please login"}), 401
    if model is None:
        return jsonify({"error": "Model not loaded on server"}), 500
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    try:
        img = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": "Could not read uploaded image", "details": str(e)}), 400

    try:
        # Transform & predict
        tensor = transform(img).unsqueeze(0).to(device)
        tensor.requires_grad_()
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        prob = probs[0][1].item()
        label_idx = int(prob > threshold)
        label = classes[label_idx]

        # Grad-CAM
        gradcam_img = generate_gradcam(model, tensor, label_idx)
        gradcam_b64 = encode_image_to_base64(overlay_heatmap_on_image(img, gradcam_img))

        # Risk message
        if label.lower() == "benign":
            confidence = 100
            message = "Safe ✅ Normal skin issue"
            risk_level = "No issues detected"
        else:
            confidence = 100
            message = "CANCER ALERT ❌ Immediate action required!"
            risk_level = "Call emergency"

        # Save feedback
        save_feedback(file.filename, label, confidence, message)

        # Generate PDF & encode
        pdf_path = "temp_report.pdf"
        generate_pdf_report(pdf_path, message, confidence, risk_level, gradcam_b64)
        with open(pdf_path, "rb") as f:
            pdf_b64 = base64.b64encode(f.read()).decode("utf-8")
        os.remove(pdf_path)

        return jsonify({
            "prediction": label.lower(),
            "confidence": confidence,
            "message": message,
            "risk_level": risk_level,
            "gradcam": gradcam_b64,
            "pdf_b64": pdf_b64
        })

    except Exception as e:
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
