import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_file
import numpy as np
from PIL import Image
import io, os, json, uuid
from datetime import datetime
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
import base64
import cv2

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "development_key_change_in_production")
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file

# ========== CONFIGURATION ==========
CLASS_NAMES = [
    "Melanoma",
    "Melanocytic Nevus",
    "Basal Cell Carcinoma",
    "Actinic Keratosis",
    "Benign Keratosis",
    "Dermatofibroma",
    "Vascular Lesion"
]

CLASS_DESCRIPTIONS = {
    "Melanoma": "Most dangerous skin cancer. Requires immediate specialist referral.",
    "Melanocytic Nevus": "Common mole. Usually benign but monitor for changes.",
    "Basal Cell Carcinoma": "Slow-growing, rarely metastatic but requires excision.",
    "Actinic Keratosis": "Pre-cancerous lesion. Should be treated to prevent progression.",
    "Benign Keratosis": "Harmless skin growth. No treatment needed unless symptomatic.",
    "Dermatofibroma": "Benign fibrous nodule. No treatment required.",
    "Vascular Lesion": "Blood vessel abnormality. Usually benign."
}

RISK_LEVELS = {
    "Melanoma": "HIGH",
    "Basal Cell Carcinoma": "MEDIUM",
    "Actinic Keratosis": "MEDIUM",
    "Melanocytic Nevus": "LOW",
    "Benign Keratosis": "LOW",
    "Dermatofibroma": "LOW",
    "Vascular Lesion": "LOW"
}

# ========== ENHANCED MODEL LOADING ==========
class SkinCancerEnsemble(nn.Module):
    def __init__(self, num_classes=7, dropout_rate=0.3):
        super().__init__()
        # Load multiple pre-trained models
        from torchvision import models
        
        # Model 1: EfficientNet-B4
        self.effnet = models.efficientnet_b4(pretrained=True)
        effnet_features = self.effnet.classifier[1].in_features
        self.effnet.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(effnet_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # Model 2: DenseNet-201
        self.densenet = models.densenet201(pretrained=True)
        densenet_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(densenet_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # Attention-based fusion
        self.attention = nn.MultiheadAttention(embed_dim=num_classes*2, num_heads=2)
        self.final_classifier = nn.Sequential(
            nn.Linear(num_classes*2, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        # Uncertainty estimation
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x, n_samples=5, return_uncertainty=False):
        # Enable dropout for uncertainty estimation
        self.train()
        
        predictions = []
        for _ in range(n_samples if return_uncertainty else 1):
            # Get predictions from both models
            eff_out = self.effnet(x)
            dens_out = self.densenet(x)
            
            # Apply dropout for uncertainty
            eff_out = self.dropout(eff_out)
            dens_out = self.dropout(dens_out)
            
            # Concatenate features
            combined = torch.cat([eff_out, dens_out], dim=1)
            
            # Attention fusion
            combined = combined.unsqueeze(0)
            attn_out, _ = self.attention(combined, combined, combined)
            attn_out = attn_out.squeeze(0)
            
            # Final classification
            final = self.final_classifier(attn_out)
            predictions.append(final)
        
        if return_uncertainty:
            predictions = torch.stack(predictions)
            mean_pred = torch.mean(predictions, dim=0)
            std_pred = torch.std(predictions, dim=0)
            uncertainty = torch.mean(std_pred, dim=1)
            return mean_pred, uncertainty
        else:
            return predictions[0]

def load_enhanced_model(model_path="best_model_ensemble.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SkinCancerEnsemble(num_classes=7)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded model from {model_path}")
        print(f"   Validation AUC: {checkpoint.get('val_auc', 0):.4f}")
        print(f"   Best epoch: {checkpoint.get('epoch', 0)}")
    else:
        print("⚠️ No saved model found, using randomly initialized")
    
    model.to(device)
    model.eval()
    return model, device

# ========== IMAGE PROCESSING ==========
class AdvancedImageProcessor:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),  # Higher resolution
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess(self, image):
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply CLAHE for better contrast (medical images)
        img_cv = np.array(image)
        if len(img_cv.shape) == 3:
            lab = cv2.cvtColor(img_cv, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            img_cv = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        image = Image.fromarray(img_cv)
        return self.transform(image)
    
    def calculate_abcde(self, image):
        """Calculate ABCDE rule for melanoma detection"""
        img_cv = np.array(image)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
        
        # A: Asymmetry
        height, width = gray.shape
        left_half = gray[:, :width//2]
        right_half = gray[:, width//2:]
        asymmetry = np.abs(left_half - right_half).mean() / 255.0
        
        # B: Border irregularity
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)
            border_irregularity = (perimeter ** 2) / (4 * np.pi * area) if area > 0 else 1.0
        else:
            border_irregularity = 1.0
        
        # C: Color variation
        if len(img_cv.shape) == 3:
            color_std = np.std(img_cv, axis=(0, 1)).mean() / 255.0
        else:
            color_std = np.std(gray) / 255.0
        
        # D: Diameter (estimate in pixels)
        diameter = np.sqrt(area) if 'area' in locals() and area > 0 else 0
        
        # E: Evolution (cannot determine from single image)
        
        return {
            'asymmetry_score': round(asymmetry, 3),
            'border_irregularity': round(border_irregularity, 3),
            'color_variation': round(color_std, 3),
            'diameter_pixels': round(diameter, 1),
            'abcde_risk': 'HIGH' if (asymmetry > 0.3 or border_irregularity > 1.5) else 'MEDIUM' if (asymmetry > 0.2) else 'LOW'
        }

# ========== ENHANCED GRAD-CAM ==========
class AdvancedGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = []
        self.gradients = []
        
    def save_activation(self, module, input, output):
        self.activations.append(output.cpu().detach())
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients.append(grad_output[0].cpu().detach())
        
    def generate(self, input_tensor, target_class):
        # Register hooks
        h1 = self.target_layer.register_forward_hook(self.save_activation)
        h2 = self.target_layer.register_full_backward_hook(self.save_gradient)
        
        # Forward pass
        output = self.model(input_tensor)
        self.model.zero_grad()
        
        # Backward pass for target class
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot)
        
        # Remove hooks
        h1.remove()
        h2.remove()
        
        # Process activations and gradients
        activations = self.activations[0].squeeze().numpy()
        gradients = self.gradients[0].squeeze().numpy()
        
        # Weight channels by gradient importance
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU and normalization
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (384, 384))
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        
        return cam

# ========== DATABASE (SQLite) ==========
import sqlite3

class PatientDatabase:
    def __init__(self, db_path="patients.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_tables()
    
    def create_tables(self):
        cursor = self.conn.cursor()
        
        # Patients table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patients (
                patient_id TEXT PRIMARY KEY,
                hospital_id TEXT,
                age INTEGER,
                gender TEXT,
                skin_type TEXT,
                family_history BOOLEAN,
                created_at TIMESTAMP
            )
        ''')
        
        # Examinations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS examinations (
                exam_id TEXT PRIMARY KEY,
                patient_id TEXT,
                image_path TEXT,
                primary_diagnosis TEXT,
                confidence REAL,
                secondary_diagnoses TEXT,
                abcde_scores TEXT,
                risk_level TEXT,
                recommendation TEXT,
                doctor_notes TEXT,
                created_at TIMESTAMP,
                FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
            )
        ''')
        
        self.conn.commit()
    
    def add_patient(self, patient_data):
        cursor = self.conn.cursor()
        patient_id = str(uuid.uuid4())[:8]
        
        cursor.execute('''
            INSERT INTO patients VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            patient_id,
            patient_data.get('hospital_id'),
            patient_data.get('age'),
            patient_data.get('gender'),
            patient_data.get('skin_type'),
            patient_data.get('family_history'),
            datetime.now()
        ))
        
        self.conn.commit()
        return patient_id
    
    def add_examination(self, exam_data):
        cursor = self.conn.cursor()
        exam_id = str(uuid.uuid4())[:8]
        
        cursor.execute('''
            INSERT INTO examinations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            exam_id,
            exam_data['patient_id'],
            exam_data['image_path'],
            exam_data['primary_diagnosis'],
            exam_data['confidence'],
            json.dumps(exam_data.get('secondary_diagnoses', [])),
            json.dumps(exam_data.get('abcde_scores', {})),
            exam_data['risk_level'],
            exam_data['recommendation'],
            exam_data.get('doctor_notes', ''),
            datetime.now()
        ))
        
        self.conn.commit()
        return exam_id

# ========== INITIALIZE COMPONENTS ==========
print("🚀 Initializing DermAI Advanced System...")
model, device = load_enhanced_model()
processor = AdvancedImageProcessor()
db = PatientDatabase()

# ========== FLASK ROUTES ==========
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    hospital_id = data.get('hospital_id', '').strip()
    password = data.get('password', '')  # In production, use hashed passwords
    
    # Simulated authentication - replace with real DB
    valid_hospitals = {
        'SUSEE': 'hospital123',
        'HSTGROUP': 'secure456',
        'MAYO': 'mayo789',
        'JOHNSHOPKINS': 'jh2024'
    }
    
    if hospital_id in valid_hospitals and password == valid_hospitals[hospital_id]:
        session['hospital_id'] = hospital_id
        session['authenticated'] = True
        return jsonify({
            'success': True,
            'hospital_name': hospital_id,
            'message': 'Authentication successful'
        })
    
    return jsonify({
        'success': False,
        'message': 'Invalid credentials'
    }), 401

@app.route('/api/patient/register', methods=['POST'])
def register_patient():
    if not session.get('authenticated'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    data = request.json
    patient_id = db.add_patient(data)
    
    return jsonify({
        'success': True,
        'patient_id': patient_id,
        'message': 'Patient registered successfully'
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    if not session.get('authenticated'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    if 'file' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['file']
    patient_id = request.form.get('patient_id', 'unknown')
    
    try:
        # Load and preprocess image
        image = Image.open(io.BytesIO(file.read()))
        
        # Calculate ABCDE scores
        abcde_scores = processor.calculate_abcde(image)
        
        # Preprocess for model
        input_tensor = processor.preprocess(image).unsqueeze(0).to(device)
        
        # Get predictions with uncertainty
        with torch.no_grad():
            model.eval()
            logits = model(input_tensor)
            probabilities = F.softmax(logits, dim=1)
            
            # Get uncertainty using Monte Carlo dropout
            predictions = []
            for _ in range(10):
                model.train()  # Enable dropout
                pred = model(input_tensor)
                predictions.append(F.softmax(pred, dim=1))
            
            predictions = torch.stack(predictions)
            mean_probs = torch.mean(predictions, dim=0)
            std_probs = torch.std(predictions, dim=0)
            uncertainty = torch.mean(std_probs).item()
        
        # Get top 3 predictions
        probs = mean_probs[0].cpu().numpy()
        top3_indices = np.argsort(probs)[-3:][::-1]
        
        primary_diagnosis = CLASS_NAMES[top3_indices[0]]
        primary_confidence = float(probs[top3_indices[0]] * 100)
        
        secondary_diagnoses = [
            {
                'name': CLASS_NAMES[idx],
                'confidence': float(probs[idx] * 100),
                'description': CLASS_DESCRIPTIONS[CLASS_NAMES[idx]]
            }
            for idx in top3_indices[1:]
        ]
        
        # Risk assessment
        risk_level = RISK_LEVELS.get(primary_diagnosis, 'LOW')
        
        # Generate Grad-CAM
        target_layer = model.effnet.features[-1]  # Last conv layer
        gradcam = AdvancedGradCAM(model.effnet, target_layer)
        cam = gradcam.generate(input_tensor, top3_indices[0])
        
        # Create heatmap overlay
        orig_image = np.array(image.resize((384, 384)))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(orig_image, 0.6, heatmap, 0.4, 0)
        
        # Encode images
        _, buffer = cv2.imencode('.png', overlay)
        gradcam_b64 = base64.b64encode(buffer).decode('utf-8')
        
        _, buffer = cv2.imencode('.png', orig_image)
        orig_b64 = base64.b64encode(buffer).decode('utf-8')
        
        # Generate recommendations
        recommendations = generate_recommendations(
            primary_diagnosis, 
            primary_confidence,
            abcde_scores,
            risk_level
        )
        
        # Save to database
        exam_data = {
            'patient_id': patient_id,
            'image_path': f"uploads/{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            'primary_diagnosis': primary_diagnosis,
            'confidence': primary_confidence,
            'secondary_diagnoses': secondary_diagnoses,
            'abcde_scores': abcde_scores,
            'risk_level': risk_level,
            'recommendation': recommendations['action']
        }
        
        exam_id = db.add_examination(exam_data)
        
        # Generate PDF report
        pdf_path = generate_medical_report(exam_data, exam_id, gradcam_b64)
        
        with open(pdf_path, 'rb') as f:
            pdf_b64 = base64.b64encode(f.read()).decode('utf-8')
        
        # Cleanup
        os.remove(pdf_path)
        
        return jsonify({
            'success': True,
            'exam_id': exam_id,
            'primary_diagnosis': primary_diagnosis,
            'confidence': round(primary_confidence, 1),
            'uncertainty': round(uncertainty * 100, 1),
            'secondary_diagnoses': secondary_diagnoses,
            'risk_level': risk_level,
            'abcde_scores': abcde_scores,
            'recommendations': recommendations,
            'gradcam': gradcam_b64,
            'original_image': orig_b64,
            'pdf_base64': pdf_b64,
            'patient_id': patient_id
        })
        
    except Exception as e:
        print(f"Analysis error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/patient/<patient_id>/history')
def patient_history(patient_id):
    cursor = db.conn.cursor()
    cursor.execute('''
        SELECT exam_id, primary_diagnosis, confidence, risk_level, created_at
        FROM examinations
        WHERE patient_id = ?
        ORDER BY created_at DESC
    ''', (patient_id,))
    
    exams = cursor.fetchall()
    
    return jsonify({
        'patient_id': patient_id,
        'examinations': [
            {
                'exam_id': exam[0],
                'diagnosis': exam[1],
                'confidence': exam[2],
                'risk_level': exam[3],
                'date': exam[4]
            }
            for exam in exams
        ]
    })

def generate_recommendations(diagnosis, confidence, abcde_scores, risk_level):
    """Generate clinical recommendations based on diagnosis and risk"""
    recommendations = {
        'immediate_action': '',
        'follow_up': '',
        'referral': '',
        'monitoring': '',
        'action': ''
    }
    
    if diagnosis == "Melanoma":
        recommendations.update({
            'immediate_action': 'Refer to dermatologist within 24-48 hours',
            'follow_up': 'Biopsy and surgical excision required',
            'referral': 'Dermatology oncology specialist',
            'monitoring': 'Complete skin examination every 3-6 months',
            'action': 'URGENT: Requires immediate specialist consultation'
        })
    elif diagnosis == "Basal Cell Carcinoma":
        recommendations.update({
            'immediate_action': 'Schedule dermatology appointment within 2 weeks',
            'follow_up': 'Surgical excision or Mohs surgery recommended',
            'referral': 'General dermatologist',
            'monitoring': 'Annual skin check',
            'action': 'Schedule dermatology appointment'
        })
    elif risk_level == "HIGH" or abcde_scores['abcde_risk'] == "HIGH":
        recommendations.update({
            'immediate_action': 'Dermatology review within 1 week',
            'follow_up': 'Consider biopsy for definitive diagnosis',
            'referral': 'Dermatologist',
            'monitoring': 'Monthly self-examination',
            'action': 'Seek dermatology opinion'
        })
    else:
        recommendations.update({
            'immediate_action': 'Routine follow-up',
            'follow_up': 'Annual skin examination',
            'referral': 'None required',
            'monitoring': 'Self-examination every 6 months',
            'action': 'No immediate action needed'
        })
    
    return recommendations

def generate_medical_report(exam_data, exam_id, gradcam_b64):
    """Generate comprehensive medical report PDF"""
    pdf_path = f"reports/report_{exam_id}.pdf"
    
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor='#2c3e50',
        spaceAfter=30
    )
    story.append(Paragraph("DermAI - Advanced Skin Cancer Diagnostic Report", title_style))
    
    # Patient Info
    story.append(Paragraph(f"<b>Report ID:</b> {exam_id}", styles["Normal"]))
    story.append(Paragraph(f"<b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
    story.append(Spacer(1, 20))
    
    # Diagnosis
    diag_style = ParagraphStyle(
        'Diagnosis',
        parent=styles['Heading2'],
        fontSize=18,
        textColor='#e74c3c' if exam_data['risk_level'] == 'HIGH' else 
                 '#f39c12' if exam_data['risk_level'] == 'MEDIUM' else '#27ae60',
        spaceAfter=15
    )
    story.append(Paragraph(f"Primary Diagnosis: {exam_data['primary_diagnosis']}", diag_style))
    story.append(Paragraph(f"Confidence: {exam_data['confidence']:.1f}%", styles["Normal"]))
    story.append(Paragraph(f"Risk Level: {exam_data['risk_level']}", styles["Normal"]))
    story.append(Spacer(1, 20))
    
    # ABCDE Scores
    story.append(Paragraph("<b>ABCDE Rule Assessment:</b>", styles["Heading3"]))
    abcde = exam_data['abcde_scores']
    story.append(Paragraph(f"Asymmetry: {abcde['asymmetry_score']} (0-1 scale)", styles["Normal"]))
    story.append(Paragraph(f"Border Irregularity: {abcde['border_irregularity']} (>1.3 suggests malignancy)", styles["Normal"]))
    story.append(Paragraph(f"Color Variation: {abcde['color_variation']} (0-1 scale)", styles["Normal"]))
    story.append(Paragraph(f"ABCDE Risk: {abcde['abcde_risk']}", styles["Normal"]))
    story.append(Spacer(1, 20))
    
    # Recommendations
    story.append(Paragraph("<b>Clinical Recommendations:</b>", styles["Heading3"]))
    story.append(Paragraph(exam_data['recommendation'], styles["Normal"]))
    
    # Add Grad-CAM image if available
    if gradcam_b64:
        try:
            img_data = base64.b64decode(gradcam_b64)
            img_path = f"temp_gradcam_{exam_id}.png"
            with open(img_path, 'wb') as f:
                f.write(img_data)
            
            story.append(Spacer(1, 20))
            story.append(Paragraph("<b>AI Attention Map (Grad-CAM):</b>", styles["Heading3"]))
            story.append(Paragraph("Red areas indicate regions the AI focused on for diagnosis", styles["Italic"]))
            
            # Add image to PDF
            img = RLImage(img_path, width=400, height=300)
            story.append(img)
            
            os.remove(img_path)
        except:
            pass
    
    # Footer
    story.append(Spacer(1, 30))
    disclaimer = Paragraph(
        "<i>This report is generated by AI and should be used as a decision support tool. "
        "Final diagnosis must be made by a qualified healthcare professional.</i>",
        ParagraphStyle('Disclaimer', parent=styles['Italic'], fontSize=10, textColor='#7f8c8d')
    )
    story.append(disclaimer)
    
    doc.build(story)
    return pdf_path

@app.route('/api/report/<exam_id>')
def download_report(exam_id):
    # In production, generate on-demand or serve from storage
    return jsonify({'message': 'Report generation endpoint'})

@app.route('/api/stats')
def get_statistics():
    """Return system statistics for dashboard"""
    cursor = db.conn.cursor()
    
    # Total examinations
    cursor.execute('SELECT COUNT(*) FROM examinations')
    total_exams = cursor.fetchone()[0]
    
    # Risk distribution
    cursor.execute('''
        SELECT risk_level, COUNT(*) 
        FROM examinations 
        GROUP BY risk_level
    ''')
    risk_dist = {row[0]: row[1] for row in cursor.fetchall()}
    
    # Recent activity
    cursor.execute('''
        SELECT primary_diagnosis, created_at 
        FROM examinations 
        ORDER BY created_at DESC 
        LIMIT 10
    ''')
    recent = cursor.fetchall()
    
    return jsonify({
        'total_examinations': total_exams,
        'risk_distribution': risk_dist,
        'recent_activity': recent,
        'system_uptime': '24/7',
        'model_version': 'Ensemble v2.0'
    })

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    print("=" * 50)
    print("🏥 DermAI Advanced System")
    print(f"📊 Model: Ensemble (7-class classification)")
    print(f"⚡ Device: {device}")
    print(f"📁 Database: patients.db")
    print("=" * 50)
    
    app.run(host='0.0.0.0', port=5000, debug=True)