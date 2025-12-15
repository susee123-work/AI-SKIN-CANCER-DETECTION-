import base64
import csv
import os
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors

# ---------- Save Feedback to CSV ----------
def save_feedback(image_name, predicted, confidence, feedback):
    """
    Save each model prediction as a record in feedback_db.csv
    (Creates file with header if missing)
    """
    try:
        file_exists = os.path.isfile("feedback_db.csv")
        with open("feedback_db.csv", "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "image", "predicted", "confidence", "feedback"])
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                image_name,
                predicted,
                confidence,
                feedback
            ])
        print("✅ Feedback saved successfully.")
    except Exception as e:
        print(f"⚠️ Failed to save feedback: {e}")


# ---------- Generate Enhanced PDF Report ----------
def generate_pdf_report(output_path, diagnosis, confidence, risk_level, gradcam_img_b64):
    """
    Creates a professional AI diagnostic report as a PDF file.
    Embeds Grad-CAM image if available.
    """
    try:
        c = canvas.Canvas(output_path, pagesize=A4)
        width, height = A4

        # --- Header ---
        c.setFont("Helvetica-Bold", 22)
        c.setFillColor(colors.darkblue)
        c.drawCentredString(width / 2, height - 50, "AI-Powered Skin Cancer Diagnostic Report")

        # --- Basic Info ---
        c.setFont("Helvetica", 14)
        c.setFillColor(colors.black)
        c.drawString(50, height - 100, f"Report Generated On: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # --- Diagnosis ---
        c.setFont("Helvetica-Bold", 16)
        if "Safe" in diagnosis:
            c.setFillColor(colors.green)
        else:
            c.setFillColor(colors.red)
        c.drawString(50, height - 130, f"Diagnosis: {diagnosis}")

        c.setFont("Helvetica", 14)
        c.setFillColor(colors.black)
        c.drawString(50, height - 155, f"Confidence: {confidence}%")
        c.drawString(50, height - 175, f"Risk Level: {risk_level}")

        # --- Grad-CAM Visualization ---
        if gradcam_img_b64:
            try:
                img_data = base64.b64decode(gradcam_img_b64)
                temp_path = "temp_gradcam.png"
                with open(temp_path, "wb") as f:
                    f.write(img_data)
                # Maintain aspect ratio (width, height)
                c.drawImage(temp_path, 100, height - 430, width=400, height=250, preserveAspectRatio=True)
                os.remove(temp_path)
            except Exception as e:
                print(f"⚠️ Could not render Grad-CAM in PDF: {e}")

        # --- Footer Note ---
        c.setFont("Helvetica-Oblique", 11)
        c.setFillColor(colors.darkgray)
        c.drawString(
            50,
            80,
            "⚠️ Note: This AI-generated report is for clinical support only. Always consult a medical professional."
        )

        # --- Emergency Info ---
        if "Alert" in diagnosis or "CANCER" in diagnosis.upper():
            c.setFont("Helvetica-Bold", 13)
            c.setFillColor(colors.red)
            c.drawString(50, 60, "🚨 Emergency: Contact your dermatologist or nearest hospital immediately!")

        c.save()
        print(f"✅ PDF report generated: {output_path}")

    except Exception as e:
        print(f"❌ Failed to generate PDF report: {e}")
