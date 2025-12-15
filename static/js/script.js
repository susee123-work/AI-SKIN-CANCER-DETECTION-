// ======= Elements =======
const loginBtn = document.getElementById("loginBtn");
const hospitalIdInput = document.getElementById("hospitalId");
const loginMsg = document.getElementById("login-msg");
const loginPanel = document.getElementById("login-panel");
const predictionPanel = document.getElementById("prediction-panel");

const predictBtn = document.getElementById("predictBtn");
const imageFile = document.getElementById("imageFile");
const diagnosisText = document.getElementById("diagnosis");
const confidenceText = document.getElementById("confidence");
const riskText = document.getElementById("risk");
const gradcamImg = document.getElementById("gradcamImg");
const downloadPdf = document.getElementById("downloadPdf");
const resultDiv = document.getElementById("result");
const predictMsg = document.getElementById("predict-msg");
const confidenceBar = document.getElementById("confidence-bar");

const logoutBtn = document.getElementById("logoutBtn");

// Keep last PDF base64 from server here
let lastPdfB64 = null;

// ======= Login =======
loginBtn.addEventListener("click", async () => {
    const hospitalId = hospitalIdInput.value.trim();
    if (!hospitalId) {
        loginMsg.innerText = "⚠️ Please enter Hospital ID";
        loginMsg.style.color = "#e74c3c";
        return;
    }

    loginBtn.disabled = true;
    loginBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Logging in...';
    loginMsg.innerText = "Logging in... ⏳";
    loginMsg.style.color = "#3498db";

    try {
        const res = await fetch("/login", {
            method: "POST",
            headers: { "Content-Type": "application/x-www-form-urlencoded" },
            body: `hospitalId=${encodeURIComponent(hospitalId)}`,
            credentials: "same-origin"
        });

        const data = await res.json();

        if (data.success) {
            loginMsg.innerText = "✅ Login successful!";
            loginMsg.style.color = "#27ae60";
            loginPanel.classList.add("hidden");
            predictionPanel.classList.remove("hidden");
        } else {
            loginMsg.innerText = "❌ " + (data.message || "Login failed");
            loginMsg.style.color = "#e74c3c";
        }
    } catch (err) {
        loginMsg.innerText = "⚠️ Server error: " + err;
        loginMsg.style.color = "#e67e22";
        console.error(err);
    } finally {
        loginBtn.disabled = false;
        loginBtn.innerHTML = '<i class="fas fa-sign-in-alt"></i> Login';
    }
});

// ======= Logout =======
logoutBtn.addEventListener("click", async () => {
    try {
        await fetch("/logout", { credentials: "same-origin" });
        loginPanel.classList.remove("hidden");
        predictionPanel.classList.add("hidden");
        loginMsg.innerText = "";
        hospitalIdInput.value = "";
        resultDiv.classList.add("hidden");
        imageFile.value = "";
        confidenceBar.style.width = "0";
        gradcamImg.src = "";
        gradcamImg.style.display = "none";
        predictMsg.innerText = "";
        downloadPdf.style.display = "none";
        lastPdfB64 = null;
    } catch (err) {
        console.error("Logout failed:", err);
    }
});

// ======= Prediction =======
predictBtn.addEventListener("click", async () => {
    if (!imageFile.files[0]) {
        predictMsg.innerText = "⚠️ Please select an image first!";
        predictMsg.style.color = "#e74c3c";
        return;
    }

    const formData = new FormData();
    formData.append("file", imageFile.files[0]);

    predictBtn.disabled = true;
    predictBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Predicting...';
    predictMsg.innerText = "Predicting... ⏳";
    predictMsg.style.color = "#3498db";

    try {
        const res = await fetch("/predict", {
            method: "POST",
            body: formData,
            credentials: "same-origin"
        });

        // If server returned non-JSON error body
        if (!res.ok) {
            let text;
            try { text = await res.text(); } catch { text = res.statusText; }
            predictMsg.innerText = `❌ Server error: ${res.status} ${text}`;
            predictMsg.style.color = "#e74c3c";
            return;
        }

        const data = await res.json();

        if (data.error) {
            predictMsg.innerText = "❌ Error: " + (data.error || "Prediction failed");
            predictMsg.style.color = "#e74c3c";
            return;
        }

        // ===== Show result =====
        resultDiv.classList.remove("hidden");
        diagnosisText.innerText = "Diagnosis: " + (data.prediction || "UNKNOWN").toUpperCase();
        riskText.innerText = "Risk Level: " + (data.risk_level || "-");
        confidenceText.innerText = "Confidence: " + (data.confidence ?? 100) + "%";

        // ===== Animate Confidence Bar =====
        confidenceBar.style.width = "0";
        confidenceBar.classList.remove("prediction-danger", "prediction-safe");
        setTimeout(() => {
            const conf = data.confidence ?? 100;
            confidenceBar.style.width = `${conf}%`;
            if ((data.prediction || "").toLowerCase() === "malignant") {
                confidenceBar.classList.add("prediction-danger");
            } else {
                confidenceBar.classList.add("prediction-safe");
            }
        }, 100);

        // ===== Grad-CAM =====
        if (data.gradcam) {
            gradcamImg.style.display = "block";
            gradcamImg.src = "data:image/png;base64," + data.gradcam;
            // optional fade-in handled via CSS class
            gradcamImg.classList.remove("fade-in");
            gradcamImg.onload = () => gradcamImg.classList.add("fade-in");
        } else {
            gradcamImg.style.display = "none";
        }

        // ===== PDF Download (use base64 from predict response) =====
        if (data.pdf_b64) {
            lastPdfB64 = data.pdf_b64; // store for download action
            downloadPdf.style.display = "inline-block";
            downloadPdf.onclick = () => {
                // create blob for better browser support
                const byteCharacters = atob(lastPdfB64);
                const byteNumbers = new Array(byteCharacters.length);
                for (let i = 0; i < byteCharacters.length; i++) {
                    byteNumbers[i] = byteCharacters.charCodeAt(i);
                }
                const byteArray = new Uint8Array(byteNumbers);
                const blob = new Blob([byteArray], { type: "application/pdf" });
                const url = URL.createObjectURL(blob);
                const link = document.createElement("a");
                link.href = url;
                link.download = "SkinCancerReport.pdf";
                document.body.appendChild(link);
                link.click();
                link.remove();
                URL.revokeObjectURL(url);
            };
        } else {
            lastPdfB64 = null;
            downloadPdf.style.display = "none";
            downloadPdf.onclick = null;
        }

        predictMsg.innerText = "✅ Prediction complete!";
        predictMsg.style.color = "#27ae60";

    } catch (err) {
        predictMsg.innerText = "⚠️ Server error: " + err;
        predictMsg.style.color = "#e67e22";
        console.error(err);
    } finally {
        predictBtn.disabled = false;
        predictBtn.innerHTML = '<i class="fas fa-microscope"></i> Predict';
    }
});
