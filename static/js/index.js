
// Elements
const loginBtn = document.getElementById("loginBtn");
const hospitalIdInput = document.getElementById("hospitalId");
const loginMsg = document.getElementById("login-msg");
const loginPanel = document.getElementById("login-panel");
const predictionPanel = document.getElementById("prediction-panel");

// Login Event
loginBtn.addEventListener("click", async () => {
    const hospitalId = hospitalIdInput.value.trim();
    if (!hospitalId) {
        loginMsg.innerText = "⚠️ Please enter Hospital ID";
        return;
    }

    loginBtn.disabled = true;
    loginBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Logging in...';

    try {
        const res = await fetch("/login", {
            method: "POST",
            headers: { "Content-Type": "application/x-www-form-urlencoded" },
            body: `hospitalId=${encodeURIComponent(hospitalId)}`,
            credentials: "same-origin"  // include session cookies
        });

        const data = await res.json();

        if (data.success) {
            loginMsg.innerText = "✅ Login successful!";
            loginPanel.classList.add("hidden");
            predictionPanel.classList.remove("hidden");
        } else {
            loginMsg.innerText = "❌ " + (data.message || "Login failed");
        }

    } catch (err) {
        loginMsg.innerText = "⚠️ Server error: " + err;
        console.error(err);
    } finally {
        loginBtn.disabled = false;
        loginBtn.innerHTML = '<i class="fas fa-sign-in-alt"></i> Login';
    }
});
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

predictBtn.addEventListener("click", async () => {
    if (!imageFile.files[0]) {
        alert("⚠️ Please select an image first!");
        return;
    }

    const formData = new FormData();
    formData.append("file", imageFile.files[0]);

    predictBtn.disabled = true;
    predictBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Predicting...';
    predictMsg.innerText = "Predicting... ⏳";

    try {
        const res = await fetch("/predict", {
            method: "POST",
            body: formData,
            credentials: "same-origin"  // include Flask session cookie
        });

        const data = await res.json();

        if (res.status !== 200 || data.error) {
            predictMsg.innerText = "❌ Error: " + (data.error || "Prediction failed");
            return;
        }

        // Show result panel
        resultDiv.classList.remove("hidden");

        // Diagnosis & risk
        diagnosisText.innerText = "Diagnosis: " + data.prediction;
        riskText.innerText = "Risk Level: " + data.risk_level;

        // Confidence & bar animation
        const confidence = data.confidence;
        confidenceText.innerText = "Confidence: " + confidence + "%";

        confidenceBar.style.width = "0";
        confidenceBar.classList.remove("prediction-danger");

        setTimeout(() => {
            confidenceBar.style.width = `${confidence}%`;
            if (data.prediction.toLowerCase() === "malignant") {
                confidenceBar.classList.add("prediction-danger");
            }
        }, 100);

        // Grad-CAM
        gradcamImg.src = "data:image/png;base64," + data.gradcam;

        // PDF download
        downloadPdf.href = "/download_pdf";

        predictMsg.innerText = "";

    } catch (err) {
        predictMsg.innerText = "⚠️ Server error: " + err;
        console.error(err);
    } finally {
        predictBtn.disabled = false;
        predictBtn.innerHTML = '<i class="fas fa-microscope"></i> Predict';
    }
});
