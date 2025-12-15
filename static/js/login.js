document.getElementById("loginBtn").addEventListener("click", async () => {
    const hospitalId = document.getElementById("hospitalId").value.trim();
    const messageEl = document.getElementById("message");

    if (!hospitalId) {
        messageEl.textContent = "Please enter your Hospital ID";
        messageEl.style.color = "red";
        return;
    }

    try {
        const response = await fetch("/login", {
            method: "POST",
            headers: { "Content-Type": "application/x-www-form-urlencoded" },
            body: `hospitalId=${hospitalId}`
        });

        const data = await response.json();
        if (data.success) {
            messageEl.textContent = "Login successful! Redirecting...";
            messageEl.style.color = "green";
            setTimeout(() => {
                window.location.href = "/";
            }, 1000);
        } else {
            messageEl.textContent = data.message || "Unauthorized";
            messageEl.style.color = "red";
        }
    } catch (err) {
        messageEl.textContent = "Server error. Try again later.";
        messageEl.style.color = "red";
    }
});
