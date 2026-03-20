const API_URL = 'http://127.0.0.1:5000';

document.addEventListener("DOMContentLoaded", async () => {
    const modelSelect = document.getElementById("model-select");
    const predictBtn = document.getElementById("predict-btn");
    const errorMsg = document.getElementById("error-msg");
    const scoreValue = document.getElementById("score-value");
    const modelBadge = document.getElementById("current-model-badge");

    let metricsData = {};

    // 1. Fetch available models and metrics on load
    try {
        const response = await fetch(`${API_URL}/models`);
        const data = await response.json();

        metricsData = data.metrics;

        // Populate dropdown
        modelSelect.innerHTML = '';
        data.models.forEach(model => {
            const option = document.createElement("option");
            option.value = model;
            // Format name nicely: random_forest -> Random Forest
            const formattedName = model.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
            option.textContent = formattedName;
            modelSelect.appendChild(option);
        });

        // Initialize metrics for first model
        if (data.models.length > 0) {
            modelBadge.textContent = modelSelect.options[modelSelect.selectedIndex].text;
        }

    } catch (err) {
        console.error("Failed to load models.", err);
        modelSelect.innerHTML = '<option value="">Server Offline</option>';
        errorMsg.textContent = "Could not connect to the backend server.";
    }

    // 2. Handle Model Change
    modelSelect.addEventListener("change", (e) => {
        modelBadge.textContent = e.target.options[e.target.selectedIndex].text;
    });

    // 3. Handle Prediction Request
    predictBtn.addEventListener("click", async () => {
        // Clear previous error
        errorMsg.textContent = "";

        // Get values
        const hours = parseFloat(document.getElementById("hours-studied").value);
        const sleep = parseFloat(document.getElementById("sleep-hours").value);
        const attendance = parseFloat(document.getElementById("attendance").value);
        const previous = parseFloat(document.getElementById("previous-scores").value);

        if (isNaN(hours) || isNaN(sleep) || isNaN(attendance) || isNaN(previous)) {
            errorMsg.textContent = "Please fill in all fields with valid numbers.";
            return;
        }

        const selectedModel = modelSelect.value;
        if (!selectedModel) {
            errorMsg.textContent = "Please select a model.";
            return;
        }

        // Animate button
        predictBtn.style.opacity = '0.7';
        predictBtn.textContent = 'Predicting...';

        try {
            const res = await fetch(`${API_URL}/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    model: selectedModel,
                    features: [hours, sleep, attendance, previous]
                })
            });

            let result;
            try {
                result = await res.json();
            } catch {
                result = { error: 'Invalid server response' };
            }

            if (res.ok) {
                animateScore(result.prediction);
                modelBadge.textContent = modelSelect.options[modelSelect.selectedIndex].text;
            } else {
                errorMsg.textContent = result.error || "An error occurred.";
            }

        } catch (err) {
            errorMsg.textContent = "Failed to connect to the server.";
            console.error(err);
        } finally {
            predictBtn.style.opacity = '1';
            predictBtn.innerHTML = '<span>Generate Prediction</span>';
        }
    });

    // Animate score counting up
    function animateScore(targetScore) {
        const duration = 1500; // ms
        const frameRate = 30; // ms
        const totalFrames = duration / frameRate;
        let currentFrame = 0;

        const target = Math.min(100, Math.max(0, targetScore)); // Clamp between 0 and 100

        const counter = setInterval(() => {
            currentFrame++;
            const progress = currentFrame / totalFrames;
            // Ease out quad
            const easeProgress = progress * (2 - progress);

            const currentVal = (target * easeProgress).toFixed(1);
            scoreValue.textContent = currentVal;

            if (currentFrame >= totalFrames) {
                clearInterval(counter);
                scoreValue.textContent = target.toFixed(1);
            }
        }, frameRate);
    }
});
