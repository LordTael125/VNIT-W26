// Classifiers Studio (Assignment 09) Interactive JavaScript Engine

document.addEventListener('DOMContentLoaded', function () {
    // =========================================================
    // 1. SCROLL POSITION PRESERVATION & HASH ANCHORING
    // =========================================================
    // If an explicit target hash exists (e.g. #step-results, #step-summary, #step-prediction), prioritize navigating to it!
    if (window.location.hash) {
        sessionStorage.removeItem('restore_scroll_pos');
        const target = document.querySelector(window.location.hash);
        if (target) {
            setTimeout(() => {
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }, 80);
        }
    } else {
        const savedScroll = sessionStorage.getItem('restore_scroll_pos');
        if (savedScroll !== null) {
            sessionStorage.removeItem('restore_scroll_pos');
            const scrollY = parseInt(savedScroll, 10);
            window.scrollTo({
                top: scrollY,
                behavior: 'instant'
            });
        }
    }

    // Only save scroll position for prediction form or class renaming, NOT for training or target switching!
    document.querySelectorAll('form').forEach(form => {
        form.addEventListener('submit', function () {
            if (form.id === 'prediction-form' || form.id === 'rename-form') {
                sessionStorage.setItem('restore_scroll_pos', window.scrollY.toString());
            } else {
                sessionStorage.removeItem('restore_scroll_pos');
            }
        });
    });

    // =========================================================
    // 2. CLASSIFIER DROPDOWN SWITCHER (Parzen vs Ho-Kashyap vs KNN vs Bayes)
    // =========================================================
    const classifierSelect = document.getElementById('classifier_choice');
    const panelParzen = document.getElementById('panel-parzen');
    const panelHk = document.getElementById('panel-hokashyap');
    const panelKnn = document.getElementById('panel-knn');
    const panelBayes = document.getElementById('panel-bayes');

    function updateClassifierUI() {
        if (!classifierSelect) return;
        const selected = classifierSelect.value;
        const panels = {
            'parzen': panelParzen,
            'hokashyap': panelHk,
            'knn': panelKnn,
            'bayes': panelBayes
        };
        for (const [key, panel] of Object.entries(panels)) {
            if (!panel) continue;
            const isActive = (selected === key);
            panel.style.display = isActive ? 'block' : 'none';
            // Disable inputs in hidden panels so browser doesn't validate inactive inputs
            panel.querySelectorAll('input, select').forEach(el => {
                el.disabled = !isActive;
            });
        }
    }

    if (classifierSelect) {
        classifierSelect.addEventListener('change', updateClassifierUI);
        updateClassifierUI();
    }

    // =========================================================
    // 3. TRAINING PROGRESS SPINNER & FEEDBACK (NON-BLOCKING)
    // =========================================================
    const trainingForm = document.getElementById('classifier-training-form');
    const spinnerContainer = document.getElementById('training-spinner-container');
    const trainSubmitBtn = document.getElementById('train-submit-btn');
    const spinnerText = document.getElementById('training-spinner-text');

    if (trainingForm && trainSubmitBtn) {
        trainingForm.addEventListener('submit', function () {
            sessionStorage.removeItem('restore_scroll_pos');
            // Use pointerEvents & opacity instead of button.disabled so form submission is NEVER aborted!
            trainSubmitBtn.style.pointerEvents = 'none';
            trainSubmitBtn.style.opacity = '0.7';
            trainSubmitBtn.innerHTML = `
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" style="display:inline-block; vertical-align:middle; margin-right:8px; animation: spin 1s linear infinite;">
                    <circle cx="12" cy="12" r="10" stroke-opacity="0.25"/>
                    <path d="M12 2a10 10 0 0 1 10 10" stroke="#ffffff"/>
                </svg>
                Executing Classifier... Please wait
            `;
            if (spinnerContainer) {
                spinnerContainer.style.display = 'block';
                if (classifierSelect && spinnerText) {
                    const cname = classifierSelect.options[classifierSelect.selectedIndex].text.split('(')[0].trim();
                    spinnerText.textContent = `Executing ${cname}... Evaluating model and computing performance plots.`;
                }
            }
        });
    }

    // =========================================================
    // 4. FILE UPLOAD NAME INDICATOR
    // =========================================================
    const fileInput = document.getElementById('dataset_file');
    const fileNameDisplay = document.getElementById('chosen-file-display');
    if (fileInput) {
        fileInput.addEventListener('change', function () {
            if (this.files && this.files.length > 0) {
                const name = this.files[0].name;
                const sizeKb = (this.files[0].size / 1024).toFixed(1);
                if (fileNameDisplay) {
                    fileNameDisplay.textContent = 'Selected: ' + name + ' (' + sizeKb + ' KB)';
                    fileNameDisplay.style.color = '#34d399';
                    fileNameDisplay.style.fontWeight = 'bold';
                }
            }
        });
    }

    // =========================================================
    // 5. MATHEMATICAL FORMULA TYPESETTING (KATEX)
    // =========================================================
    if (window.renderMathInElement) {
        try {
            renderMathInElement(document.body, {
                delimiters: [
                    {left: "$$", right: "$$", display: true},
                    {left: "\\[", right: "\\]", display: true},
                    {left: "$", right: "$", display: false},
                    {left: "\\(", right: "\\)", display: false}
                ],
                throwOnError: false
            });
        } catch (e) {
            console.warn("KaTeX rendering warning:", e);
        }
    }
});
