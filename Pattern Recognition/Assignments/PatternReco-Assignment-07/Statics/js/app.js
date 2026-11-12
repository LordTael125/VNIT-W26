// Classifiers Studio Interactive Enhancements

document.addEventListener('DOMContentLoaded', function () {
    // =========================================================
    // 1. SCROLL POSITION PRESERVATION (PREVENTS RESET TO TOP)
    // =========================================================
    const savedScroll = sessionStorage.getItem('restore_scroll_pos');

    if (savedScroll !== null) {
        sessionStorage.removeItem('restore_scroll_pos');
        const scrollY = parseInt(savedScroll, 10);

        const predictionBox = document.querySelector('.prediction-result-box');
        if (window.location.hash === '#step-prediction' && predictionBox) {
            setTimeout(() => {
                predictionBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }, 60);
        } else {
            // Restore exact scroll position instantly to prevent page jump to top
            window.scrollTo({
                top: scrollY,
                behavior: 'instant'
            });
        }
    } else if (window.location.hash) {
        const target = document.querySelector(window.location.hash);
        if (target) {
            setTimeout(() => {
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }, 60);
        }
    }

    // Save scroll position on any form submission
    document.querySelectorAll('form').forEach(form => {
        form.addEventListener('submit', function () {
            sessionStorage.setItem('restore_scroll_pos', window.scrollY.toString());
        });
    });

    // =========================================================
    // 2. DYNAMIC NUMERIC TARGET CLASS RENAMING
    // =========================================================
    const targetSelect = document.getElementById('target_col_select');
    const renameContainer = document.getElementById('rename-classes-container');
    const renameGrid = document.getElementById('rename-inputs-grid');

    function checkTargetNumeric() {
        if (!targetSelect || !window.COLUMN_INFO) return;
        const chosenCol = targetSelect.value;
        const colMeta = window.COLUMN_INFO[chosenCol];

        if (colMeta && colMeta.is_numeric && colMeta.unique_values && colMeta.unique_values.length >= 2 && colMeta.unique_values.length <= 15) {
            if (renameContainer) renameContainer.style.display = 'block';
            if (renameGrid) {
                // Collect existing typed values
                const existingInputs = {};
                renameGrid.querySelectorAll('input').forEach(inp => {
                    const code = inp.name.replace('rename_class_', '');
                    existingInputs[code] = inp.value;
                });

                const currentCodes = Array.from(renameGrid.querySelectorAll('input')).map(inp => inp.name.replace('rename_class_', ''));
                const newCodes = colMeta.unique_values.map(String);
                const isSame = currentCodes.length === newCodes.length && currentCodes.every((val, idx) => val === newCodes[idx]);

                if (!isSame) {
                    renameGrid.innerHTML = '';
                    colMeta.unique_values.forEach(val => {
                        const item = document.createElement('div');
                        item.className = 'rename-input-item';
                        const currentVal = existingInputs[val] || '';
                        item.innerHTML = `
                            <span class="rename-code-badge">Class ${val}</span>
                            <span class="rename-arrow">&rarr;</span>
                            <input type="text" name="rename_class_${val}" class="form-input rename-field"
                                   placeholder="Enter name for Class ${val}"
                                   value="${currentVal}">
                        `;
                        renameGrid.appendChild(item);
                    });
                }
            }
        } else {
            if (renameContainer) renameContainer.style.display = 'none';
        }
    }

    if (targetSelect) {
        targetSelect.addEventListener('change', checkTargetNumeric);
    }

    // =========================================================
    // 3. FILE UPLOAD NAME INDICATOR
    // =========================================================
    const fileInput = document.getElementById('csv-file-input');
    const fileNameDisplay = document.getElementById('chosen-file-display');

    if (fileInput && fileNameDisplay) {
        fileInput.addEventListener('change', function () {
            if (this.files && this.files.length > 0) {
                fileNameDisplay.textContent = 'Selected: ' + this.files[0].name + ' (' + (this.files[0].size / 1024).toFixed(1) + ' KB)';
                fileNameDisplay.style.color = '#34d399';
            } else {
                fileNameDisplay.textContent = 'No file selected';
                fileNameDisplay.style.color = '#a5b4fc';
            }
        });
    }

    // =========================================================
    // 4. SELECT ALL / DESELECT ALL FEATURES CHECKBOXES
    // =========================================================
    const selectAllBtn = document.getElementById('btn-select-all-features');
    const deselectAllBtn = document.getElementById('btn-deselect-all-features');
    const featureCheckboxes = document.querySelectorAll('.feature-checkbox');

    if (selectAllBtn && featureCheckboxes.length > 0) {
        selectAllBtn.addEventListener('click', function (e) {
            e.preventDefault();
            featureCheckboxes.forEach(cb => cb.checked = true);
        });
    }

    if (deselectAllBtn && featureCheckboxes.length > 0) {
        deselectAllBtn.addEventListener('click', function (e) {
            e.preventDefault();
            featureCheckboxes.forEach(cb => cb.checked = false);
        });
    }

    // =========================================================
    // 5. CLASSIFIER DROPDOWN SWITCHER (KNN vs Bayes)
    // =========================================================
    const classifierSelect = document.getElementById('classifier_choice_select');
    const knnControls = document.getElementById('knn-specific-controls');
    const bayesNotice = document.getElementById('bayes-specific-notice');

    function updateClassifierUI() {
        if (!classifierSelect) return;
        const selected = classifierSelect.value;
        if (selected === 'knn') {
            if (knnControls) knnControls.style.display = 'block';
            if (bayesNotice) bayesNotice.style.display = 'none';
        } else {
            if (knnControls) knnControls.style.display = 'none';
            if (bayesNotice) bayesNotice.style.display = 'block';
        }
    }

    if (classifierSelect) {
        classifierSelect.addEventListener('change', updateClassifierUI);
        updateClassifierUI();
    }

    // =========================================================
    // 6. MAX K SLIDER LIVE LABEL
    // =========================================================
    const kSlider = document.getElementById('max_k_slider');
    const kDisplay = document.getElementById('max_k_display');
    if (kSlider && kDisplay) {
        kSlider.addEventListener('input', function () {
            kDisplay.textContent = this.value;
        });
    }

    // =========================================================
    // 7. MATHEMATICAL FORMULA TYPESETTING (KATEX + FALLBACK)
    // =========================================================
    function formatFormulaToHtml(formula) {
        let f = formula.trim();

        // Fractions: \frac{A}{B}
        f = f.replace(/\\frac\{([^{}]+)\}\{([^{}]+)\}/g, function(_, num, den) {
            return `<span class="math-frac"><span class="math-num">${formatFormulaToHtml(num)}</span><span class="math-den">${formatFormulaToHtml(den)}</span></span>`;
        });

        // Bold & Calligraphic
        f = f.replace(/\\mathbf\{([^{}]+)\}/g, '<b>$1</b>');
        f = f.replace(/\\mathcal\{([^{}]+)\}/g, '<b>$1</b>');

        // Square roots: \sqrt{...}
        f = f.replace(/\\sqrt\{([^{}]+)\}/g, function(_, inner) {
            return `&radic;<span class="math-sqrt-body">${formatFormulaToHtml(inner)}</span>`;
        });

        // Sums: \sum_{j=1}^D
        f = f.replace(/\\sum_\{([^{}]+)\}\^\{?([a-zA-Z0-9]+)\}?/g, '&sum;<sub>$1</sub><sup>$2</sup>');
        f = f.replace(/\\sum/g, '&sum;');

        // Greek letters & math symbols
        f = f.replace(/\\mu/g, '&mu;');
        f = f.replace(/\\sigma/g, '&sigma;');
        f = f.replace(/\\pi/g, '&pi;');
        f = f.replace(/\\mid/g, '|');
        f = f.replace(/\\neq/g, '&ne;');
        f = f.replace(/\\times/g, '&times;');

        // Superscripts & Subscripts
        f = f.replace(/\^\{([^{}]+)\}/g, '<sup>$1</sup>');
        f = f.replace(/\^([a-zA-Z0-9])/g, '<sup>$1</sup>');
        f = f.replace(/_\{([^{}]+)\}/g, '<sub>$1</sub>');
        f = f.replace(/_([a-zA-Z0-9])/g, '<sub>$1</sub>');
        return `<span class="math-expr">${f}</span>`;
    }

    function applyMathFallback(element) {
        if (!element) return;
        const walker = document.createTreeWalker(
            element,
            NodeFilter.SHOW_TEXT,
            {
                acceptNode: function(node) {
                    if (!node.parentElement) return NodeFilter.FILTER_REJECT;
                    const tag = node.parentElement.tagName;
                    if (['SCRIPT', 'STYLE', 'CODE', 'PRE', 'TEXTAREA'].includes(tag)) {
                        return NodeFilter.FILTER_REJECT;
                    }
                    return (node.nodeValue.includes('\\(') || node.nodeValue.includes('\\[')) 
                        ? NodeFilter.FILTER_ACCEPT 
                        : NodeFilter.FILTER_SKIP;
                }
            }
        );

        const nodesToReplace = [];
        while (walker.nextNode()) {
            nodesToReplace.push(walker.currentNode);
        }

        nodesToReplace.forEach(node => {
            const text = node.nodeValue;
            if (!text.includes('\\(') && !text.includes('\\[')) return;

            let html = text.replace(/\\\[([\s\S]*?)\\\]/g, function(_, formula) {
                return `<div class="math-display">${formatFormulaToHtml(formula)}</div>`;
            });
            html = html.replace(/\\\(([\s\S]*?)\\\)/g, function(_, formula) {
                return formatFormulaToHtml(formula);
            });

            const span = document.createElement('span');
            span.innerHTML = html;
            node.parentElement.replaceChild(span, node);
        });
    }

    function renderMathFormulas() {
        if (typeof window.renderMathInElement === 'function') {
            try {
                window.renderMathInElement(document.body, {
                    delimiters: [
                        { left: '$$', right: '$$', display: true },
                        { left: '\\[', right: '\\]', display: true },
                        { left: '\\(', right: '\\)', display: false },
                        { left: '$', right: '$', display: false }
                    ],
                    throwOnError: false
                });
                return;
            } catch (e) {
                console.warn("KaTeX rendering encountered an error, falling back:", e);
            }
        }
        applyMathFallback(document.body);
    }

    // Render immediately if KaTeX is loaded, otherwise poll or fallback
    if (typeof window.renderMathInElement === 'function') {
        renderMathFormulas();
    } else {
        let attempts = 0;
        const mathPoll = setInterval(function () {
            attempts++;
            if (typeof window.renderMathInElement === 'function') {
                clearInterval(mathPoll);
                renderMathFormulas();
            } else if (attempts >= 10) {
                clearInterval(mathPoll);
                renderMathFormulas();
            }
        }, 50);
        window.addEventListener('load', renderMathFormulas);
    }
});
