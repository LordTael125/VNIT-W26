/* ==============================================================================
   Pattern Recognition Lab - Assignment 10
   Client-Side Application Scripts
   ============================================================================== */

document.addEventListener('DOMContentLoaded', () => {
    // 1. Initialize KaTeX math rendering
    if (typeof renderMathInElement === 'function') {
        renderMathInElement(document.body, {
            delimiters: [
                { left: '$$', right: '$$', display: true },
                { left: '\\[', right: '\\]', display: true },
                { left: '$', right: '$', display: false },
                { left: '\\(', right: '\\)', display: false }
            ],
            throwOnError: false
        });
    }

    // 2. Safe form submission handlers:
    // IMPORTANT: DO NOT set submitButton.disabled = true synchronously in submit handlers!
    // That cancels form POST submission in Chromium. Use pointerEvents = 'none' and styling instead.
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', (e) => {
            const submitBtn = form.querySelector('button[type="submit"]');
            if (submitBtn) {
                submitBtn.style.pointerEvents = 'none';
                submitBtn.style.opacity = '0.65';
                const originalText = submitBtn.innerHTML;
                submitBtn.innerHTML = `<span>Processing...</span>`;
            }
        });
    });

    // 3. Image file dropzone & input preview (Main upload)
    const fileInput = document.getElementById('image_file');
    const dropzone = document.getElementById('main_dropzone');
    const fileSelectedLabel = document.getElementById('file_selected_label');

    if (fileInput && dropzone) {
        dropzone.addEventListener('click', () => fileInput.click());

        fileInput.addEventListener('change', () => {
            if (fileInput.files && fileInput.files[0]) {
                const file = fileInput.files[0];
                if (fileSelectedLabel) {
                    fileSelectedLabel.textContent = `Selected: ${file.name} (${Math.round(file.size / 1024)} KB)`;
                    fileSelectedLabel.style.display = 'block';
                }
            }
        });

        // Drag & drop visual feedback
        ['dragenter', 'dragover'].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                dropzone.style.borderColor = '#38bdf8';
                dropzone.style.background = 'rgba(56, 189, 248, 0.12)';
            });
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                dropzone.style.borderColor = '';
                dropzone.style.background = '';
            });
        });

        dropzone.addEventListener('drop', (e) => {
            if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                fileInput.files = e.dataTransfer.files;
                const file = e.dataTransfer.files[0];
                if (fileSelectedLabel) {
                    fileSelectedLabel.textContent = `Selected: ${file.name} (${Math.round(file.size / 1024)} KB)`;
                    fileSelectedLabel.style.display = 'block';
                }
            }
        });
    }

    // 4. Query Image file dropzone & input preview (Pattern Recognition query)
    const queryInput = document.getElementById('query_image_file');
    const queryDropzone = document.getElementById('query_dropzone');
    const querySelectedLabel = document.getElementById('query_selected_label');

    if (queryInput && queryDropzone) {
        queryDropzone.addEventListener('click', () => queryInput.click());

        queryInput.addEventListener('change', () => {
            if (queryInput.files && queryInput.files[0]) {
                const file = queryInput.files[0];
                if (querySelectedLabel) {
                    querySelectedLabel.textContent = `Query Selected: ${file.name} (${Math.round(file.size / 1024)} KB)`;
                    querySelectedLabel.style.display = 'block';
                }
            }
        });

        queryDropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                queryInput.files = e.dataTransfer.files;
                const file = e.dataTransfer.files[0];
                if (querySelectedLabel) {
                    querySelectedLabel.textContent = `Query Selected: ${file.name} (${Math.round(file.size / 1024)} KB)`;
                    querySelectedLabel.style.display = 'block';
                }
            }
        });
    }

    // 5. Garbage Collector / Cleanup button handler
    const cleanupBtn = document.getElementById('btn_cleanup');
    if (cleanupBtn) {
        cleanupBtn.addEventListener('click', async (e) => {
            e.preventDefault();
            cleanupBtn.style.pointerEvents = 'none';
            cleanupBtn.style.opacity = '0.6';
            cleanupBtn.textContent = 'Cleaning...';

            try {
                const response = await fetch('/cleanup', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' }
                });
                const data = await response.json();
                showToast(data.message || 'Temporary files cleaned successfully!');
            } catch (err) {
                showToast('Error during garbage collection.');
            } finally {
                cleanupBtn.style.pointerEvents = 'auto';
                cleanupBtn.style.opacity = '1';
                cleanupBtn.textContent = 'Clean Temp Files';
            }
        });
    }
});

// Toast notification helper
function showToast(message) {
    const toast = document.getElementById('toast');
    if (!toast) return;
    toast.textContent = message;
    toast.classList.add('show');
    setTimeout(() => {
        toast.classList.remove('show');
    }, 4000);
}
