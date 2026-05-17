document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const uploadSection = document.getElementById('upload-stage');
    const loadingOverlay = document.getElementById('loading-overlay');
    const resultsSection = document.getElementById('results-section');
    const resetBtn = document.getElementById('reset-btn');
    const originalImage = document.getElementById('original-image');
    const annotatedImage = document.getElementById('annotated-image');
    const detectionsList = document.getElementById('detections-list');
    const noDetections = document.getElementById('no-detections');

    // Drag and Drop Events
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.remove('dragover');
        }, false);
    });

    dropZone.addEventListener('drop', handleDrop, false);
    fileInput.addEventListener('change', handleFileSelect, false);
    resetBtn.addEventListener('click', resetUI, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files && files.length > 0) {
            handleFiles(files[0]);
        }
    }

    function handleFileSelect(e) {
        const files = e.target.files;
        if (files && files.length > 0) {
            handleFiles(files[0]);
        }
    }

    function handleFiles(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file.');
            return;
        }

        // Show original image
        const reader = new FileReader();
        reader.onload = (e) => {
            originalImage.src = e.target.result;
        };
        reader.readAsDataURL(file);

        // Upload and process
        processImage(file);
    }

    async function processImage(file) {
        // Show loading state
        loadingOverlay.classList.remove('hidden');

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                displayResults(data);
            } else {
                alert('Error processing image: ' + data.error);
                loadingOverlay.classList.add('hidden');
            }
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while communicating with the server.');
            loadingOverlay.classList.add('hidden');
        }
    }

    function displayResults(data) {
        // Hide upload section, show results
        uploadSection.classList.add('hidden');
        loadingOverlay.classList.add('hidden');
        resultsSection.classList.remove('hidden');

        // Set annotated image
        annotatedImage.src = 'data:image/jpeg;base64,' + data.image;

        // Clear previous detections
        detectionsList.innerHTML = '';

        if (data.detections && data.detections.length > 0) {
            noDetections.classList.add('hidden');
            detectionsList.classList.remove('hidden');

            // Sort detections by confidence (descending)
            const sortedDetections = data.detections.sort((a, b) => b.confidence - a.confidence);

            sortedDetections.forEach((det) => {
                const item = document.createElement('div');
                item.className = 'detection-item';
                item.innerHTML = `
                    <span class="class-name">${det.class}</span>
                    <span class="confidence-val">${det.confidence}%</span>
                `;
                detectionsList.appendChild(item);
            });
        } else {
            detectionsList.classList.add('hidden');
            noDetections.classList.remove('hidden');
        }
    }

    function resetUI() {
        resultsSection.classList.add('hidden');
        uploadSection.classList.remove('hidden');
        fileInput.value = '';
        originalImage.src = '';
        annotatedImage.src = '';
    }
});
