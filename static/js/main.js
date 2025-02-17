document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const progressSection = document.getElementById('progressSection');
    const resultsSection = document.getElementById('resultsSection');
    const progressBar = document.querySelector('.progress-bar');
    const statusText = document.getElementById('statusText');
    const detectedEmotion = document.getElementById('detectedEmotion');
    const colorPreview = document.getElementById('colorPreview');
    const downloadBtn = document.getElementById('downloadBtn');
    const processBtn = document.getElementById('processBtn');
    const previewVideo = document.getElementById('previewVideo');

    let outputVideoPath = null;

    function showToast(message, isError = false) {
        Toastify({
            text: message,
            duration: 3000,
            gravity: "top",
            position: "right",
            backgroundColor: isError ? "#dc3545" : "#198754",
            stopOnFocus: true,
        }).showToast();
    }

    // Preview uploaded video
    document.getElementById('videoFile').addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            if (file.size > 100 * 1024 * 1024) {
                showToast('Video file size must be less than 100MB', true);
                this.value = '';
                return;
            }
            const url = URL.createObjectURL(file);
            previewVideo.src = url;
            previewVideo.classList.remove('d-none');
        }
    });

    uploadForm.addEventListener('submit', async function(e) {
        e.preventDefault();

        const formData = new FormData();
        const scriptFile = document.getElementById('scriptFile').files[0];
        const videoFile = document.getElementById('videoFile').files[0];

        if (!scriptFile || !videoFile) {
            showToast('Please select both script and video files', true);
            return;
        }

        formData.append('script', scriptFile);
        formData.append('video', videoFile);

        // Show progress section
        progressSection.classList.remove('d-none');
        resultsSection.classList.add('d-none');
        processBtn.disabled = true;
        statusText.classList.remove('text-danger');

        try {
            // Start progress animation
            let progress = 0;
            const progressInterval = setInterval(() => {
                progress += 2;
                if (progress <= 95) {
                    progressBar.style.width = `${progress}%`;
                    statusText.textContent = `Processing... ${progress}%`;
                }
            }, 500);

            // Make API call
            const response = await fetch('/process', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Processing failed');
            }

            // Complete progress
            clearInterval(progressInterval);
            progressBar.style.width = '100%';
            statusText.textContent = 'Processing complete!';

            // Show results
            outputVideoPath = data.output_video;
            detectedEmotion.textContent = data.dominant_emotion;
            colorPreview.style.backgroundColor = data.color;
            resultsSection.classList.remove('d-none');

            // Update preview video with processed result
            const processedVideoUrl = `/download/${outputVideoPath}`;
            previewVideo.src = processedVideoUrl;
            previewVideo.classList.remove('d-none');

            showToast('Video processed successfully!');

        } catch (error) {
            console.error('Error:', error);
            clearInterval(progressInterval);
            progressBar.style.width = '0%';
            statusText.textContent = `Error: ${error.message}`;
            statusText.classList.add('text-danger');
            showToast(error.message, true);
        } finally {
            processBtn.disabled = false;
        }
    });

    downloadBtn.addEventListener('click', function() {
        if (outputVideoPath) {
            window.location.href = `/download/${encodeURIComponent(outputVideoPath)}`;
        } else {
            showToast('No processed video available', true);
        }
    });
});