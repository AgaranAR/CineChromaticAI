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

    // Preview uploaded video
    document.getElementById('videoFile').addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
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
            alert('Please select both script and video files');
            return;
        }

        formData.append('script', scriptFile);
        formData.append('video', videoFile);

        // Show progress section
        progressSection.classList.remove('d-none');
        resultsSection.classList.add('d-none');
        processBtn.disabled = true;

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

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Processing failed');
            }

            const data = await response.json();

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

        } catch (error) {
            console.error('Error:', error);
            statusText.textContent = `Error: ${error.message}`;
            statusText.classList.add('text-danger');
        } finally {
            processBtn.disabled = false;
        }
    });

    downloadBtn.addEventListener('click', function() {
        if (outputVideoPath) {
            window.location.href = `/download/${encodeURIComponent(outputVideoPath)}`;
        }
    });
});