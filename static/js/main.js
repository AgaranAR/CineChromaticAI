document.addEventListener('DOMContentLoaded', function () {

    const uploadForm      = document.getElementById('uploadForm');
    const progressSection = document.getElementById('progressSection');
    const resultsSection  = document.getElementById('resultsSection');
    const progressBar     = document.querySelector('.progress-bar');
    const statusText      = document.getElementById('statusText');
    const detectedEmotion = document.getElementById('detectedEmotion');
    const colorPreview    = document.getElementById('colorPreview');
    const downloadBtn     = document.getElementById('downloadBtn');
    const processBtn      = document.getElementById('processBtn');
    const previewVideo    = document.getElementById('previewVideo');

    // Emotion → representative color (for the color preview swatch)
    const EMOTION_COLORS = {
        happy:   '#FFD700',
        sad:     '#4682B4',
        angry:   '#FF4500',
        fearful: '#4B0082',
        neutral: '#888888',
    };

    let currentJobId      = null;
    let outputVideoPath   = null;
    let activeEventSource = null;

    // -----------------------------------------------------------------------
    // Toast helper
    // -----------------------------------------------------------------------
    function showToast(message, isError = false) {
        Toastify({
            text: message,
            duration: 3000,
            gravity: 'top',
            position: 'right',
            backgroundColor: isError ? '#dc3545' : '#198754',
            stopOnFocus: true,
        }).showToast();
    }

    // -----------------------------------------------------------------------
    // Progress bar helper
    // -----------------------------------------------------------------------
    function setProgress(pct, label) {
        progressBar.style.width = pct + '%';
        progressBar.setAttribute('aria-valuenow', pct);
        statusText.textContent = label;
    }

    // -----------------------------------------------------------------------
    // Video file preview
    // -----------------------------------------------------------------------
    document.getElementById('videoFile').addEventListener('change', function (e) {
        const file = e.target.files[0];
        if (!file) return;
        if (file.size > 100 * 1024 * 1024) {
            showToast('Video file size must be less than 100MB', true);
            this.value = '';
            return;
        }
        previewVideo.src = URL.createObjectURL(file);
        previewVideo.classList.remove('d-none');
    });

    // -----------------------------------------------------------------------
    // Form submit — three clear phases:
    //   Phase 1: POST /process   → get job_id + act breakdown immediately
    //   Phase 2: GET  /progress  → SSE stream, update bar with real numbers
    //   Phase 3: done=true       → trigger /download with job_id guard
    // -----------------------------------------------------------------------
    uploadForm.addEventListener('submit', async function (e) {
        e.preventDefault();

        const scriptFile = document.getElementById('scriptFile').files[0];
        const videoFile  = document.getElementById('videoFile').files[0];

        if (!scriptFile || !videoFile) {
            showToast('Please select both script and video files', true);
            return;
        }

        // Close any previous SSE stream that might still be open
        if (activeEventSource) {
            activeEventSource.close();
            activeEventSource = null;
        }

        const formData = new FormData();
        formData.append('script', scriptFile);
        formData.append('video', videoFile);

        // Reset UI
        progressSection.classList.remove('d-none');
        resultsSection.classList.add('d-none');
        processBtn.disabled = true;
        statusText.classList.remove('text-danger');
        setProgress(0, 'Uploading files…');

        // ------------------------------------------------------------------
        // Phase 1: submit files, get job_id back immediately
        // ------------------------------------------------------------------
        let data;
        try {
            const response = await fetch('/process', {
                method: 'POST',
                body: formData,
            });
            data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Upload failed');
            }
        } catch (err) {
            setProgress(0, 'Error: ' + err.message);
            statusText.classList.add('text-danger');
            processBtn.disabled = false;
            showToast(err.message, true);
            return;
        }

        currentJobId    = data.job_id;
        outputVideoPath = data.output_video;

        // Show the act breakdown the server detected
        if (data.acts && data.acts.length) {
            const dominant = data.acts.reduce(
                (best, a) => (a.tension > best.tension ? a : best),
                data.acts[0]
            );
            detectedEmotion.textContent = dominant.emotion;
            colorPreview.style.backgroundColor =
                EMOTION_COLORS[dominant.emotion] || '#888888';

            // Build a small act summary string for the status line
            const actSummary = data.acts
                .map(a => `Act ${a.act}: ${a.emotion} (tension ${a.tension})`)
                .join(' → ');
            console.log('Acts:', actSummary);
        }

        setProgress(2, 'Processing started…');

        // ------------------------------------------------------------------
        // Phase 2: open SSE stream — real frame-by-frame progress
        // ------------------------------------------------------------------
        activeEventSource = new EventSource(`/progress/${currentJobId}`);

        activeEventSource.onmessage = function (e) {
            let state;
            try {
                state = JSON.parse(e.data);
            } catch {
                return;
            }

            if (state.error) {
                activeEventSource.close();
                activeEventSource = null;
                setProgress(0, 'Error: ' + state.error);
                statusText.classList.add('text-danger');
                processBtn.disabled = false;
                showToast(state.error, true);
                return;
            }

            // Update the real progress bar
            const pct = Math.max(state.pct, 2);  // never show 0% once started
            setProgress(pct, `Processing… ${pct}%`);

            // ----------------------------------------------------------
            // Phase 3: processing finished — NOW trigger the download
            // ----------------------------------------------------------
            if (state.done) {
                activeEventSource.close();
                activeEventSource = null;

                setProgress(100, 'Processing complete!');
                processBtn.disabled = false;

                // Show results panel
                resultsSection.classList.remove('d-none');

                // Update the preview player with the processed video
                // Pass job_id so /download confirms the job is truly done
                const downloadUrl =
                    `/download/${encodeURIComponent(outputVideoPath)}?job_id=${currentJobId}`;
                previewVideo.src = downloadUrl;
                previewVideo.classList.remove('d-none');

                showToast('Video processed successfully!');
            }
        };

        activeEventSource.onerror = function () {
            // SSE connection dropped — could be normal after done,
            // or a real network error mid-stream
            if (activeEventSource) {
                activeEventSource.close();
                activeEventSource = null;
            }
            // Only show an error if we haven't completed successfully
            if (progressBar.style.width !== '100%') {
                setProgress(0, 'Connection lost — please try again');
                statusText.classList.add('text-danger');
                processBtn.disabled = false;
                showToast('Connection lost during processing', true);
            }
        };
    });

    // -----------------------------------------------------------------------
    // Download button — same guard as the preview: requires job to be done
    // -----------------------------------------------------------------------
    downloadBtn.addEventListener('click', function () {
        if (!outputVideoPath) {
            showToast('No processed video available', true);
            return;
        }
        const url = currentJobId
            ? `/download/${encodeURIComponent(outputVideoPath)}?job_id=${currentJobId}`
            : `/download/${encodeURIComponent(outputVideoPath)}`;
        window.location.href = url;
    });

});
