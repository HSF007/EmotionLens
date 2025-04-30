document.addEventListener('DOMContentLoaded', function() {
    const imageUpload = document.getElementById('imageUpload');
    const preview = document.getElementById('preview');
    const predictBtn = document.getElementById('predictBtn');
    const resultContainer = document.getElementById('resultContainer');
    const emotionResult = document.getElementById('emotionResult');
    const confidenceBar = document.getElementById('confidenceBar');
    const confidenceText = document.getElementById('confidenceText');

    // Handle image selection
    imageUpload.addEventListener('change', function() {
        const file = this.files[0];
        if (file) {
            const reader = new FileReader();
            
            reader.onload = function(e) {
                preview.src = e.target.result;
                preview.style.display = 'block';
                predictBtn.disabled = false;
                resultContainer.style.display = 'none';
            }
            
            reader.readAsDataURL(file);
        }
    });

    // Handle predict button click
    predictBtn.addEventListener('click', function() {
        if (imageUpload.files.length === 0) {
            return;
        }

        // Show loading state
        predictBtn.disabled = true;
        predictBtn.textContent = 'Processing...';
        
        const formData = new FormData();
        formData.append('file', imageUpload.files[0]);
        
        fetch('/predict/', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            // Display results
            emotionResult.textContent = capitalizeFirstLetter(data.emotion);
            
            const confidencePercent = Math.round(data.confidence * 100);
            confidenceBar.style.width = `${confidencePercent}%`;
            confidenceText.textContent = `Confidence: ${confidencePercent}%`;
            
            // Apply color based on emotion
            applyEmotionColor(data.emotion);
            
            resultContainer.style.display = 'block';
            predictBtn.textContent = 'Predict Emotion';
            predictBtn.disabled = false;
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Error processing the image. Please try again.');
            predictBtn.textContent = 'Predict Emotion';
            predictBtn.disabled = false;
        });
    });
    
    // Helper function to capitalize first letter
    function capitalizeFirstLetter(string) {
        return string.charAt(0).toUpperCase() + string.slice(1);
    }
    
    // Apply color based on detected emotion
    function applyEmotionColor(emotion) {
        const colors = {
            anger: '#e74c3c',
            disgust: '#8e44ad',
            fear: '#7f8c8d',
            happiness: '#f1c40f',
            sadness: '#3498db',
            surprise: '#e67e22',
            neutral: '#95a5a6'
        };
        
        emotionResult.style.color = colors[emotion] || '#333';
        confidenceBar.style.backgroundColor = colors[emotion] || '#3498db';
    }
});