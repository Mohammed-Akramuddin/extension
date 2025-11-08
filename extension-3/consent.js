const consentCheckbox = document.getElementById('consentCheckbox');
const acceptBtn = document.getElementById('acceptBtn');
const declineBtn = document.getElementById('declineBtn');

// Enable/disable accept button based on checkbox
consentCheckbox.addEventListener('change', () => {
    acceptBtn.disabled = !consentCheckbox.checked;
});

// Handle accept button click
acceptBtn.addEventListener('click', async () => {
    try {
        acceptBtn.disabled = true;
        const btnText = acceptBtn.querySelector('.btn-text');
        if (btnText) {
            btnText.textContent = 'Saving...';
        } else {
            acceptBtn.textContent = 'Saving...';
        }
        
        // Save consent to storage
        await chrome.storage.local.set({ 
            consentGiven: true, 
            consentTimestamp: Date.now() 
        });
        
        // Update button to show success
        if (btnText) {
            btnText.textContent = 'Accepted!';
        } else {
            acceptBtn.textContent = 'Accepted!';
        }
        acceptBtn.style.background = 'linear-gradient(135deg, #10b981, #059669)';
        
        // Close window after short delay
        setTimeout(() => {
            window.close();
        }, 800);
    } catch (error) {
        console.error('Error saving consent:', error);
        acceptBtn.disabled = false;
        const btnText = acceptBtn.querySelector('.btn-text');
        if (btnText) {
            btnText.textContent = 'Accept & Continue';
        } else {
            acceptBtn.textContent = 'Accept & Continue';
        }
        alert('Error saving consent. Please try again.');
    }
});

// Handle decline button click
declineBtn.addEventListener('click', () => {
    const confirmed = confirm('You must accept the terms to use this extension. Do you want to close the consent window?');
    if (confirmed) {
        window.close();
    }
});
