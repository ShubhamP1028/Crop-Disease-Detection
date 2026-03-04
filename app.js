// Plantia AI - Modern Glassmorphism UI
// Basic navigation setup for Task 1

// StateManager Class - Manages application state and LocalStorage persistence
class StateManager {
    constructor() {
        this.currentScan = null;
        this.recentScans = [];
        this.loadFromLocalStorage();
    }
    
    getCurrentScan() {
        return this.currentScan;
    }
    
    setCurrentScan(scanData) {
        this.currentScan = scanData;
    }
    
    getRecentScans() {
        return this.recentScans;
    }
    
    addRecentScan(scanData) {
        // Add to beginning of array
        this.recentScans.unshift(scanData);
        
        // Keep only last 10 items
        if (this.recentScans.length > 10) {
            this.recentScans = this.recentScans.slice(0, 10);
        }
        
        this.saveToLocalStorage();
    }
    
    clearRecentScans() {
        this.recentScans = [];
        localStorage.removeItem('recentScans');
    }
    
    saveToLocalStorage() {
        try {
            localStorage.setItem('recentScans', JSON.stringify(this.recentScans));
        } catch (e) {
            console.error('Failed to save to localStorage:', e);
            // Handle QuotaExceededError by removing oldest scans
            if (e.name === 'QuotaExceededError') {
                console.warn('LocalStorage quota exceeded, reducing to 5 items');
                // Remove oldest items and retry
                this.recentScans = this.recentScans.slice(0, 5);
                try {
                    localStorage.setItem('recentScans', JSON.stringify(this.recentScans));
                } catch (retryError) {
                    console.error('Failed to save even after reducing items:', retryError);
                }
            }
        }
    }
    
    loadFromLocalStorage() {
        try {
            const stored = localStorage.getItem('recentScans');
            if (stored) {
                this.recentScans = JSON.parse(stored);
                // Ensure we don't exceed 10 items on load
                if (this.recentScans.length > 10) {
                    this.recentScans = this.recentScans.slice(0, 10);
                    this.saveToLocalStorage();
                }
            }
        } catch (e) {
            console.error('Failed to load from localStorage:', e);
            this.recentScans = [];
        }
    }
}

// NavigationManager Class - Manages client-side routing between screens
class NavigationManager {
    constructor(stateManager) {
        this.stateManager = stateManager;
        this.currentScreen = 'dashboard';
        this.navigationHistory = ['dashboard'];
        this.setupBottomNav();
        this.setupBackButtons();
    }
    
    /**
     * Navigate to a specific screen and optionally pass data
     * @param {string} screenName - Name of the screen to navigate to (dashboard, camera, result)
     * @param {object} data - Optional data to pass to the screen
     */
    navigateTo(screenName, data = null) {
        // If data is provided, pass it to StateManager
        if (data) {
            this.stateManager.setCurrentScan(data);
        }
        
        // Hide all screens
        document.querySelectorAll('.screen').forEach(screen => {
            screen.classList.remove('active');
        });
        
        // Show target screen
        const targetScreen = document.getElementById(`${screenName}-screen`);
        if (targetScreen) {
            targetScreen.classList.add('active');
            
            // Update current screen
            this.currentScreen = screenName;
            
            // Add to navigation history
            this.navigationHistory.push(screenName);
        } else {
            console.error(`Screen not found: ${screenName}`);
        }
        
        // Update bottom navigation active state
        this.updateBottomNavState(screenName);
    }
    
    /**
     * Get the current active screen name
     * @returns {string} Current screen name
     */
    getCurrentScreen() {
        return this.currentScreen;
    }
    
    /**
     * Navigate back to the previous screen or dashboard
     */
    goBack() {
        // Remove current screen from history
        if (this.navigationHistory.length > 1) {
            this.navigationHistory.pop();
            // Get previous screen
            const previousScreen = this.navigationHistory[this.navigationHistory.length - 1];
            // Navigate without adding to history again
            this.navigateToWithoutHistory(previousScreen);
        } else {
            // Default to dashboard if no history
            this.navigateTo('dashboard');
        }
    }
    
    /**
     * Navigate without adding to history (used by goBack)
     * @param {string} screenName - Name of the screen to navigate to
     */
    navigateToWithoutHistory(screenName) {
        // Hide all screens
        document.querySelectorAll('.screen').forEach(screen => {
            screen.classList.remove('active');
        });
        
        // Show target screen
        const targetScreen = document.getElementById(`${screenName}-screen`);
        if (targetScreen) {
            targetScreen.classList.add('active');
            this.currentScreen = screenName;
        }
        
        // Update bottom navigation active state
        this.updateBottomNavState(screenName);
    }
    
    /**
     * Update bottom navigation active state
     * @param {string} screenName - Name of the active screen
     */
    updateBottomNavState(screenName) {
        document.querySelectorAll('.nav-item').forEach(item => {
            item.classList.remove('active');
            if (item.dataset.screen === screenName) {
                item.classList.add('active');
            }
        });
    }
    
    /**
     * Set up bottom navigation click handlers
     */
    setupBottomNav() {
        const navItems = document.querySelectorAll('.nav-item');
        
        navItems.forEach(item => {
            item.addEventListener('click', () => {
                const screenName = item.dataset.screen;
                this.navigateTo(screenName);
            });
        });
    }
    
    /**
     * Set up back button handlers for Camera and Result screens
     */
    setupBackButtons() {
        const backButtons = document.querySelectorAll('.back-btn');
        
        backButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                // Navigate to dashboard when back button is clicked
                this.navigateTo('dashboard');
            });
        });
    }
}

// DashboardScreen Class - Manages dashboard functionality
class DashboardScreen {
    constructor(stateManager, navigationManager) {
        this.stateManager = stateManager;
        this.navigationManager = navigationManager;
        this.currentCarouselIndex = 0;
        this.carouselAutoAdvanceTimer = null;
        this.init();
    }
    
    /**
     * Initialize dashboard screen
     */
    init() {
        this.setupCarousel();
        this.setupScanButton();
        this.renderRecentScans();
    }
    
    /**
     * Set up carousel functionality with paging dots
     */
    setupCarousel() {
        const carouselContainer = document.querySelector('.carousel-container');
        const dots = document.querySelectorAll('.paging-dots .dot');
        
        if (!carouselContainer || dots.length === 0) {
            console.error('Carousel elements not found');
            return;
        }
        
        // Set up paging dot click handlers
        dots.forEach((dot, index) => {
            dot.addEventListener('click', () => {
                this.navigateCarousel(index);
                // Reset auto-advance timer when user manually navigates
                this.resetAutoAdvanceTimer();
            });
        });
        
        // Start auto-advance timer
        this.startAutoAdvanceTimer();
    }
    
    /**
     * Navigate carousel to specific index
     * @param {number} targetIndex - Index to navigate to
     */
    navigateCarousel(targetIndex) {
        const carouselContainer = document.querySelector('.carousel-container');
        const dots = document.querySelectorAll('.paging-dots .dot');
        const totalItems = dots.length;
        
        // Ensure index is within bounds
        if (targetIndex < 0 || targetIndex >= totalItems) {
            console.error('Invalid carousel index:', targetIndex);
            return;
        }
        
        // Update current index
        this.currentCarouselIndex = targetIndex;
        
        // Animate carousel using CSS transform
        const translateX = -targetIndex * 100;
        carouselContainer.style.transform = `translateX(${translateX}%)`;
        
        // Update paging dots active state
        dots.forEach((dot, index) => {
            if (index === targetIndex) {
                dot.classList.add('active');
            } else {
                dot.classList.remove('active');
            }
        });
    }
    
    /**
     * Start auto-advance timer for carousel
     */
    startAutoAdvanceTimer() {
        this.carouselAutoAdvanceTimer = setInterval(() => {
            const dots = document.querySelectorAll('.paging-dots .dot');
            const totalItems = dots.length;
            
            // Calculate next index (wrap around to 0 if at end)
            const nextIndex = (this.currentCarouselIndex + 1) % totalItems;
            
            // Navigate to next item
            this.navigateCarousel(nextIndex);
        }, 5000); // 5 seconds
    }
    
    /**
     * Reset auto-advance timer
     */
    resetAutoAdvanceTimer() {
        if (this.carouselAutoAdvanceTimer) {
            clearInterval(this.carouselAutoAdvanceTimer);
        }
        this.startAutoAdvanceTimer();
    }
    
    /**
     * Set up scan button to navigate to Camera screen
     */
    setupScanButton() {
        const scanButtons = document.querySelectorAll('.scan-now-btn');
        
        scanButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                this.navigationManager.navigateTo('camera');
            });
        });
    }
    
    /**
     * Render recent scans from StateManager
     */
    renderRecentScans() {
        const container = document.getElementById('recent-scans-container');
        const recentScans = this.stateManager.getRecentScans();
        
        if (!container) {
            console.error('Recent scans container not found');
            return;
        }
        
        // Clear container
        container.innerHTML = '';
        
        // Check if there are any recent scans
        if (recentScans.length === 0) {
            container.innerHTML = '<p style="color: var(--text-secondary); text-align: center;">No recent scans</p>';
            return;
        }
        
        // Create card for each recent scan
        recentScans.forEach(scan => {
            const card = this.createRecentScanCard(scan);
            container.appendChild(card);
        });
    }
    
    /**
     * Create a recent scan card element
     * @param {object} scan - Scan result object
     * @returns {HTMLElement} Card element
     */
    createRecentScanCard(scan) {
        const card = document.createElement('div');
        card.className = 'recent-scan-card glass-card';
        
        // Format timestamp
        const timeAgo = this.formatTimestamp(scan.timestamp);
        
        // Create card HTML
        card.innerHTML = `
            <img src="${scan.image}" alt="${scan.disease}" class="scan-thumbnail">
            <div class="scan-info">
                <h4>${scan.disease.replace(/_/g, ' ')}</h4>
                <p>${scan.plantType}</p>
                <span class="scan-time">${timeAgo}</span>
            </div>
        `;
        
        // Add click handler to navigate to result screen
        card.addEventListener('click', () => {
            this.handleRecentScanClick(scan);
        });
        
        return card;
    }
    
    /**
     * Handle click on recent scan card
     * @param {object} scan - Scan result object
     */
    handleRecentScanClick(scan) {
        // Navigate to result screen with scan data
        this.navigationManager.navigateTo('result', scan);
    }
    
    /**
     * Format timestamp to relative time string
     * @param {number} timestamp - Unix timestamp in milliseconds
     * @returns {string} Formatted time string
     */
    formatTimestamp(timestamp) {
        const date = new Date(timestamp);
        const now = new Date();
        const diffMs = now - date;
        const diffMins = Math.floor(diffMs / 60000);
        
        if (diffMins < 1) return 'Just now';
        if (diffMins < 60) return `${diffMins} min ago`;
        
        const diffHours = Math.floor(diffMins / 60);
        if (diffHours < 24) return `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`;
        
        const diffDays = Math.floor(diffHours / 24);
        if (diffDays < 7) return `${diffDays} day${diffDays > 1 ? 's' : ''} ago`;
        
        return date.toLocaleDateString();
    }
}

// Global instances
let stateManager;
let navigationManager;
let dashboardScreen;
let cameraScreen;
let resultScreen;

document.addEventListener('DOMContentLoaded', () => {
    console.log('Plantia AI initialized');
    
    // Initialize StateManager
    stateManager = new StateManager();
    
    // Initialize NavigationManager with StateManager dependency
    navigationManager = new NavigationManager(stateManager);
    
    // Initialize DashboardScreen
    dashboardScreen = new DashboardScreen(stateManager, navigationManager);
    
    // Initialize CameraScreen
    cameraScreen = new CameraScreen(stateManager, navigationManager);
    
    // Initialize ResultScreen
    resultScreen = new ResultScreen(stateManager, navigationManager);
    
    // Navigate to dashboard as initial screen
    navigationManager.navigateTo('dashboard');
});

// CameraScreen Class - Manages camera/upload screen functionality
class CameraScreen {
    constructor(stateManager, navigationManager) {
        this.stateManager = stateManager;
        this.navigationManager = navigationManager;
        this.stream = null;
        this.cameraSupported = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
        this.init();
    }
    
    /**
     * Initialize camera screen
     */
    init() {
        this.setupFileInput();
        this.setupCancelButton();
        this.setupScreenActivation();
    }
    
    /**
     * Set up file input to trigger on upload button click
     */
    setupFileInput() {
        const uploadBtn = document.getElementById('upload-btn');
        const imageInput = document.getElementById('image-input');
        this.videoElement = document.getElementById('camera-video');
        this.canvasElement = document.getElementById('camera-canvas');

        if (!uploadBtn || !imageInput) {
            console.error('Upload button or image input not found');
            return;
        }

        // Primary action: capture from camera when available, otherwise fall back to file picker
        uploadBtn.addEventListener('click', () => {
            if (this.cameraSupported) {
                this.captureFromCamera();
            } else {
                imageInput.click();
            }
        });

        // Handle file selection fallback
        imageInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                this.handleImageCapture(file);
            }
        });
    }
    
    /**
     * Set up cancel button functionality
     */
    setupCancelButton() {
        const cancelBtn = document.getElementById('cancel-btn');
        
        if (!cancelBtn) {
            console.error('Cancel button not found');
            return;
        }
        
        cancelBtn.addEventListener('click', () => {
            this.handleCancel();
        });
    }

    /**
     * Start camera when camera screen becomes active
     */
    setupScreenActivation() {
        const cameraScreenEl = document.getElementById('camera-screen');

        if (!cameraScreenEl) {
            console.error('Camera screen not found');
            return;
        }

        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                if (mutation.type === 'attributes' && mutation.attributeName === 'class') {
                    if (cameraScreenEl.classList.contains('active')) {
                        // Screen became active
                        this.startCameraIfSupported();
                    } else {
                        // Screen hidden
                        this.stopCamera();
                    }
                }
            });
        });

        observer.observe(cameraScreenEl, { attributes: true });
    }

    async startCameraIfSupported() {
        if (!this.cameraSupported) {
            console.warn('Camera not supported, falling back to file upload.');
            return;
        }

        if (this.stream) {
            return;
        }

        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: 'environment' }
            });
            this.stream = stream;
            if (this.videoElement) {
                this.videoElement.srcObject = stream;
            }
        } catch (error) {
            console.warn('Camera access denied or failed:', error);
            this.showError('Unable to access camera. Please allow camera or use gallery upload.');
        }
    }

    stopCamera() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        if (this.videoElement && this.videoElement.srcObject) {
            this.videoElement.srcObject = null;
        }
    }

    /**
     * Capture a frame from the live camera preview and send to classifier
     */
    captureFromCamera() {
        if (!this.stream || !this.videoElement || !this.canvasElement) {
            // If camera is not ready yet, try to start it and fall back
            this.startCameraIfSupported();
            return;
        }

        const video = this.videoElement;
        const canvas = this.canvasElement;
        const width = video.videoWidth;
        const height = video.videoHeight;

        if (!width || !height) {
            console.warn('Video not ready yet for capture.');
            return;
        }

        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, width, height);

        canvas.toBlob(async (blob) => {
            if (!blob) {
                this.showError('Failed to capture image from camera.');
                return;
            }

            const file = new File([blob], 'camera-capture.png', { type: 'image/png' });
            await this.handleImageCapture(file);
            // Stop camera after successful capture
            this.stopCamera();
        }, 'image/png');
    }
    
    /**
     * Handle image capture/selection with validation
     * @param {File} file - Selected image file
     */
    async handleImageCapture(file) {
        console.log('Image captured:', file.name);
        
        // Validate file type - must be an image (JPEG, PNG, JPG)
        const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
        if (!validTypes.includes(file.type.toLowerCase())) {
            this.showError('Please upload a valid image (JPG, PNG)');
            return;
        }
        
        // Validate file size - max 16MB
        const maxSize = 16 * 1024 * 1024; // 16MB in bytes
        if (file.size > maxSize) {
            this.showError('File size exceeds 16MB. Please upload a smaller image.');
            return;
        }
        
        // If validation passes, proceed with upload
        console.log('Validation passed, proceeding with upload');
        await this.uploadImage(file);
    }
    
    /**
     * Upload image to backend and process response
     * @param {File} file - Image file to upload
     */
    async uploadImage(file) {
        try {
            // Step 1: Initialize upload - 0% progress
            this.updateProgress(0);
            
            // Step 2: Prepare FormData - 30% progress
            const formData = new FormData();
            formData.append('file', file);
            this.updateProgress(30);
            
            // Step 3: Send POST request to /predict endpoint - 60% progress
            this.updateProgress(60);
            
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`Upload failed with status: ${response.status}`);
            }
            
            // Parse backend response
            const data = await response.json();
            console.log('Backend response:', data);
            
            // Step 4: Process response and create ScanResult object
            // Fetch treatment data for the disease
            const treatment = await this.fetchTreatmentData(data.label);
            
            // Calculate risk level
            const riskLevel = this.calculateRiskLevel(data.confidence, data.label);
            
            // Create ScanResult object with all required fields
            const scanResult = {
                id: this.generateUUID(),
                timestamp: Date.now(),
                image: `data:image/png;base64,${data.image_b64}`,
                disease: data.label,
                plantType: this.extractPlantType(data.label),
                confidence: data.confidence,
                allPredictions: data.scores || [],
                treatment: treatment,
                riskLevel: riskLevel
            };
            
            // Step 5: Update progress to 100%
            this.updateProgress(100);
            
            // Step 6: Save scan result to StateManager
            this.stateManager.addRecentScan(scanResult);
            
            // Step 7: Navigate to Result screen after 500ms delay
            setTimeout(() => {
                this.navigationManager.navigateTo('result', scanResult);
                // Reset UI state
                this.handleCancel();
            }, 500);
            
        } catch (error) {
            console.error('Upload error:', error);
            this.showError('Failed to analyze image. Please try again.');
            this.handleCancel();
        }
    }
    
    /**
     * Generate UUID v4 for unique scan identifiers
     * @returns {string} UUID string
     */
    generateUUID() {
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
            const r = Math.random() * 16 | 0;
            const v = c === 'x' ? r : (r & 0x3 | 0x8);
            return v.toString(16);
        });
    }
    
    /**
     * Extract plant type from PlantVillage disease label format
     * @param {string} diseaseLabel - Disease label in format "PlantName___DiseaseName"
     * @returns {string} Plant type with spaces
     */
    extractPlantType(diseaseLabel) {
        // Split by "___" separator
        const parts = diseaseLabel.split('___');
        
        // Get plant name (first part) and replace underscores with spaces
        const plantName = parts[0].replace(/_/g, ' ');
        
        return plantName;
    }
    
    /**
     * Calculate risk level based on confidence and disease type
     * @param {number} confidence - Confidence score (0.0 to 1.0)
     * @param {string} disease - Disease name
     * @returns {string} Risk level: "Low", "Medium", or "High"
     */
    calculateRiskLevel(confidence, disease) {
        // Check if plant is healthy
        if (disease.toLowerCase().includes('healthy')) {
            return 'Low';
        }
        
        // High-severity diseases based on PlantVillage dataset
        const highSeverityDiseases = [
            'Late_blight',
            'Black_rot',
            'Haunglongbing',
            'Esca',
            'Northern_Leaf_Blight'
        ];
        
        // Check if disease is high severity
        const isHighSeverity = highSeverityDiseases.some(severeDiseasePattern => 
            disease.includes(severeDiseasePattern)
        );
        
        // Calculate risk based on confidence and severity
        if (isHighSeverity && confidence >= 0.7) {
            return 'High';
        } else if (confidence >= 0.8) {
            return 'Medium';
        } else {
            return 'Low';
        }
    }
    
    /**
     * Fetch treatment data for a specific disease
     * @param {string} disease - Disease name
     * @returns {Promise<object>} Treatment information object
     */
    async fetchTreatmentData(disease) {
        try {
            // Try to fetch from backend /treatment endpoint
            const response = await fetch(`/treatment/${encodeURIComponent(disease)}`);
            
            if (response.ok) {
                const treatmentData = await response.json();
                return treatmentData;
            }
        } catch (error) {
            console.warn('Failed to fetch treatment data from backend:', error);
        }
        
        // Fallback treatment data if backend request fails
        return {
            disease: disease,
            steps: [
                'Remove affected leaves immediately',
                'Apply appropriate fungicide or treatment',
                'Improve air circulation around plants',
                'Monitor plant regularly for recurrence'
            ],
            prevention: [
                'Maintain proper plant spacing',
                'Water at base of plant, avoid wetting leaves',
                'Practice crop rotation',
                'Use disease-resistant varieties'
            ],
            expertNote: 'Early detection and prompt action are key to managing plant diseases effectively.',
            severity: 'Medium'
        };
    }
    
    /**
     * Handle cancel button click
     */
    handleCancel() {
        this.stopCamera();
        // Hide progress indicator
        const progressIndicator = document.querySelector('.progress-indicator');
        if (progressIndicator) {
            progressIndicator.style.display = 'none';
        }
        
        // Reset file input
        const imageInput = document.getElementById('image-input');
        if (imageInput) {
            imageInput.value = '';
        }
        
        // Hide cancel button, show upload button
        const cancelBtn = document.getElementById('cancel-btn');
        const uploadBtn = document.getElementById('upload-btn');
        
        if (cancelBtn) cancelBtn.style.display = 'none';
        if (uploadBtn) uploadBtn.style.display = 'block';
    }
    
    /**
     * Show progress indicator with percentage
     * @param {number} percentage - Progress percentage (0-100)
     */
    updateProgress(percentage) {
        const progressIndicator = document.querySelector('.progress-indicator');
        const progressText = document.getElementById('progress-text');
        
        if (!progressIndicator || !progressText) {
            console.error('Progress indicator elements not found');
            return;
        }
        
        // Show progress indicator
        progressIndicator.style.display = 'flex';
        
        // Update progress text
        progressText.textContent = `${percentage}% Analyse`;
        
        // Show cancel button, hide upload button
        const cancelBtn = document.getElementById('cancel-btn');
        const uploadBtn = document.getElementById('upload-btn');
        
        if (cancelBtn) cancelBtn.style.display = 'block';
        if (uploadBtn) uploadBtn.style.display = 'none';
    }
    
    /**
     * Show error message to user
     * @param {string} message - Error message to display
     */
    showError(message) {
        // Create error toast element
        const errorToast = document.createElement('div');
        errorToast.className = 'error-toast';
        errorToast.textContent = message;
        errorToast.style.cssText = `
            position: fixed;
            top: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(244, 67, 54, 0.95);
            color: white;
            padding: 16px 24px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            z-index: 10000;
            font-size: 14px;
            max-width: 90%;
            text-align: center;
            animation: slideDown 0.3s ease-out;
        `;
        
        // Add to body
        document.body.appendChild(errorToast);
        
        // Remove after 3 seconds
        setTimeout(() => {
            errorToast.style.animation = 'slideUp 0.3s ease-out';
            setTimeout(() => {
                document.body.removeChild(errorToast);
            }, 300);
        }, 3000);
        
        // Reset UI state
        this.handleCancel();
    }
}

// ResultScreen Class - Manages diagnosis result display
class ResultScreen {
    constructor(stateManager, navigationManager) {
        this.stateManager = stateManager;
        this.navigationManager = navigationManager;
        this.init();
    }
    
    /**
     * Initialize result screen
     */
    init() {
        this.setupButtons();
        this.setupScreenActivation();
    }
    
    /**
     * Set up screen activation listener to render when screen becomes active
     */
    setupScreenActivation() {
        // Use MutationObserver to detect when result screen becomes active
        const resultScreen = document.getElementById('result-screen');
        
        if (!resultScreen) {
            console.error('Result screen not found');
            return;
        }
        
        // Create observer to watch for class changes
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                if (mutation.type === 'attributes' && mutation.attributeName === 'class') {
                    if (resultScreen.classList.contains('active')) {
                        // Screen became active, render the current scan
                        this.render();
                    }
                }
            });
        });
        
        // Start observing
        observer.observe(resultScreen, { attributes: true });
    }
    
    /**
     * Render result screen with current scan data
     */
    render() {
        const currentScan = this.stateManager.getCurrentScan();
        
        if (!currentScan) {
            console.error('No current scan data available');
            return;
        }
        
        console.log('Rendering result screen with scan:', currentScan);
        this.displayDiagnosis(currentScan);
    }
    
    /**
     * Display diagnosis information on the result screen
     * @param {object} result - Scan result object
     */
    displayDiagnosis(result) {
        // Display image
        const resultImage = document.getElementById('result-image');
        if (resultImage) {
            resultImage.src = result.image;
            resultImage.alt = result.disease;
        }
        
        // Display disease name (replace underscores with spaces)
        const diseaseName = document.getElementById('disease-name');
        if (diseaseName) {
            diseaseName.textContent = result.disease.replace(/_/g, ' ');
        }
        
        // Display plant type
        const plantType = document.getElementById('plant-type');
        if (plantType) {
            plantType.textContent = result.plantType;
        }
        
        // Display formatted timestamp
        const scanTime = document.getElementById('scan-time');
        if (scanTime) {
            scanTime.textContent = this.formatTimestamp(result.timestamp);
        }
        
        // Render treatment section
        if (result.treatment) {
            this.renderTreatmentSection(result.treatment);
        }
        
        // Render risk indicator
        this.renderRiskIndicator(result.confidence, result.riskLevel);
        
        // Display expert note
        if (result.treatment && result.treatment.expertNote) {
            const expertNoteText = document.getElementById('expert-note-text');
            if (expertNoteText) {
                expertNoteText.textContent = result.treatment.expertNote;
            }
        }
    }
    
    /**
     * Render treatment section with steps and prevention tips
     * @param {object} treatment - Treatment information object
     */
    renderTreatmentSection(treatment) {
        const treatmentList = document.getElementById('treatment-list');
        
        if (!treatmentList) {
            console.error('Treatment list element not found');
            return;
        }
        
        // Clear existing content
        treatmentList.innerHTML = '';
        
        // Add treatment steps
        if (treatment.steps && treatment.steps.length > 0) {
            treatment.steps.forEach(step => {
                const li = document.createElement('li');
                li.innerHTML = `<i class="fas fa-check-circle"></i> ${step}`;
                treatmentList.appendChild(li);
            });
        }
        
        // Add prevention tips
        if (treatment.prevention && treatment.prevention.length > 0) {
            // Add a separator or subheading if desired
            const preventionHeader = document.createElement('li');
            preventionHeader.innerHTML = '<strong>Prevention:</strong>';
            preventionHeader.style.marginTop = '12px';
            preventionHeader.style.listStyle = 'none';
            treatmentList.appendChild(preventionHeader);
            
            treatment.prevention.forEach(tip => {
                const li = document.createElement('li');
                li.innerHTML = `<i class="fas fa-check-circle"></i> ${tip}`;
                treatmentList.appendChild(li);
            });
        }
    }
    
    /**
     * Render risk indicator with marker positioned based on confidence
     * @param {number} confidence - Confidence score (0.0 to 1.0)
     * @param {string} riskLevel - Risk level: "Low", "Medium", or "High"
     */
    renderRiskIndicator(confidence, riskLevel) {
        const riskMarker = document.getElementById('risk-marker');
        
        if (!riskMarker) {
            console.error('Risk marker element not found');
            return;
        }
        
        // Calculate marker position based on confidence (0-100%)
        const position = confidence * 100;
        
        // Position marker (accounting for marker width to center it)
        riskMarker.style.left = `${position}%`;
        
        // Remove all risk level classes
        riskMarker.classList.remove('low', 'medium', 'high');
        
        // Add appropriate risk level class
        riskMarker.classList.add(riskLevel.toLowerCase());
    }
    
    /**
     * Set up action button handlers
     */
    setupButtons() {
        // Re-generate button
        const regenerateBtn = document.querySelector('.regenerate-btn');
        if (regenerateBtn) {
            regenerateBtn.addEventListener('click', () => {
                this.handleRegenerate();
            });
        }
        
        // Share button
        const shareBtn = document.querySelector('.share-btn');
        if (shareBtn) {
            shareBtn.addEventListener('click', () => {
                this.handleShare();
            });
        }
    }
    
    /**
     * Handle re-generate button click - navigate back to camera screen
     */
    handleRegenerate() {
        console.log('Re-generate clicked');
        this.navigationManager.navigateTo('camera');
    }
    
    /**
     * Handle share button click - share result using Web Share API or clipboard
     */
    async handleShare() {
        const currentScan = this.stateManager.getCurrentScan();
        
        if (!currentScan) {
            console.error('No scan data to share');
            return;
        }
        
        // Format share text
        const shareText = `Plantia AI Diagnosis:\n\nDisease: ${currentScan.disease.replace(/_/g, ' ')}\nPlant: ${currentScan.plantType}\nConfidence: ${(currentScan.confidence * 100).toFixed(1)}%\nRisk Level: ${currentScan.riskLevel}`;
        
        // Try Web Share API first (mobile devices)
        if (navigator.share) {
            try {
                await navigator.share({
                    title: 'Plantia AI Diagnosis',
                    text: shareText
                });
                console.log('Shared successfully');
            } catch (error) {
                console.log('Share cancelled or failed:', error);
            }
        } else {
            // Fallback to clipboard copy
            try {
                await navigator.clipboard.writeText(shareText);
                this.showShareSuccess('Diagnosis copied to clipboard!');
            } catch (error) {
                console.error('Failed to copy to clipboard:', error);
                this.showShareSuccess('Unable to share. Please try again.');
            }
        }
    }
    
    /**
     * Show share success message
     * @param {string} message - Message to display
     */
    showShareSuccess(message) {
        // Create success toast element
        const toast = document.createElement('div');
        toast.className = 'share-toast';
        toast.textContent = message;
        toast.style.cssText = `
            position: fixed;
            top: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(76, 175, 80, 0.95);
            color: white;
            padding: 16px 24px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            z-index: 10000;
            font-size: 14px;
            max-width: 90%;
            text-align: center;
            animation: slideDown 0.3s ease-out;
        `;
        
        // Add to body
        document.body.appendChild(toast);
        
        // Remove after 2 seconds
        setTimeout(() => {
            toast.style.animation = 'slideUp 0.3s ease-out';
            setTimeout(() => {
                document.body.removeChild(toast);
            }, 300);
        }, 2000);
    }
    
    /**
     * Format timestamp to relative time string
     * @param {number} timestamp - Unix timestamp in milliseconds
     * @returns {string} Formatted time string
     */
    formatTimestamp(timestamp) {
        const date = new Date(timestamp);
        const now = new Date();
        const diffMs = now - date;
        const diffMins = Math.floor(diffMs / 60000);
        
        if (diffMins < 1) return 'Just now';
        if (diffMins < 60) return `${diffMins} min ago`;
        
        const diffHours = Math.floor(diffMins / 60);
        if (diffHours < 24) return `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`;
        
        const diffDays = Math.floor(diffHours / 24);
        if (diffDays < 7) return `${diffDays} day${diffDays > 1 ? 's' : ''} ago`;
        
        return date.toLocaleDateString();
    }
}
