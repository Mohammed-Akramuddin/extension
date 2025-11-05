'use strict';

/* ========================================
   CONFIGURATION & CONSTANTS
======================================== */
const CONFIG = {
  TARGET_SIZE: 224,
  MEAN: [0.485, 0.456, 0.406],
  STD: [0.229, 0.224, 0.225],
  FACE_PADDING: 25, // 25% padding around detected face
  MIN_FACE_SIZE: 80, // Minimum face size in pixels
  CONFIDENCE_THRESHOLD: 0.55, // Adjusted threshold for better classification
  GUIDELINE_DURATION: 3000, // Show guidelines for 3 seconds
  AUTO_STOP_DELAY: 3000, // Auto-stop camera after 3 seconds
  VIDEO_CONSTRAINTS: {
    ideal: { width: 1280, height: 720, facingMode: "user" },
    fallback: { facingMode: "user" }
  }
};

/* ========================================
   STATE MANAGEMENT
======================================== */
const state = {
  stream: null,
  running: false,
  session: null,
  inputName: null,
  outputName: null,
  lastMinorState: null,
  blaze: null,
  hasDetector: false,
  isAnalyzing: false,
  guidelineTimer: null
};

/* ========================================
   DOM REFERENCES
======================================== */
const elements = {
  video: document.getElementById("video"),
  canvas: document.getElementById("work"),
  overlayCanvas: document.getElementById("overlayCanvas"),
  statusText: document.getElementById("statusText"),
  statusCard: document.getElementById("statusCard"),
  statusIcon: document.getElementById("statusIcon"),
  statusProgress: document.getElementById("statusProgress"),
  resultCard: document.getElementById("resultCard"),
  resultText: document.getElementById("resultText"),
  resultSub: document.getElementById("resultSub"),
  resultIcon: document.getElementById("resultIcon"),
  safesearchBadge: document.getElementById("safesearchBadge"),
  badgeText: document.getElementById("badgeText"),
  badgeIconWrapper: document.getElementById("badgeIconWrapper"),
  errorBadge: document.getElementById("errorBadge"),
  errorText: document.getElementById("errorText"),
  scanOverlay: document.getElementById("scanOverlay"),
  faceGuide: document.getElementById("faceGuide"),
  guidelines: document.getElementById("guidelines"),
  startBtn: document.getElementById("startBtn"),
  stopBtn: document.getElementById("stopBtn"),
  openSettingsBtn: document.getElementById("openSettingsBtn")
};

/* ========================================
   UTILITY FUNCTIONS
======================================== */
const utils = {
  sleep: (ms) => new Promise(resolve => setTimeout(resolve, ms)),

  log: (...args) => {
    try {
      console.log('[FaceAgeVerify]', ...args);
    } catch (e) {}
  },

  setElementText: (element, text) => {
    if (element && element.textContent !== undefined) {
      element.textContent = text;
    }
  },

  addClass: (element, ...classes) => {
    if (element) element.classList.add(...classes);
  },

  removeClass: (element, ...classes) => {
    if (element) element.classList.remove(...classes);
  },

  toggleClass: (element, className, force) => {
    if (element) element.classList.toggle(className, force);
  },

  setDisplay: (element, display) => {
    if (element) element.style.display = display;
  },

  formatError: (error) => {
    if (!error) return 'Unknown error';
    if (typeof error === 'string') return error;
    return error.message || error.toString();
  }
};

/* ========================================
   CONSENT CHECK
======================================== */
async function checkConsent() {
  try {
    const { consentGiven } = await chrome.storage.local.get('consentGiven');
    if (!consentGiven) {
      const consentUrl = chrome.runtime.getURL('consent.html');
      try {
        await chrome.windows.create({
          url: consentUrl,
          type: 'popup',
          width: 650,
          height: 700,
          left: Math.max(0, screen.availWidth - 670),
          top: 20
        });
        window.close();
      } catch (e) {
        utils.log('Failed to open consent window', e);
      }
      return false;
    }
    return true;
  } catch (e) {
    utils.log('Consent check error', e);
    return false;
  }
}

/* ========================================
   ONNX RUNTIME CONFIGURATION
======================================== */
function configureONNXRuntime() {
  try {
    if (window.ort) {
      const wasmBase = chrome.runtime.getURL('lib/');
      if (!window.ort.env) window.ort.env = {};
      if (!window.ort.env.wasm) window.ort.env.wasm = {};
      window.ort.env.wasm.wasmPaths = wasmBase;
      // MV3-friendly: disable worker proxy and advanced threading to avoid blob/importScripts errors
      window.ort.env.wasm.proxy = false;
      window.ort.env.wasm.numThreads = 1;
      window.ort.env.wasm.simd = false;
      utils.log('ONNX Runtime configured:', wasmBase);
    }
  } catch (e) {
    utils.log('ONNX Runtime configuration warning', e);
  }
}

/* ========================================
   MODEL INITIALIZATION
======================================== */
async function initializeModel() {
  try {
    const modelUrl = chrome.runtime.getURL("swin_face_classifier.onnx");
    utils.log('Loading model from:', modelUrl);
    utils.setElementText(elements.statusText, "Loading AI model...");

    const response = await fetch(modelUrl);
    if (!response.ok) {
      throw new Error(`Model fetch failed: ${response.status}`);
    }

    const arrayBuffer = await response.arrayBuffer();
    if (!arrayBuffer || arrayBuffer.byteLength < 4) {
      throw new Error('Model file is empty or corrupt');
    }

    const modelBytes = new Uint8Array(arrayBuffer);
    utils.log('Model size:', modelBytes.byteLength, 'bytes');

    // Try WASM first (main-thread, no proxy), fallback to WebGL
    try {
      state.session = await ort.InferenceSession.create(modelBytes, {
        executionProviders: ['wasm']
      });
      utils.log('Model loaded with WASM provider');
    } catch (wasmError) {
      utils.log('WASM failed, trying WebGL', wasmError);
      state.session = await ort.InferenceSession.create(modelBytes, {
        executionProviders: ['webgl']
      });
      utils.log('Model loaded with WebGL provider');
    }

    state.inputName = state.session.inputNames[0];
    state.outputName = state.session.outputNames[0];

    utils.log('Model ready. Input:', state.inputName, 'Output:', state.outputName);
    utils.setElementText(elements.statusText, "AI model ready");

    return true;
  } catch (error) {
    utils.log('Model initialization error:', error);
    utils.setElementText(elements.statusText, `Model error: ${utils.formatError(error)}`);
    return false;
  }
}

/* ========================================
   BLAZEFACE DETECTOR INITIALIZATION
======================================== */
async function initializeDetector() {
  try {
    if (window.blazeface && window.tf) {
      state.blaze = await window.blazeface.load();
      state.hasDetector = true;
      utils.log('BlazeFace detector loaded');
    } else {
      state.hasDetector = false;
      utils.log('BlazeFace not available, using full frame');
    }
  } catch (error) {
    state.hasDetector = false;
    utils.log('BlazeFace initialization failed:', error);
  }
}

/* ========================================
   IMAGE PREPROCESSING
======================================== */
function preprocessImage(sourceCanvas, faces, overridePaddingPct) {
  // Find largest face
  let faceRegion = null;

  if (faces && faces.length > 0) {
    let largestFace = faces[0];
    let maxArea = 0;

    for (const face of faces) {
      const width = face.bottomRight[0] - face.topLeft[0];
      const height = face.bottomRight[1] - face.topLeft[1];
      const area = width * height;

      if (area > maxArea) {
        maxArea = area;
        largestFace = face;
      }
    }

    // Check if face is large enough
    const faceWidth = largestFace.bottomRight[0] - largestFace.topLeft[0];
    const faceHeight = largestFace.bottomRight[1] - largestFace.topLeft[1];

    if (faceWidth >= CONFIG.MIN_FACE_SIZE && faceHeight >= CONFIG.MIN_FACE_SIZE) {
      faceRegion = largestFace;
      utils.log('Face detected:', Math.round(faceWidth), 'x', Math.round(faceHeight));
    } else {
      utils.log('Face too small, using full frame');
    }
  }

  // Create working canvas
  let workCanvas = document.createElement('canvas');
  let workCtx = workCanvas.getContext('2d');

  if (faceRegion) {
    // Crop face with padding (allow override for TTA)
    const paddingPercent = typeof overridePaddingPct === 'number' ? overridePaddingPct : CONFIG.FACE_PADDING;
    const padding = (Math.max(0, Math.min(50, paddingPercent))) / 100;
    const x1 = Math.floor(faceRegion.topLeft[0]);
    const y1 = Math.floor(faceRegion.topLeft[1]);
    const x2 = Math.ceil(faceRegion.bottomRight[0]);
    const y2 = Math.ceil(faceRegion.bottomRight[1]);

    const faceWidth = x2 - x1;
    const faceHeight = y2 - y1;
    const padX = Math.floor(faceWidth * padding);
    const padY = Math.floor(faceHeight * padding);

    const cropX = Math.max(0, x1 - padX);
    const cropY = Math.max(0, y1 - padY);
    const cropWidth = Math.min(sourceCanvas.width - cropX, faceWidth + 2 * padX);
    const cropHeight = Math.min(sourceCanvas.height - cropY, faceHeight + 2 * padY);

    workCanvas.width = cropWidth;
    workCanvas.height = cropHeight;
    workCtx.drawImage(
      sourceCanvas,
      cropX, cropY, cropWidth, cropHeight,
      0, 0, cropWidth, cropHeight
    );
  } else {
    // Use full frame
    workCanvas.width = sourceCanvas.width;
    workCanvas.height = sourceCanvas.height;
    workCtx.drawImage(sourceCanvas, 0, 0);
  }

  // Resize to model input size
  const resizedCanvas = document.createElement('canvas');
  resizedCanvas.width = CONFIG.TARGET_SIZE;
  resizedCanvas.height = CONFIG.TARGET_SIZE;
  const resizedCtx = resizedCanvas.getContext('2d');
  resizedCtx.drawImage(workCanvas, 0, 0, CONFIG.TARGET_SIZE, CONFIG.TARGET_SIZE);

  // Convert to tensor with normalization
  const imageData = resizedCtx.getImageData(0, 0, CONFIG.TARGET_SIZE, CONFIG.TARGET_SIZE);
  const pixels = imageData.data;
  const inputTensor = new Float32Array(3 * CONFIG.TARGET_SIZE * CONFIG.TARGET_SIZE);

  for (let i = 0; i < CONFIG.TARGET_SIZE * CONFIG.TARGET_SIZE; i++) {
    const pixelIndex = i * 4;
    const r = pixels[pixelIndex] / 255.0;
    const g = pixels[pixelIndex + 1] / 255.0;
    const b = pixels[pixelIndex + 2] / 255.0;

    inputTensor[i] = (r - CONFIG.MEAN[0]) / CONFIG.STD[0];
    inputTensor[CONFIG.TARGET_SIZE * CONFIG.TARGET_SIZE + i] = (g - CONFIG.MEAN[1]) / CONFIG.STD[1];
    inputTensor[2 * CONFIG.TARGET_SIZE * CONFIG.TARGET_SIZE + i] = (b - CONFIG.MEAN[2]) / CONFIG.STD[2];
  }

  return new ort.Tensor('float32', inputTensor, [1, 3, CONFIG.TARGET_SIZE, CONFIG.TARGET_SIZE]);
}

/* ========================================
   FACE DETECTION & VISUALIZATION
======================================== */
async function detectAndDrawFaces(sourceCanvas) {
  let faces = null;

  if (state.hasDetector && state.blaze) {
    try {
      faces = await state.blaze.estimateFaces(sourceCanvas, false);
      utils.log('Detected', faces ? faces.length : 0, 'face(s)');

      // Draw face box on overlay
      if (faces && faces.length > 0 && elements.overlayCanvas) {
        const ctx = elements.overlayCanvas.getContext('2d');
        elements.overlayCanvas.width = sourceCanvas.width;
        elements.overlayCanvas.height = sourceCanvas.height;
        ctx.clearRect(0, 0, elements.overlayCanvas.width, elements.overlayCanvas.height);

        // Draw largest face
        let largestFace = faces[0];
        let maxArea = 0;

        for (const face of faces) {
          const width = face.bottomRight[0] - face.topLeft[0];
          const height = face.bottomRight[1] - face.topLeft[1];
          const area = width * height;
          if (area > maxArea) {
            maxArea = area;
            largestFace = face;
          }
        }

        const x = largestFace.topLeft[0];
        const y = largestFace.topLeft[1];
        const width = largestFace.bottomRight[0] - largestFace.topLeft[0];
        const height = largestFace.bottomRight[1] - largestFace.topLeft[1];

        // Draw with gradient
        ctx.strokeStyle = '#06b6d4';
        ctx.lineWidth = 3;
        ctx.shadowColor = 'rgba(6, 182, 212, 0.5)';
        ctx.shadowBlur = 10;
        ctx.strokeRect(x, y, width, height);

        // Draw corner accents
        const cornerLength = 20;
        ctx.lineWidth = 4;

        // Top-left
        ctx.beginPath();
        ctx.moveTo(x, y + cornerLength);
        ctx.lineTo(x, y);
        ctx.lineTo(x + cornerLength, y);
        ctx.stroke();

        // Top-right
        ctx.beginPath();
        ctx.moveTo(x + width - cornerLength, y);
        ctx.lineTo(x + width, y);
        ctx.lineTo(x + width, y + cornerLength);
        ctx.stroke();

        // Bottom-left
        ctx.beginPath();
        ctx.moveTo(x, y + height - cornerLength);
        ctx.lineTo(x, y + height);
        ctx.lineTo(x + cornerLength, y + height);
        ctx.stroke();

        // Bottom-right
        ctx.beginPath();
        ctx.moveTo(x + width - cornerLength, y + height);
        ctx.lineTo(x + width, y + height);
        ctx.lineTo(x + width, y + height - cornerLength);
        ctx.stroke();
      }
    } catch (error) {
      utils.log('Face detection error:', error);
      faces = null;
    }
  }

  return faces;
}

/* ========================================
   MODEL INFERENCE
======================================== */
async function runInference(inputTensor) {
  if (!state.session) {
    throw new Error('Model not initialized');
  }

  const feeds = { [state.inputName]: inputTensor };
  const results = await state.session.run(feeds);
  const output = results[state.outputName] || Object.values(results)[0];

  if (!output) {
    throw new Error('Model returned no output');
  }

  return output.data;
}

/* ========================================
   CLASSIFICATION LOGIC
======================================== */
function interpretModelOutput(outputData) {
  utils.log('Raw model output:', Array.from(outputData));

  let probability;

  if (outputData.length === 1) {
    const rawValue = outputData[0];
    if (rawValue >= 0 && rawValue <= 1) {
      probability = rawValue;
      utils.log('Single output interpreted as probability');
    } else {
      probability = 1 / (1 + Math.exp(-rawValue));
      utils.log('Single output interpreted as logit, applied sigmoid');
    }
  } else if (outputData.length === 2) {
    const max = Math.max(outputData[0], outputData[1]);
    const exp0 = Math.exp(outputData[0] - max);
    const exp1 = Math.exp(outputData[1] - max);
    const sum = exp0 + exp1;
    probability = exp1 / sum; // Index 1 = Major
    utils.log('Two outputs interpreted as logits, applied softmax');
  } else {
    throw new Error(`Unexpected output length: ${outputData.length}`);
  }

  utils.log('Major probability:', (probability * 100).toFixed(2) + '%');

  const prediction = probability >= CONFIG.CONFIDENCE_THRESHOLD ? 'MAJOR' : 'MINOR';
  const confidence = probability >= CONFIG.CONFIDENCE_THRESHOLD ? probability : (1 - probability);

  return {
    prediction,
    confidence,
    isMinor: prediction === 'MINOR',
    probability
  };
}

/* ========================================
   UI UPDATES
======================================== */
function updateResultUI(result) {
  utils.setElementText(elements.resultText, result.prediction);
  utils.setElementText(elements.resultSub, `Confidence: ${(result.confidence * 100).toFixed(1)}%`);

  utils.removeClass(elements.resultCard, 'major', 'minor');
  utils.addClass(elements.resultCard, 'active', result.isMinor ? 'minor' : 'major');
}

async function updateSafeSearch(isMinor) {
  if (state.lastMinorState === isMinor) {
    utils.log('SafeSearch already set to', isMinor ? 'ON' : 'OFF');
    return;
  }

  state.lastMinorState = isMinor;
  utils.log('Updating SafeSearch:', isMinor ? 'ON' : 'OFF');

  try {
    const response = await new Promise((resolve) => {
      chrome.runtime.sendMessage(
        { type: "SET_SAFE", minor: isMinor },
        resolve
      );
    });

    if (response && response.ok) {
      await utils.sleep(500);

      utils.removeClass(elements.safesearchBadge, 'on', 'off');
      utils.addClass(elements.safesearchBadge, 'active', isMinor ? 'on' : 'off');
      utils.setElementText(elements.badgeText, isMinor ? 'SafeSearch ON' : 'SafeSearch OFF');

      utils.removeClass(elements.errorBadge, 'active');
    } else if (response && response.error) {
      utils.log('SafeSearch update failed:', response.error);
      await utils.sleep(500);

      utils.addClass(elements.errorBadge, 'active');
      utils.setElementText(
        elements.errorText,
        'SafeSearch is locked by browser settings. Classification successful, but cannot change automatically.'
      );
    }
  } catch (error) {
    utils.log('SafeSearch update error:', error);
    await utils.sleep(500);

    utils.addClass(elements.errorBadge, 'active');
    utils.setElementText(elements.errorText, 'Could not update SafeSearch. Classification successful.');
  }
}

function showGuidelines() {
  utils.addClass(elements.guidelines, 'active');
  utils.addClass(elements.faceGuide, 'active');

  if (state.guidelineTimer) {
    clearTimeout(state.guidelineTimer);
  }

  state.guidelineTimer = setTimeout(() => {
    utils.removeClass(elements.guidelines, 'active');
  }, CONFIG.GUIDELINE_DURATION);
}

function hideGuidelines() {
  utils.removeClass(elements.guidelines, 'active');
  utils.removeClass(elements.faceGuide, 'active');

  if (state.guidelineTimer) {
    clearTimeout(state.guidelineTimer);
    state.guidelineTimer = null;
  }
}

function updateStatus(text, processing = false) {
  utils.setElementText(elements.statusText, text);
  utils.toggleClass(elements.statusCard, 'processing', processing);
  utils.toggleClass(elements.statusProgress, 'active', processing);
}

/* ========================================
   MAIN ANALYSIS FUNCTION
======================================== */
async function analyzeFrame() {
  if (state.isAnalyzing) {
    utils.log('Analysis already in progress');
    return;
  }

  state.isAnalyzing = true;

  try {
    // Reset UI
    utils.removeClass(elements.resultCard, 'active', 'major', 'minor');
    utils.setElementText(elements.resultText, '—');
    utils.setElementText(elements.resultSub, '');
    utils.removeClass(elements.safesearchBadge, 'active');
    utils.removeClass(elements.errorBadge, 'active');

    // Show scanning effect
    utils.addClass(elements.scanOverlay, 'active');
    updateStatus('Capturing frame...', true);

    await utils.sleep(500);

    if (!state.session) {
      throw new Error('Model not loaded');
    }

    // Capture frame from video
    const sourceCanvas = document.createElement('canvas');
    const videoWidth = elements.video.videoWidth || 640;
    const videoHeight = elements.video.videoHeight || 480;

    sourceCanvas.width = videoWidth;
    sourceCanvas.height = videoHeight;

    const sourceCtx = sourceCanvas.getContext('2d');
    sourceCtx.drawImage(elements.video, 0, 0, videoWidth, videoHeight);

    // Detect faces
    updateStatus('Detecting face...', true);
    const faces = await detectAndDrawFaces(sourceCanvas);

    if (!faces || faces.length === 0) {
      utils.log('No face detected, analyzing full frame');
    }

    // Preprocess image with TTA (multi-padding) and average probabilities
    updateStatus('Analyzing age...', true);

    const paddings = Array.from(new Set([
      CONFIG.FACE_PADDING,
      Math.max(0, CONFIG.FACE_PADDING - 10),
      Math.min(50, CONFIG.FACE_PADDING + 10)
    ]));

    let probabilities = [];
    for (const pad of paddings) {
      try {
        const tensor = preprocessImage(sourceCanvas, faces, pad);
        const data = await runInference(tensor);
        // Convert to probability-of-major
        let prob;
        if (data.length === 1) {
          const v = data[0];
          prob = (v >= 0 && v <= 1) ? v : 1 / (1 + Math.exp(-v));
        } else if (data.length === 2) {
          const m = Math.max(data[0], data[1]);
          const e0 = Math.exp(data[0] - m);
          const e1 = Math.exp(data[1] - m);
          prob = e1 / (e0 + e1);
        } else {
          continue;
        }
        probabilities.push(prob);
      } catch (_) {
        // Skip failed TTA branch
      }
    }

    // Fallback: single pass if TTA failed
    if (probabilities.length === 0) {
      const fallbackTensor = preprocessImage(sourceCanvas, faces);
      const fallbackData = await runInference(fallbackTensor);
      const fallbackResult = interpretModelOutput(fallbackData);
      probabilities.push(fallbackResult.probability);
    }

    // Average probabilities
    const avgProb = probabilities.reduce((a,b)=>a+b,0) / probabilities.length;
    const averagedData = new Float32Array([1-avgProb, avgProb]);
    const result = interpretModelOutput(averagedData);
    utils.log('Classification:', result.prediction, 'Confidence:', (result.confidence * 100).toFixed(1) + '%');

    // Update UI
    updateResultUI(result);
    updateStatus('Analysis complete!', false);

    // Update SafeSearch
    await updateSafeSearch(result.isMinor);

    utils.removeClass(elements.scanOverlay, 'active');

    // Auto-stop camera
    setTimeout(() => {
      stopCamera();
    }, CONFIG.AUTO_STOP_DELAY);

  } catch (error) {
    utils.log('Analysis error:', error);
    updateStatus(`Error: ${utils.formatError(error)}`, false);
    utils.removeClass(elements.scanOverlay, 'active');
    stopCamera();
  } finally {
    state.isAnalyzing = false;
  }
}

/* ========================================
   CAMERA CONTROL
======================================== */
async function startCamera() {
  if (state.running) {
    utils.log('Camera already running');
    return;
  }

  try {
    // Check consent
    const hasConsent = await checkConsent();
    if (!hasConsent) {
      updateStatus('Please accept terms and conditions first', false);
      return;
    }

    // Update UI
    updateStatus('Requesting camera access...', true);
    elements.startBtn.disabled = true;
    elements.stopBtn.disabled = true;

    // Request camera with constraints
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: CONFIG.VIDEO_CONSTRAINTS.ideal,
        audio: false
      });
    } catch (constraintError) {
      if (constraintError.name === "OverconstrainedError") {
        utils.log('Trying fallback constraints');
        stream = await navigator.mediaDevices.getUserMedia({
          video: CONFIG.VIDEO_CONSTRAINTS.fallback,
          audio: false
        });
      } else {
        throw constraintError;
      }
    }

    state.stream = stream;
    elements.video.srcObject = stream;

    // Wait for video metadata
    await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error('Video timeout')), 5000);

      if (elements.video.videoWidth > 0) {
        clearTimeout(timeout);
        resolve();
        return;
      }

      elements.video.onloadedmetadata = () => {
        clearTimeout(timeout);
        resolve();
      };

      elements.video.onerror = (error) => {
        clearTimeout(timeout);
        reject(error);
      };
    });

    await elements.video.play();
    utils.log('Camera started:', elements.video.videoWidth, 'x', elements.video.videoHeight);

    // Initialize detector and model
    if (!state.hasDetector) {
      await initializeDetector();
    }

    if (!state.session) {
      const success = await initializeModel();
      if (!success) {
        throw new Error('Model initialization failed');
      }
    }

    // Update state
    state.running = true;
    elements.startBtn.disabled = true;
    elements.stopBtn.disabled = false;

    // Show guidelines
    showGuidelines();
    updateStatus('Camera ready. Keep face in frame...', false);

    // Auto-analyze after guidelines
    await utils.sleep(CONFIG.GUIDELINE_DURATION);
    hideGuidelines();

    await analyzeFrame();

  } catch (error) {
    utils.log('Camera error:', error);

    state.running = false;
    elements.startBtn.disabled = false;
    elements.stopBtn.disabled = true;

    const errorName = error.name || 'Unknown';

    if (errorName === 'NotAllowedError' || errorName === 'PermissionDeniedError') {
      updateStatus('Camera access denied. Please allow camera.', false);
      utils.addClass(elements.openSettingsBtn, 'active');
    } else if (errorName === 'NotFoundError') {
      updateStatus('No camera found. Please connect a camera.', false);
    } else if (errorName === 'NotReadableError' || errorName === 'TrackStartError') {
      updateStatus('Camera in use by another app. Please close it.', false);
    } else {
      updateStatus(`Camera error: ${utils.formatError(error)}`, false);
    }
  }
}

function stopCamera() {
  if (!state.stream && !state.running) {
    updateStatus('Ready to start verification', false);
    return;
  }

  utils.log('Stopping camera');

  state.running = false;
  elements.startBtn.disabled = false;
  elements.stopBtn.disabled = true;

  if (state.stream) {
    state.stream.getTracks().forEach(track => track.stop());
    state.stream = null;
  }

  if (elements.video) {
    elements.video.pause();
    elements.video.srcObject = null;
  }

  if (elements.overlayCanvas) {
    const ctx = elements.overlayCanvas.getContext('2d');
    ctx.clearRect(0, 0, elements.overlayCanvas.width, elements.overlayCanvas.height);
  }

  hideGuidelines();
  utils.removeClass(elements.scanOverlay, 'active');
  utils.removeClass(elements.faceGuide, 'active');

  updateStatus('Camera stopped. Ready to start.', false);
}

/* ========================================
   EVENT LISTENERS
======================================== */
function setupEventListeners() {
  elements.startBtn.addEventListener('click', startCamera);
  elements.stopBtn.addEventListener('click', stopCamera);

  elements.openSettingsBtn.addEventListener('click', async () => {
    try {
      await chrome.tabs.create({ url: 'chrome://settings/content/camera' });
      updateStatus('Please allow camera access and try again.', false);
    } catch (error) {
      updateStatus('Open chrome://settings/content/camera manually', false);
    }
  });
}

/* ========================================
   INITIALIZATION
======================================== */
async function initialize() {
  utils.log('Initializing extension');

  // Configure ONNX Runtime
  configureONNXRuntime();

  // Check consent
  const hasConsent = await checkConsent();
  if (!hasConsent) {
    updateStatus('Please accept terms to continue', false);
    elements.startBtn.disabled = true;

    const checkInterval = setInterval(async () => {
      const { consentGiven } = await chrome.storage.local.get('consentGiven');
      if (consentGiven) {
        clearInterval(checkInterval);
        updateStatus('Ready to start verification', false);
        elements.startBtn.disabled = false;
      }
    }, 500);
    return;
  }

  // Load last state
  const { isMinor } = await chrome.storage.local.get('isMinor');
  state.lastMinorState = typeof isMinor === 'boolean' ? isMinor : null;
  utils.log('Last state - isMinor:', state.lastMinorState);

  // Pre-load resources
  initializeDetector().catch(() => {});
  initializeModel().then(success => {
    if (success) {
      updateStatus('Ready to start verification', false);
    }
  }).catch(() => {});

  // Setup event listeners
  setupEventListeners();

  utils.log('Initialization complete');
}

/* ========================================
   START APPLICATION
======================================== */
initialize();
