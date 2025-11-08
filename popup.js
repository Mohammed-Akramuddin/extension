'use strict';

/* ========================================
   CONFIGURATION & CONSTANTS
======================================== */
const CONFIG = {
  TARGET_SIZE: 224,
  MEAN: [0.485, 0.456, 0.406],  // ImageNet mean
  STD: [0.229, 0.224, 0.225],   // ImageNet std
  FACE_PADDING: 25, // 25% padding around detected face (default)
  MIN_FACE_SIZE: 50, // Minimum face size in pixels (reduced for better detection)
  CONFIDENCE_THRESHOLD: 0.5, // Standard threshold (0.5 = 50%)
  GUIDELINE_DURATION: 3000, // Show guidelines for 3 seconds
  AUTO_STOP_DELAY: 4000, // Auto-stop camera after 4 seconds
  VIDEO_CONSTRAINTS: {
    ideal: { width: 1280, height: 720, facingMode: "user" },
    fallback: { facingMode: "user" }
  },
  // Optimized TTA configuration for speed and reliability
  TTA_PADDINGS: [25], // Single padding for speed (can add more if needed: [20, 25, 30])
  TTA_FLIPS: false, // Disable flips for speed (set to true for slightly better accuracy but slower)
  TTA_BRIGHTNESS: [1.0], // Single brightness for speed (disabled variations)
  NUM_FRAMES_TO_AVERAGE: 1, // Single frame for speed (increased reliability over quantity)
  FRAME_CAPTURE_DELAY: 50, // Minimal delay
  INFERENCE_BATCH_SIZE: 1, // Process one inference at a time to prevent blocking
  INFERENCE_DELAY: 50 // Delay between inferences to prevent UI freeze (increased for better responsiveness)
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
          focused: true
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
   IMAGE PREPROCESSING - Enhanced
======================================== */
function cropFaceRegion(sourceCanvas, faces, overridePaddingPct) {
  if (!faces || faces.length === 0) {
    utils.log('No faces detected or face detection disabled, using full image');
    return sourceCanvas;
  }

  // Find largest face by area
  let largestFace = faces[0];
  let maxArea = 0;

  for (const face of faces) {
    const tl = face.topLeft;
    const br = face.bottomRight;
    const width = br[0] - tl[0];
    const height = br[1] - tl[1];
    const area = width * height;

    if (area > maxArea) {
      maxArea = area;
      largestFace = face;
    }
  }

  // Get padding percentage (clamp to valid range)
  const paddingPercent = typeof overridePaddingPct === 'number' 
    ? Math.max(0, Math.min(50, overridePaddingPct))
    : CONFIG.FACE_PADDING;
  const padding = paddingPercent / 100;

  // Calculate face bounds
  const x1 = Math.floor(largestFace.topLeft[0]);
  const y1 = Math.floor(largestFace.topLeft[1]);
  const x2 = Math.ceil(largestFace.bottomRight[0]);
  const y2 = Math.ceil(largestFace.bottomRight[1]);

  const faceWidth = x2 - x1;
  const faceHeight = y2 - y1;

  // Check if face is large enough
  if (faceWidth < CONFIG.MIN_FACE_SIZE || faceHeight < CONFIG.MIN_FACE_SIZE) {
    utils.log(`Face too small (${Math.round(faceWidth)}x${Math.round(faceHeight)}), using full frame`);
    return sourceCanvas;
  }

  // Calculate padding
  const padX = Math.floor(faceWidth * padding);
  const padY = Math.floor(faceHeight * padding);

  // Calculate crop region (ensure we don't go outside canvas bounds)
  const cropX = Math.max(0, x1 - padX);
  const cropY = Math.max(0, y1 - padY);
  const cropWidth = Math.min(sourceCanvas.width - cropX, faceWidth + 2 * padX);
  const cropHeight = Math.min(sourceCanvas.height - cropY, faceHeight + 2 * padY);

  utils.log(`Cropping face region: x=${cropX}, y=${cropY}, w=${cropWidth}, h=${cropHeight} (padding: ${paddingPercent.toFixed(1)}%)`);

  // Create cropped canvas
  const croppedCanvas = document.createElement('canvas');
  croppedCanvas.width = cropWidth;
  croppedCanvas.height = cropHeight;
  const ctx = croppedCanvas.getContext('2d');

  // Use high-quality image smoothing for better results
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';

  ctx.drawImage(
    sourceCanvas,
    cropX, cropY, cropWidth, cropHeight,
    0, 0, cropWidth, cropHeight
  );

  return croppedCanvas;
}

/* ========================================
   IMAGE ENHANCEMENT FUNCTIONS
======================================== */
function enhanceImage(canvas) {
  // Simplified enhancement for speed - only essential adjustments
  const enhancedCanvas = document.createElement('canvas');
  enhancedCanvas.width = canvas.width;
  enhancedCanvas.height = canvas.height;
  const ctx = enhancedCanvas.getContext('2d');
  ctx.drawImage(canvas, 0, 0);
  
  // Skip expensive image processing for speed
  // The model should handle raw images well
  // Only do minimal enhancement if really needed
  
  return enhancedCanvas;
}

function applyBrightnessAdjustment(canvas, brightnessFactor) {
  if (brightnessFactor === 1.0) return canvas;
  
  const enhancedCanvas = document.createElement('canvas');
  enhancedCanvas.width = canvas.width;
  enhancedCanvas.height = canvas.height;
  const ctx = enhancedCanvas.getContext('2d');
  ctx.drawImage(canvas, 0, 0);
  
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const data = imageData.data;
  
  for (let i = 0; i < data.length; i += 4) {
    data[i] = Math.max(0, Math.min(255, data[i] * brightnessFactor));
    data[i + 1] = Math.max(0, Math.min(255, data[i + 1] * brightnessFactor));
    data[i + 2] = Math.max(0, Math.min(255, data[i + 2] * brightnessFactor));
  }
  
  ctx.putImageData(imageData, 0, 0);
  return enhancedCanvas;
}

function flipImageHorizontally(canvas) {
  const flippedCanvas = document.createElement('canvas');
  flippedCanvas.width = canvas.width;
  flippedCanvas.height = canvas.height;
  const ctx = flippedCanvas.getContext('2d');
  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(canvas, 0, 0);
  return flippedCanvas;
}

function preprocessImage(sourceCanvas, faces, overridePaddingPct, flip = false, brightnessFactor = 1.0) {
  // Crop face region (or use full frame) - simplified for speed
  let processCanvas = cropFaceRegion(sourceCanvas, faces, overridePaddingPct);
  
  // Skip brightness and flip adjustments for speed (they're disabled in config anyway)
  // Only apply if explicitly needed (currently disabled)
  
  // Skip expensive enhancement for speed
  // processCanvas = enhanceImage(processCanvas);

  // Resize to target size with high quality
  const resizedCanvas = document.createElement('canvas');
  resizedCanvas.width = CONFIG.TARGET_SIZE;
  resizedCanvas.height = CONFIG.TARGET_SIZE;
  const ctx = resizedCanvas.getContext('2d');

  // Enable high-quality image smoothing
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';

  ctx.drawImage(processCanvas, 0, 0, CONFIG.TARGET_SIZE, CONFIG.TARGET_SIZE);

  // Extract image data
  const imageData = ctx.getImageData(0, 0, CONFIG.TARGET_SIZE, CONFIG.TARGET_SIZE);
  const pixels = imageData.data;

  // Create tensor with proper normalization
  const inputTensor = new Float32Array(3 * CONFIG.TARGET_SIZE * CONFIG.TARGET_SIZE);

  // Normalize: (pixel / 255.0 - mean) / std
  for (let i = 0; i < CONFIG.TARGET_SIZE * CONFIG.TARGET_SIZE; i++) {
    const pixelIndex = i * 4;
    
    // Get RGB values [0, 255] and normalize to [0, 1]
    const r = pixels[pixelIndex] / 255.0;
    const g = pixels[pixelIndex + 1] / 255.0;
    const b = pixels[pixelIndex + 2] / 255.0;

    // Apply ImageNet normalization: (x - mean) / std
    // Channel order: R, G, B (CHW format)
    inputTensor[i] = (r - CONFIG.MEAN[0]) / CONFIG.STD[0];
    inputTensor[CONFIG.TARGET_SIZE * CONFIG.TARGET_SIZE + i] = (g - CONFIG.MEAN[1]) / CONFIG.STD[1];
    inputTensor[2 * CONFIG.TARGET_SIZE * CONFIG.TARGET_SIZE + i] = (b - CONFIG.MEAN[2]) / CONFIG.STD[2];
  }

  // Return tensor in NCHW format: [1, 3, 224, 224]
  return new ort.Tensor('float32', inputTensor, [1, 3, CONFIG.TARGET_SIZE, CONFIG.TARGET_SIZE]);
}

/* ========================================
   FACE DETECTION & VISUALIZATION - Enhanced
======================================== */
async function detectAndDrawFaces(sourceCanvas) {
  let faces = null;

  if (state.hasDetector && state.blaze) {
    try {
      // Estimate faces with returnTensors=false for better performance
      // Try with different detection parameters for better accuracy
      let detectionResults = await state.blaze.estimateFaces(sourceCanvas, false);
      
      // Filter faces by confidence if available
      if (detectionResults && detectionResults.length > 0) {
        // BlazeFace returns faces with probability scores
        // Filter low-confidence detections (if probability < 0.5, skip)
        faces = detectionResults.filter(face => {
          // Check if face has probability property
          if (face.probability !== undefined) {
            return face.probability > 0.5;
          }
          // If no probability, assume it's valid
          return true;
        });
        
        // If all faces were filtered out, use the original results
        if (faces.length === 0 && detectionResults.length > 0) {
          faces = detectionResults;
        }
      } else {
        faces = detectionResults;
      }
      
      utils.log(`Detected ${faces ? faces.length : 0} face(s)`);

      // Draw face box on overlay if faces detected
      if (faces && faces.length > 0 && elements.overlayCanvas) {
        const ctx = elements.overlayCanvas.getContext('2d');
        elements.overlayCanvas.width = sourceCanvas.width;
        elements.overlayCanvas.height = sourceCanvas.height;
        ctx.clearRect(0, 0, elements.overlayCanvas.width, elements.overlayCanvas.height);

        // Find largest face
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

        const x = Math.floor(largestFace.topLeft[0]);
        const y = Math.floor(largestFace.topLeft[1]);
        const width = Math.ceil(largestFace.bottomRight[0] - largestFace.topLeft[0]);
        const height = Math.ceil(largestFace.bottomRight[1] - largestFace.topLeft[1]);

        // Draw face bounding box with enhanced styling
        ctx.strokeStyle = '#06b6d4';
        ctx.lineWidth = 3;
        ctx.shadowColor = 'rgba(6, 182, 212, 0.5)';
        ctx.shadowBlur = 10;
        ctx.strokeRect(x, y, width, height);

        // Draw corner accents for better visual feedback
        const cornerLength = 20;
        ctx.lineWidth = 4;
        ctx.shadowBlur = 5;

        // Top-left corner
        ctx.beginPath();
        ctx.moveTo(x, y + cornerLength);
        ctx.lineTo(x, y);
        ctx.lineTo(x + cornerLength, y);
        ctx.stroke();

        // Top-right corner
        ctx.beginPath();
        ctx.moveTo(x + width - cornerLength, y);
        ctx.lineTo(x + width, y);
        ctx.lineTo(x + width, y + cornerLength);
        ctx.stroke();

        // Bottom-left corner
        ctx.beginPath();
        ctx.moveTo(x, y + height - cornerLength);
        ctx.lineTo(x, y + height);
        ctx.lineTo(x + cornerLength, y + height);
        ctx.stroke();

        // Bottom-right corner
        ctx.beginPath();
        ctx.moveTo(x + width - cornerLength, y + height);
        ctx.lineTo(x + width, y + height);
        ctx.lineTo(x + width, y + height - cornerLength);
        ctx.stroke();
      } else if (elements.overlayCanvas) {
        // Clear overlay if no faces detected
        const ctx = elements.overlayCanvas.getContext('2d');
        ctx.clearRect(0, 0, elements.overlayCanvas.width, elements.overlayCanvas.height);
      }
    } catch (error) {
      utils.log('Face detection error:', error && error.message ? error.message : error);
      faces = null;
      
      // Clear overlay on error
      if (elements.overlayCanvas) {
        const ctx = elements.overlayCanvas.getContext('2d');
        ctx.clearRect(0, 0, elements.overlayCanvas.width, elements.overlayCanvas.height);
      }
    }
  } else {
    utils.log('Face detector not available, using full frame');
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

  // Get input name - more robust handling
  const inputName = state.inputName || 
    (state.session.inputNames && state.session.inputNames[0]) || 
    Object.keys(state.session.inputMetadata || {})[0];

  if (!inputName) {
    throw new Error('Unable to determine model input name');
  }

  const feeds = { [inputName]: inputTensor };
  utils.log('Running inference with input:', inputName);
  
  const results = await state.session.run(feeds);
  
  // Get output name - more robust handling
  const outputName = state.outputName || 
    (state.session.outputNames && state.session.outputNames[0]) || 
    Object.keys(results)[0];
    
  const output = results[outputName] || Object.values(results)[0];

  if (!output) {
    throw new Error('Model returned no output');
  }

  return output.data;
}

/* ========================================
   HELPER: Interpret output to probability
======================================== */
function interpretOutputToProbability(outputData) {
  if (!outputData || outputData.length === 0) {
    return null;
  }

  let probability;

  if (outputData.length === 1) {
    const rawValue = outputData[0];
    if (rawValue >= 0 && rawValue <= 1) {
      probability = rawValue;
    } else {
      probability = 1 / (1 + Math.exp(-rawValue));
    }
  } else if (outputData.length === 2) {
    const max = Math.max(outputData[0], outputData[1]);
    const exps = [
      Math.exp(outputData[0] - max),
      Math.exp(outputData[1] - max)
    ];
    const sum = exps[0] + exps[1];
    probability = exps[1] / sum; // Index 1 = Major
  } else {
    return null;
  }

  return Math.max(0, Math.min(1, probability));
}

/* ========================================
   CLASSIFICATION LOGIC - Enhanced
======================================== */
function interpretModelOutput(outputData) {
  if (!outputData || outputData.length === 0) {
    throw new Error('Empty model output');
  }

  utils.log('Raw model output:', Array.from(outputData).map(v => v.toFixed(4)));

  let probability;

  if (outputData.length === 1) {
    const rawValue = outputData[0];
    // Check if it's already a probability [0,1] or a logit
    if (rawValue >= 0 && rawValue <= 1) {
      probability = rawValue;
      utils.log('Single output interpreted as probability [0..1]');
    } else {
      // Apply sigmoid for logit
      probability = 1 / (1 + Math.exp(-rawValue));
      utils.log('Single output interpreted as logit; applied sigmoid');
    }
  } else if (outputData.length === 2) {
    // Two outputs: assume logits for [MINOR, MAJOR] or [MAJOR, MINOR]
    // Apply softmax to get probabilities
    const max = Math.max(outputData[0], outputData[1]);
    const exps = [
      Math.exp(outputData[0] - max),
      Math.exp(outputData[1] - max)
    ];
    const sum = exps[0] + exps[1];
    // Index 1 typically represents MAJOR class
    probability = exps[1] / sum;
    utils.log('Two outputs interpreted as 2-class logits; applied softmax');
    utils.log(`  Logits: [${outputData[0].toFixed(3)}, ${outputData[1].toFixed(3)}]`);
    utils.log(`  Probabilities: [${(exps[0]/sum*100).toFixed(1)}%, ${(probability*100).toFixed(1)}%]`);
  } else {
    throw new Error(`Unexpected model output length: ${outputData.length}`);
  }

  // Clamp probability to [0, 1] range for safety
  probability = Math.max(0, Math.min(1, probability));
  
  utils.log(`Final probability (Major): ${(probability * 100).toFixed(2)}%`);

  // Use threshold for classification, but calculate confidence differently
  const prediction = probability >= 0.5 ? 'MAJOR' : 'MINOR';
  // Confidence is the distance from 0.5, scaled to [0, 1]
  // Higher confidence when prediction is further from 0.5
  const confidence = Math.abs(probability - 0.5) * 2;

  return {
    prediction,
    confidence,
    isMinor: prediction === 'MINOR',
    probability,
    rawProbability: probability
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

    // Detect faces with enhanced detection
    updateStatus('Detecting face...', true);
    const faces = await detectAndDrawFaces(sourceCanvas);

    if (!faces || faces.length === 0) {
      utils.log('No face detected, analyzing full frame');
    } else {
      utils.log(`Face detected! Analyzing with face cropping...`);
    }

    // Ensure video is playing and ready
    if (elements.video.paused) {
      utils.log('Video is paused, playing...');
      try {
        await elements.video.play();
      } catch (e) {
        utils.log('Failed to play video:', e);
      }
    }
    
    // Wait for video to have valid dimensions
    let attempts = 0;
    while ((elements.video.videoWidth === 0 || elements.video.videoHeight === 0) && attempts < 10) {
      await utils.sleep(100);
      attempts++;
    }
    
    if (elements.video.videoWidth === 0 || elements.video.videoHeight === 0) {
      utils.log('Warning: Video dimensions are still 0, using fallback');
    }

    // Use single best frame for reliable, fast analysis
    updateStatus('Preparing analysis...', true);
    
    // Use the source canvas we already captured
    const analysisCanvas = sourceCanvas;
    const currentWidth = elements.video.videoWidth || videoWidth;
    const currentHeight = elements.video.videoHeight || videoHeight;
    
    // Ensure we have valid dimensions
    if (analysisCanvas.width === 0 || analysisCanvas.height === 0) {
      if (currentWidth > 0 && currentHeight > 0) {
        analysisCanvas.width = currentWidth;
        analysisCanvas.height = currentHeight;
        const ctx = analysisCanvas.getContext('2d');
        ctx.drawImage(elements.video, 0, 0, currentWidth, currentHeight);
      }
    }
    
    utils.log(`Analyzing frame: ${analysisCanvas.width}x${analysisCanvas.height}`);

    // Simplified TTA with async breaks to prevent UI freezing
    updateStatus('Analyzing age...', true);
    
    // Helper function to yield control and prevent UI blocking
    // Uses requestAnimationFrame for smooth UI updates when available
    const yieldToUI = () => {
      return new Promise(resolve => {
        if (typeof requestAnimationFrame !== 'undefined') {
          requestAnimationFrame(() => {
            setTimeout(resolve, CONFIG.INFERENCE_DELAY);
          });
        } else {
          setTimeout(resolve, CONFIG.INFERENCE_DELAY);
        }
      });
    };

    let allProbabilities = [];
    const paddingValues = CONFIG.TTA_PADDINGS;
    
    utils.log(`Starting analysis with ${paddingValues.length} padding value(s)`);

    // Process with small delays to prevent blocking
    for (let padIdx = 0; padIdx < paddingValues.length; padIdx++) {
      const pad = paddingValues[padIdx];
      
      // Yield to UI between operations
      if (padIdx > 0) {
        await yieldToUI();
      }
      
      try {
        updateStatus(`Analyzing... (${padIdx + 1}/${paddingValues.length})`, true);
        
        // Single inference with optimal settings
        const tensor = preprocessImage(analysisCanvas, faces, pad, false, 1.0);
        
        // Yield before inference
        await yieldToUI();
        
        const outputData = await runInference(tensor);
        
        // Yield after inference
        await yieldToUI();
        
        const prob = interpretOutputToProbability(outputData);
        if (prob !== null && !isNaN(prob) && prob >= 0 && prob <= 1) {
          allProbabilities.push(prob);
          utils.log(`Inference ${padIdx + 1}: Major probability = ${(prob * 100).toFixed(2)}%`);
        }
      } catch (e) {
        utils.log(`Inference failed for padding ${pad}%:`, e);
        // Continue with next padding
      }
    }
    
    utils.log(`Analysis complete: ${allProbabilities.length} successful inference(s)`);

    // Fallback: single pass if all attempts failed
    if (allProbabilities.length === 0) {
      utils.log('All inference attempts failed, using fallback');
      updateStatus('Using fallback analysis...', true);
      try {
        await yieldToUI();
        const fallbackTensor = preprocessImage(analysisCanvas, faces, CONFIG.FACE_PADDING, false, 1.0);
        await yieldToUI();
        const fallbackData = await runInference(fallbackTensor);
        const prob = interpretOutputToProbability(fallbackData);
        if (prob !== null && !isNaN(prob) && prob >= 0 && prob <= 1) {
          allProbabilities.push(prob);
          utils.log('Fallback inference successful:', (prob * 100).toFixed(2) + '%');
        } else {
          throw new Error('Fallback inference returned invalid result');
        }
      } catch (e) {
        utils.log('Fallback inference failed:', e);
        throw new Error('Failed to analyze image. Please ensure your face is clearly visible and try again.');
      }
    }
    
    if (allProbabilities.length === 0) {
      throw new Error('No valid predictions obtained');
    }

    // Calculate final probability (simple average for reliability)
    const avgProb = allProbabilities.reduce((a, b) => a + b, 0) / allProbabilities.length;
    
    // Calculate confidence based on:
    // 1. Distance from threshold (0.5)
    // 2. Consistency of predictions (if multiple)
    const distanceFromThreshold = Math.abs(avgProb - 0.5);
    
    let confidence;
    if (allProbabilities.length === 1) {
      // Single inference: confidence based on distance from threshold
      confidence = Math.min(0.95, 0.5 + (distanceFromThreshold * 1.5));
    } else {
      // Multiple inferences: confidence based on agreement and consistency
      const minProb = Math.min(...allProbabilities);
      const maxProb = Math.max(...allProbabilities);
      const range = maxProb - minProb;
      const stdDev = Math.sqrt(
        allProbabilities.reduce((sum, p) => sum + Math.pow(p - avgProb, 2), 0) / allProbabilities.length
      );
      
      // Agreement: how many predictions agree with the majority
      const majorVotes = allProbabilities.filter(p => p >= 0.5).length;
      const minorVotes = allProbabilities.filter(p => p < 0.5).length;
      const agreement = Math.max(majorVotes, minorVotes) / allProbabilities.length;
      
      // Base confidence from agreement
      confidence = agreement;
      
      // Boost if consistent (low std dev and range)
      if (stdDev < 0.1 && range < 0.2) {
        confidence = Math.min(0.95, confidence * 1.2);
      }
      
      // Boost if far from threshold
      confidence = Math.min(0.95, confidence + (distanceFromThreshold * 0.4));
      
      // Penalty if inconsistent
      if (stdDev > 0.15 || range > 0.3) {
        confidence = confidence * 0.85;
      }
      
      utils.log(`Statistics: Avg=${(avgProb*100).toFixed(1)}%, Range=[${(minProb*100).toFixed(1)}%, ${(maxProb*100).toFixed(1)}%], StdDev=${(stdDev*100).toFixed(1)}%, Agreement=${(agreement*100).toFixed(0)}%`);
    }
    
    // Ensure minimum confidence
    confidence = Math.max(0.55, Math.min(0.95, confidence));
    
    // Apply threshold with slight bias toward safety (minor = SafeSearch ON)
    // avgProb is the probability of MAJOR (0 = MINOR, 1 = MAJOR)
    // Use 0.52 threshold for MAJOR (more conservative - requires higher confidence)
    const MAJOR_THRESHOLD = 0.52;
    const prediction = avgProb >= MAJOR_THRESHOLD ? 'MAJOR' : 'MINOR';
    const isMinor = prediction === 'MINOR';
    
    // Adjust confidence based on threshold distance
    const thresholdDistance = Math.abs(avgProb - (isMinor ? (1 - MAJOR_THRESHOLD) : MAJOR_THRESHOLD));
    if (thresholdDistance < 0.08) {
      // Close to threshold: reduce confidence
      confidence = Math.max(0.55, confidence * 0.85);
      utils.log(`Close to threshold (distance: ${(thresholdDistance*100).toFixed(1)}%), reducing confidence`);
    }

    const result = {
      prediction,
      confidence,
      isMinor,
      probability: avgProb,
      rawProbability: avgProb,
      inferenceCount: allProbabilities.length
    };
    
    utils.log('Final Classification:', result.prediction);
    utils.log('  Confidence:', (result.confidence * 100).toFixed(1) + '%');
    utils.log('  Probability (Major):', (avgProb * 100).toFixed(2) + '%');
    utils.log('  Threshold used:', isMinor ? '< 52% (Major)' : '>= 52% (Major)');
    utils.log('  Distance from threshold:', (thresholdDistance * 100).toFixed(1) + '%');

    // Update UI with enhanced result display
    utils.setElementText(elements.resultText, result.prediction);
    const confidencePercent = (result.confidence * 100).toFixed(1);
    utils.setElementText(elements.resultSub, `Confidence: ${confidencePercent}% | ${result.inferenceCount} analyses`);
    
    utils.removeClass(elements.resultCard, 'major', 'minor');
    utils.addClass(elements.resultCard, 'active', result.isMinor ? 'minor' : 'major');
    
    if (elements.resultIcon) {
      elements.resultIcon.textContent = result.isMinor ? '👶' : '👨';
    }
    
    updateStatus('Analysis complete!', false);

    // Update SafeSearch
    await updateSafeSearch(result.isMinor);

    // Set 1-hour verification window
    try {
      const allowedUntil = Date.now() + 60 * 60 * 1000;
      await chrome.storage.local.set({ verificationAllowedUntil: allowedUntil });
    } catch (_) {}

    utils.removeClass(elements.scanOverlay, 'active');

    // Auto-stop camera
    setTimeout(() => {
      stopCamera();
    }, CONFIG.AUTO_STOP_DELAY);

  } catch (error) {
    utils.log('Analysis error:', error);
    const errorMsg = error && error.message ? error.message : String(error);
    updateStatus(`Analysis failed: ${errorMsg}. Please try again.`, false);
    utils.removeClass(elements.scanOverlay, 'active');
    
    // Show error in UI
    utils.setElementText(elements.resultText, 'Error');
    utils.setElementText(elements.resultSub, 'Please try again');
    utils.removeClass(elements.resultCard, 'major', 'minor');
    
    // Don't stop camera immediately - let user try again
    setTimeout(() => {
      if (!state.running) {
        stopCamera();
      }
    }, 2000);
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
    if (elements.openSettingsBtn) {
      elements.openSettingsBtn.style.display = 'none';
    }

    // Check if getUserMedia is available
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error('Camera API not supported in this browser');
    }

    // Request camera with constraints - this will show native browser permission prompt
    let stream;
    try {
      utils.log('Requesting camera with ideal constraints...');
      // First try with ideal constraints - this triggers the native permission prompt
      stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user'
        },
        audio: false
      });
      utils.log('Camera access granted with ideal constraints');
    } catch (constraintError) {
      utils.log('Camera request error:', constraintError.name, constraintError.message);
      
      if (constraintError.name === "OverconstrainedError") {
        // Try with simpler constraints
        utils.log('Trying fallback constraints...');
        try {
          stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'user' },
            audio: false
          });
          utils.log('Camera access granted with fallback constraints');
        } catch (fallbackError) {
          utils.log('Fallback constraints also failed:', fallbackError);
          throw fallbackError;
        }
      } else if (constraintError.name === "NotAllowedError" || constraintError.name === "PermissionDeniedError") {
        // Permission denied - show user-friendly message
        utils.log('Camera permission denied by user');
        throw new Error('CAMERA_PERMISSION_DENIED');
      } else {
        // Other error - rethrow
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
    
    // Wait for video to be fully ready before proceeding
    await new Promise((resolve) => {
      if (elements.video.readyState >= 2 && elements.video.videoWidth > 0) {
        resolve();
      } else {
        const checkReady = () => {
          if (elements.video.readyState >= 2 && elements.video.videoWidth > 0) {
            resolve();
          }
        };
        elements.video.onloadedmetadata = checkReady;
        elements.video.oncanplay = checkReady;
        // Timeout after 3 seconds
        setTimeout(() => {
          if (elements.video.videoWidth > 0) {
            resolve();
          } else {
            utils.log('Video metadata timeout, proceeding anyway');
            resolve();
          }
        }, 3000);
      }
    });
    
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

    // Wait a bit for camera to stabilize
    await utils.sleep(500);

    // Auto-analyze after guidelines
    await utils.sleep(CONFIG.GUIDELINE_DURATION);
    hideGuidelines();

    // Ensure video is still playing before analysis
    if (elements.video.paused) {
      await elements.video.play();
    }

    await analyzeFrame();

  } catch (error) {
    utils.log('Camera error:', error);

    state.running = false;
    elements.startBtn.disabled = false;
    elements.stopBtn.disabled = true;

    const errorName = error.name || error.message || 'Unknown';
    const errorMessage = error.message || '';

    // Handle specific error cases
    if (errorName === 'NotAllowedError' || errorName === 'PermissionDeniedError' || errorMessage === 'CAMERA_PERMISSION_DENIED') {
      updateStatus('Camera access denied. Click "Allow" when prompted, or check browser settings.', false);
      if (elements.openSettingsBtn) {
        elements.openSettingsBtn.style.display = 'block';
        elements.openSettingsBtn.textContent = 'Open Camera Settings';
      }
      // Provide helpful instructions
      utils.log('To enable camera access:');
      utils.log('1. Click the camera icon in the address bar');
      utils.log('2. Select "Allow" for camera access');
      utils.log('3. Or open chrome://settings/content/camera and allow the extension');
    } else if (errorName === 'NotFoundError') {
      updateStatus('No camera found. Please connect a camera device.', false);
    } else if (errorName === 'NotReadableError' || errorName === 'TrackStartError') {
      updateStatus('Camera is in use by another app. Please close other apps using the camera.', false);
    } else if (errorMessage.includes('not supported')) {
      updateStatus('Camera API not supported in this browser. Please use Chrome or Edge.', false);
    } else {
      updateStatus(`Camera error: ${utils.formatError(error)}. Please try again.`, false);
      if (elements.openSettingsBtn) {
        elements.openSettingsBtn.style.display = 'block';
      }
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
