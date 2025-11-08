// web_script.js - adapted from provided script.js for Chrome extension

let ort, tf, blazeface;
let faceDetector = null;
let session = null;
let stream = null;
let uploadedImage = null;
let isProcessing = false;
let currentMode = 'none'; // 'none', 'camera', 'upload'
let lastMinorState = null;

const elements = {
	video: document.getElementById('video'),
	hiddenCanvas: document.getElementById('hiddenCanvas'),
	overlayCanvas: document.getElementById('overlayCanvas'),
	previewImage: document.getElementById('previewImage'),
	imagePreviewContainer: document.getElementById('imagePreviewContainer'),
	videoContainer: document.querySelector('.video-container'),
	startCamBtn: document.getElementById('startCamBtn'),
	stopCamBtn: document.getElementById('stopCamBtn'),
	captureBtn: document.getElementById('captureBtn'),
	fileInput: document.getElementById('fileInput'),
	modelSelect: document.getElementById('modelSelect'),
	useFaceDetection: document.getElementById('useFaceDetection'),
	facePadding: document.getElementById('facePadding'),
	paddingValue: document.getElementById('paddingValue'),
	result: document.getElementById('result'),
	debugLog: document.getElementById('debugLog'),
	toggleDebug: document.getElementById('toggleDebug')
};

function log(...args) {
	try { console.log(...args); } catch (_) {}
	const message = args.map(arg =>
		typeof arg === 'object' ? JSON.stringify(arg, null, 2) : String(arg)
	).join(' ');
	if (elements.debugLog) {
		elements.debugLog.textContent += message + '\n';
		elements.debugLog.scrollTop = elements.debugLog.scrollHeight;
	}
}

function setStatus(message) {
	elements.result.innerHTML = `<p class="status">${message}</p>`;
	elements.result.className = 'result';
}

function setResult(prediction, confidence) {
	const resultClass = prediction.toLowerCase();
	elements.result.className = `result ${resultClass}`;
	elements.result.innerHTML = `
    <p class="prediction">${prediction}</p>
    <p class="confidence">Confidence: ${(confidence * 100).toFixed(1)}%</p>
  `;
}

async function initializeLibraries() {
	try {
		log('Initializing libraries...');
		ort = window.ort;
		// Explicit ORT WASM mapping and preflight
		try {
			const wasmMap = {
				"ort-wasm.wasm": chrome.runtime.getURL('lib/ort-wasm.wasm'),
				"ort-wasm-simd.wasm": chrome.runtime.getURL('lib/ort-wasm-simd.wasm'),
				"ort-wasm-threaded.wasm": chrome.runtime.getURL('lib/ort-wasm-threaded.wasm'),
				"ort-wasm-simd-threaded.wasm": chrome.runtime.getURL('lib/ort-wasm-simd-threaded.wasm')
			};
			if (ort && ort.env && ort.env.wasm) {
				ort.env.wasm.wasmPaths = wasmMap;
				ort.env.wasm.numThreads = 1;
				log('ORT wasm mapping set', wasmMap);
				const availability = {};
				for (const [name, url] of Object.entries(wasmMap)){
					try { const r = await fetch(url, { method: 'HEAD' }); availability[name] = r.ok; log('WASM preflight', name, r.ok ? 'OK' : `HTTP ${r.status}`); }
					catch(e){ availability[name] = false; log('WASM preflight error', name, e && e.message ? e.message : e); }
				}
				try {
					if (!availability["ort-wasm-simd.wasm"]) {
						ort.env.wasm.simd = false;
						log('Disabling SIMD: simd wasm not available');
					}
					if (!availability["ort-wasm-threaded.wasm"] || !availability["ort-wasm-simd-threaded.wasm"]) {
						ort.env.wasm.numThreads = 1;
						log('Disabling threading: threaded wasm not available');
					}
				} catch (e){ log('Failed to set ORT fallbacks', e && e.message ? e.message : e); }
			}
		} catch (e) {
			log('Could not set ORT wasm mapping:', e && e.message ? e.message : e);
		}
		tf = window.tf;
		blazeface = window.blazeface;
		if (!ort) throw new Error('ONNX Runtime (ort) not loaded');
		if (!tf) throw new Error('TensorFlow.js (tf) not loaded');
		if (!blazeface) throw new Error('BlazeFace (blazeface) not loaded');
		log('Loading BlazeFace model...');
		faceDetector = await blazeface.load();
		log('BlazeFace model loaded successfully');
		return true;
	} catch (error) {
		log('Error initializing libraries:', error && error.message ? error.message : error);
		log('Face detection will be disabled');
		return false;
	}
}

async function loadONNXModel(modelPath) {
	try {
		if (!modelPath) throw new Error('Empty model path');
		log(`Loading ONNX model (fetch): ${modelPath}`);
		setStatus('Loading model...');
		let response = await fetch(modelPath, { method: 'GET' });
		if (!response.ok) {
			const altPath = modelPath.startsWith('./') ? modelPath.slice(2) : `./${modelPath}`;
			log(`Primary fetch failed (${response.status}). Trying fallback: ${altPath}`);
			response = await fetch(altPath, { method: 'GET' });
		}
		if (!response.ok) {
			const msg = `Model fetch failed: HTTP ${response.status} ${response.statusText}`;
			log(msg);
			setStatus('Failed to fetch model (network/404). Check path.');
			return false;
		}
		const contentType = response.headers.get('Content-Type') || '';
		log('Model fetch OK. Content-Type:', contentType);
		if (contentType.includes('text/html')) {
			const text = await response.text();
			log('Returned HTML instead of ONNX. First 400 chars:\n', text.slice(0,400));
			setStatus('Unexpected HTML when fetching model.');
			return false;
		}
		const buffer = await response.arrayBuffer();
		if (!buffer || buffer.byteLength < 4) {
			log('Model buffer too small/empty');
			setStatus('Model file seems empty or corrupted.');
			return false;
		}
		log('Creating InferenceSession from buffer:', buffer.byteLength);
		const opts = { executionProviders: ['wasm'] };
		session = await ort.InferenceSession.create(new Uint8Array(buffer), opts);
		log('Model loaded successfully');
		const inputNames = session.inputNames || (session.inputMetadata ? Object.keys(session.inputMetadata) : []);
		const outputNames = session.outputNames || (session.outputMetadata ? Object.keys(session.outputMetadata) : []);
		log('Input names:', inputNames);
		log('Output names:', outputNames);
		setStatus('Model loaded. Ready to analyze.');
		return true;
	} catch (error) {
		log('Error loading model:', error && error.message ? error.message : error);
		setStatus('Failed to load model. Ensure ORT wasm files are present in lib/.');
		return false;
	}
}

async function detectFaces(imageCanvas) {
	if (!faceDetector || !elements.useFaceDetection.checked) {
		return null;
	}
	try {
		log('Detecting faces...');
		const predictions = await faceDetector.estimateFaces(imageCanvas, false);
		log(`Found ${predictions.length} face(s)`);
		return predictions;
	} catch (error) {
		log('Face detection error:', error && error.message ? error.message : error);
		return null;
	}
}

function drawFaceBox(face, sourceW, sourceH) {
	const canvas = elements.overlayCanvas;
	canvas.width = sourceW || elements.video.videoWidth;
	canvas.height = sourceH || elements.video.videoHeight;
	const ctx = canvas.getContext('2d');
	ctx.clearRect(0, 0, canvas.width, canvas.height);
	ctx.strokeStyle = '#00ff00';
	ctx.lineWidth = 3;
	const x = face.topLeft[0];
	const y = face.topLeft[1];
	const width = face.bottomRight[0] - face.topLeft[0];
	const height = face.bottomRight[1] - face.topLeft[1];
	ctx.strokeRect(x, y, width, height);
}

function cropFaceRegion(sourceCanvas, faces) {
	if (!faces || faces.length === 0) {
		log('No faces detected or detection disabled, using full image');
		const ctxClear = elements.overlayCanvas.getContext('2d');
		ctxClear.clearRect(0, 0, elements.overlayCanvas.width, elements.overlayCanvas.height);
		return sourceCanvas;
	}
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
	const padding = parseInt(elements.facePadding.value, 10) / 100;
	const x1 = Math.floor(largestFace.topLeft[0]);
	const y1 = Math.floor(largestFace.topLeft[1]);
	const x2 = Math.ceil(largestFace.bottomRight[0]);
	const y2 = Math.ceil(largestFace.bottomRight[1]);
	const faceWidth = x2 - x1;
	const faceHeight = y2 - y1;
	const padX = Math.floor(faceWidth * padding);
	const padY = Math.floor(faceHeight * padding);
	const cropX = Math.max(0, x1 - padX);
	const cropY = Math.max(0, y1 - padY);
	const cropWidth = Math.min(sourceCanvas.width - cropX, faceWidth + 2 * padX);
	const cropHeight = Math.min(sourceCanvas.height - cropY, faceHeight + 2 * padY);
	log(`Cropping face region: x=${cropX}, y=${cropY}, w=${cropWidth}, h=${cropHeight}`);
	const croppedCanvas = document.createElement('canvas');
	croppedCanvas.width = cropWidth;
	croppedCanvas.height = cropHeight;
	const ctx = croppedCanvas.getContext('2d');
	ctx.drawImage(
		sourceCanvas,
		cropX, cropY, cropWidth, cropHeight,
		0, 0, cropWidth, cropHeight
	);
	drawFaceBox(largestFace, sourceCanvas.width, sourceCanvas.height);
	return croppedCanvas;
}

function preprocessImage(canvas) {
	const INPUT_SIZE = 224;
	const resizedCanvas = document.createElement('canvas');
	resizedCanvas.width = INPUT_SIZE;
	resizedCanvas.height = INPUT_SIZE;
	const ctx = resizedCanvas.getContext('2d');
	ctx.drawImage(canvas, 0, 0, INPUT_SIZE, INPUT_SIZE);
	const imageData = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
	const pixels = imageData.data;
	const mean = [0.485, 0.456, 0.406];
	const std = [0.229, 0.224, 0.225];
	const inputTensor = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
	for (let i = 0; i < INPUT_SIZE * INPUT_SIZE; i++) {
		const pixelIndex = i * 4;
		const r = pixels[pixelIndex] / 255.0;
		const g = pixels[pixelIndex + 1] / 255.0;
		const b = pixels[pixelIndex + 2] / 255.0;
		inputTensor[i] = (r - mean[0]) / std[0];
		inputTensor[INPUT_SIZE * INPUT_SIZE + i] = (g - mean[1]) / std[1];
		inputTensor[2 * INPUT_SIZE * INPUT_SIZE + i] = (b - mean[2]) / std[2];
	}
	return new ort.Tensor('float32', inputTensor, [1, 3, INPUT_SIZE, INPUT_SIZE]);
}

async function runInference(imageSource) {
	if (!session) {
		throw new Error('Model not loaded');
	}
	const sourceCanvas = document.createElement('canvas');
	if (imageSource instanceof HTMLVideoElement) {
		sourceCanvas.width = imageSource.videoWidth;
		sourceCanvas.height = imageSource.videoHeight;
		const ctx = sourceCanvas.getContext('2d');
		ctx.drawImage(imageSource, 0, 0, sourceCanvas.width, sourceCanvas.height);
	} else {
		sourceCanvas.width = imageSource.naturalWidth || imageSource.width;
		sourceCanvas.height = imageSource.naturalHeight || imageSource.height;
		const ctx = sourceCanvas.getContext('2d');
		ctx.drawImage(imageSource, 0, 0, sourceCanvas.width, sourceCanvas.height);
	}
	elements.overlayCanvas.width = sourceCanvas.width;
	elements.overlayCanvas.height = sourceCanvas.height;
	const faces = await detectFaces(sourceCanvas);
	const processCanvas = cropFaceRegion(sourceCanvas, faces);
	const inputTensor = preprocessImage(processCanvas);
	const inputName = (session.inputNames && session.inputNames[0]) || Object.keys(session.inputMetadata || {})[0];
	if (!inputName) throw new Error('Unable to determine model input name');
	const feeds = { [inputName]: inputTensor };
	log('Running inference...');
	const results = await session.run(feeds);
	const outputName = (session.outputNames && session.outputNames[0]) || Object.keys(results)[0];
	const output = results[outputName] || Object.values(results)[0];
	if (!output) throw new Error('Model returned no outputs');
	const data = output.data;
	log('Raw model output:', Array.from(data));
	let probability;
	if (data.length === 1) {
		const rawValue = data[0];
		probability = (rawValue >= 0 && rawValue <= 1) ? rawValue : 1 / (1 + Math.exp(-rawValue));
		log('Using single-output (prob or sigmoid)');
	} else if (data.length === 2) {
		const max = Math.max(data[0], data[1]);
		const exps = data.map(v => Math.exp(v - max));
		const sum = exps[0] + exps[1];
		probability = exps[1] / sum; // assume index 1 = Major
		log('Using 2-class logits (softmax)');
	} else {
		throw new Error(`Unexpected model output length: ${data.length}`);
	}
	log(`Final probability (Major): ${(probability * 100).toFixed(2)}%`);
	return probability;
}

async function analyzeImage() {
	if (isProcessing) {
		log('Already processing...');
		return;
	}
	let imageSource;
	if (currentMode === 'upload' && uploadedImage) {
		imageSource = uploadedImage;
		log('Using uploaded image for analysis');
	} else if (currentMode === 'camera' && stream && elements.video.srcObject) {
		imageSource = elements.video;
		log('Using camera video for analysis');
	} else {
		setStatus('No image source available');
		log('No valid image source in current mode');
		return;
	}
	isProcessing = true;
	setStatus('Analyzing...');
	elements.captureBtn.disabled = true;
	try {
		const probability = await runInference(imageSource);
		const prediction = probability >= 0.5 ? 'MAJOR' : 'MINOR';
		const confidence = probability >= 0.5 ? probability : (1 - probability);
		setResult(prediction, confidence);
		const isMinor = prediction === 'MINOR';
		if (lastMinorState !== isMinor) {
			lastMinorState = isMinor;
			try { await new Promise((resolve) => chrome.runtime.sendMessage({ type: 'SET_SAFE', minor: isMinor }, resolve)); } catch (_) {}
		}
		log(`Prediction: ${prediction} with ${(confidence * 100).toFixed(1)}% confidence`);
	} catch (error) {
		log('Analysis error:', error && error.message ? error.message : error);
		setStatus('Analysis failed. Check debug log.');
	} finally {
		isProcessing = false;
		elements.captureBtn.disabled = false;
	}
}

async function startCamera() {
	try {
		log('Starting camera...');
		setStatus('Starting camera...');
		clearUploadedImage();
		const constraints = {
			video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
			audio: false
		};
		stream = await navigator.mediaDevices.getUserMedia(constraints);
		elements.video.srcObject = stream;
		await elements.video.play();
		elements.overlayCanvas.width = elements.video.videoWidth;
		elements.overlayCanvas.height = elements.video.videoHeight;
		currentMode = 'camera';
		elements.startCamBtn.disabled = true;
		elements.stopCamBtn.disabled = false;
		elements.captureBtn.disabled = false;
		elements.videoContainer.style.display = 'block';
		elements.imagePreviewContainer.style.display = 'none';
		setStatus('Camera ready. Click "Capture & Analyze" or keep capturing manually.');
		log('Camera started successfully');
	} catch (error) {
		log('Camera error:', error && error.message ? error.message : error);
		setStatus('Failed to start camera. Check permissions.');
		alert('Camera access denied or unavailable. Please allow camera access.');
	}
}

function stopCamera() {
	if (stream) {
		stream.getTracks().forEach(track => track.stop());
		stream = null;
	}
	elements.video.srcObject = null;
	const ctx = elements.overlayCanvas.getContext('2d');
	ctx.clearRect(0, 0, elements.overlayCanvas.width, elements.overlayCanvas.height);
	currentMode = 'none';
	elements.startCamBtn.disabled = false;
	elements.stopCamBtn.disabled = true;
	elements.captureBtn.disabled = true;
	elements.videoContainer.style.display = 'block';
	setStatus('Camera stopped.');
	log('Camera stopped');
}

function clearUploadedImage() {
	uploadedImage = null;
	elements.previewImage.src = '';
	elements.fileInput.value = '';
	const ctx = elements.overlayCanvas.getContext('2d');
	ctx.clearRect(0, 0, elements.overlayCanvas.width, elements.overlayCanvas.height);
	log('Uploaded image cleared');
}

elements.startCamBtn.addEventListener('click', startCamera);

elements.stopCamBtn.addEventListener('click', stopCamera);

elements.captureBtn.addEventListener('click', analyzeImage);

elements.fileInput.addEventListener('change', (e) => {
	const file = e.target.files[0];
	if (!file) return;
	if (!file.type.startsWith('image/')) {
		alert('Please select an image file');
		return;
	}
	log(`Loading uploaded file: ${file.name}`);
	if (stream) {
		stopCamera();
	}
	const reader = new FileReader();
	reader.onload = (event) => {
		const img = new Image();
		img.onload = () => {
			uploadedImage = img;
			currentMode = 'upload';
			elements.previewImage.src = event.target.result;
			elements.imagePreviewContainer.style.display = 'block';
			elements.videoContainer.style.display = 'none';
			elements.captureBtn.disabled = false;
			setStatus('Image uploaded. Click "Capture & Analyze".');
			log('Image loaded successfully');
		};
		img.src = event.target.result;
	};
	reader.readAsDataURL(file);
});

elements.modelSelect.addEventListener('change', async () => {
	const modelPath = elements.modelSelect.value;
	elements.captureBtn.disabled = true;
	const ok = await loadONNXModel(modelPath);
	elements.captureBtn.disabled = !(ok && (currentMode === 'camera' || currentMode === 'upload'));
});

elements.facePadding.addEventListener('input', (e) => {
	elements.paddingValue.textContent = `${e.target.value}%`;
});

elements.toggleDebug.addEventListener('click', () => {
	const isHidden = elements.debugLog.style.display === 'none';
	elements.debugLog.style.display = isHidden ? 'block' : 'none';
	elements.toggleDebug.textContent = isHidden ? 'Hide Debug Info' : 'Show Debug Info';
});

async function initialize() {
	log('=== Face Age Detector Initializing ===');
	await initializeLibraries();
	let modelPath = elements.modelSelect.value || 'swin_face_classifier.onnx';
	elements.modelSelect.value = modelPath;
	await loadONNXModel(modelPath);
	log('=== Initialization Complete ===');
}

initialize();
