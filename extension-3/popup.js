const videoEl = document.getElementById("video");
const canvasEl = document.getElementById("work");
const statusText = document.getElementById("statusText");
const resultText = document.getElementById("resultText");
const resultSub = document.getElementById("resultSub");
const resultCard = document.getElementById("resultCard");
const resultIcon = document.getElementById("resultIcon");
const safesearchBadge = document.getElementById("safesearchBadge");
const badgeText = document.getElementById("badgeText");
const badgeIcon = document.getElementById("badgeIcon");
const errorBadge = document.getElementById("errorBadge");
const errorText = document.getElementById("errorText");
const videoOverlay = document.getElementById("videoOverlay");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const openSettingsBtn = document.getElementById("openSettingsBtn");
const openFullPageBtn = document.getElementById("openFullPageBtn");

let stream = null;
let running = false;
let session = null;
let inputName = null;
let outputName = null;
let lastMinorState = null;
let blaze = null; // BlazeFace model
let hasDetector = false;

const TARGET_SIZE = 224;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

function setStatus(text){ statusText.textContent = text; }
function setResult(text){ resultText.textContent = text; }

function showResult(prediction, confidence, isMinor){
	resultCard.classList.add('has-result', isMinor ? 'minor' : 'major');
	resultCard.classList.remove(isMinor ? 'major' : 'minor');
	resultText.textContent = prediction;
	resultSub.textContent = `Confidence: ${(confidence * 100).toFixed(1)}%`;
	resultIcon.textContent = isMinor ? '👶' : '👨';
}

// Configure ORT WASM path to extension lib/ (base path), disable SIMD and threading for MV3 compatibility
try {
	const wasmBase = chrome.runtime.getURL('lib/');
	if (window.ort){
		if (!window.ort.env) window.ort.env = {};
		if (!window.ort.env.wasm) window.ort.env.wasm = {};
		window.ort.env.wasm.wasmPaths = wasmBase;
		window.ort.env.wasm.numThreads = 1;
		window.ort.env.wasm.simd = false;
		console.log('[popup] ORT wasm base path', wasmBase, 'threads=1 simd=false');
	}
} catch (e){
	console.warn('[popup] Failed to set ORT wasm base path', e);
}

async function createSessionWithProvider(modelBytes, provider){
	console.log('[popup] Creating session with provider', provider);
	return await ort.InferenceSession.create(modelBytes, { executionProviders: [provider] });
}

async function initSession(){
	try{
		const modelUrl = chrome.runtime.getURL("swin_face_classifier.onnx");
		console.log('[popup] Loading model (fetch bytes) from', modelUrl);
		setStatus("Loading model...");
		const resp = await fetch(modelUrl, { method: 'GET' });
		if (!resp.ok){
			throw new Error(`Model HTTP ${resp.status}`);
		}
		const buf = await resp.arrayBuffer();
		if (!buf || buf.byteLength < 4){
			throw new Error('Model buffer empty/corrupt');
		}
		const bytes = new Uint8Array(buf);
		console.log('[popup] Model bytes', bytes.byteLength);
		try {
			session = await createSessionWithProvider(bytes, 'wasm');
		} catch (eWas){
			console.warn('[popup] WASM provider failed, trying WEBGL', eWas && eWas.message ? eWas.message : eWas);
			session = await createSessionWithProvider(bytes, 'webgl');
		}
		inputName = session.inputNames[0];
		outputName = session.outputNames[0];
		console.log('[popup] Model ready. Inputs:', session.inputNames, 'Outputs:', session.outputNames);
		setStatus("Model ready");
	}catch(e){
		console.error('[popup] Model load error', e);
		setStatus(`Model load error: ${e && e.message ? e.message : e}`);
		throw e;
	}
}

async function ensureDetector(){
	try{
		if (window.blazeface && window.tf){
			blaze = await window.blazeface.load();
			hasDetector = true;
			console.log('[popup] BlazeFace loaded for detection');
		}else{
			hasDetector = false;
			console.log('[popup] BlazeFace not available; using full-frame');
		}
	}catch(e){
		hasDetector = false;
		console.warn('[popup] BlazeFace init failed; using full-frame', e && e.message ? e.message : e);
	}
}

function preprocessFromVideoBox(box){
	const ctx = canvasEl.getContext("2d", { willReadFrequently: true });
	const sx = Math.max(0, Math.floor(box.x));
	const sy = Math.max(0, Math.floor(box.y));
	const sw = Math.max(1, Math.floor(box.width));
	const sh = Math.max(1, Math.floor(box.height));
	canvasEl.width = TARGET_SIZE;
	canvasEl.height = TARGET_SIZE;
	ctx.drawImage(videoEl, sx, sy, sw, sh, 0, 0, TARGET_SIZE, TARGET_SIZE);
	const imageData = ctx.getImageData(0, 0, TARGET_SIZE, TARGET_SIZE);
	const { data } = imageData;
	const floatData = new Float32Array(3 * TARGET_SIZE * TARGET_SIZE);
	let p = 0;
	for (let i = 0; i < data.length; i += 4){
		const r = data[i] / 255;
		const g = data[i+1] / 255;
		const b = data[i+2] / 255;
		floatData[p] = (r - MEAN[0]) / STD[0];
		floatData[p + TARGET_SIZE*TARGET_SIZE] = (g - MEAN[1]) / STD[1];
		floatData[p + 2*TARGET_SIZE*TARGET_SIZE] = (b - MEAN[2]) / STD[2];
		p++;
	}
	return new ort.Tensor("float32", floatData, [1, 3, TARGET_SIZE, TARGET_SIZE]);
}

function softmax(arr){
	const max = Math.max(...arr);
	const exps = arr.map(v => Math.exp(v - max));
	const sum = exps.reduce((a,b)=>a+b,0);
	return exps.map(v => v / sum);
}

async function analyzeSingleFrame(){
	try {
		// Reset UI state
		resultCard.classList.remove('has-result', 'major', 'minor');
		resultText.textContent = '-';
		resultSub.textContent = '';
		resultIcon.textContent = '👤';
		safesearchBadge.style.display = 'none';
		errorBadge.style.display = 'none';
		videoOverlay.classList.remove('active');
		
		setStatus("Capturing frame...");
		videoOverlay.classList.add('active');
		
		// Wait a moment for video to be ready
		await new Promise(resolve => setTimeout(resolve, 500));
		
		let box;
		if (hasDetector && blaze){
			const faces = await blaze.estimateFaces(videoEl, false);
			if (faces && faces.length){
				const f = faces[0];
				const x1 = f.topLeft[0];
				const y1 = f.topLeft[1];
				const x2 = f.bottomRight[0];
				const y2 = f.bottomRight[1];
				box = { x: x1, y: y1, width: x2 - x1, height: y2 - y1 };
				console.log('[popup] Face detected at', box);
			}
		}
		if (!box){
			const vw = videoEl.videoWidth || videoEl.clientWidth || 224;
			const vh = videoEl.videoHeight || videoEl.clientHeight || 224;
			box = { x: 0, y: 0, width: vw, height: vh };
			console.log('[popup] Using full frame', box);
		}
		
		setStatus("Analyzing...");
		const input = preprocessFromVideoBox(box);
		const feeds = {}; feeds[inputName] = input;
		const outputMap = await session.run(feeds);
		const scores = Array.from(outputMap[outputName].data);
		const probs = softmax(scores);
		const idx = probs[1] >= probs[0] ? 1 : 0; // 0: minor, 1: major (assumed order)
		const isMinor = idx === 0;
		const confidence = Math.max(probs[0], probs[1]);
		
		const prediction = isMinor ? 'MINOR' : 'MAJOR';
		showResult(prediction, confidence, isMinor);
		console.log('[popup] Classification:', prediction, 'confidence:', (confidence * 100).toFixed(1) + '%');
		
		// Hide error badge initially
		errorBadge.style.display = 'none';
		safesearchBadge.style.display = 'none';
		
		if (lastMinorState !== isMinor){
			lastMinorState = isMinor;
			console.log('[popup] Updating SafeSearch:', isMinor ? 'ON' : 'OFF');
			try {
				const response = await new Promise((resolve) => 
					chrome.runtime.sendMessage({ type: "SET_SAFE", minor: isMinor }, resolve)
				);
				
				if (response && response.ok) {
					// SafeSearch update successful
					setTimeout(() => {
						safesearchBadge.style.display = 'flex';
						safesearchBadge.classList.remove('on', 'off');
						safesearchBadge.classList.add(isMinor ? 'on' : 'off');
						badgeIcon.textContent = isMinor ? '🛡️' : '🔓';
						badgeText.textContent = `SafeSearch is ${isMinor ? 'ON' : 'OFF'}`;
					}, 800);
				} else if (response && response.error) {
					// SafeSearch is locked - show error message
					console.warn('[popup] SafeSearch update failed:', response.error);
					setTimeout(() => {
						errorBadge.style.display = 'flex';
						errorText.textContent = 'SafeSearch is locked by browser/admin settings. Classification successful, but SafeSearch cannot be changed automatically.';
					}, 800);
				}
			} catch (err) {
				console.error('[popup] Error updating SafeSearch:', err);
				setTimeout(() => {
					errorBadge.style.display = 'flex';
					errorText.textContent = 'Could not update SafeSearch. Classification successful.';
				}, 800);
			}
		} else {
			console.log('[popup] SafeSearch already set to', isMinor ? 'ON' : 'OFF');
		}
		
		setStatus("Analysis complete!");
		videoOverlay.classList.remove('active');
		
		// Automatically stop camera after showing result
		setTimeout(() => {
			stop();
		}, 3000); // Give user 3 seconds to see the result
		
	} catch (e){
		console.error('[popup] Analysis error', e);
		setStatus(`Analysis error: ${e && e.message ? e.message : e}`);
		videoOverlay.classList.remove('active');
		stop();
	}
}

async function start(){
	if (running) return;
	try{
		openSettingsBtn.style.display = "none";
		if (openFullPageBtn) openFullPageBtn.style.display = "none";
		console.log('[popup] Starting camera request');
		setStatus("Requesting camera...");
		startBtn.disabled = true;
		stopBtn.disabled = true;
		stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
		videoEl.srcObject = stream;
		await videoEl.play();
		console.log('[popup] Camera stream acquired. video size', videoEl.videoWidth, videoEl.videoHeight);
		await ensureDetector();
		if (!session) await initSession();
		
		running = true;
		startBtn.disabled = true;
		stopBtn.disabled = false;
		
		// Analyze single frame instead of continuous loop
		await analyzeSingleFrame();
	}catch(e){
		console.error('[popup] start() error', e);
		running = false;
		startBtn.disabled = false;
		stopBtn.disabled = true;
		if (e && (e.name === "NotAllowedError" || e.name === "PermissionDismissedError")){
			setStatus("Camera blocked or dismissed. Use full-page to trigger prompt or open settings.");
			openSettingsBtn.style.display = "block";
			if (openFullPageBtn) openFullPageBtn.style.display = "block";
		} else if (e && e.name === "NotFoundError"){
			setStatus("No camera found. Check your device.");
		} else {
			setStatus(`Camera error: ${e.message}`);
		}
	}
}

function stop(){
	if (!stream && !running) return;
	running = false;
	startBtn.disabled = false;
	stopBtn.disabled = true;
	setStatus("Ready to analyze");
	videoOverlay.classList.remove('active');
	if (stream){
		console.log('[popup] Stopping camera');
		stream.getTracks().forEach(t => t.stop());
		stream = null;
		videoEl.srcObject = null;
	}
}

startBtn.addEventListener("click", start);
stopBtn.addEventListener("click", stop);
openSettingsBtn.addEventListener("click", async () => {
	try {
		await chrome.tabs.create({ url: "chrome://settings/content/camera" });
		setStatus("Open settings tab and allow camera, then click Start again.");
	} catch (e){
		setStatus("Open chrome://settings/content/camera, allow camera for Chrome.");
	}
});
if (openFullPageBtn){
	openFullPageBtn.addEventListener("click", async () => {
		try{
			const url = chrome.runtime.getURL("camera.html");
			await chrome.tabs.create({ url });
			setStatus("Full-page camera opened in a new tab. Use it to Allow camera.");
		}catch(e){
			setStatus("Could not open full-page. Use camera settings instead.");
		}
	});
}

(async function init(){
	const { isMinor } = await chrome.storage.local.get("isMinor");
	lastMinorState = typeof isMinor === "boolean" ? isMinor : null;
	console.log('[popup] Initial state isMinor =', lastMinorState);
})();
