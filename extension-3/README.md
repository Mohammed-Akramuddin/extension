# Face Age SafeSearch Toggle (Chrome Extension)

Analyzes a face via an ONNX model (`swin_face_classifier.onnx`) in the popup and toggles SafeSearch on Google/Bing using dynamic redirect rules.

## Requirements
- Chrome 116+ (FaceDetector API + MV3)
- `swin_face_classifier.onnx` in the extension root (already present)
- onnxruntime-web browser bundle placed as `lib/ort.min.js`

## Setup
1. Download onnxruntime-web browser bundle and place it at:
   `lib/ort.min.js`
2. In Chrome, open `chrome://extensions`, enable Developer mode, click "Load unpacked", and select this folder.

## Usage
- Click the extension icon to open the popup.
- Click "Start" to start the camera and analysis.
- When detected as Minor, SafeSearch is enabled (Google: `safe=active`, Bing: `adlt=strict`). When Major, rules are removed.

## Notes
- The model is expected to output 2 logits: index 0 = Minor, index 1 = Major. Adjust `popup.js` if your order differs.
- Uses FaceDetector API for face bounding box; ensure Chrome supports it. If not supported, consider replacing with another detector (e.g., MediaPipe) inside `popup.js`.
