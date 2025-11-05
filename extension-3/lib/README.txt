Place browser bundles here:

1) onnxruntime-web (ort.min.js)
   Save as: C:\Users\akram\OneDrive\Desktop\extension-3\lib\ort.min.js
   Also place ORT wasm assets (from the SAME onnxruntime-web release):
   - ort-wasm.wasm
   - ort-wasm-simd.wasm
   - ort-wasm-threaded.wasm
   - ort-wasm-simd-threaded.wasm  <-- your runtime attempted to load this

2) TensorFlow.js core/browser bundle (tf.min.js)
   Save as: C:\Users\akram\OneDrive\Desktop\extension-3\lib\tf.min.js

3) BlazeFace model bundle (blazeface.min.js)
   Save as: C:\Users\akram\OneDrive\Desktop\extension-3\lib\blazeface.min.js

Tip: Download the ORT release zip for the browser and copy ALL wasm files into lib/ to avoid 404s.
MV3 forbids remote scripts; these must be local.
