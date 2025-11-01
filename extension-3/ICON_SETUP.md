# Icon Setup Instructions

## Current Setup
The extension is configured to use `extension_img.jpg` as the icon.

## To Fix the Icon Issue:

### Option 1: Use the Icon Converter (Recommended)
1. Open `icon_converter.html` in your browser
2. Select your `extension_img.jpg` file
3. Click "Convert to PNG Icons"
4. The tool will download icon files:
   - `icon.png` (128x128 - for toolbar)
   - `icon16.png`, `icon32.png`, `icon48.png`, `icon128.png`
5. Move all downloaded PNG files to your extension folder
6. The manifest will need to be updated (see below)

### Option 2: Manual Conversion
1. Convert `extension_img.jpg` to PNG format
2. Resize to square dimensions (recommended: 128x128 pixels)
3. Rename to `icon.png`
4. Ensure it's in the root of the extension folder

### Option 3: Fix Current JPG
If you want to keep using JPG:
1. Ensure `extension_img.jpg` is a square image (same width and height)
2. Recommended size: at least 128x128 pixels
3. Reload the extension

## After Setup:
1. Reload the extension in `chrome://extensions`
2. The icon should appear correctly

## Troubleshooting:
- If you still see the gray "F" icon, the image file may be corrupted or in an unsupported format
- Chrome prefers PNG format for icons (better compression and transparency support)
- Ensure the image file is in the root directory of your extension

