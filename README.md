# ClearText AI

A browser-first document clarity and OCR app built from a clean port of the Axion Marketplace starter's visual direction. The Axion Marketplace repository itself is unchanged.

## What it does

- Accepts JPG, PNG, WEBP and PDF documents (up to 18 MB).
- Renders up to the first 8 PDF pages locally in the browser.
- Offers quick contrast and sharpening filters.
- Offers an experimental NAFNet deblurring model using ONNX Runtime Web and browser WASM. The model is fetched from the public OpenCV Zoo Hugging Face repository only after the user presses **AI deblur beta**. The model is roughly 92 MB, so this is optional and may be slow on phones.
- Uses Tesseract.js (English) for on-device OCR.
- Lets a user edit, copy, and download extracted text; enhanced images can be saved as PNG.

## Privacy and costs

The app has no application backend, account system, database, Google Gemini key, paid AI endpoint, or server-side image upload. Images/PDF pages and inference stay in the user's browser. Tesseract trained data, OCR worker assets, PDF.js worker assets and the optional ONNX model are downloaded from public CDNs/model hosting when required. Those providers may process ordinary request metadata. Browsers may cache these downloads.

This avoids AI inference charges to the developer's account. It does **not** mean hosting is unlimited or that Vercel's free plan has no usage limits; check the current plan limits before public launch.

## Important limitations

- Deblurring cannot recreate detail that is absent from the pixels. Severe blur, very low resolution, glare, motion blur, occlusion and redactions can remain unreadable.
- Generative-looking restoration can alter fine strokes. Always compare the enhanced image with the original and verify important words, numbers, dates and names.
- OCR confidence is an indicator, not a guarantee. The current OCR language is English.
- PDF uploads are limited to 8 pages per run to reduce browser memory use.
- The AI deblur model is experimental; use Quick clean-up if WebAssembly/model loading fails.
- This tool is for legitimate document recovery/accessibility, not to bypass access controls or reconstruct deliberately hidden information.

## Run locally

Requires Node.js 22 or newer.

```bash
npm install
npm run dev
```

Open http://localhost:3000.

```bash
npm run lint
npm run build
```

No environment variables are required.

## Technology

- Next.js / React
- Tesseract.js (Apache-2.0) for OCR
- ONNX Runtime Web (MIT) for optional local inference
- PDF.js (Apache-2.0) for rendering PDF pages
- NAFNet model via OpenCV Zoo / Hugging Face. Review and retain the model repository's license/attribution when redistributing or modifying model weights.

## Hosting

Import this repository into a separate Vercel project named `cleartext-ai` with the Next.js preset. No environment variables are required. Keep it separate from all Axion Marketplace Vercel projects.
