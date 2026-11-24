# ClearText AI

A browser-first document clarity and OCR app. It uses the Axion Marketplace visual foundation but lives in a separate repository. The original Axion Marketplace repository is not modified.

## What it does

- Accepts JPG, PNG, WEBP and PDF documents up to 25 MB.
- Renders up to the first 8 PDF pages locally in the browser.
- Extracts selectable text directly from text-based PDFs instead of rasterizing it through OCR.
- Uses the official PaddleOCR.js SDK with PP-OCRv6 for scanned pages and images.
- Supports English, Turkish, French, German, Spanish, Portuguese, Italian, Dutch and Polish.
- Uses a second high-contrast pass when the first OCR result has low confidence or very little text.
- Falls back to Tesseract.js if the PaddleOCR model host cannot be reached.
- Offers quick contrast/sharpening filters and an experimental NAFNet deblurring model.
- Lets users edit, copy, and download TXT and JSON results, and save enhanced images as PNG.
- Includes a sample page so users can check the OCR workflow without uploading their own file.

## Privacy and costs

There is no application backend, account system, database, Gemini key, paid AI endpoint, or server-side document upload. Images and PDFs are processed in the user's browser. PaddleOCR.js uses ONNX Runtime Web. The PP-OCRv6 small model is about 31 MB and is fetched from Paddle model hosting; Tesseract fallback data and PDF.js worker code are also downloaded when needed. Public hosts may process ordinary request metadata.

This avoids developer-side AI inference charges. It does not make Vercel hosting, visitor bandwidth, or the user's device resources unlimited.

## OCR accuracy notes

PaddleOCR uses text detection followed by recognition, and returns per-line confidence scores. The app also checks for selectable text in PDFs and uses that exact text when available. For images and scanned PDF pages, the app upsamples small pages and can run a contrast-normalized second pass.

OCR cannot restore characters that are missing from the image. Severe motion blur, glare, low resolution, handwriting, curved pages and unusual fonts can remain difficult. Always verify important names, numbers, dates, amounts and technical terms against the original. The current browser SDK supports the Latin-script languages listed above; Arabic, Cyrillic, Thai, Devanagari and some other scripts are not included in this browser model.

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
- PaddleOCR.js 0.4.2 (Apache-2.0) + PP-OCRv6
- ONNX Runtime Web (MIT)
- Tesseract.js 7 (Apache-2.0) as OCR fallback
- PDF.js 6 (Apache-2.0) for PDF text extraction and page rendering
- NAFNet / OpenCV Zoo for experimental local deblurring

## Hosting

Import this repository into the separate Vercel project `cleartext-ai` with the Next.js preset. No environment variables are required. Keep it separate from all Axion Marketplace Vercel projects.
