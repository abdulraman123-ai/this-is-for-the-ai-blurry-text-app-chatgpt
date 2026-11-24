"use client";

import { ChangeEvent, ClipboardEvent, DragEvent, useCallback, useEffect, useRef, useState } from "react";
import {
  ArrowDownToLine, Check, Clipboard, FileImage, FileText, ImagePlus, LoaderCircle,
  LockKeyhole, ScanText, Sparkles, UploadCloud, WandSparkles, X
} from "lucide-react";
import {
  createPaddleOcr,
  makeHighContrastVariant,
  orderOcrLines,
  prepareOcrCanvas,
  scoreOcrResult,
  type PaddleLanguage,
  type PaddleOcrEngine,
  type PaddleOcrLine
} from "@/lib/paddle-ocr";

type PageItem = {
  id: string;
  name: string;
  beforeUrl: string;
  sourceCanvas: HTMLCanvasElement;
  enhancedUrl: string;
  enhancedCanvas: HTMLCanvasElement;
  embeddedText: string;
};

type OCRResult = {
  text: string;
  confidence: number | null;
  pages: number;
  lines: number;
  lowConfidenceLines: number;
  pdfTextPages: number;
  imageOcrPages: number;
  engine: string;
};

type PdfTextItem = {
  str?: string;
  transform?: number[];
  width?: number;
  height?: number;
};

const MAX_IMAGE_DIMENSION = 2600;
const MAX_PDF_PAGES = 8;
const MAX_FILE_SIZE = 25 * 1024 * 1024;
const NAFNET_URL =
  "https://huggingface.co/opencv/opencv_zoo/resolve/main/models/deblurring_nafnet/deblurring_nafnet_2025may.onnx";

const OCR_LANGUAGES: { value: PaddleLanguage; label: string; tess: string }[] = [
  { value: "en", label: "English", tess: "eng" },
  { value: "tr", label: "Turkish", tess: "tur" },
  { value: "fr", label: "French", tess: "fra" },
  { value: "de", label: "German", tess: "deu" },
  { value: "es", label: "Spanish", tess: "spa" },
  { value: "pt", label: "Portuguese", tess: "por" },
  { value: "it", label: "Italian", tess: "ita" },
  { value: "nl", label: "Dutch", tess: "nld" },
  { value: "pl", label: "Polish", tess: "pol" },
];

function makeCanvas(width: number, height: number) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

function dataUrl(canvas: HTMLCanvasElement) {
  return canvas.toDataURL("image/png");
}

function loadImageCanvas(file: File) {
  return new Promise<HTMLCanvasElement>((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const image = new Image();
    image.onload = () => {
      const scale = Math.min(1, MAX_IMAGE_DIMENSION / Math.max(image.naturalWidth, image.naturalHeight));
      const canvas = makeCanvas(
        Math.max(1, Math.round(image.naturalWidth * scale)),
        Math.max(1, Math.round(image.naturalHeight * scale))
      );
      const context = canvas.getContext("2d");
      if (!context) {
        URL.revokeObjectURL(url);
        reject(new Error("Your browser could not prepare this image."));
        return;
      }
      context.imageSmoothingEnabled = true;
      context.imageSmoothingQuality = "high";
      context.drawImage(image, 0, 0, canvas.width, canvas.height);
      URL.revokeObjectURL(url);
      resolve(canvas);
    };
    image.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("This image could not be opened. Try JPG, PNG, or WEBP."));
    };
    image.src = url;
  });
}

function extractPdfText(items: unknown[]) {
  const textItems = items.filter((item): item is PdfTextItem =>
    typeof item === "object" &&
    item !== null &&
    "str" in item &&
    typeof (item as PdfTextItem).str === "string" &&
    Boolean((item as PdfTextItem).str?.trim())
  );

  const rows: { y: number; fontSize: number; items: PdfTextItem[] }[] = [];
  for (const item of textItems) {
    const transform = item.transform ?? [];
    const x = Number(transform[4] ?? 0);
    const y = Number(transform[5] ?? 0);
    const fontSize = Math.max(6, Math.abs(Number(transform[3] ?? item.height ?? 10)));
    let row = rows.find((candidate) => Math.abs(candidate.y - y) <= Math.max(3, fontSize * 0.45));
    if (!row) {
      row = { y, fontSize, items: [] };
      rows.push(row);
    }
    row.items.push({ ...item, transform: [x, y, fontSize] });
  }

  rows.sort((a, b) => b.y - a.y);
  return rows
    .map((row) => {
      const sorted = row.items.sort(
        (a, b) => Number(a.transform?.[0] ?? 0) - Number(b.transform?.[0] ?? 0)
      );
      let line = "";
      let previousEnd = Number.NEGATIVE_INFINITY;
      for (const item of sorted) {
        const x = Number(item.transform?.[0] ?? 0);
        const width = Math.max(0, Number(item.width ?? 0));
        const nextText = item.str ?? "";
        if (
          line &&
          x - previousEnd > Math.max(1.5, row.fontSize * 0.2) &&
          !line.endsWith(" ") &&
          !nextText.startsWith(" ")
        ) {
          line += " ";
        }
        line += nextText;
        previousEnd = x + width;
      }
      return line.trim();
    })
    .filter(Boolean)
    .join("\n")
    .replace(/[ \t]+\n/g, "\n")
    .trim();
}

async function loadPdfPages(file: File) {
  const pdfjs = await import("pdfjs-dist/legacy/build/pdf.mjs");
  pdfjs.GlobalWorkerOptions.workerSrc =
    "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/" + pdfjs.version + "/pdf.worker.min.mjs";
  const bytes = new Uint8Array(await file.arrayBuffer());
  const loadingTask = pdfjs.getDocument({ data: bytes });
  const pdf = await loadingTask.promise;
  const total = Math.min(pdf.numPages, MAX_PDF_PAGES);
  const pages: { canvas: HTMLCanvasElement; embeddedText: string }[] = [];

  try {
    for (let pageNumber = 1; pageNumber <= total; pageNumber += 1) {
      const page = await pdf.getPage(pageNumber);
      const [textContent] = await Promise.all([page.getTextContent()]);
      const embeddedText = extractPdfText(textContent.items as unknown[]);
      const rawViewport = page.getViewport({ scale: 2 });
      const scale = Math.min(1, MAX_IMAGE_DIMENSION / Math.max(rawViewport.width, rawViewport.height));
      const viewport = page.getViewport({ scale: 2 * scale });
      const canvas = makeCanvas(Math.ceil(viewport.width), Math.ceil(viewport.height));
      const context = canvas.getContext("2d", { willReadFrequently: true });
      if (!context) throw new Error("Your browser could not prepare this PDF page.");
      await page.render({ canvas, canvasContext: context, viewport }).promise;
      pages.push({ canvas, embeddedText });
    }
  } finally {
    await loadingTask.destroy();
  }

  return { pages, totalPages: pdf.numPages, processedPages: total };
}

function enhanceCanvas(source: HTMLCanvasElement, strength: number) {
  const canvas = makeCanvas(source.width, source.height);
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("Could not enhance this image in your browser.");
  context.drawImage(source, 0, 0);
  const image = context.getImageData(0, 0, canvas.width, canvas.height);
  const data = image.data;
  const contrast = 1 + strength * 0.62;
  const brightness = 1 + strength * 0.035;
  for (let i = 0; i < data.length; i += 4) {
    const gray = data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114;
    const value = Math.max(0, Math.min(255, (gray - 128) * contrast + 128));
    data[i] = value * brightness;
    data[i + 1] = value * brightness;
    data[i + 2] = value * brightness;
  }
  context.putImageData(image, 0, 0);

  const filtered = context.getImageData(0, 0, canvas.width, canvas.height);
  const output = context.createImageData(canvas.width, canvas.height);
  output.data.set(filtered.data);
  const input = filtered.data;
  const out = output.data;
  const amount = 0.22 + strength * 0.36;
  for (let y = 1; y < canvas.height - 1; y += 1) {
    for (let x = 1; x < canvas.width - 1; x += 1) {
      const index = (y * canvas.width + x) * 4;
      for (let channel = 0; channel < 3; channel += 1) {
        const center = input[index + channel] * (1 + 4 * amount);
        const neighbors =
          input[index - 4 + channel] + input[index + 4 + channel] +
          input[index - canvas.width * 4 + channel] + input[index + canvas.width * 4 + channel];
        out[index + channel] = Math.max(0, Math.min(255, center - neighbors * amount));
      }
      out[index + 3] = 255;
    }
  }
  context.putImageData(output, 0, 0);
  return canvas;
}

async function runNafNet(source: HTMLCanvasElement, onProgress: (value: number) => void) {
  onProgress(5);
  const ort = await import("onnxruntime-web");
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths =
    "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/";
  const session = await ort.InferenceSession.create(NAFNET_URL, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all"
  });

  try {
    onProgress(35);
    const maxSize = 512;
    const scale = Math.min(1, maxSize / Math.max(source.width, source.height));
    const width = Math.max(16, Math.floor(source.width * scale / 8) * 8);
    const height = Math.max(16, Math.floor(source.height * scale / 8) * 8);
    const inputCanvas = makeCanvas(width, height);
    const inputContext = inputCanvas.getContext("2d", { willReadFrequently: true });
    if (!inputContext) throw new Error("Could not prepare the AI model input.");
    inputContext.drawImage(source, 0, 0, width, height);
    const pixels = inputContext.getImageData(0, 0, width, height).data;
    const planar = new Float32Array(3 * width * height);
    const plane = width * height;
    for (let i = 0; i < plane; i += 1) {
      planar[i] = pixels[i * 4] / 255;
      planar[plane + i] = pixels[i * 4 + 1] / 255;
      planar[2 * plane + i] = pixels[i * 4 + 2] / 255;
    }
    onProgress(55);
    const tensor = new ort.Tensor("float32", planar, [1, 3, height, width]);
    const outputMap = await session.run({ [session.inputNames[0]]: tensor });
    const result = outputMap[session.outputNames[0]];
    const outputData = result.data as Float32Array;
    const outputCanvas = makeCanvas(width, height);
    const outputContext = outputCanvas.getContext("2d");
    if (!outputContext) throw new Error("Could not render the AI result.");
    const outputImage = outputContext.createImageData(width, height);
    for (let i = 0; i < plane; i += 1) {
      outputImage.data[i * 4] = Math.max(0, Math.min(255, outputData[i] * 255));
      outputImage.data[i * 4 + 1] = Math.max(0, Math.min(255, outputData[plane + i] * 255));
      outputImage.data[i * 4 + 2] = Math.max(0, Math.min(255, outputData[2 * plane + i] * 255));
      outputImage.data[i * 4 + 3] = 255;
    }
    outputContext.putImageData(outputImage, 0, 0);
    const finalCanvas = makeCanvas(source.width, source.height);
    const finalContext = finalCanvas.getContext("2d");
    if (!finalContext) throw new Error("Could not resize the AI result.");
    finalContext.drawImage(outputCanvas, 0, 0, source.width, source.height);
    onProgress(100);
    return finalCanvas;
  } finally {
    await session.release();
  }
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function createSampleCanvas() {
  const canvas = makeCanvas(1600, 1000);
  const context = canvas.getContext("2d");
  if (!context) throw new Error("Could not create the sample page.");
  context.fillStyle = "#ffffff";
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = "#111827";
  context.font = "700 48px Arial";
  context.fillText("CLEAR DOCUMENT SAMPLE", 105, 130);
  context.font = "28px Arial";
  context.fillStyle = "#263244";
  context.fillText("Invoice reference: CT-2026-0148", 105, 225);
  context.fillText("Customer: Ayşe Demir", 105, 285);
  context.fillText("Date: 24 November 2026", 105, 345);
  context.fillText("Description", 105, 465);
  context.fillText("Document recovery and text extraction", 105, 525);
  context.fillText("Quantity: 2", 105, 590);
  context.fillText("Total: 1,250.00 TRY", 105, 655);
  context.font = "italic 22px Arial";
  context.fillStyle = "#5b6472";
  context.fillText("Sample text for checking OCR quality. No file is uploaded.", 105, 820);
  context.strokeStyle = "#d5dce6";
  context.lineWidth = 2;
  context.beginPath();
  context.moveTo(105, 170);
  context.lineTo(1495, 170);
  context.moveTo(105, 430);
  context.lineTo(1495, 430);
  context.stroke();
  return canvas;
}

function buildPage(fileName: string, canvas: HTMLCanvasElement, embeddedText = ""): PageItem {
  return {
    id: String(Date.now()) + "-" + Math.random().toString(36).slice(2),
    name: fileName,
    beforeUrl: dataUrl(canvas),
    sourceCanvas: canvas,
    enhancedUrl: dataUrl(canvas),
    enhancedCanvas: canvas,
    embeddedText,
  };
}

function average(values: number[]) {
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
}

export function ClearTextWorkspace() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const paddleRef = useRef<{ language: PaddleLanguage; engine: PaddleOcrEngine } | null>(null);
  const [pages, setPages] = useState<PageItem[]>([]);
  const [activePage, setActivePage] = useState(0);
  const [dragging, setDragging] = useState(false);
  const [strength, setStrength] = useState(0.55);
  const [ocrLanguage, setOcrLanguage] = useState<PaddleLanguage>("en");
  const [sensitivity, setSensitivity] = useState(0.45);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState("Your documents stay in this browser. Text-based PDFs can be extracted without OCR.");
  const [statusType, setStatusType] = useState<"normal" | "success" | "error">("normal");
  const [ocr, setOcr] = useState<OCRResult | null>(null);
  const [ocrText, setOcrText] = useState("");
  const [copied, setCopied] = useState(false);
  const [enhanceMode, setEnhanceMode] = useState<"original" | "quick" | "ai">("original");

  const active = pages[activePage];

  const announce = useCallback((message: string, kind: "normal" | "success" | "error" = "normal") => {
    setStatus(message);
    setStatusType(kind);
  }, []);

  const addFile = useCallback(async (file?: File) => {
    if (!file) return;
    const isPdf = file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf");
    const isImage = file.type.startsWith("image/");
    if (!isPdf && !isImage) {
      announce("Choose a PDF, JPG, PNG, or WEBP file.", "error");
      return;
    }
    if (file.size > MAX_FILE_SIZE) {
      announce("This file is over 25 MB. Try a smaller image or PDF.", "error");
      return;
    }

    setBusy(true);
    setProgress(0);
    setOcr(null);
    setOcrText("");
    setPages([]);
    setActivePage(0);
    setEnhanceMode("original");
    try {
      if (isPdf) {
        announce("Reading PDF pages and checking for selectable text...");
        const loaded = await loadPdfPages(file);
        const nextPages = loaded.pages.map((page, index) =>
          buildPage(file.name + " · page " + (index + 1), page.canvas, page.embeddedText)
        );
        setPages(nextPages);
        const pagesWithText = nextPages.filter((page) => page.embeddedText.trim().length > 12).length;
        if (loaded.totalPages > loaded.processedPages) {
          announce(
            "Loaded the first " + loaded.processedPages + " pages of " + loaded.totalPages +
            ". This free browser workspace processes up to " + MAX_PDF_PAGES + " pages per upload."
          );
        } else if (pagesWithText === nextPages.length) {
          announce("Ready. This PDF contains selectable text, so ClearText can extract it directly without OCR.", "success");
        } else {
          announce("PDF loaded. Scanned pages will be read with PaddleOCR; selectable text will be extracted directly.", "success");
        }
      } else {
        const canvas = await loadImageCanvas(file);
        setPages([buildPage(file.name, canvas)]);
        announce("Image ready. Choose an enhancement or extract text with PaddleOCR.", "success");
      }
    } catch (error) {
      announce(error instanceof Error ? error.message : "That file could not be opened.", "error");
    } finally {
      setBusy(false);
      setProgress(0);
    }
  }, [announce]);

  const onInputChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    void addFile(file);
    event.target.value = "";
  };

  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(false);
    void addFile(event.dataTransfer.files?.[0]);
  };

  const useSample = () => {
    const canvas = createSampleCanvas();
    setPages([buildPage("ClearText sample page", canvas)]);
    setActivePage(0);
    setOcr(null);
    setOcrText("");
    setEnhanceMode("original");
    announce("Sample ready. Run OCR to check text recognition, Turkish characters and layout.", "success");
  };

  const applyQuickEnhance = useCallback(() => {
    if (!active) return;
    try {
      const enhancedCanvas = enhanceCanvas(active.sourceCanvas, strength);
      const url = dataUrl(enhancedCanvas);
      setPages((current) => current.map((page, index) =>
        index === activePage ? { ...page, enhancedUrl: url, enhancedCanvas } : page
      ));
      setEnhanceMode("quick");
      announce("Quick clean-up applied. Compare it with the original and run OCR.", "success");
    } catch (error) {
      announce(error instanceof Error ? error.message : "Enhancement failed.", "error");
    }
  }, [active, activePage, announce, strength]);

  const applyAiDeblur = useCallback(async () => {
    if (!active) return;
    setBusy(true);
    setProgress(3);
    announce("Loading the optional on-device deblurring model. The first download is large and can take a while.");
    try {
      const enhancedCanvas = await runNafNet(active.sourceCanvas, setProgress);
      const url = dataUrl(enhancedCanvas);
      setPages((current) => current.map((page, index) =>
        index === activePage ? { ...page, enhancedUrl: url, enhancedCanvas } : page
      ));
      setEnhanceMode("ai");
      announce("AI deblur finished on this device. Check the output carefully; restoration can alter fine details.", "success");
    } catch (error) {
      announce(
        (error instanceof Error ? error.message : "The AI model could not run.") +
        " Try Quick clean-up instead, or use a smaller image / newer browser.",
        "error"
      );
    } finally {
      setBusy(false);
      setProgress(0);
    }
  }, [active, activePage, announce]);

  const getPaddleEngine = useCallback(async () => {
    if (paddleRef.current?.language === ocrLanguage) return paddleRef.current.engine;
    if (paddleRef.current) {
      await paddleRef.current.engine.dispose();
      paddleRef.current = null;
    }
    const engine = await createPaddleOcr(ocrLanguage);
    paddleRef.current = { language: ocrLanguage, engine };
    return engine;
  }, [ocrLanguage]);

  const runOcr = useCallback(async () => {
    if (!pages.length) {
      announce("Upload an image or PDF first.", "error");
      return;
    }

    setBusy(true);
    setProgress(0);
    setOcr(null);
    setOcrText("");

    try {
      const pageTexts = new Map<number, string>();
      const scannedPageIndexes: number[] = [];
      let pdfTextPages = 0;

      pages.forEach((page, index) => {
        const text = page.embeddedText.trim();
        if (text.length > 12) {
          pageTexts.set(index, text);
          pdfTextPages += 1;
        } else {
          scannedPageIndexes.push(index);
        }
      });

      const confidenceScores: number[] = [];
      let lineCount = 0;
      let lowConfidenceLines = 0;
      let engineLabel = pdfTextPages ? "PDF text layer" : "PaddleOCR PP-OCRv6";

      if (scannedPageIndexes.length) {
        let paddleEngine: PaddleOcrEngine | null = null;
        let paddleFailed = false;
        try {
          announce(
            "Preparing local OCR. First use downloads the PaddleOCR model (about 31 MB); it is cached by your browser."
          );
          paddleEngine = await getPaddleEngine();
        } catch {
          paddleFailed = true;
          engineLabel = pdfTextPages ? "PDF text + Tesseract backup" : "Tesseract backup";
        }

        if (paddleEngine && !paddleFailed) {
          try {
            for (let position = 0; position < scannedPageIndexes.length; position += 1) {
              const pageIndex = scannedPageIndexes[position];
              announce(
                "Reading page " + (pageIndex + 1) + " of " + pages.length + " with PaddleOCR..."
              );
              setProgress(Math.round((position / scannedPageIndexes.length) * 100));

              const source = pages[pageIndex].enhancedCanvas;
              const prepared = prepareOcrCanvas(source);
              const firstRun = await paddleEngine.predict(prepared, {
                textDetThresh: 0.3,
                textDetBoxThresh: sensitivity,
                textDetUnclipRatio: 1.8,
                textRecScoreThresh: 0.05,
                textDetMaxSideLimit: 4000,
              });
              let bestItems = orderOcrLines(firstRun[0]?.items ?? []);
              let bestScore = scoreOcrResult(bestItems);

              if (bestScore.confidence < 0.76 || bestScore.textLength < 20) {
                const alternate = makeHighContrastVariant(prepared);
                const secondRun = await paddleEngine.predict(alternate, {
                  textDetThresh: 0.3,
                  textDetBoxThresh: sensitivity,
                  textDetUnclipRatio: 1.8,
                  textRecScoreThresh: 0.05,
                  textDetMaxSideLimit: 4000,
                });
                const alternateItems = orderOcrLines(secondRun[0]?.items ?? []);
                const alternateScore = scoreOcrResult(alternateItems);
                if (alternateScore.quality > bestScore.quality) {
                  bestItems = alternateItems;
                  bestScore = alternateScore;
                }
              }

              const text = bestItems.map((item) => item.text.trim()).filter(Boolean).join("\n");
              pageTexts.set(pageIndex, text);
              lineCount += bestItems.length;
              lowConfidenceLines += bestItems.filter((item) => item.score < 0.5).length;
              bestItems.forEach((item) => {
                if (Number.isFinite(item.score) && item.score >= 0 && item.score <= 1) {
                  confidenceScores.push(item.score);
                }
              });
              setProgress(Math.round(((position + 1) / scannedPageIndexes.length) * 100));
            }
          } catch {
            paddleFailed = true;
            for (const pageIndex of scannedPageIndexes) pageTexts.delete(pageIndex);
            lineCount = 0;
            lowConfidenceLines = 0;
            confidenceScores.length = 0;
            engineLabel = pdfTextPages ? "PDF text + Tesseract backup" : "Tesseract backup";
          }
        }

        if (paddleFailed) {
          announce("PaddleOCR could not load from its model host. Switching to the local Tesseract backup...");
          const { createWorker } = await import("tesseract.js");
          const languageCode = OCR_LANGUAGES.find((language) => language.value === ocrLanguage)?.tess ?? "eng";
          const worker = await createWorker(languageCode, 1, {
            logger: (message) => {
              if (typeof message.progress === "number") setProgress(Math.round(message.progress * 100));
              if (message.status) announce("Backup OCR: " + message.status);
            }
          });
          try {
            for (let position = 0; position < scannedPageIndexes.length; position += 1) {
              const pageIndex = scannedPageIndexes[position];
              const prepared = prepareOcrCanvas(pages[pageIndex].enhancedCanvas);
              const result = await worker.recognize(prepared);
              pageTexts.set(pageIndex, result.data.text.trim());
              lineCount += result.data.text.split(/\\r?\\n/).filter((line) => line.trim().length > 0).length;
              if (typeof result.data.confidence === "number") {
                confidenceScores.push(result.data.confidence / 100);
              }
              setProgress(Math.round(((position + 1) / scannedPageIndexes.length) * 100));
            }
          } finally {
            await worker.terminate();
          }
        }
      }

      const combined = pages
        .map((page, index) => {
          const pageText = (pageTexts.get(index) ?? "").trim();
          if (!pageText) return "";
          return pages.length > 1 ? "PAGE " + (index + 1) + "\n\n" + pageText : pageText;
        })
        .filter(Boolean)
        .join("\n\n────────────────────────\n\n");

      const confidence = average(confidenceScores);
      const result: OCRResult = {
        text: combined,
        confidence: confidence === null ? null : Math.round(confidence * 100),
        pages: pages.length,
        lines: lineCount,
        lowConfidenceLines,
        pdfTextPages,
        imageOcrPages: scannedPageIndexes.length,
        engine: engineLabel,
      };

      setOcrText(combined);
      setOcr(result);

      if (!combined.trim()) {
        announce("No readable text was found. Try a sharper image, better lighting, or a different language.", "error");
      } else if (lowConfidenceLines > 0) {
        announce(
          "Text extracted. " + lowConfidenceLines + " lines have lower confidence; check them against the original.",
          "success"
        );
      } else {
        announce(
          "Text extraction complete. " +
          (pdfTextPages ? pdfTextPages + " PDF page(s) used selectable text. " : "") +
          "Review names, numbers and dates before using the result.",
          "success"
        );
      }
    } catch (error) {
      announce(
        error instanceof Error ? error.message : "OCR did not finish. Check your connection and try again.",
        "error"
      );
    } finally {
      setBusy(false);
      setProgress(0);
    }
  }, [announce, getPaddleEngine, ocrLanguage, pages, sensitivity]);

  const copyText = async () => {
    try {
      await navigator.clipboard.writeText(ocrText);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
      announce("Extracted text copied to clipboard.", "success");
    } catch {
      announce("Clipboard access was blocked. Select the text and copy it manually.", "error");
    }
  };

  const downloadText = () => {
    if (!ocrText.trim()) return;
    downloadBlob(new Blob([ocrText], { type: "text/plain;charset=utf-8" }), "cleartext-extracted-text.txt");
  };

  const downloadJson = () => {
    if (!ocr) return;
    downloadBlob(
      new Blob([JSON.stringify({ ...ocr, text: ocrText }, null, 2)], { type: "application/json;charset=utf-8" }),
      "cleartext-ocr-result.json"
    );
  };

  const downloadImage = () => {
    if (!active) return;
    active.enhancedCanvas.toBlob((blob) => {
      if (blob) downloadBlob(blob, "cleartext-enhanced-page.png");
    }, "image/png");
  };

  const reset = () => {
    setPages([]);
    setActivePage(0);
    setOcr(null);
    setOcrText("");
    setEnhanceMode("original");
    announce("Workspace cleared. Upload another document whenever you're ready.");
  };

  useEffect(() => {
    return () => {
      const current = paddleRef.current;
      paddleRef.current = null;
      if (current) void current.engine.dispose();
    };
  }, []);

  const handlePaste = (event: ClipboardEvent<HTMLElement>) => {
    const file = Array.from(event.clipboardData.files).find((item) => item.type.startsWith("image/"));
    if (file) void addFile(file);
  };

  return (
    <main onPaste={handlePaste}>
      <header className="topbar">
        <div className="shell topbar-inner">
          <a href="/" className="brand" aria-label="ClearText AI home">
            <span className="brand-mark"><ScanText size={19} /></span>
            <span>cleartext<span style={{ color: "#85b5ff" }}>.</span><small>AI DOCUMENT RECOVERY</small></span>
          </a>
          <div className="nav-note"><LockKeyhole size={13} /> Private by design · Runs in your browser</div>
        </div>
      </header>

      <section className="shell hero">
        <div className="eyebrow"><Sparkles size={13} /> DOCUMENT CLARITY TOOLKIT</div>
        <h1>Turn scans into text <span>you can actually use.</span></h1>
        <p>
          Improve difficult document photos, read scanned pages, and extract editable text.
          Selectable PDF text is copied directly; scanned pages use AI OCR on your device.
        </p>
        <div className="hero-points">
          <span><LockKeyhole size={14} /> Files stay on your device</span>
          <span><FileText size={14} /> Images and multi-page PDFs</span>
          <span><ScanText size={14} /> AI text detection and export</span>
        </div>
      </section>

      <section className="shell app-section">
        <div className="workspace">
          <section className="panel">
            <div className="panel-head">
              <div><span className="step">01 / INPUT</span><h2>Upload your document</h2></div>
              <FileImage size={18} color="#80b3ff" />
            </div>
            <div className="panel-body">
              <div
                className={"dropzone" + (dragging ? " dragging" : "")}
                onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
                onDragLeave={() => setDragging(false)}
                onDrop={onDrop}
                aria-label="Document upload drop area"
              >
                <div className="upload-icon"><UploadCloud size={25} /></div>
                <strong>Drop a page or document here</strong>
                <p>PDF, JPG, PNG or WEBP · Up to 25 MB<br />PDFs: first 8 pages per upload</p>
                <div className="action-row">
                  <button className="primary-btn" onClick={() => fileInputRef.current?.click()} disabled={busy}>
                    <ImagePlus size={16} /> Choose file
                  </button>
                  <button className="ghost-btn" onClick={useSample} disabled={busy}>Try a sample</button>
                </div>
                <input
                  ref={fileInputRef}
                  className="hidden-input"
                  type="file"
                  accept="image/png,image/jpeg,image/webp,application/pdf"
                  onChange={onInputChange}
                  aria-label="Choose a PDF or image"
                />
              </div>

              {pages.length > 0 && (
                <div style={{ marginTop: 16 }}>
                  <div className="field-label">
                    <span>Loaded document</span>
                    <button className="ghost-btn small-btn" onClick={reset} disabled={busy}><X size={12} /> Clear</button>
                  </div>
                  <div className="control-box">
                    <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
                      <FileImage size={18} color="#80b3ff" />
                      <div style={{ minWidth: 0 }}>
                        <div style={{ fontSize: 12, fontWeight: 650, overflowWrap: "anywhere" }}>{pages[0].name}</div>
                        <div style={{ fontSize: 10, color: "var(--muted)", marginTop: 3 }}>
                          {pages.length} page{pages.length === 1 ? "" : "s"} loaded
                          {pages.some((page) => page.embeddedText.trim().length > 12) ? " · PDF text layer detected" : ""}
                        </div>
                      </div>
                    </div>
                    {pages.length > 1 && (
                      <div className="page-strip">
                        {pages.map((page, index) => (
                          <button
                            key={page.id}
                            className={"page-pill" + (index === activePage ? " active" : "")}
                            onClick={() => setActivePage(index)}
                            aria-pressed={index === activePage}
                          >
                            Page {index + 1}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              )}

              <div style={{ marginTop: 17 }}>
                <div className="field-label"><span>Quick clean-up strength</span><b>{Math.round(strength * 100)}%</b></div>
                <input
                  className="range"
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={strength}
                  onChange={(event) => setStrength(Number(event.target.value))}
                  disabled={!active || busy}
                  aria-label="Quick clean-up strength"
                />
                <div className="action-row" style={{ marginTop: 12 }}>
                  <button className="secondary-btn" onClick={applyQuickEnhance} disabled={!active || busy}>
                    <WandSparkles size={15} /> Quick clean-up
                  </button>
                  <button className="ghost-btn" onClick={applyAiDeblur} disabled={!active || busy}>
                    <Sparkles size={15} /> AI deblur beta
                  </button>
                </div>
                <p className="helper-copy">
                  Quick clean-up adjusts contrast and sharpening. AI deblur is experimental and downloads a separate model on first use.
                </p>
              </div>
            </div>
          </section>

          <section className="panel">
            <div className="panel-head">
              <div><span className="step">02 / ENHANCE</span><h2>Preview & compare</h2></div>
                  <Sparkles size={18} color="#80b3ff" />
            </div>
            <div className="panel-body">
              <div className="field-label">
                <span>{enhanceMode === "original" ? "Original page" : enhanceMode === "ai" ? "AI-restored preview" : "Quick enhanced preview"}</span>
                <span className="muted-value">{active ? active.name : "No document"}</span>
              </div>
              <div className="preview-frame">
                {active ? (
                  <img
                    src={enhanceMode === "original" ? active.beforeUrl : active.enhancedUrl}
                    alt={"Preview of " + active.name}
                  />
                ) : (
                  <div className="preview-empty">Your selected document will appear here. Upload a page or try the sample to get started.</div>
                )}
              </div>
              <div className="action-row" style={{ marginTop: 12 }}>
                <button className="ghost-btn" onClick={() => setEnhanceMode("original")} disabled={!active || busy}>Show original</button>
                <button className="ghost-btn" onClick={() => setEnhanceMode("quick")} disabled={!active || busy}>Show enhanced</button>
                <button className="ghost-btn" onClick={downloadImage} disabled={!active || busy}>
                  <ArrowDownToLine size={14} /> Save PNG
                </button>
              </div>
              <div style={{ marginTop: 17 }} className="disclaimer">
                <b>Important:</b> Enhancement cannot recover detail that is absent from the pixels. Compare the result with the original, especially before relying on small numbers or names.
              </div>
            </div>
          </section>

          <section className="panel ocr-panel">
            <div className="panel-head">
              <div><span className="step">03 / OCR</span><h2>Extract editable text</h2></div>
              <ScanText size={18} color="#80b3ff" />
            </div>
            <div className="panel-body">
              <div className="ocr-settings">
                <label className="setting">
                  <span>Document language</span>
                  <select
                    value={ocrLanguage}
                    onChange={(event) => setOcrLanguage(event.target.value as PaddleLanguage)}
                    disabled={busy}
                    aria-label="OCR document language"
                  >
                    {OCR_LANGUAGES.map((language) => (
                      <option key={language.value} value={language.value}>{language.label}</option>
                    ))}
                  </select>
                </label>
                <label className="setting sensitivity-setting">
                  <span>Faint text sensitivity <b>{Math.round(sensitivity * 100)}%</b></span>
                  <input
                    className="range"
                    type="range"
                    min="0.35"
                    max="0.7"
                    step="0.05"
                    value={sensitivity}
                    onChange={(event) => setSensitivity(Number(event.target.value))}
                    disabled={busy}
                    aria-label="Faint text sensitivity"
                  />
                </label>
              </div>

              <div className="action-row">
                <button className="primary-btn" onClick={runOcr} disabled={!pages.length || busy}>
                  {busy ? <LoaderCircle size={15} className="animate-spin" /> : <ScanText size={15} />}
                  {busy ? "Reading document..." : "Extract text"}
                </button>
                <button className="ghost-btn" onClick={copyText} disabled={!ocrText.trim()}>
                  {copied ? <Check size={14} /> : <Clipboard size={14} />}{copied ? "Copied" : "Copy text"}
                </button>
                <button className="ghost-btn" onClick={downloadText} disabled={!ocrText.trim()}>
                  <ArrowDownToLine size={14} /> Download TXT
                </button>
                <button className="ghost-btn" onClick={downloadJson} disabled={!ocr}>
                  <ArrowDownToLine size={14} /> Download JSON
                </button>
              </div>

              <div className="stat-row">
                <span className="stat">Engine <b>{ocr?.engine ?? "PaddleOCR v6"}</b></span>
                <span className="stat">Pages <b>{ocr ? ocr.pages : pages.length || "—"}</b></span>
                <span className="stat">Lines <b>{ocr ? ocr.lines || "—" : "—"}</b></span>
                <span className="stat">Confidence <b>{ocr?.confidence !== null && ocr?.confidence !== undefined ? ocr.confidence + "%" : "—"}</b></span>
                {Boolean(ocr?.lowConfidenceLines) && <span className="stat warning-stat">Review <b>{ocr?.lowConfidenceLines} low-confidence lines</b></span>}
              </div>

              <textarea
                className="result-area"
                value={ocrText}
                onChange={(event) => setOcrText(event.target.value)}
                placeholder="Extracted text will appear here. You can edit it before copying or downloading."
                aria-label="Extracted text"
                spellCheck={false}
              />
              {busy && (
                <div className="progress-track" aria-label={"OCR progress " + progress + "%"}>
                  <div className="progress-bar" style={{ width: progress + "%" }} />
                </div>
              )}
              <div className={"status " + (statusType === "error" ? "error" : statusType === "success" ? "success" : "")} role="status" aria-live="polite">
                {status}
              </div>
              <p className="helper-copy">
                Scanned documents use PaddleOCR PP-OCRv6 locally in your browser. On the first run, the model downloads (about 31 MB) and is then reused by your browser cache. Text-based PDFs are extracted directly, without raster OCR. OCR confidence is only a guide: verify names, numbers, dates and technical terms.
              </p>
            </div>
          </section>
        </div>

        <div className="info-grid">
          <article className="info-card"><WandSparkles color="#83b6ff" size={19} /><b>Enhance before reading</b><p>Gentle clean-up can improve faint print. Experimental AI deblurring is optional and should always be checked against the source.</p></article>
          <article className="info-card"><ScanText color="#83b6ff" size={19} /><b>More accurate extraction</b><p>AI text detection reads individual text lines; selectable PDF text is extracted directly to avoid unnecessary OCR errors.</p></article>
          <article className="info-card"><LockKeyhole color="#83b6ff" size={19} /><b>Keep documents private</b><p>Your images and PDFs are processed in this tab. The OCR code, WASM runtime and model files are downloaded from public hosts; your documents are not uploaded by this app.</p></article>
        </div>

        <footer className="footer">
          <div style={{ marginBottom: 7, color: "#9baec7", fontWeight: 650 }}>cleartext.ai</div>
          Built for personal document recovery and accessibility. Do not use this tool to bypass access controls or attempt to reconstruct deliberately redacted information.
          <br />Your files are not uploaded by this app. Standard hosting and model/CDN request logs may still be processed by those providers.
        </footer>
      </section>
    </main>
  );
}
