"use client";

import { ChangeEvent, DragEvent, useCallback, useEffect, useRef, useState } from "react";
import {
  ArrowDownToLine, Check, ChevronRight, Clipboard, FileImage, FileText,
  ImagePlus, LoaderCircle, LockKeyhole, ScanText, Sparkles, UploadCloud, WandSparkles, X
} from "lucide-react";

type PageItem = {
  id: string;
  name: string;
  beforeUrl: string;
  sourceCanvas: HTMLCanvasElement;
  enhancedUrl: string;
  enhancedCanvas: HTMLCanvasElement;
};

type OCRResult = {
  text: string;
  confidence: number | null;
  pages: number;
};

const MAX_IMAGE_DIMENSION = 2200;
const MAX_PDF_PAGES = 8;
const NAFNET_URL =
  "https://huggingface.co/opencv/opencv_zoo/resolve/main/models/deblurring_nafnet/deblurring_nafnet_2025may.onnx";

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

async function loadPdfPages(file: File) {
  const pdfjs = await import("pdfjs-dist/legacy/build/pdf.mjs");
  pdfjs.GlobalWorkerOptions.workerSrc =
    "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/" + pdfjs.version + "/pdf.worker.min.mjs";
  const bytes = new Uint8Array(await file.arrayBuffer());
  const pdf = await pdfjs.getDocument({ data: bytes }).promise;
  const total = Math.min(pdf.numPages, MAX_PDF_PAGES);
  const canvases: HTMLCanvasElement[] = [];
  for (let pageNumber = 1; pageNumber <= total; pageNumber += 1) {
    const page = await pdf.getPage(pageNumber);
    const rawViewport = page.getViewport({ scale: 1.6 });
    const scale = Math.min(1, MAX_IMAGE_DIMENSION / Math.max(rawViewport.width, rawViewport.height));
    const viewport = page.getViewport({ scale: 1.6 * scale });
    const canvas = makeCanvas(Math.ceil(viewport.width), Math.ceil(viewport.height));
    const context = canvas.getContext("2d");
    if (!context) throw new Error("Your browser could not prepare this PDF page.");
    await page.render({ canvas, canvasContext: context, viewport }).promise;
    canvases.push(canvas);
  }
  return { canvases, totalPages: pdf.numPages, processedPages: total };
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
    const gray = (data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114);
    const value = Math.max(0, Math.min(255, (gray - 128) * contrast + 128));
    data[i] = value * brightness;
    data[i + 1] = value * brightness;
    data[i + 2] = value * brightness;
  }
  context.putImageData(image, 0, 0);

  const original = context.getImageData(0, 0, canvas.width, canvas.height);
  const output = context.createImageData(canvas.width, canvas.height);
  const input = original.data;
  const out = output.data;
  out.set(input);
  const amount = 0.35 + strength * 0.9;
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
  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];
  const tensor = new ort.Tensor("float32", planar, [1, 3, height, width]);
  const outputMap = await session.run({ [inputName]: tensor });
  const result = outputMap[outputName];
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
  await session.release();
  onProgress(100);
  return finalCanvas;
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

export function ClearTextWorkspace() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [pages, setPages] = useState<PageItem[]>([]);
  const [activePage, setActivePage] = useState(0);
  const [dragging, setDragging] = useState(false);
  const [strength, setStrength] = useState(0.55);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState("Your files stay in this browser. Nothing is uploaded to a ClearText server.");
  const [statusType, setStatusType] = useState<"normal" | "success" | "error">("normal");
  const [ocr, setOcr] = useState<OCRResult | null>(null);
  const [ocrText, setOcrText] = useState("");
  const [copied, setCopied] = useState(false);
  const [dragMode, setDragMode] = useState(false);
  const [enhanceMode, setEnhanceMode] = useState<"original" | "quick" | "ai">("original");
  const [aiLoaded, setAiLoaded] = useState(false);

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
    if (file.size > 18 * 1024 * 1024) {
      announce("This file is over 18 MB. Try a smaller image or PDF.", "error");
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
      let canvases: HTMLCanvasElement[] = [];
      let names: string[] = [];
      if (isPdf) {
        const loaded = await loadPdfPages(file);
        canvases = loaded.canvases;
        names = canvases.map((_, index) => file.name + " · page " + (index + 1));
        if (loaded.totalPages > loaded.processedPages) {
          announce("Loaded the first " + loaded.processedPages + " pages of " + loaded.totalPages + ". The free browser workspace limits each PDF to " + MAX_PDF_PAGES + " pages at a time.");
        }
      } else {
        canvases = [await loadImageCanvas(file)];
        names = [file.name];
      }
      const nextPages = canvases.map((canvas, index) => ({
        id: String(Date.now()) + "-" + index,
        name: names[index],
        beforeUrl: dataUrl(canvas),
        sourceCanvas: canvas,
        enhancedUrl: dataUrl(canvas),
        enhancedCanvas: canvas
      }));
      setPages(nextPages);
      setEnhanceMode("original");
      announce("Ready. Choose Quick clean-up or experimental AI deblur, then extract the text.", "success");
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

  const applyQuickEnhance = useCallback(() => {
    if (!active) return;
    try {
      const enhancedCanvas = enhanceCanvas(active.sourceCanvas, strength);
      const url = dataUrl(enhancedCanvas);
      setPages((current) => current.map((page, index) =>
        index === activePage ? { ...page, enhancedUrl: url, enhancedCanvas } : page
      ));
      setEnhanceMode("quick");
      announce("Quick enhancement applied. Compare it with the original and run OCR.", "success");
    } catch (error) {
      announce(error instanceof Error ? error.message : "Enhancement failed.", "error");
    }
  }, [active, activePage, announce, strength]);

  const applyAiDeblur = useCallback(async () => {
    if (!active) return;
    setBusy(true);
    setProgress(3);
    announce("Loading the on-device AI model. The first download is about 92 MB and can take a while.", "normal");
    try {
      const enhancedCanvas = await runNafNet(active.sourceCanvas, setProgress);
      const url = dataUrl(enhancedCanvas);
      setPages((current) => current.map((page, index) =>
        index === activePage ? { ...page, enhancedUrl: url, enhancedCanvas } : page
      ));
      setEnhanceMode("ai");
      setAiLoaded(true);
      announce("AI deblur finished on this device. Check the output carefully; AI restoration can alter fine details.", "success");
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
      const { createWorker } = await import("tesseract.js");
      const worker = await createWorker("eng", 1, {
        logger: (message) => {
          if (typeof message.progress === "number") {
            setProgress(Math.round(message.progress * 100));
          }
          if (message.status) announce("OCR: " + message.status + (message.progress ? " · " + Math.round(message.progress * 100) + "%" : ""));
        }
      });
      const results: string[] = [];
      const confidences: number[] = [];
      try {
        for (let index = 0; index < pages.length; index += 1) {
          const canvas = index === activePage ? active.enhancedCanvas : pages[index].enhancedCanvas;
          const result = await worker.recognize(canvas);
          results.push(
            pages.length > 1 ? "PAGE " + (index + 1) + "\n\n" + result.data.text.trim() : result.data.text.trim()
          );
          if (typeof result.data.confidence === "number") confidences.push(result.data.confidence);
          setProgress(Math.round(((index + 1) / pages.length) * 100));
        }
      } finally {
        await worker.terminate();
      }
      const combined = results.filter(Boolean).join("\n\n────────────────────────\n\n");
      const confidence = confidences.length
        ? Math.round(confidences.reduce((sum, value) => sum + value, 0) / confidences.length)
        : null;
      setOcrText(combined);
      setOcr({ text: combined, confidence, pages: pages.length });
      announce("Text extraction complete. Review uncertain words before using the result.", "success");
    } catch (error) {
      announce(error instanceof Error ? error.message : "OCR did not finish. Check your connection for the one-time OCR engine download.", "error");
    } finally {
      setBusy(false);
      setProgress(0);
    }
  }, [active, activePage, announce, pages]);

  const copyText = async () => {
    try {
      await navigator.clipboard.writeText(ocrText);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
      announce("Extracted text copied to clipboard.", "success");
    } catch {
      announce("Clipboard access was blocked by this browser. Select the text and copy it manually.", "error");
    }
  };

  const downloadText = () => {
    if (!ocrText.trim()) return;
    downloadBlob(new Blob([ocrText], { type: "text/plain;charset=utf-8" }), "cleartext-extracted-text.txt");
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
      // There are no server uploads or persistent file copies to clean up.
    };
  }, []);

  return (
    <main>
      <header className="topbar">
        <div className="shell topbar-inner">
          <a href="/" className="brand" aria-label="ClearText AI home">
            <span className="brand-mark"><ScanText size={19} /></span>
            <span>cleartext<span style={{ color: "#85b5ff" }}>.</span><small>AI DOCUMENT RECOVERY</small></span>
          </a>
          <div className="nav-note">Private by design · Runs in your browser</div>
        </div>
      </header>

      <section className="shell hero">
        <div className="eyebrow"><Sparkles size={13} /> DOCUMENT CLARITY TOOLKIT</div>
        <h1>Make difficult text <span>easier to read.</span></h1>
        <p>
          Clean up blurry document photos, enhance scans, and extract editable text.
          Your files are processed on your device — no account or AI API key required.
        </p>
        <div className="hero-points">
          <span><LockKeyhole size={14} /> Local-first processing</span>
          <span><FileText size={14} /> Images and PDFs</span>
          <span><ScanText size={14} /> OCR text export</span>
        </div>
      </section>

      <section className="shell">
        <div className="workspace">
          <section className="panel">
            <div className="panel-head">
              <h2>Upload your document</h2><span className="step">01 / INPUT</span>
            </div>
            <div className="panel-body">
              <div
                className={"dropzone" + (dragging ? " dragging" : "")}
                onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
                onDragLeave={() => setDragging(false)}
                onDrop={onDrop}
              >
                <div className="upload-icon"><UploadCloud size={25} /></div>
                <strong>Drop a page or document here</strong>
                <p>PDF, JPG, PNG or WEBP · Up to 18 MB<br />PDFs: first 8 pages per upload</p>
                <button className="primary-btn" onClick={() => fileInputRef.current?.click()} disabled={busy}>
                  <ImagePlus size={16} /> Choose file
                </button>
                <input
                  ref={fileInputRef}
                  className="hidden-input"
                  type="file"
                  accept="image/png,image/jpeg,image/webp,application/pdf"
                  onChange={onInputChange}
                />
              </div>

              {pages.length > 0 && (
                <div style={{ marginTop: 16 }}>
                  <div className="field-label"><span>Loaded document</span><button className="ghost-btn" style={{ minHeight: 29, padding: "0 9px", fontSize: 10 }} onClick={reset}><X size={12} /> Clear</button></div>
                  <div className="control-box">
                    <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
                      <FileImage size={18} color="#80b3ff" />
                      <div style={{ minWidth: 0 }}>
                        <div style={{ fontSize: 12, fontWeight: 650, overflowWrap: "anywhere" }}>{pages[0].name}</div>
                        <div style={{ fontSize: 10, color: "var(--muted)", marginTop: 3 }}>{pages.length} page{pages.length === 1 ? "" : "s"} loaded</div>
                      </div>
                    </div>
                    {pages.length > 1 && (
                      <div className="page-strip">
                        {pages.map((page, index) => (
                          <button key={page.id} className={"page-pill" + (index === activePage ? " active" : "")} onClick={() => setActivePage(index)}>
                            Page {index + 1}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              )}

              <div style={{ marginTop: 15 }}>
                <div className="field-label"><span>Quick clean-up strength</span><b>{Math.round(strength * 100)}%</b></div>
                <input className="range" type="range" min="0" max="1" step="0.05" value={strength} onChange={(event) => setStrength(Number(event.target.value))} disabled={!active || busy} />
                <div className="action-row" style={{ marginTop: 12 }}>
                  <button className="secondary-btn" onClick={applyQuickEnhance} disabled={!active || busy}><WandSparkles size={15} /> Quick clean-up</button>
                  <button className="ghost-btn" onClick={applyAiDeblur} disabled={!active || busy}><Sparkles size={15} /> AI deblur beta</button>
                </div>
                <p style={{ margin: "9px 0 0", color: "var(--muted)", fontSize: 10, lineHeight: 1.6 }}>
                  Quick clean-up uses browser image filters. AI deblur downloads a roughly 92 MB open model on first use and runs it locally.
                </p>
              </div>
            </div>
          </section>

          <section className="panel">
            <div className="panel-head">
              <h2>Preview & compare</h2><span className="step">02 / ENHANCE</span>
            </div>
            <div className="panel-body">
              <div className="field-label">
                <span>{enhanceMode === "original" ? "Original page" : enhanceMode === "ai" ? "AI-restored preview" : "Quick enhanced preview"}</span>
                <span style={{ color: "var(--muted)" }}>{active ? active.name : "No document"}</span>
              </div>
              <div className="preview-frame">
                {active ? <img src={enhanceMode === "original" ? active.beforeUrl : active.enhancedUrl} alt="Document preview" /> : <div className="preview-empty">Your selected document will appear here. Upload a page to get started.</div>}
              </div>
              <div className="action-row" style={{ marginTop: 12 }}>
                <button className="ghost-btn" onClick={() => setEnhanceMode("original")} disabled={!active || busy}>Show original</button>
                <button className="ghost-btn" onClick={() => setEnhanceMode("quick")} disabled={!active || busy}>Show enhanced</button>
                <button className="ghost-btn" onClick={downloadImage} disabled={!active || busy}><ArrowDownToLine size={14} /> Save PNG</button>
              </div>
              <div style={{ marginTop: 17 }} className="disclaimer">
                <b>Important:</b> Enhancement cannot guarantee recovery of characters lost to severe blur, low resolution, or missing pixels. AI restoration can alter small details. It cannot reveal text that has been covered, redacted, or removed.
              </div>
            </div>
          </section>

          <section className="panel" style={{ gridColumn: "1 / -1" }}>
            <div className="panel-head">
              <h2>Extract readable text</h2><span className="step">03 / OCR</span>
            </div>
            <div className="panel-body">
              <div className="action-row">
                <button className="primary-btn" onClick={runOcr} disabled={!pages.length || busy}>
                  {busy ? <LoaderCircle size={15} className="animate-spin" /> : <ScanText size={15} />}
                  {busy ? "Working..." : "Extract text from document"}
                </button>
                <button className="ghost-btn" onClick={copyText} disabled={!ocrText.trim()}>{copied ? <Check size={14} /> : <Clipboard size={14} />}{copied ? "Copied" : "Copy text"}</button>
                <button className="ghost-btn" onClick={downloadText} disabled={!ocrText.trim()}><ArrowDownToLine size={14} /> Download TXT</button>
              </div>
              <div className="stat-row">
                <span className="stat">OCR engine <b>Tesseract · on-device</b></span>
                <span className="stat">Pages <b>{ocr ? ocr.pages : pages.length || "—"}</b></span>
                <span className="stat">Average OCR confidence <b>{ocr?.confidence !== null && ocr?.confidence !== undefined ? ocr.confidence + "%" : "—"}</b></span>
              </div>
              <textarea
                className="result-area"
                value={ocrText}
                onChange={(event) => setOcrText(event.target.value)}
                placeholder="Extracted text will appear here. You can edit it before copying or downloading."
                aria-label="Extracted text"
              />
              {busy && <div style={{ height: 4, background: "rgba(255,255,255,.08)", borderRadius: 99, overflow: "hidden", marginTop: 12 }}><div style={{ height: "100%", width: progress + "%", background: "linear-gradient(90deg,#397fe4,#89baff)", transition: "width .2s" }} /></div>}
              <div className={"status " + (statusType === "error" ? "error" : statusType === "success" ? "success" : "")} role="status">{status}</div>
              <p style={{ color: "var(--muted)", fontSize: 10, lineHeight: 1.6, margin: "6px 0 0" }}>
                OCR confidence is a guide, not a guarantee. Review names, numbers, dates and technical terms against the original page. OCR language is currently English.
              </p>
            </div>
          </section>
        </div>

        <div className="info-grid">
          <article className="info-card"><WandSparkles color="#83b6ff" size={19} /><b>Enhance before reading</b><p>Quick contrast and sharpening filters improve many ordinary phone scans. Experimental AI deblurring is optional.</p></article>
          <article className="info-card"><ScanText color="#83b6ff" size={19} /><b>Turn pixels into text</b><p>OCR converts readable characters into editable text you can copy or save as a TXT file.</p></article>
          <article className="info-card"><LockKeyhole color="#83b6ff" size={19} /><b>Keep documents private</b><p>Images are processed in your browser. The OCR engine and optional model files are downloaded from public CDNs.</p></article>
        </div>

        <footer className="footer">
          <div style={{ marginBottom: 7, color: "#9baec7", fontWeight: 650 }}>cleartext.ai</div>
          Built for personal document recovery and accessibility. Do not use this tool to bypass access controls or attempt to reconstruct deliberately redacted information.
          <br />Your files are not uploaded by this app. Standard hosting and CDN request logs may still be processed by those providers.
        </footer>
      </section>
    </main>
  );
}