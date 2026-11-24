"use client";

export type PaddleLanguage =
  | "en" | "tr" | "fr" | "de" | "es" | "pt" | "it" | "nl" | "pl";

export type PaddleOcrLine = {
  text: string;
  score: number;
  poly?: number[][];
};

export type PaddleOcrResult = {
  items: PaddleOcrLine[];
  metrics?: {
    detMs?: number;
    recMs?: number;
    totalMs?: number;
    detectedBoxes?: number;
    recognizedCount?: number;
  };
};

export type PaddleOcrEngine = {
  predict: (
    image: HTMLCanvasElement,
    params?: {
      textDetThresh?: number;
      textDetBoxThresh?: number;
      textDetUnclipRatio?: number;
      textRecScoreThresh?: number;
      textDetLimitSideLen?: number;
      textDetLimitType?: "min" | "max";
      textDetMaxSideLimit?: number;
    }
  ) => Promise<PaddleOcrResult[]>;
  dispose: () => Promise<void> | void;
};

export async function createPaddleOcr(language: PaddleLanguage): Promise<PaddleOcrEngine> {
  const { PaddleOCR } = await import("@paddleocr/paddleocr-js");
  const engine = await PaddleOCR.create({
    worker: false,
    lang: language,
    ocrVersion: "PP-OCRv6",
    ortOptions: {
      backend: "auto",
      wasmPaths: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/",
      numThreads: 1,
      simd: true,
    },
  });
  return engine as unknown as PaddleOcrEngine;
}

export function prepareOcrCanvas(source: HTMLCanvasElement): HTMLCanvasElement {
  const longest = Math.max(source.width, source.height);
  const scale =
    longest < 1500
      ? Math.min(2, 1500 / Math.max(1, longest))
      : Math.min(1, 2600 / longest);

  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(source.width * scale));
  canvas.height = Math.max(1, Math.round(source.height * scale));
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("Your browser could not prepare the page for OCR.");

  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  context.fillStyle = "#ffffff";
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.drawImage(source, 0, 0, canvas.width, canvas.height);
  return canvas;
}

export function makeHighContrastVariant(source: HTMLCanvasElement): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.width = source.width;
  canvas.height = source.height;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("Your browser could not prepare the alternate OCR image.");

  context.drawImage(source, 0, 0);
  const frame = context.getImageData(0, 0, canvas.width, canvas.height);
  const pixels = frame.data;
  const histogram = new Uint32Array(256);
  const total = canvas.width * canvas.height;

  for (let i = 0; i < pixels.length; i += 4) {
    const gray = Math.max(
      0,
      Math.min(
        255,
        Math.round(pixels[i] * 0.299 + pixels[i + 1] * 0.587 + pixels[i + 2] * 0.114)
      )
    );
    histogram[gray] += 1;
  }

  const percentile = (fraction: number) => {
    const target = Math.floor(total * fraction);
    let seen = 0;
    for (let value = 0; value < histogram.length; value += 1) {
      seen += histogram[value];
      if (seen >= target) return value;
    }
    return 255;
  };

  const black = percentile(0.015);
  const white = Math.max(black + 24, percentile(0.985));
  const range = Math.max(1, white - black);

  for (let i = 0; i < pixels.length; i += 4) {
    const gray = pixels[i] * 0.299 + pixels[i + 1] * 0.587 + pixels[i + 2] * 0.114;
    const value = Math.max(0, Math.min(255, Math.round(((gray - black) / range) * 255)));
    pixels[i] = value;
    pixels[i + 1] = value;
    pixels[i + 2] = value;
    pixels[i + 3] = 255;
  }

  context.putImageData(frame, 0, 0);
  return canvas;
}

export function orderOcrLines(items: PaddleOcrLine[]): PaddleOcrLine[] {
  const withPosition = items.map((item, index) => {
    const points = Array.isArray(item.poly) ? item.poly : [];
    const xs = points.map((point) => Number(point?.[0])).filter(Number.isFinite);
    const ys = points.map((point) => Number(point?.[1])).filter(Number.isFinite);
    return {
      item,
      index,
      x: xs.length ? Math.min(...xs) : 0,
      y: ys.length ? ys.reduce((sum, value) => sum + value, 0) / ys.length : index,
    };
  });

  withPosition.sort((a, b) => {
    if (Math.abs(a.y - b.y) > 12) return a.y - b.y;
    return a.x - b.x;
  });

  return withPosition.map((entry) => entry.item);
}

export function scoreOcrResult(items: PaddleOcrLine[]) {
  const textLength = items.reduce((total, item) => total + item.text.trim().length, 0);
  const scores = items
    .map((item) => item.score)
    .filter((score) => Number.isFinite(score) && score >= 0 && score <= 1);
  const confidence = scores.length
    ? scores.reduce((sum, value) => sum + value, 0) / scores.length
    : 0;
  return {
    textLength,
    confidence,
    quality: textLength * (0.55 + confidence),
  };
}
