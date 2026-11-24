import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ClearText AI — Make blurry documents readable",
  description:
    "Enhance blurry document photos, run OCR, and copy or download extracted text. Processing happens locally in your browser.",
  applicationName: "ClearText AI",
  keywords: ["blurry text", "document enhancement", "OCR", "scan cleanup", "text extraction"],
  robots: { index: true, follow: true },
  icons: { icon: "/icon.svg", shortcut: "/icon.svg" },
  openGraph: {
    title: "ClearText AI",
    description: "Turn difficult-to-read document photos into clearer scans and usable text.",
    type: "website",
  },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: "#080c14",
  colorScheme: "dark",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}