/** @type {import('next').NextConfig} */
const securityHeaders = [
  { key: "X-Content-Type-Options", value: "nosniff" },
  { key: "X-Frame-Options", value: "DENY" },
  { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
  { key: "Permissions-Policy", value: "camera=(), microphone=(), geolocation=()" },
  {
    key: "Content-Security-Policy",
    value: "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval' 'wasm-unsafe-eval' https://cdn.jsdelivr.net; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; font-src 'self' data:; connect-src 'self' https://cdn.jsdelivr.net https://huggingface.co https://cdn-lfs.huggingface.co https://cdn-lfs-us-1.huggingface.co https://cdn-lfs-eu-1.huggingface.co https://cdn-lfs-us-1.hf.co https://cdn-lfs-eu-1.hf.co https://cdn-lfs.hf.co https://cas-bridge.xethub.hf.co https://cas-server.xethub.hf.co https://transfer.xethub.hf.co https://*.aws.cdn.hf.co https://cdnjs.cloudflare.com https://paddle-model-ecology.bj.bcebos.com https://*.bcebos.com https://tessdata.projectnaptha.com; worker-src 'self' blob: https://cdn.jsdelivr.net https://cdnjs.cloudflare.com; object-src 'none'; base-uri 'self'; frame-ancestors 'none';",
  },
];

const nextConfig = {
  poweredByHeader: false,
  compress: true,
  turbopack: {
    root: process.cwd(),
    resolveAlias: {
      fs: { browser: "./lib/empty-node-module.ts" },
      path: { browser: "./lib/empty-node-module.ts" },
      crypto: { browser: "./lib/empty-node-module.ts" },
      "ort.bundle.min.mjs": "./lib/empty-node-module.ts",
    },
  },
  async headers() {
    return [{ source: "/(.*)", headers: securityHeaders }];
  },
};

export default nextConfig;