import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import { Client } from "@gradio/client";
import path from "path";
import { fileURLToPath } from "url";

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);

const app      = express();
const PORT     = process.env.PORT || 3000;
const HF_TOKEN = process.env.HF_TOKEN;
const HF_SPACE = process.env.HF_SPACE || "yisol/IDM-VTON";

// ─── CORS — must be first, before every route ──────────────────────────────────
app.use((req, res, next) => {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");
  res.setHeader("Access-Control-Max-Age", "86400");
  if (req.method === "OPTIONS") return res.status(204).end();
  next();
});

app.use(cors({
  origin: "*",
  methods: ["GET", "POST", "OPTIONS"],
  allowedHeaders: ["Content-Type", "Authorization"],
}));
app.options("*", cors());

// ─── Body parser — wrap to catch malformed JSON ───────────────────────────────
app.use((req, res, next) => {
  express.json({ limit: "50mb" })(req, res, (err) => {
    if (err) {
      console.error("[Body Parser] error:", err.message);
      return res.status(400).json({ success: false, error: "Invalid JSON body — " + err.message });
    }
    next();
  });
});
app.use(express.urlencoded({ extended: true, limit: "50mb" }));
app.use(express.static(path.join(__dirname, "public")));

// ─── Startup checks ────────────────────────────────────────────────────────────
if (!HF_TOKEN || !HF_TOKEN.startsWith("hf_")) {
  console.warn("\n⚠️  WARNING: HF_TOKEN is missing or invalid.");
  console.warn("   Set it in .env:  HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx\n");
}

// ─── Helpers ───────────────────────────────────────────────────────────────────
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function base64ToBlob(b64) {
  const match    = b64.match(/^data:([^;]+);base64,(.+)$/s);
  const mimeType = match ? match[1] : "image/jpeg";
  const raw      = match ? match[2] : b64;
  return new Blob([Buffer.from(raw.trim(), "base64")], { type: mimeType });
}

function buildInputs(personBlob, garmentBlob, opts) {
  return [
    { background: personBlob, layers: [], composite: null },
    garmentBlob,
    String(opts.description),
    Boolean(opts.is_checked),
    Boolean(opts.is_checked_crop),
    Number(opts.denoise_steps),
    Number(opts.seed),
  ];
}

function extractUrl(data) {
  if (!Array.isArray(data) || data.length === 0) return null;
  const item = data[0];
  if (typeof item === "string" && item.length > 4) return item;
  return item?.url ?? item?.path ?? null;
}

// ─── Connect with retry ────────────────────────────────────────────────────────
async function connectWithRetry(maxRetries = 5, baseDelayMs = 4000) {
  let lastErr;
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      console.log(`[Connect] Attempt ${attempt}/${maxRetries} → ${HF_SPACE}`);
      const client = await Client.connect(HF_SPACE, {
        hf_token: HF_TOKEN,
        status_callback: (s) =>
          console.log(`[Space] stage=${s?.stage} message=${s?.message ?? ""}`),
      });
      console.log("[Connect] ✓ Connected.");
      return client;
    } catch (err) {
      lastErr = err;
      const recoverable =
        err.message?.includes("app config") ||
        err.message?.includes("Could not resolve") ||
        err.message?.includes("503") ||
        err.message?.includes("502") ||
        err.message?.includes("sleeping") ||
        err.message?.includes("building") ||
        err.message?.includes("starting");
      if (!recoverable) throw err;
      const delay = baseDelayMs * attempt;
      console.warn(`[Connect] Not ready (${err.message}). Retrying in ${delay / 1000}s…`);
      await sleep(delay);
    }
  }
  throw new Error(
    `Space "${HF_SPACE}" not ready after ${maxRetries} attempts. ` +
    `Last error: ${lastErr?.message}. ` +
    `Visit https://huggingface.co/spaces/${HF_SPACE} to wake it.`
  );
}

// ─── Discover endpoint ─────────────────────────────────────────────────────────
async function discoverEndpoint(client) {
  try {
    const apiInfo = await client.view_api();
    console.log("\n[API] view_api():\n", JSON.stringify(apiInfo, null, 2));
    const named   = apiInfo?.named_endpoints   ?? {};
    const unnamed = apiInfo?.unnamed_endpoints ?? {};
    if (named["/tryon"])  return "/tryon";
    const key = Object.keys(named).find((k) => /try.?on/i.test(k));
    if (key) return key;
    const idx = unnamed[6] !== undefined ? 6 : Number(Object.keys(unnamed)[0] ?? 0);
    return { fn_index: idx };
  } catch (err) {
    console.error("[API] view_api() failed:", err.message);
    return "/tryon";
  }
}

// ─── Predict helpers ───────────────────────────────────────────────────────────
async function predictNamed(client, name, inputs) {
  const result = await client.predict(name, inputs);
  return result?.data ?? null;
}

async function predictByIndex(client, fn_index, inputs) {
  return new Promise((resolve, reject) => {
    const job     = client.submit(null, inputs, { fn_index });
    const results = [];
    job.on("data",   (e) => { if (e?.data) results.push(...e.data); });
    job.on("status", (s) => { if (s?.stage === "error") reject(new Error(s?.message ?? "fn_index error")); });
    job.on("error",  reject);
    job.on("done",   () => resolve(results.length ? results : null));
  });
}

// ─── POST /try-on ──────────────────────────────────────────────────────────────
app.post("/try-on", async (req, res) => {
  if (!req.body || typeof req.body !== "object") {
    return res.status(400).json({ success: false, error: "Request body missing or not valid JSON." });
  }

  const {
    person_image,
    garment_image,
    description     = "fashion item",
    is_checked      = true,
    is_checked_crop = false,
    denoise_steps   = 30,
    seed            = 42,
  } = req.body;

  if (!person_image  || typeof person_image  !== "string" || person_image.length  < 50)
    return res.status(400).json({ success: false, error: "person_image is missing or invalid." });
  if (!garment_image || typeof garment_image !== "string" || garment_image.length < 50)
    return res.status(400).json({ success: false, error: "garment_image is missing or invalid." });

  const opts = { description, is_checked, is_checked_crop, denoise_steps, seed };

  try {
    if (!HF_TOKEN) throw new Error("HF_TOKEN is not set in .env");

    const client      = await connectWithRetry();
    const endpoint    = await discoverEndpoint(client);
    const personBlob  = base64ToBlob(person_image);
    const garmentBlob = base64ToBlob(garment_image);
    const inputs      = buildInputs(personBlob, garmentBlob, opts);

    let data;
    if (typeof endpoint === "string") {
      console.log(`[VTO] Predicting via named="${endpoint}" …`);
      data = await predictNamed(client, endpoint, inputs);
    } else {
      console.log(`[VTO] Predicting via fn_index=${endpoint.fn_index} …`);
      data = await predictByIndex(client, endpoint.fn_index, inputs);
    }

    console.log("[VTO] Raw data:", JSON.stringify(data, null, 2));

    const outputUrl = extractUrl(data);
    if (!outputUrl) throw new Error("No output URL in response: " + JSON.stringify(data));

    console.log("[VTO] ✓ Done:", outputUrl);
    return res.json({ success: true, output_url: outputUrl, data });

  } catch (err) {
    console.error("\n[VTO] Error:", err.message);

    if (err.message?.includes("app config") || err.message?.includes("not become ready")) {
      return res.status(503).json({
        success: false,
        error: "The Hugging Face Space is still starting up.",
        detail: err.message,
        fix: `Open https://huggingface.co/spaces/${HF_SPACE} to wake it, wait ~1 min, then retry.`,
      });
    }
    if (err.message?.toLowerCase().includes("quota")) {
      return res.status(429).json({
        success: false,
        error: "ZeroGPU quota exceeded.",
        detail: err.message,
        fix: "Make sure HF_SPACE points to YOUR duplicated Space and HF_TOKEN is set as a Secret inside it.",
      });
    }
    if (err.message?.toLowerCase().includes("unauthorized") ||
        err.message?.toLowerCase().includes("token")) {
      return res.status(401).json({
        success: false,
        error: "Hugging Face authentication failed.",
        fix: "Check HF_TOKEN in your environment variables.",
      });
    }
    return res.status(500).json({
      success: false,
      error: err.message || "Unknown error",
      hint: "Check server logs for details.",
    });
  }
});

// ─── GET /fetch-image — proxy product images to avoid Shopify CDN CORS ─────────
app.get("/fetch-image", async (req, res) => {
  const { url } = req.query;
  if (!url) return res.status(400).json({ error: "url param required" });

  try {
    const safeUrl  = url.replace(/^http:\/\//i, "https://");
    const response = await fetch(safeUrl);
    if (!response.ok) throw new Error(`Upstream HTTP ${response.status}`);

    const buffer      = await response.arrayBuffer();
    const contentType = response.headers.get("content-type") || "image/jpeg";

    res.set("Access-Control-Allow-Origin", "*");
    res.set("Content-Type", contentType);
    res.send(Buffer.from(buffer));
  } catch (err) {
    console.error("[fetch-image]", err.message);
    res.status(500).json({ error: err.message });
  }
});

// ─── GET /space-status ─────────────────────────────────────────────────────────
app.get("/space-status", async (req, res) => {
  try {
    const client    = await Client.connect(HF_SPACE, { hf_token: HF_TOKEN });
    const info      = await client.view_api();
    const endpoints = Object.keys(info?.named_endpoints ?? {});
    return res.json({ success: true, space: HF_SPACE, status: "ready", endpoints });
  } catch (err) {
    return res.status(503).json({
      success: false, space: HF_SPACE, status: "not ready", error: err.message,
      fix: `Visit https://huggingface.co/spaces/${HF_SPACE} to wake the Space.`,
    });
  }
});

// ─── GET /health ───────────────────────────────────────────────────────────────
app.get("/health", (_req, res) =>
  res.json({ status: "ok", hf_token_set: !!HF_TOKEN, space: HF_SPACE, timestamp: new Date().toISOString() })
);

// ─── Start (local only — Vercel handles this in production) ───────────────────
if (process.env.VERCEL !== "1") {
  app.listen(PORT, () => {
    console.log(`\n✅  VTO Backend → http://localhost:${PORT}`);
    console.log(`   HF_TOKEN : ${HF_TOKEN ? "✓ (" + HF_TOKEN.slice(0, 8) + "…)" : "✗ MISSING"}`);
    console.log(`   HF_SPACE : ${HF_SPACE}\n`);
    console.log(`   Space status → http://localhost:${PORT}/space-status\n`);
  });
}

export default app;