import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import { Client } from "@gradio/client";
import path from "path";
import { fileURLToPath } from "url";

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);

const app  = express();
const PORT = process.env.PORT || 3000;

// ─── Middleware ────────────────────────────────────────────────────────────────
app.use(cors({
  origin: [
    'https://mohamed-ezzeldine-48-teststore.myshopify.com/',
    'https://your-custom-domain.com',
    'http://localhost:3000',
  ]
}));
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

// ─── Startup token check ───────────────────────────────────────────────────────
const HF_TOKEN = process.env.HF_TOKEN;
if (!HF_TOKEN || !HF_TOKEN.startsWith("hf_")) {
  console.warn("\n⚠️  WARNING: HF_TOKEN is missing or looks invalid.");
  console.warn("   Without it you will hit ZeroGPU anonymous quota limits immediately.");
  console.warn("   Set it in your .env file:  HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx\n");
}

// ─── Helper: base64 data-URI → Blob ───────────────────────────────────────────
function base64ToBlob(b64) {
  const match    = b64.match(/^data:([^;]+);base64,(.+)$/s);
  const mimeType = match ? match[1] : "image/jpeg";
  const raw      = match ? match[2] : b64;
  return new Blob([Buffer.from(raw.trim(), "base64")], { type: mimeType });
}

// ─── Helper: build IDM-VTON inputs ────────────────────────────────────────────
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

// ─── Helper: extract URL from Gradio response data ────────────────────────────
function extractUrl(data) {
  if (!Array.isArray(data) || data.length === 0) return null;
  const item = data[0];
  if (typeof item === "string" && item.length > 4) return item;
  return item?.url ?? item?.path ?? null;
}

// ─── Helper: auto-discover the /tryon endpoint name ───────────────────────────
async function discoverEndpoint(client) {
  try {
    const apiInfo = await client.view_api();
    console.log("\n[API Discovery] view_api() structure:\n", JSON.stringify(apiInfo, null, 2));

    const named   = apiInfo?.named_endpoints   ?? {};
    const unnamed = apiInfo?.unnamed_endpoints ?? {};

    if (named["/tryon"]) {
      console.log("[API Discovery] Found named endpoint: /tryon");
      return "/tryon";               // always return a STRING
    }

    const key = Object.keys(named).find((k) => /try.?on/i.test(k));
    if (key) {
      console.log("[API Discovery] Found named endpoint:", key);
      return key;
    }

    // For fn_index we return an object so the caller can use submit() instead
    const idx = unnamed[6] !== undefined ? 6 : Number(Object.keys(unnamed)[0] ?? 0);
    console.log("[API Discovery] No named endpoint found, will use fn_index:", idx);
    return { fn_index: idx };

  } catch (err) {
    console.error("[API Discovery] view_api() failed:", err.message);
    return "/tryon";
  }
}

// ─── Helper: predict via named endpoint (string) ──────────────────────────────
async function predictNamed(client, endpointName, inputs) {
  const result = await client.predict(endpointName, inputs);
  return result?.data ?? null;
}

// ─── Helper: predict via fn_index using submit() to bypass the inputs bug ─────
//   Passing a raw number to client.predict() in @gradio/client ≥1.x triggers
//   "Cannot read properties of undefined (reading 'inputs')" because the
//   library tries to look up the dependency schema by name, not by index.
//   Using client.submit() with fn_index in the options object avoids that path.
async function predictByIndex(client, fn_index, inputs) {
  return new Promise((resolve, reject) => {
    const job = client.submit(null, inputs, { fn_index });
    const results = [];

    job.on("data", (event) => {
      if (event?.data) results.push(...event.data);
    });

    job.on("status", (status) => {
      if (status?.stage === "error") {
        reject(new Error(status?.message ?? "fn_index predict error"));
      }
    });

    job.on("error", reject);

    job.on("done", () => resolve(results.length > 0 ? results : null));
  });
}

// ─── POST /try-on ──────────────────────────────────────────────────────────────
app.post("/try-on", async (req, res) => {

  if (!req.body || typeof req.body !== "object") {
    return res.status(400).json({
      success: false,
      error: "Request body missing or not valid JSON.",
    });
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

  if (!person_image || typeof person_image !== "string" || person_image.length < 50) {
    return res.status(400).json({ success: false, error: "person_image is missing or invalid." });
  }
  if (!garment_image || typeof garment_image !== "string" || garment_image.length < 50) {
    return res.status(400).json({ success: false, error: "garment_image is missing or invalid." });
  }

  const opts = { description, is_checked, is_checked_crop, denoise_steps, seed };

  let client;
  try {
    console.log("\n[VTO] Connecting to yisol/IDM-VTON …");

    // HF_TOKEN is required — without it you are treated as anonymous and
    // immediately exhaust the free ZeroGPU quota on the Space.
    if (!HF_TOKEN) throw new Error("HF_TOKEN is not set. Add it to your .env file.");

    // Use your own duplicated Space (HF_SPACE) to avoid ZeroGPU anonymous quota.
    // Falls back to the original Space if HF_SPACE is not set.
    const SPACE = process.env.HF_SPACE || "yisol/IDM-VTON";
    console.log("[VTO] Using Space:", SPACE);
    client = await Client.connect(SPACE, { hf_token: HF_TOKEN });
    console.log("[VTO] Connected.");

    const endpoint    = await discoverEndpoint(client);
    const personBlob  = base64ToBlob(person_image);
    const garmentBlob = base64ToBlob(garment_image);
    const inputs      = buildInputs(personBlob, garmentBlob, opts);

    let data;

    if (typeof endpoint === "string") {
      // ── Named endpoint path (preferred) ────────────────────────────────────
      console.log(`[VTO] Predicting via named = "${endpoint}" …`);
      data = await predictNamed(client, endpoint, inputs);
    } else {
      // ── fn_index path — use submit() to avoid the inputs bug ───────────────
      console.log(`[VTO] Predicting via fn_index = ${endpoint.fn_index} …`);
      data = await predictByIndex(client, endpoint.fn_index, inputs);
    }

    console.log("[VTO] Raw data:", JSON.stringify(data, null, 2));

    const outputUrl = extractUrl(data);
    if (!outputUrl) {
      throw new Error("Could not extract output URL. Raw data: " + JSON.stringify(data));
    }

    console.log("[VTO] ✓ Success:", outputUrl);
    return res.json({ success: true, output_url: outputUrl, data });

  } catch (err) {
    console.error("\n[VTO] Error:", err.message);

    // Surface quota / auth errors clearly
    if (err.message?.toLowerCase().includes("quota")) {
      return res.status(429).json({
        success: false,
        error: "Hugging Face ZeroGPU quota exceeded.",
        detail: err.message,
        fix: "Ensure HF_TOKEN is set in .env to a valid logged-in token, then restart the server.",
      });
    }

    if (err.message?.toLowerCase().includes("unauthorized") ||
        err.message?.toLowerCase().includes("token")) {
      return res.status(401).json({
        success: false,
        error: "Hugging Face authentication failed.",
        detail: err.message,
        fix: "Check that HF_TOKEN in .env is correct and has read access.",
      });
    }

    return res.status(500).json({
      success: false,
      error: err.message || "Unknown server error",
      hint: "Check server console for the full view_api() output.",
    });
  }
});

// ─── Health check ──────────────────────────────────────────────────────────────
app.get("/health", (_req, res) =>
  res.json({
    status: "ok",
    hf_token_set: !!HF_TOKEN,
    timestamp: new Date().toISOString(),
  })
);
// ─── GET /fetch-image — proxy Shopify CDN images to avoid CORS ────────────────
app.get("/fetch-image", async (req, res) => {
  const { url } = req.query;
  if (!url) return res.status(400).json({ error: "url param required" });

  try {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const buffer = await response.arrayBuffer();
    const contentType = response.headers.get("content-type") || "image/jpeg";
    res.set("Content-Type", contentType);
    res.set("Access-Control-Allow-Origin", "*");
    res.send(Buffer.from(buffer));
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});
// ─── Start ─────────────────────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`\n✅  VTO Backend → http://localhost:${PORT}`);
  console.log(`   HF_TOKEN : ${HF_TOKEN ? "✓ loaded (" + HF_TOKEN.slice(0,8) + "…)" : "✗ MISSING — set HF_TOKEN in .env"}`);
  console.log(`   HF_SPACE : ${process.env.HF_SPACE || "⚠️  not set — using yisol/IDM-VTON (may hit quota)"}\n`);
});