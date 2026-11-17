// Thin client for the FastAPI backend (see backend/api.py).

function resolveApiBase() {
  if (import.meta.env.VITE_API_URL) {
    return import.meta.env.VITE_API_URL.replace(/\/+$/, "");
  }
  // When running locally in browser (e.g. localhost:5173), prioritize local backend on port 8000
  if (typeof window !== "undefined") {
    const host = window.location.hostname;
    if (host === "localhost" || host === "127.0.0.1" || host === "0.0.0.0" || host === "::1") {
      return "http://localhost:8000";
    }
  }
  return "https://protmind-ai-final.onrender.com";
}

let BASE = resolveApiBase();
const isLocalHost =
  typeof window !== "undefined" &&
  ["localhost", "127.0.0.1", "0.0.0.0", "::1"].includes(window.location.hostname);

const FALLBACK_BASE = isLocalHost ? "https://protmind-ai-final.onrender.com" : null;

async function parseErrorMessage(res, path) {
  let errMsg = `Server error (${res.status}) on ${path}`;
  try {
    const errData = await res.json();
    if (errData?.detail) {
      errMsg = Array.isArray(errData.detail)
        ? errData.detail.map((d) => d.msg || JSON.stringify(d)).join(", ")
        : String(errData.detail);
    }
  } catch {
    // Response body wasn't JSON, keep default status message
  }
  return errMsg;
}

async function request(path, options) {
  const url = `${BASE}${path}`;
  let res;

  try {
    res = await fetch(url, options);
  } catch (err) {
    // Only attempt localhost/Render fallback if a local dev fallback is configured
    if (FALLBACK_BASE && BASE !== FALLBACK_BASE) {
      try {
        const altRes = await fetch(`${FALLBACK_BASE}${path}`, options);
        if (altRes.ok) {
          BASE = FALLBACK_BASE;
          return altRes;
        }
        throw new Error(await parseErrorMessage(altRes, path));
      } catch (fallbackErr) {
        if (!fallbackErr.message.includes("Failed to fetch")) {
          throw fallbackErr;
        }
      }
    }
    throw new Error(`Failed to reach backend (${err.message}) on ${path}`);
  }

  // If primary server returns 404 on an upgraded endpoint and a fallback exists, try it
  if (!res.ok && res.status === 404 && FALLBACK_BASE && BASE !== FALLBACK_BASE) {
    try {
      const altRes = await fetch(`${FALLBACK_BASE}${path}`, options);
      if (altRes.ok) {
        BASE = FALLBACK_BASE;
        return altRes;
      }
    } catch {
      // Ignore fallback network failure and report primary error below
    }
  }

  if (!res.ok) {
    throw new Error(await parseErrorMessage(res, path));
  }

  return res;
}

export async function postJson(path, payload) {
  const res = await request(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return res.json();
}

export async function validateInput({ uniprotId, mutation, sequence, file }) {
  const fd = new FormData();
  fd.append("uniprot_id", uniprotId);
  fd.append("mutation", mutation);
  fd.append("sequence_input", sequence);
  if (file) fd.append("file", file);
  const res = await request("/api/step1/validate", { method: "POST", body: fd });
  return res.json();
}

export async function fetchReportPdf(payload) {
  const res = await request("/api/step10/pdf", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return res.blob();
}

export async function predictStep9({ uniprotId = "P04637", mutation = "Y220C", sequence = "" } = {}) {
  const payload = {
    uniprot_id: uniprotId,
    mutation: mutation,
    sequence: sequence
  };
  if (isLocalHost) {
    try {
      const res = await fetch("http://127.0.0.1:8000/predict-step9", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      if (res.ok) return await res.json();
    } catch {
      // Fall through to postJson
    }
  }
  return await postJson("/predict-step9", payload);
}

export async function registerUser({ name, full_name, email, password, captcha_token }) {
  const payload = {
    full_name: full_name || name,
    email,
    password,
    captcha_token
  };
  try {
    return await postJson("/register", payload);
  } catch (err) {
    // Only retry /api/register if the route wasn't found (404), not on 400 validation errors
    if (err.message.includes("404")) {
      return await postJson("/api/register", payload);
    }
    throw err;
  }
}

export async function loginUser({ email, password, captcha_token }) {
  const payload = {
    email,
    password,
    captcha_token
  };
  try {
    return await postJson("/login", payload);
  } catch (err) {
    if (err.message.includes("404")) {
      return await postJson("/api/login", payload);
    }
    throw err;
  }
}

export async function sendChatMessage({ message, history = [], pipeline_data = {} }) {
  const payload = {
    message,
    history,
    pipeline_data
  };
  try {
    return await postJson("/chat-assistant", payload);
  } catch (err) {
    if (err.message.includes("404")) {
      return await postJson("/api/chat-assistant", payload);
    }
    throw err;
  }
}
