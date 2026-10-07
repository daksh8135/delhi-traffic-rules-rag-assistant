// Delhi Traffic Rules Assistant - Frontend Application
// Matches the logic of frontend_app.py

const DEFAULT_API_URL = "http://127.0.0.1:8000";

// State
let apiUrl = localStorage.getItem("DELHI_RAG_API_URL") || DEFAULT_API_URL;
let currentAnswerRaw = "";

// DOM Elements
const queryInput = document.getElementById("queryInput");
const askBtn = document.getElementById("askBtn");
const topKInput = document.getElementById("topKInput");
const loadingIndicator = document.getElementById("loadingIndicator");
const errorBanner = document.getElementById("errorBanner");
const errorTitle = document.getElementById("errorTitle");
const errorMessage = document.getElementById("errorMessage");
const corsTip = document.getElementById("corsTip");
const answerSection = document.getElementById("answerSection");
const answerText = document.getElementById("answerText");
const contextAccordion = document.getElementById("contextAccordion");
const contextText = document.getElementById("contextText");
const copyBtn = document.getElementById("copyBtn");
const copyBtnText = document.getElementById("copyBtnText");

// Settings Modal Elements
const settingsBtn = document.getElementById("settingsBtn");
const settingsModal = document.getElementById("settingsModal");
const closeSettingsModal = document.getElementById("closeSettingsModal");
const apiUrlInput = document.getElementById("apiUrlInput");
const testConnectionBtn = document.getElementById("testConnectionBtn");
const testConnectionStatus = document.getElementById("testConnectionStatus");
const saveSettingsBtn = document.getElementById("saveSettingsBtn");

// Suggestion chips
const chips = document.querySelectorAll(".chip");

// Configure marked.js options
if (window.marked) {
  marked.setOptions({
    breaks: true,
    gfm: true
  });
}

// Initialize
function init() {
  apiUrlInput.value = apiUrl;
  
  // Event listeners
  askBtn.addEventListener("click", handleAsk);
  queryInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleAsk();
    }
  });

  // Suggestion chips click
  chips.forEach((chip) => {
    chip.addEventListener("click", () => {
      const q = chip.getAttribute("data-question");
      if (q) {
        queryInput.value = q;
        queryInput.focus();
        handleAsk();
      }
    });
  });

  // Copy button
  copyBtn.addEventListener("click", copyAnswerToClipboard);

  // Modal events
  settingsBtn.addEventListener("click", openSettings);
  closeSettingsModal.addEventListener("click", closeSettings);
  settingsModal.addEventListener("click", (e) => {
    if (e.target === settingsModal) closeSettings();
  });
  saveSettingsBtn.addEventListener("click", saveSettings);
  testConnectionBtn.addEventListener("click", testConnection);
}

// Core Ask Handler (matches Streamlit: requests.post(f"{API_URL}/ask", json={"query": query, "top_k": 10}))
async function handleAsk() {
  const query = queryInput.value.trim();
  if (!query) {
    queryInput.focus();
    return;
  }

  const topK = parseInt(topKInput.value, 10) || 10;

  // Reset UI states
  hideError();
  hideAnswer();
  setLoading(true);

  try {
    const cleanUrl = apiUrl.replace(/\/+$/, "");
    const endpoint = `${cleanUrl}/ask`;

    const response = await fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        query: query,
        top_k: topK
      })
    });

    if (!response.ok) {
      let detail = "";
      try {
        const errorJson = await response.json();
        detail = errorJson.detail || JSON.stringify(errorJson);
      } catch (_) {
        detail = await response.text();
      }
      throw new Error(`Server returned HTTP ${response.status}: ${detail || response.statusText}`);
    }

    const result = await response.json();

    // Extract answer (handles {"answer": ...} or {"response": ...})
    const answer = result.answer || result.response || (typeof result === "string" ? result : JSON.stringify(result));
    const context = result.context || "";

    renderAnswer(answer, context);
  } catch (err) {
    console.error("Query Error:", err);
    handleError(err);
  } finally {
    setLoading(false);
  }
}

// Render answer
function renderAnswer(answer, context = "") {
  currentAnswerRaw = answer;

  let formattedHtml = answer;
  if (window.marked && window.DOMPurify) {
    formattedHtml = DOMPurify.sanitize(marked.parse(answer));
  } else {
    formattedHtml = `<p>${escapeHtml(answer)}</p>`;
  }

  answerText.innerHTML = formattedHtml;

  // Handle retrieved context if returned by FastAPI
  if (context && context.trim()) {
    contextText.textContent = context.trim();
    contextAccordion.classList.remove("hidden");
  } else {
    contextAccordion.classList.add("hidden");
    contextText.textContent = "";
  }

  answerSection.classList.remove("hidden");

  // Scroll into view smoothly
  answerSection.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// Error handling matching Streamlit messages
function handleError(err) {
  const isNetworkError = err.name === "TypeError" || (err.message && err.message.toLowerCase().includes("fetch"));

  if (isNetworkError) {
    errorTitle.textContent = "Can't reach the backend";
    errorMessage.textContent = "Make sure the FastAPI server is running (uvicorn api.backend:app --reload) and CORS is enabled.";
    corsTip.classList.remove("hidden");
  } else {
    errorTitle.textContent = "Something went wrong";
    errorMessage.textContent = err.message || "An unexpected error occurred while contacting the RAG pipeline.";
    corsTip.classList.add("hidden");
  }

  errorBanner.classList.remove("hidden");
  errorBanner.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function setLoading(isLoading) {
  if (isLoading) {
    loadingIndicator.classList.remove("hidden");
    askBtn.disabled = true;
    queryInput.disabled = true;
  } else {
    loadingIndicator.classList.add("hidden");
    askBtn.disabled = false;
    queryInput.disabled = false;
    queryInput.focus();
  }
}

function hideError() {
  errorBanner.classList.add("hidden");
  corsTip.classList.add("hidden");
}

function hideAnswer() {
  answerSection.classList.add("hidden");
  answerText.innerHTML = "";
  currentAnswerRaw = "";
  contextAccordion.classList.add("hidden");
  contextText.textContent = "";
}

// Copy to clipboard
async function copyAnswerToClipboard() {
  if (!currentAnswerRaw) return;

  try {
    await navigator.clipboard.writeText(currentAnswerRaw);
    copyBtn.classList.add("copied");
    copyBtnText.textContent = "Copied!";
    setTimeout(() => {
      copyBtn.classList.remove("copied");
      copyBtnText.textContent = "Copy";
    }, 2000);
  } catch (e) {
    console.error("Clipboard copy failed:", e);
  }
}

// Modal Settings
function openSettings() {
  apiUrlInput.value = apiUrl;
  testConnectionStatus.textContent = "";
  testConnectionStatus.className = "modal-status";
  settingsModal.classList.remove("hidden");
  apiUrlInput.focus();
}

function closeSettings() {
  settingsModal.classList.add("hidden");
}

function saveSettings() {
  const newUrl = apiUrlInput.value.trim();
  if (newUrl) {
    apiUrl = newUrl.replace(/\/+$/, "");
    localStorage.setItem("DELHI_RAG_API_URL", apiUrl);
  }
  closeSettings();
}

async function testConnection() {
  const testUrl = (apiUrlInput.value.trim() || DEFAULT_API_URL).replace(/\/+$/, "");
  testConnectionStatus.className = "modal-status";
  testConnectionStatus.textContent = "Testing connection...";

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 4000);

    const res = await fetch(`${testUrl}/`, {
      method: "GET",
      signal: controller.signal
    });
    clearTimeout(timeoutId);

    if (res.ok) {
      testConnectionStatus.className = "modal-status status-ok";
      testConnectionStatus.textContent = "✓ Server connected successfully!";
    } else {
      testConnectionStatus.className = "modal-status status-error";
      testConnectionStatus.textContent = `Server responded with HTTP ${res.status}`;
    }
  } catch (e) {
    testConnectionStatus.className = "modal-status status-error";
    testConnectionStatus.textContent = "✗ Could not reach server. Verify server is running and URL is correct.";
  }
}

function escapeHtml(string) {
  const div = document.createElement("div");
  div.innerText = string;
  return div.innerHTML;
}

// Run on load
document.addEventListener("DOMContentLoaded", init);
