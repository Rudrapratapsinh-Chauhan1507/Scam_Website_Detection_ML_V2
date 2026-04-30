const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => document.querySelectorAll(selector);

const form = $("#scanForm");
const urlInput = $("#urlInput");
const timeoutInput = $("#timeoutInput");
const timeoutValue = $("#timeoutValue");
const scanButton = $("#scanButton");
const statusBox = $("#statusBox");
const verdictCard = $("#verdictCard");
const predictionLabel = $("#predictionLabel");
const finalUrl = $("#finalUrl");
const scoreRing = $("#scoreRing");
const scoreValue = $("#scoreValue");
const confidenceValue = $("#confidenceValue");
const riskBand = $("#riskBand");
const scrapeStatus = $("#scrapeStatus");
const textLength = $("#textLength");
const signalsGrid = $("#signalsGrid");
const historyList = $("#historyList");
const previewButton = $("#previewButton");
const featurePanel = $("#featurePanel");
const featureTable = $("#featureTable");
const riskRadar = $("#riskRadar");
const driverList = $("#driverList");
const notesList = $("#notesList");
const batchInput = $("#batchInput");
const batchButton = $("#batchButton");
const clearBatchButton = $("#clearBatchButton");
const batchTable = $("#batchTable");
const simUrl = $("#simUrl");
const simKeyword = $("#simKeyword");
const simButton = $("#simButton");
const simResults = $("#simResults");

let latestUrl = "";

const percent = (value) => `${Math.round((Number(value) || 0) * 100)}%`;
const escapeHtml = (value) =>
  String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  })[char]);

function setPage(pageName) {
  $$(".nav-link").forEach((button) => button.classList.toggle("active", button.dataset.page === pageName));
  $$(".page").forEach((page) => page.classList.toggle("active", page.id === `page-${pageName}`));
}

function setStatus(message, tone = "ready") {
  if (!statusBox) return;
  statusBox.querySelector("span:last-child").textContent = message;
  const dot = statusBox.querySelector(".status-dot");
  const colors = { ready: "#0f766e", loading: "#2563eb", error: "#c2410c" };
  dot.style.background = colors[tone] || colors.ready;
}

function renderSignals(signals = []) {
  if (!signals.length) {
    signalsGrid.innerHTML = `
      <div class="signal"><span>Waiting</span><strong>URL</strong></div>
      <div class="signal"><span>Waiting</span><strong>Content</strong></div>
      <div class="signal"><span>Waiting</span><strong>Model</strong></div>
    `;
    return;
  }

  signalsGrid.innerHTML = signals.map((signal) => `
    <div class="signal ${signal.active ? "active" : ""}">
      <span>${escapeHtml(signal.label)}</span>
      <strong>${escapeHtml(signal.value)}</strong>
    </div>
  `).join("");
}

function drawRadar(items = []) {
  if (!riskRadar) return;
  const ctx = riskRadar.getContext("2d");
  const width = riskRadar.width;
  const height = riskRadar.height;
  const centerX = width / 2;
  const centerY = height / 2 + 5;
  const radius = Math.min(width, height) * 0.32;
  const sides = Math.max(items.length, 5);

  ctx.clearRect(0, 0, width, height);
  ctx.lineWidth = 1;
  ctx.font = "12px ui-sans-serif, system-ui";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";

  for (let ring = 1; ring <= 4; ring += 1) {
    const ringRadius = (radius * ring) / 4;
    ctx.beginPath();
    for (let i = 0; i < sides; i += 1) {
      const angle = -Math.PI / 2 + (i * Math.PI * 2) / sides;
      const x = centerX + Math.cos(angle) * ringRadius;
      const y = centerY + Math.sin(angle) * ringRadius;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.strokeStyle = ring === 4 ? "#8ca0b4" : "#d7e0e7";
    ctx.stroke();
  }

  items.forEach((item, i) => {
    const angle = -Math.PI / 2 + (i * Math.PI * 2) / sides;
    ctx.beginPath();
    ctx.moveTo(centerX, centerY);
    ctx.lineTo(centerX + Math.cos(angle) * radius, centerY + Math.sin(angle) * radius);
    ctx.strokeStyle = "#e0e7ee";
    ctx.stroke();
    ctx.fillStyle = "#52616f";
    ctx.fillText(item.label.split(" ")[0], centerX + Math.cos(angle) * (radius + 42), centerY + Math.sin(angle) * (radius + 32));
  });

  if (items.length) {
    ctx.beginPath();
    items.forEach((item, i) => {
      const angle = -Math.PI / 2 + (i * Math.PI * 2) / sides;
      const scoreRadius = radius * (Number(item.value) || 0);
      const x = centerX + Math.cos(angle) * scoreRadius;
      const y = centerY + Math.sin(angle) * scoreRadius;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.fillStyle = "rgba(14, 116, 144, 0.18)";
    ctx.strokeStyle = "#0f766e";
    ctx.lineWidth = 3;
    ctx.fill();
    ctx.stroke();
  }
}

function renderDrivers(insights = {}) {
  const drivers = insights.top_drivers || [];
  const notes = insights.notes || ["Run a scan to generate reliability notes."];

  driverList.innerHTML = drivers.length ? drivers.map((driver) => `
    <div class="driver">
      <div><strong>${escapeHtml(driver.label)}</strong><span>${escapeHtml(driver.detail)}</span></div>
      <div class="driver-bar"><span style="width: ${Math.round((Number(driver.value) || 0) * 100)}%"></span></div>
    </div>
  `).join("") : '<p class="empty-state">Risk drivers will appear after analysis.</p>';

  notesList.innerHTML = notes.map((note) => `<p>${escapeHtml(note)}</p>`).join("");
}

function applyVerdict(result) {
  const tone = result.risk_band?.tone || "safe";
  verdictCard.className = `verdict-card ${tone}`;
  predictionLabel.textContent = result.prediction === "SCAM" ? "Likely scam" : "Likely legitimate";
  finalUrl.textContent = result.final_url || result.url;
  scoreRing.style.setProperty("--score", Math.round((result.scam_probability || 0) * 100));
  scoreValue.textContent = percent(result.scam_probability);
  confidenceValue.textContent = percent(result.confidence);
  riskBand.textContent = result.risk_band?.label || "--";
  scrapeStatus.textContent = result.status || "--";
  textLength.textContent = result.text_length ?? "--";
  renderSignals(result.signals);
  drawRadar(result.insights?.radar || []);
  renderDrivers(result.insights || {});
}

function renderHistory(items = []) {
  if (!historyList) return;
  if (!items.length) {
    historyList.innerHTML = '<p class="empty-state">Scans will appear here during this Flask session.</p>';
    return;
  }

  historyList.innerHTML = items.map((item) => `
    <div class="history-item">
      <span title="${escapeHtml(item.url)}">${escapeHtml(item.url)}</span>
      <span class="badge ${item.risk_band?.tone || "safe"}">${escapeHtml(item.prediction)}</span>
      <strong>${percent(item.scam_probability)}</strong>
    </div>
  `).join("");
}

async function refreshHistory() {
  const response = await fetch("/api/history");
  if (response.ok) renderHistory(await response.json());
}

async function predict(url, timeout = timeoutInput.value) {
  const response = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url, timeout }),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Prediction failed.");
  return data;
}

async function analyze(url) {
  latestUrl = url;
  scanButton.disabled = true;
  scanButton.textContent = "Scanning";
  featurePanel.hidden = true;
  setStatus("Scraping page and running model prediction...", "loading");

  try {
    const data = await predict(url);
    applyVerdict(data);
    await refreshHistory();
    setStatus(data.error || "Prediction complete.", "ready");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    scanButton.disabled = false;
    scanButton.textContent = "Run Scan";
  }
}

async function previewFeatures() {
  const url = latestUrl || urlInput.value.trim();
  if (!url) {
    setStatus("Enter a URL before previewing features.", "error");
    return;
  }

  setStatus("Extracting URL-only feature preview...", "loading");
  try {
    const response = await fetch("/api/features", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Feature preview failed.");
    featureTable.innerHTML = Object.entries(data.features)
      .map(([key, value]) => `<div class="feature-row"><span>${escapeHtml(key)}</span><strong>${escapeHtml(value ?? 0)}</strong></div>`)
      .join("");
    featurePanel.hidden = false;
    setStatus("URL feature preview ready.", "ready");
  } catch (error) {
    setStatus(error.message, "error");
  }
}

async function runBatch() {
  const urls = batchInput.value.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  if (!urls.length) {
    batchTable.innerHTML = '<p class="empty-state">Add at least one URL first.</p>';
    return;
  }

  batchButton.disabled = true;
  batchButton.textContent = "Scanning...";
  batchTable.innerHTML = '<p class="empty-state">Running batch predictions...</p>';
  const rows = [];

  for (const url of urls) {
    try {
      const result = await predict(url, 5);
      rows.push({ url, result });
      await refreshHistory();
    } catch (error) {
      rows.push({ url, error: error.message });
    }
    batchTable.innerHTML = renderBatchRows(rows);
  }

  batchButton.disabled = false;
  batchButton.textContent = "Scan List";
}

function renderBatchRows(rows) {
  return `
    <div class="table-row table-head"><span>URL</span><span>Prediction</span><span>Scam Risk</span><span>Confidence</span></div>
    ${rows.map(({ url, result, error }) => `
      <div class="table-row">
        <span title="${escapeHtml(url)}">${escapeHtml(url)}</span>
        <span class="badge ${result?.risk_band?.tone || "watch"}">${escapeHtml(error ? "ERROR" : result.prediction)}</span>
        <strong>${error ? "--" : percent(result.scam_probability)}</strong>
        <strong>${error ? escapeHtml(error) : percent(result.confidence)}</strong>
      </div>
    `).join("")}
  `;
}

function makeVariants(url, keyword) {
  const cleaned = url.trim() || "https://www.example.com/login";
  const injection = (keyword.trim() || "verify-account").replace(/\s+/g, "-");
  let parsed;
  try {
    parsed = new URL(cleaned.startsWith("http") ? cleaned : `https://${cleaned}`);
  } catch {
    parsed = new URL("https://www.example.com/login");
  }
  return [
    { label: "Base URL", url: parsed.toString() },
    { label: "Keyword in path", url: `${parsed.origin}/${injection}/login` },
    { label: "Extra subdomain", url: `${parsed.protocol}//secure-${injection}.${parsed.hostname}${parsed.pathname}` },
    { label: "Query pressure", url: `${parsed.origin}${parsed.pathname}?${injection}=true&session=778899` },
  ];
}

async function runSimulation() {
  simButton.disabled = true;
  simButton.textContent = "Simulating...";
  simResults.innerHTML = '<p class="empty-state">Running URL-only model scores...</p>';
  const variants = makeVariants(simUrl.value, simKeyword.value);
  const rows = [];

  for (const item of variants) {
    try {
      const response = await fetch("/api/url-predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: item.url }),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || "Simulation failed.");
      rows.push({ ...item, result });
    } catch (error) {
      rows.push({ ...item, error: error.message });
    }
  }

  simResults.innerHTML = rows.map(({ label, url, result, error }) => `
    <div class="sim-card">
      <span>${escapeHtml(label)}</span>
      <strong>${error ? "Error" : percent(result.scam_probability)}</strong>
      <p>${escapeHtml(error || url)}</p>
    </div>
  `).join("");
  simButton.disabled = false;
  simButton.textContent = "Generate Variants";
}

$$(".nav-link").forEach((button) => button.addEventListener("click", () => setPage(button.dataset.page)));
timeoutInput.addEventListener("input", () => { timeoutValue.textContent = `${timeoutInput.value}s`; });
form.addEventListener("submit", (event) => { event.preventDefault(); analyze(urlInput.value.trim()); });
$$(".quick-tests button").forEach((button) => button.addEventListener("click", () => { urlInput.value = button.dataset.url; analyze(button.dataset.url); }));
previewButton.addEventListener("click", previewFeatures);
batchButton.addEventListener("click", runBatch);
clearBatchButton.addEventListener("click", () => { batchInput.value = ""; batchTable.innerHTML = ""; });
simButton.addEventListener("click", runSimulation);

renderSignals([]);
drawRadar([
  { label: "URL complexity", value: 0 },
  { label: "Host reputation", value: 0 },
  { label: "Brand mismatch", value: 0 },
  { label: "Content pressure", value: 0 },
  { label: "Redirect structure", value: 0 },
]);
renderDrivers();
refreshHistory();
