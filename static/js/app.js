const form = document.querySelector("#scanForm");
const urlInput = document.querySelector("#urlInput");
const timeoutInput = document.querySelector("#timeoutInput");
const timeoutValue = document.querySelector("#timeoutValue");
const scanButton = document.querySelector("#scanButton");
const statusBox = document.querySelector("#statusBox");
const verdictCard = document.querySelector("#verdictCard");
const predictionLabel = document.querySelector("#predictionLabel");
const finalUrl = document.querySelector("#finalUrl");
const scoreRing = document.querySelector("#scoreRing");
const scoreValue = document.querySelector("#scoreValue");
const confidenceValue = document.querySelector("#confidenceValue");
const riskBand = document.querySelector("#riskBand");
const scrapeStatus = document.querySelector("#scrapeStatus");
const textLength = document.querySelector("#textLength");
const signalsGrid = document.querySelector("#signalsGrid");
const historyList = document.querySelector("#historyList");
const previewButton = document.querySelector("#previewButton");
const featurePanel = document.querySelector("#featurePanel");
const featureTable = document.querySelector("#featureTable");
const riskRadar = document.querySelector("#riskRadar");
const driverList = document.querySelector("#driverList");
const notesList = document.querySelector("#notesList");

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

function setStatus(message, tone = "ready") {
  statusBox.querySelector("span:last-child").textContent = message;
  const dot = statusBox.querySelector(".status-dot");
  const colors = {
    ready: "#15803d",
    loading: "#1d4ed8",
    error: "#c2410c",
  };
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

  signalsGrid.innerHTML = signals
    .map(
      (signal) => `
        <div class="signal ${signal.active ? "active" : ""}">
          <span>${escapeHtml(signal.label)}</span>
          <strong>${escapeHtml(signal.value)}</strong>
        </div>
      `,
    )
    .join("");
}

function drawRadar(items = []) {
  const ctx = riskRadar.getContext("2d");
  const width = riskRadar.width;
  const height = riskRadar.height;
  const centerX = width / 2;
  const centerY = height / 2 + 6;
  const radius = Math.min(width, height) * 0.32;
  const sides = Math.max(items.length, 5);

  ctx.clearRect(0, 0, width, height);
  ctx.lineWidth = 1;
  ctx.font = "12px Inter, system-ui, sans-serif";
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
    ctx.strokeStyle = ring === 4 ? "#bac8d4" : "#dbe4eb";
    ctx.stroke();
  }

  items.forEach((item, i) => {
    const angle = -Math.PI / 2 + (i * Math.PI * 2) / sides;
    const endX = centerX + Math.cos(angle) * radius;
    const endY = centerY + Math.sin(angle) * radius;
    ctx.beginPath();
    ctx.moveTo(centerX, centerY);
    ctx.lineTo(endX, endY);
    ctx.strokeStyle = "#e1e8ee";
    ctx.stroke();

    const labelX = centerX + Math.cos(angle) * (radius + 42);
    const labelY = centerY + Math.sin(angle) * (radius + 32);
    ctx.fillStyle = "#52616f";
    ctx.fillText(item.label.replace(" ", "\n").split("\n")[0], labelX, labelY);
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
    ctx.fillStyle = "rgba(15, 118, 110, 0.22)";
    ctx.strokeStyle = "#0f766e";
    ctx.lineWidth = 3;
    ctx.fill();
    ctx.stroke();
  }
}

function renderDrivers(insights = {}) {
  const drivers = insights.top_drivers || [];
  const notes = insights.notes || ["Run a scan to generate reliability notes."];

  if (!drivers.length) {
    driverList.innerHTML = '<p class="empty-state">Risk drivers will appear after analysis.</p>';
  } else {
    driverList.innerHTML = drivers
      .map(
        (driver) => `
          <div class="driver">
            <div>
              <strong>${escapeHtml(driver.label)}</strong>
              <span>${escapeHtml(driver.detail)}</span>
            </div>
            <div class="driver-bar" aria-label="${escapeHtml(driver.label)} ${percent(driver.value)}">
              <span style="width: ${Math.round((Number(driver.value) || 0) * 100)}%"></span>
            </div>
          </div>
        `,
      )
      .join("");
  }

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
  if (!items.length) {
    historyList.innerHTML = '<p class="empty-state">Scans will appear here during this Flask session.</p>';
    return;
  }

  historyList.innerHTML = items
    .map(
      (item) => `
        <div class="history-item">
          <span title="${escapeHtml(item.url)}">${escapeHtml(item.url)}</span>
          <span class="badge ${item.risk_band?.tone || "safe"}">${item.prediction}</span>
          <strong>${percent(item.scam_probability)}</strong>
        </div>
      `,
    )
    .join("");
}

async function refreshHistory() {
  const response = await fetch("/api/history");
  if (response.ok) {
    renderHistory(await response.json());
  }
}

async function analyze(url) {
  latestUrl = url;
  scanButton.disabled = true;
  scanButton.textContent = "Scanning";
  featurePanel.hidden = true;
  setStatus("Scraping page and running model prediction...", "loading");

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url, timeout: timeoutInput.value }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Prediction failed.");
    }
    applyVerdict(data);
    await refreshHistory();
    setStatus(data.error || "Prediction complete.", "ready");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    scanButton.disabled = false;
    scanButton.textContent = "Analyze";
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
    if (!response.ok) {
      throw new Error(data.error || "Feature preview failed.");
    }

    featureTable.innerHTML = Object.entries(data.features)
      .map(([key, value]) => `<div class="feature-row"><span>${escapeHtml(key)}</span><strong>${escapeHtml(value ?? 0)}</strong></div>`)
      .join("");
    featurePanel.hidden = false;
    setStatus("URL feature preview ready.", "ready");
  } catch (error) {
    setStatus(error.message, "error");
  }
}

timeoutInput.addEventListener("input", () => {
  timeoutValue.textContent = `${timeoutInput.value}s`;
});

form.addEventListener("submit", (event) => {
  event.preventDefault();
  analyze(urlInput.value.trim());
});

document.querySelectorAll(".examples button").forEach((button) => {
  button.addEventListener("click", () => {
    urlInput.value = button.dataset.url;
    analyze(button.dataset.url);
  });
});

previewButton.addEventListener("click", previewFeatures);
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
