"use strict";

let auditedObjects = [];
let activeCampaign = null;

const modelContent = {
  ann01: {
    title: "ANN-01 · catalog features",
    layers: [
      { label: "Six inputs", count: 6, roles: ["Orbital period", "Transit duration", "Transit depth", "Planet radius", "Model signal-to-noise", "Equilibrium temperature"] },
      { label: "Hidden 01", count: 8, roles: Array(8).fill("Learned ReLU feature transform") },
      { label: "Hidden 02", count: 4, roles: Array(4).fill("Learned ReLU latent representation") },
      { label: "Output", count: 1, roles: ["Binary classification score"] },
    ],
    methods: [
      ["Permutation importance", "Measures validation-metric change when one input feature is shuffled."],
      ["Local feature differences", "Compares a selected record against a training-derived reference without claiming causality."],
    ],
  },
  ann02: {
    title: "ANN-02 · light-curve morphology",
    layers: [
      { label: "Global view", count: 3, roles: ["Global flux vector", "Global missing-data mask", "Phase context"] },
      { label: "Convolution", count: 4, roles: ["Learned temporal filter", "Learned temporal filter", "Learned temporal filter", "Learned temporal filter"] },
      { label: "Local view", count: 3, roles: ["Local transit vector", "Local missing-data mask", "Transit-window context"] },
      { label: "Output", count: 1, roles: ["Binary classification score"] },
    ],
    methods: [
      ["Gradient × input", "Highlights input regions with large first-order contribution to the current score."],
      ["Occlusion sensitivity", "Measures score change while controlled windows of the signal are masked."],
      ["Feature maps", "Inspects learned convolutional responses for persisted compatible observations."],
    ],
  },
};

const navItems = [...document.querySelectorAll(".nav-item[data-view]")];
const views = [...document.querySelectorAll(".view")];
const routeName = document.querySelector("#route-name");
const mainContent = document.querySelector("#main-content");

function showView(name, focus = true) {
  const target = document.querySelector(`#view-${name}`);
  if (!target) return;
  views.forEach((view) => view.classList.toggle("is-active", view === target));
  navItems.forEach((item) => {
    const selected = item.dataset.view === name;
    item.classList.toggle("is-active", selected);
    if (selected) item.setAttribute("aria-current", "page"); else item.removeAttribute("aria-current");
  });
  routeName.textContent = target.dataset.title;
  document.title = `ASTRAL — ${target.dataset.title}`;
  if (window.location.hash !== `#${name}`) window.history.replaceState(null, "", `#${name}`);
  document.body.classList.remove("nav-open");
  document.querySelector("#menu-button").setAttribute("aria-expanded", "false");
  if (focus) mainContent.focus({ preventScroll: true });
  window.scrollTo({ top: 0, behavior: "smooth" });
}

navItems.forEach((item) => item.addEventListener("click", () => showView(item.dataset.view)));

const menuButton = document.querySelector("#menu-button");
menuButton.addEventListener("click", () => {
  const open = document.body.classList.toggle("nav-open");
  menuButton.setAttribute("aria-expanded", String(open));
});
document.querySelector("#nav-scrim").addEventListener("click", () => {
  document.body.classList.remove("nav-open");
  menuButton.setAttribute("aria-expanded", "false");
});

document.addEventListener("keydown", (event) => {
  if (event.target.matches("input, select, button") || event.ctrlKey || event.metaKey || event.altKey) return;
  const index = Number(event.key) - 1;
  if (index >= 0 && index < navItems.length) showView(navItems[index].dataset.view);
});

document.querySelectorAll(".direction-switcher [data-direction]").forEach((button) => {
  button.addEventListener("click", () => {
    const direction = button.dataset.direction;
    document.body.dataset.direction = direction;
    document.querySelectorAll(".direction-switcher [data-direction]").forEach((choice) => choice.setAttribute("aria-pressed", String(choice === button)));
  });
});

function classNameForDisposition(disposition) {
  return disposition.toLowerCase().replaceAll(" ", "-");
}

function renderObjectTable() {
  const body = document.querySelector("#object-table-body");
  body.innerHTML = auditedObjects.map((object, index) => `
    <tr tabindex="0" data-object="${escapeHtml(object.id)}" class="${index === 0 ? "is-selected" : ""}" aria-selected="${index === 0}">
      <td class="mono">${escapeHtml(object.id)}</td><td>${escapeHtml(object.host)}</td>
      <td><span class="catalog-tag ${classNameForDisposition(object.disposition)}">${object.disposition === "CANDIDATE" ? "◇" : object.disposition === "CONFIRMED" ? "✓" : "×"} ${escapeHtml(object.disposition)}</span></td>
      <td>${escapeHtml(object.observation)}</td>
    </tr>`).join("");
  body.querySelectorAll("tr").forEach((row) => {
    const select = () => selectObservatoryObject(row.dataset.object);
    row.addEventListener("click", select);
    row.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); select(); } });
  });
}

async function selectObservatoryObject(id) {
  const object = auditedObjects.find((item) => item.id === id);
  if (!object) return;
  document.querySelector("#observatory-object").textContent = object.id;
  document.querySelector("#observatory-host").textContent = object.host;
  const disposition = document.querySelector("#observatory-disposition");
  disposition.className = `disposition ${classNameForDisposition(object.disposition)}`;
  disposition.textContent = `${object.disposition === "CANDIDATE" ? "◇" : object.disposition === "CONFIRMED" ? "✓" : "×"} ${object.disposition}`;
  document.querySelector("#observatory-ann01").textContent = "Loading";
  document.querySelector("#observatory-ann02").textContent = "Loading";
  document.querySelectorAll("#object-table-body tr").forEach((row) => {
    const selected = row.dataset.object === id;
    row.classList.toggle("is-selected", selected);
    row.setAttribute("aria-selected", String(selected));
  });
  await Promise.allSettled([
    loadInference("ann-01", object.id, "#observatory-ann01"),
    loadInference("ann-02", object.id, "#observatory-ann02"),
    loadObservationCurve(object.id),
  ]);
}

let activeModel = "ann01";

function renderTopology(modelKey) {
  activeModel = modelKey;
  const model = modelContent[modelKey];
  document.querySelector("#topology-title").textContent = model.title;
  const topology = document.querySelector("#topology");
  topology.innerHTML = model.layers.map((layer, layerIndex) => `
    <div class="layer"><span class="layer-label">${layer.label}</span>${layer.roles.map((role, nodeIndex) => `<button class="node" type="button" data-role="${role}" aria-label="${layer.label}, node ${nodeIndex + 1}: ${role}">${layerIndex + 1}.${nodeIndex + 1}</button>`).join("")}</div>`).join("");
  const description = document.querySelector("#node-description");
  topology.querySelectorAll(".node").forEach((node) => {
    const explain = () => { description.textContent = node.dataset.role; };
    node.addEventListener("mouseenter", explain);
    node.addEventListener("focus", explain);
  });
  document.querySelector("#explanation-title").textContent = `${modelKey.toUpperCase().replace("ANN", "ANN-")} evidence`;
  document.querySelector("#explanation-methods").innerHTML = model.methods.map(([name, copy]) => `<article class="method-card"><strong>${name}</strong><p>${copy}</p></article>`).join("");
  document.querySelectorAll("[data-model]").forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.model === modelKey)));
  updateNeuralScore();
}

async function updateNeuralScore() {
  const object = auditedObjects.find((item) => item.id === document.querySelector("#neural-object").value);
  if (!object) return;
  document.querySelector("#neural-score-label").textContent = `${activeModel.toUpperCase().replace("ANN", "ANN-")} stored score`;
  const modelId = activeModel.replace("ann", "ann-");
  document.querySelector("#neural-score").textContent = "Loading";
  await loadInference(modelId, object.id, "#neural-score");
}

document.querySelectorAll("[data-model]").forEach((button) => button.addEventListener("click", () => renderTopology(button.dataset.model)));
document.querySelector("#neural-object").addEventListener("change", updateNeuralScore);

const dialog = document.querySelector("#expedition-dialog");
document.querySelector("#expedition-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const sampleSize = Number(document.querySelector("#sample").value);
  document.querySelector("#planner-state").textContent = "Validating protected host selection";
  let plan;
  try {
    plan = (await api(`/api/v1/discovery/plan?sample_size=${sampleSize}`)).data;
  } catch (error) {
    document.querySelector("#planner-state").textContent = "Plan unavailable";
    showInterfaceError(error);
    return;
  }
  const fields = [
    ["Mission", "Kepler"],
    ["Protected targets", String(plan.sample_size)],
    ["Cached now", String(plan.cached_targets)],
    ["Frozen-host overlap", String(plan.protected_host_overlap.length)],
    ["Network acquisition", "Disabled"],
    ["Detection method", "Astropy Box Least Squares"],
  ];
  document.querySelector("#plan-summary").innerHTML = fields.map(([name, value]) => `<div><dt>${name}</dt><dd>${value}</dd></div>`).join("");
  document.querySelector("#launch-expedition").dataset.sampleSize = String(sampleSize);
  document.querySelector("#launch-expedition").disabled = !plan.execution_enabled;
  document.querySelector("#launch-expedition").textContent = plan.execution_enabled ? "Start real campaign" : "Execution disabled on this deployment";
  document.querySelector("#planner-state").textContent = `${plan.cached_targets}/${plan.sample_size} targets cached · protected overlap ${plan.protected_host_overlap.length}`;
  dialog.showModal();
});
document.querySelector("#dialog-close").addEventListener("click", () => dialog.close());
document.querySelector("#dialog-close-secondary").addEventListener("click", () => dialog.close());
dialog.addEventListener("click", (event) => { if (event.target === dialog) dialog.close(); });

document.querySelector("#launch-expedition").addEventListener("click", async (event) => {
  const button = event.currentTarget;
  const sampleSize = Number(button.dataset.sampleSize || 10);
  button.disabled = true;
  button.textContent = "Starting…";
  try {
    const job = await api("/api/v1/discovery/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sample_size: sampleSize }),
    });
    dialog.close();
    await monitorDiscoveryJob(job.job_id);
  } catch (error) {
    showInterfaceError(error);
  } finally {
    button.disabled = false;
    button.textContent = "Start real campaign";
  }
});
document.querySelector("#view-campaign").addEventListener("click", () => {
  document.querySelector("#campaign-workspace").scrollIntoView({ behavior: "smooth" });
});

const requestedView = window.location.hash.slice(1);
if (document.querySelector(`#view-${requestedView}`)) showView(requestedView, false);

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  })[character]);
}

async function api(path, options = {}) {
  const response = await fetch(path, { ...options, headers: { Accept: "application/json", ...(options.headers || {}) } });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload?.error?.message || `Scientific API returned ${response.status}`);
  return payload;
}

function showInterfaceError(error) {
  const status = document.querySelector("#api-status");
  status.textContent = "Scientific API unavailable";
  status.classList.add("is-error");
  console.error(error);
}

async function loadInference(modelId, objectId, selector) {
  const element = document.querySelector(selector);
  try {
    const result = await api(`/api/v1/inference/${modelId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ object_id: objectId }),
    });
    element.textContent = Number(result.score).toFixed(4);
  } catch (_error) {
    element.textContent = "Incompatible";
  }
}

function chartPath(xValues, yValues, width = 1000, height = 300) {
  const points = xValues.map((x, index) => [Number(x), Number(yValues[index])]).filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y));
  if (points.length < 2) return "";
  const xs = points.map(([x]) => x); const ys = points.map(([, y]) => y);
  const xMin = Math.min(...xs); const xMax = Math.max(...xs); const yMin = Math.min(...ys); const yMax = Math.max(...ys);
  const xRange = xMax - xMin || 1; const yRange = yMax - yMin || 1;
  return points.map(([x, y], index) => `${index ? "L" : "M"}${40 + ((x - xMin) / xRange) * (width - 70)} ${20 + (1 - (y - yMin) / yRange) * (height - 55)}`).join(" ");
}

function renderChart(svg, series, label) {
  if (!svg) return;
  const paths = series.map((item) => `<path d="${chartPath(item.x, item.y)}" class="${item.className || "chart-primary"}" vector-effect="non-scaling-stroke"/>`).join("");
  svg.setAttribute("viewBox", "0 0 1000 300");
  svg.innerHTML = `<title>${escapeHtml(label)}</title><path class="chart-grid" d="M40 20V265H970M40 81H970M40 142H970M40 203H970"/>${paths}`;
}

async function loadObservationCurve(objectId) {
  const payload = await api(`/api/v1/objects/${encodeURIComponent(objectId)}/light-curve?view=processed&limit=1600`);
  const curve = payload.data;
  renderChart(document.querySelector("#observation-chart"), [{ x: curve.data.time_bkjd, y: curve.data.detrended_flux }], `Processed luminosity for ${objectId}`);
  document.querySelector("#observation-chart-title").textContent = `${objectId} · processed Kepler luminosity`;
  document.querySelector("#observation-chart-state").textContent = curve.downsampled ? `${curve.records_returned}/${curve.records_total} points` : `${curve.records_total} points`;
  document.querySelector("#observation-chart-caption").textContent = "Observed persisted cadences. Browser rendering uses deterministic downsampling and never fills temporal gaps.";
}

async function loadObjects() {
  const payload = await api("/api/v1/objects?observed_only=true&limit=12");
  auditedObjects = payload.items.map((item) => ({
    id: item.object_id,
    host: `KIC ${item.kic}`,
    disposition: item.catalog_disposition,
    observation: item.has_cached_observation ? "Cached observation" : "Unavailable",
  }));
  document.querySelector("#object-count").textContent = `${auditedObjects.length} records`;
  const neuralSelect = document.querySelector("#neural-object");
  neuralSelect.innerHTML = auditedObjects.slice(0, 8).map((item) => `<option value="${escapeHtml(item.id)}">${escapeHtml(item.id)} · ${escapeHtml(item.disposition)}</option>`).join("");
  renderObjectTable();
  renderTopology("ann01");
  if (auditedObjects.length) await selectObservatoryObject(auditedObjects[0].id);
}

async function loadModelCatalog() {
  const payload = await api("/api/v1/models");
  const ann01 = payload.data.models.find((model) => model.id === "ann-01");
  const ann02 = payload.data.models.find((model) => model.id === "ann-02");
  if (ann01) modelContent.ann01.title = `${ann01.name} · ${ann01.trainable_parameters} trainable parameters`;
  if (ann02) modelContent.ann02.title = `${ann02.name} · ${ann02.trainable_parameters} trainable parameters`;
}

async function loadResearchMetrics() {
  const payload = await api("/api/v1/metrics/comparison");
  const values = {
    "ann01-accuracy": payload.data.ann01_metrics.accuracy,
    "ann02-accuracy": payload.data.ann02_metrics.accuracy,
    "ann01-balanced": payload.data.ann01_metrics.balanced_accuracy,
    "ann02-balanced": payload.data.ann02_metrics.balanced_accuracy,
  };
  Object.entries(values).forEach(([key, value]) => {
    const card = document.querySelector(`[data-metric="${key}"]`);
    if (!card) return;
    card.querySelector("strong").textContent = Number(value).toFixed(4);
    card.querySelector(".meter i").style.setProperty("--value", `${Number(value) * 100}%`);
  });
  document.querySelector("#research-cohort").textContent = `${payload.data.cohort.records} shared validation objects`;
  document.querySelector("#research-roc").textContent = `${payload.data.ann01_metrics.roc_auc.toFixed(4)} · validation only, not definitive`;
}

function campaignMetric(label, value, note) {
  return `<article class="campaign-metric panel"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong><small>${escapeHtml(note)}</small></article>`;
}

async function loadLatestCampaign(preferredId = null) {
  const page = await api("/api/v1/discovery/campaigns?limit=20");
  if (!page.items.length) throw new Error("No persisted discovery campaign is available.");
  const chosen = preferredId ? page.items.find((item) => item.campaign_id === preferredId) : page.items[0];
  const detail = await api(`/api/v1/discovery/campaigns/${encodeURIComponent((chosen || page.items[0]).campaign_id)}`);
  activeCampaign = detail.data;
  renderCampaign(activeCampaign);
}

function renderCampaign(detail) {
  const campaign = detail.campaign;
  document.querySelector("#campaign-provenance").textContent = `${campaign.campaign_id} · ${campaign.status} · persisted Kepler evidence`;
  document.querySelector("#campaign-metrics").innerHTML = [
    campaignMetric("Investigated", campaign.investigated_stars, "real Kepler stars"),
    campaignMetric("BLS screens", campaign.screening_detections, "not confirmations"),
    campaignMetric("Catalog associated", campaign.catalog_associated_detections, "local period relation"),
    campaignMetric("Inconclusive", campaign.unassociated_inconclusive_detections, "no novelty claim"),
    campaignMetric("Confirmed", 0, "scientific boundary"),
  ].join("");
  const sorted = [...detail.targets].sort((a, b) => b.depth_snr - a.depth_snr);
  document.querySelector("#campaign-console").innerHTML = `
    <header class="panel-header"><div><span class="eyebrow">CAMPAIGN TARGETS</span><h2>Signal screening ledger</h2></div><span class="state-chip">${escapeHtml(campaign.status)}</span></header>
    <div class="discovery-layout">
      <div class="table-wrap"><table><thead><tr><th>Host</th><th>Period</th><th>Depth SNR</th><th>Events</th><th>Evidence state</th></tr></thead><tbody>${sorted.map((item, index) => `
        <tr tabindex="0" data-discovery-kic="${item.kic}" class="${index === 0 ? "is-selected" : ""}"><td class="mono">KIC ${item.kic}</td><td>${item.period_days.toFixed(6)} d</td><td>${item.depth_snr.toFixed(2)}</td><td>${item.observed_events}</td><td>${escapeHtml(item.interpretation.replaceAll("_", " "))}</td></tr>`).join("")}</tbody></table></div>
      <aside class="signal-inspector" id="signal-inspector"><div class="loading-state">Select a measured signal.</div></aside>
    </div>`;
  document.querySelectorAll("[data-discovery-kic]").forEach((row) => {
    const select = () => selectDiscoveryTarget(Number(row.dataset.discoveryKic));
    row.addEventListener("click", select);
    row.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); select(); } });
  });
  renderCandidateEvidence(detail);
  if (sorted.length) selectDiscoveryTarget(sorted[0].kic);
}

async function selectDiscoveryTarget(kic) {
  const target = activeCampaign.targets.find((item) => item.kic === kic);
  document.querySelectorAll("[data-discovery-kic]").forEach((row) => row.classList.toggle("is-selected", Number(row.dataset.discoveryKic) === kic));
  const inspector = document.querySelector("#signal-inspector");
  inspector.innerHTML = `<div class="loading-state">Loading persisted arrays for KIC ${kic}…</div>`;
  const payload = await api(`/api/v1/discovery/campaigns/${encodeURIComponent(activeCampaign.campaign.campaign_id)}/targets/${kic}/series?limit=1600`);
  const data = payload.data;
  inspector.innerHTML = `
    <div class="signal-heading"><div><span class="eyebrow">MEASURED SIGNAL</span><h3>KIC ${kic}</h3></div><span class="state-chip">${escapeHtml(target.status)}</span></div>
    <dl class="signal-facts"><div><dt>Period</dt><dd>${target.period_days.toFixed(6)} d</dd></div><div><dt>Duration</dt><dd>${target.duration_hours.toFixed(2)} h</dd></div><div><dt>Depth</dt><dd>${target.depth_ppm.toFixed(1)} ppm</dd></div><div><dt>Coverage</dt><dd>${(target.event_coverage * 100).toFixed(1)}%</dd></div></dl>
    <div class="mini-chart"><span>Detrended luminosity</span><svg id="discovery-light-chart"></svg></div>
    <div class="mini-chart"><span>BLS periodogram</span><svg id="discovery-period-chart"></svg></div>
    <div class="mini-chart"><span>Phase-folded evidence</span><svg id="discovery-phase-chart"></svg></div>
    <div class="evidence-note ${target.catalog_period_association ? "is-associated" : ""}"><strong>${target.catalog_period_association ? "Catalog period association" : "Unassociated · inconclusive"}</strong><span>${target.possible_eclipsing_binary ? "Eclipsing-binary compatible diagnostic present." : "No automatic confirmation is performed."}</span></div>`;
  renderChart(document.querySelector("#discovery-light-chart"), [{ x: data.time_bkjd, y: data.detrended_flux }], `Detrended luminosity KIC ${kic}`);
  renderChart(document.querySelector("#discovery-period-chart"), [{ x: data.periods, y: data.powers, className: "chart-signal" }], `BLS periodogram KIC ${kic}`);
  renderChart(document.querySelector("#discovery-phase-chart"), [
    { x: data.phase, y: data.phase_flux },
    { x: data.phase_bin_centers, y: data.phase_bin_flux, className: "chart-accent" },
  ], `Phase-folded signal KIC ${kic}`);
}

function renderCandidateEvidence(detail) {
  const inconclusive = detail.targets.filter((target) => target.interpretation === "unassociated_inconclusive_signal");
  document.querySelector("#candidate-count").textContent = `${inconclusive.length} signals`;
  document.querySelector("#candidate-table-body").innerHTML = inconclusive.map((item) => `<tr><td class="mono">KIC ${item.kic}</td><td>${item.period_days.toFixed(6)} d</td><td>${item.depth_snr.toFixed(2)}</td><td>${item.observed_events}</td><td>Unassociated · inconclusive</td></tr>`).join("") || `<tr><td colspan="5">No unassociated screening signal in this campaign.</td></tr>`;
}

async function monitorDiscoveryJob(jobId) {
  showView("discover", false);
  const state = document.querySelector("#planner-state");
  while (true) {
    const job = await api(`/api/v1/discovery/jobs/${jobId}`);
    state.textContent = `${job.stage.toUpperCase()} · ${job.current}/${job.total} · ${job.message}`;
    if (job.status === "completed") {
      await loadLatestCampaign(job.campaign_id);
      document.querySelector("#campaign-workspace").scrollIntoView({ behavior: "smooth" });
      return;
    }
    if (job.status === "failed") throw new Error(`Campaign stopped safely: ${job.error_code || "unknown error"}`);
    await new Promise((resolve) => window.setTimeout(resolve, 400));
  }
}

async function initializeApplication() {
  try {
    const health = await api("/health");
    document.querySelector("#api-status").textContent = `Scientific API ${health.version} · live`;
    document.querySelector("#api-status").classList.add("is-live");
    await loadModelCatalog();
    await Promise.all([loadObjects(), loadLatestCampaign(), loadResearchMetrics()]);
    document.querySelector("#planner-state").textContent = "Ready · cache-only execution";
    if (new URLSearchParams(window.location.search).get("section") === "campaign") {
      document.querySelector("#campaign-workspace").scrollIntoView({ behavior: "instant" });
    }
  } catch (error) {
    showInterfaceError(error);
    document.querySelector("#campaign-console").innerHTML = `<div class="loading-state">Scientific evidence is unavailable. Start the FastAPI service and retry.</div>`;
  }
}

initializeApplication();
