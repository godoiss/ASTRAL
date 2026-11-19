"use strict";

const auditedObjects = [
  { id: "K00753.01", host: "KIC 10811496", disposition: "CANDIDATE", observation: "Cached Q3", ann01: 0.4147668955, ann02: 0.7588245273 },
  { id: "K00097.01", host: "KIC 5780885", disposition: "CONFIRMED", observation: "Cached Q3", ann01: 0.8280777101, ann02: 0.6251904964 },
  { id: "K07977.01", host: "KIC 9912977", disposition: "FALSE POSITIVE", observation: "Cached Q3", ann01: 6.4357e-25, ann02: 0.0589553453 },
];

const modelContent = {
  ann01: {
    title: "ANN-01 · catalog features",
    layers: [
      { label: "Six inputs", count: 6, roles: ["Orbital period", "Transit duration", "Transit depth", "Planet radius", "Model signal-to-noise", "Equilibrium temperature"] },
      { label: "Hidden 01", count: 4, roles: ["Learned feature transform", "Learned feature transform", "Learned feature transform", "Learned feature transform"] },
      { label: "Hidden 02", count: 3, roles: ["Learned latent representation", "Learned latent representation", "Learned latent representation"] },
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
    <tr tabindex="0" data-object="${object.id}" class="${index === 0 ? "is-selected" : ""}" aria-selected="${index === 0}">
      <td class="mono">${object.id}</td><td>${object.host}</td>
      <td><span class="catalog-tag ${classNameForDisposition(object.disposition)}">${object.disposition === "CANDIDATE" ? "◇" : object.disposition === "CONFIRMED" ? "✓" : "×"} ${object.disposition}</span></td>
      <td>${object.observation}</td>
    </tr>`).join("");
  body.querySelectorAll("tr").forEach((row) => {
    const select = () => selectObservatoryObject(row.dataset.object);
    row.addEventListener("click", select);
    row.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); select(); } });
  });
}

function selectObservatoryObject(id) {
  const object = auditedObjects.find((item) => item.id === id);
  if (!object) return;
  document.querySelector("#observatory-object").textContent = object.id;
  document.querySelector("#observatory-host").textContent = object.host;
  const disposition = document.querySelector("#observatory-disposition");
  disposition.className = `disposition ${classNameForDisposition(object.disposition)}`;
  disposition.textContent = `${object.disposition === "CANDIDATE" ? "◇" : object.disposition === "CONFIRMED" ? "✓" : "×"} ${object.disposition}`;
  document.querySelector("#observatory-ann01").textContent = object.ann01.toFixed(4);
  document.querySelector("#observatory-ann02").textContent = object.ann02.toFixed(4);
  document.querySelectorAll("#object-table-body tr").forEach((row) => {
    const selected = row.dataset.object === id;
    row.classList.toggle("is-selected", selected);
    row.setAttribute("aria-selected", String(selected));
  });
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

function updateNeuralScore() {
  const object = auditedObjects.find((item) => item.id === document.querySelector("#neural-object").value);
  document.querySelector("#neural-score-label").textContent = `${activeModel.toUpperCase().replace("ANN", "ANN-")} stored score`;
  document.querySelector("#neural-score").textContent = object[activeModel].toFixed(4);
}

document.querySelectorAll("[data-model]").forEach((button) => button.addEventListener("click", () => renderTopology(button.dataset.model)));
document.querySelector("#neural-object").addEventListener("change", updateNeuralScore);

const dialog = document.querySelector("#expedition-dialog");
document.querySelector("#expedition-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const fields = [
    ["Mission", document.querySelector("#mission").selectedOptions[0].text],
    ["Sample", document.querySelector("#sample").selectedOptions[0].text],
    ["Strategy", document.querySelector("#strategy").selectedOptions[0].text],
    ["Detection method", document.querySelector("#method").selectedOptions[0].text],
  ];
  document.querySelector("#plan-summary").innerHTML = fields.map(([name, value]) => `<div><dt>${name}</dt><dd>${value}</dd></div>`).join("");
  dialog.showModal();
});
document.querySelector("#dialog-close").addEventListener("click", () => dialog.close());
dialog.addEventListener("click", (event) => { if (event.target === dialog) dialog.close(); });

renderObjectTable();
renderTopology("ann01");

const requestedDirection = new URLSearchParams(window.location.search).get("direction");
const directionButton = document.querySelector(`.direction-switcher [data-direction="${requestedDirection}"]`);
if (directionButton) directionButton.click();
const requestedView = window.location.hash.slice(1);
if (document.querySelector(`#view-${requestedView}`)) showView(requestedView, false);
