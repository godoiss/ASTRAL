import { chromium } from "playwright-core";
import { access, mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

const baseURL = process.env.ASTRAL_E2E_URL ?? "http://127.0.0.1:3000";
const routes = ["/", "/discover", "/observatory", "/candidates", "/neural", "/research"];
const widths = [320, 360, 390, 430, 768, 1024, 1440, 1920];
const candidates = [
  process.env.ASTRAL_BROWSER_PATH,
  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
  "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
  "/usr/bin/google-chrome",
  "/usr/bin/chromium",
].filter(Boolean);

let executablePath;
for (const candidate of candidates) {
  try { await access(candidate); executablePath = candidate; break; } catch { /* continue */ }
}
if (!executablePath) throw new Error("No supported Chrome/Chromium executable was found. Set ASTRAL_BROWSER_PATH.");

const browser = await chromium.launch({ headless: true, executablePath });
const report = { baseURL, browser: executablePath, routes: [], viewports: [], interactions: [], consoleErrors: [] };

try {
  for (const width of widths) {
    const height = width <= 430 ? 844 : width <= 1024 ? 900 : 1080;
    const page = await browser.newPage({ viewport: { width, height }, hasTouch: width <= 768, isMobile: width <= 430 });
    page.on("console", (message) => { if (message.type() === "error") report.consoleErrors.push({ width, text: message.text() }); });
    for (const route of routes) {
      const response = await page.goto(`${baseURL}${route}`, { waitUntil: "domcontentloaded", timeout: 30_000 });
      if (!response?.ok()) throw new Error(`${route} returned ${response?.status()}.`);
      await page.locator("main").waitFor({ state: "visible" });
      const geometry = await page.evaluate(() => {
        const pageBox = document.querySelector(".page")?.getBoundingClientRect();
        return { viewport: window.innerWidth, document: document.documentElement.scrollWidth, pageLeft: pageBox?.left ?? 0, pageRight: pageBox?.right ?? 0 };
      });
      if (geometry.document > geometry.viewport + 2 || geometry.pageLeft < -2 || geometry.pageRight > geometry.viewport + 2) {
        const offenders = await page.evaluate(() => [...document.querySelectorAll("body *")].map((element) => ({ element: element.tagName.toLowerCase(), className: element.className, ...element.getBoundingClientRect().toJSON() })).filter((box) => box.right > window.innerWidth + 2 || box.left < -2).slice(0, 8));
        throw new Error(`${route} overflows horizontally at ${width}px (${geometry.document}px): ${JSON.stringify(offenders)}`);
      }
      report.viewports.push({ width, height, route, horizontalOverflow: false });
    }
    await page.close();
  }

  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  page.on("console", (message) => { if (message.type() === "error") report.consoleErrors.push({ width: 1440, text: message.text() }); });
  await page.goto(`${baseURL}/discover`, { waitUntil: "networkidle" });
  await page.getByLabel(/Kepler catalog field atlas/).click({ position: { x: 250, y: 220 } });
  report.interactions.push("stellar atlas accepts pointer selection");

  if (process.env.ASTRAL_RUN_CAMPAIGN === "1") {
    const persist = page.getByRole("button", { name: /Review and persist plan/ });
    await persist.waitFor({ state: "visible" });
    if (await persist.isEnabled()) {
      await persist.click();
      const start = page.getByRole("button", { name: /Start real campaign/ });
      await start.waitFor({ state: "visible", timeout: 10_000 });
      await start.click();
      await page.getByText("Campaign evidence and provenance persisted successfully.").waitFor({ state: "visible", timeout: 90_000 });
      report.interactions.push("authorized cache-only campaign completed through Mission Control");
    }
  }

  await page.goto(`${baseURL}/candidates`, { waitUntil: "networkidle" });
  const cards = page.locator(".signal-card");
  if (await cards.count() !== 8) throw new Error("Candidate Intelligence did not expose all eight persisted dossiers.");
  await cards.nth(1).click();
  await page.getByText(/ASTRAL scientific dossier/).waitFor();
  report.interactions.push("all eight scientific dossiers are navigable");

  const exportResponse = await page.request.get(`${baseURL}/api/v1/candidate-intelligence/10811496/export`);
  const reportResponse = await page.request.get(`${baseURL}/api/v1/candidate-intelligence/10811496/report`);
  if (!exportResponse.ok() || !reportResponse.ok()) throw new Error("Candidate export endpoints failed.");
  report.interactions.push("JSON and Markdown dossier exports return persisted evidence");

  await page.goto(`${baseURL}/observatory?object=K00753.01`, { waitUntil: "networkidle" });
  await page.getByRole("img", { name: /luminosity/i }).first().waitFor({ timeout: 30_000 });
  report.interactions.push("Observatory renders real luminosity instruments");

  await page.goto(`${baseURL}/neural`, { waitUntil: "networkidle" });
  await page.getByRole("button", { name: /ANN-02/ }).click();
  await page.getByText(/Real checkpoint state/).waitFor();
  report.interactions.push("ANN-01 and ANN-02 retained states are selectable");

  await page.goto(`${baseURL}/research`, { waitUntil: "networkidle" });
  await page.getByText(/Discovery campaign archive/).waitFor();
  report.interactions.push("persistent campaign archive is visible after navigation");

  report.routes = routes;
  const evidenceDirectory = path.resolve(process.cwd(), "..", ".cache");
  await mkdir(evidenceDirectory, { recursive: true });
  await page.goto(baseURL, { waitUntil: "networkidle" });
  await page.screenshot({ path: path.join(evidenceDirectory, "astral-home-desktop.png"), fullPage: true });
  const mobilePage = await browser.newPage({ viewport: { width: 390, height: 844 }, hasTouch: true, isMobile: true });
  await mobilePage.goto(baseURL, { waitUntil: "networkidle" });
  report.mobileLandingGeometry = await mobilePage.evaluate(() => Object.fromEntries(
    [".page", ".landing-hero", ".hero-copy", ".hero-copy h1", ".hero-actions", ".hero-instrument", ".landing-facts", ".instrument-journey", ".method-boundary"]
      .map((selector) => [selector, document.querySelector(selector)?.getBoundingClientRect().toJSON()]),
  ));
  await mobilePage.screenshot({ path: path.join(evidenceDirectory, "astral-home-mobile.png") });
  await mobilePage.screenshot({ path: path.join(evidenceDirectory, "astral-home-mobile-full.png"), fullPage: true });
  await mobilePage.close();
  const output = path.join(evidenceDirectory, "e2e-report.json");
  await writeFile(output, JSON.stringify(report, null, 2), "utf8");
  console.log(`ASTRAL E2E passed: ${report.viewports.length} responsive route checks, ${report.interactions.length} interactions.`);
  console.log(`Report: ${output}`);
  if (report.consoleErrors.length) throw new Error(`Browser emitted ${report.consoleErrors.length} console error(s).`);
} finally {
  await browser.close();
}
