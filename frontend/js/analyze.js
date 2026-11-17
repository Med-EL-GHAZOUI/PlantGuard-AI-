import { api } from "./api.js";
import { bar, clear, el, percent, statusBadge } from "./dom.js";

const MAX_BYTES = 5 * 1024 * 1024;
const ALLOWED = ["image/jpeg", "image/png"];

let diseaseInfo = null; // cache des fiches maladies

async function infoFor(className) {
  if (!diseaseInfo) {
    try { diseaseInfo = await api.diseases(); } catch { diseaseInfo = []; }
  }
  return diseaseInfo.find((d) => d.class_name === className) || null;
}

export function renderResult(container, prediction, imageSrc, info) {
  clear(container);
  const rows = prediction.top_k.map((item) =>
    el("div", { class: "row" },
      el("span", { class: "label", title: item.class_name, text: item.class_name.replaceAll("_", " ") }),
      bar(item.confidence),
      el("span", { text: percent(item.confidence) })));
  container.append(el("div", { class: "card result" },
    el("img", { class: "result-img", src: imageSrc, alt: "Image analysée" }),
    el("h3", { text: `${prediction.plant} — ${prediction.disease}` }),
    el("p", {}, statusBadge(prediction), " ", el("strong", { text: `Confiance : ${percent(prediction.confidence)}` })),
    prediction.low_confidence && el("p", { class: "muted",
      text: "La confiance est faible : le résultat est peu fiable. Essayez une photo plus nette d'une seule feuille." }),
    info && info.description && el("p", { text: info.description }),
    info && info.advice && el("p", {}, el("strong", { text: "Conseil : " }), info.advice),
    el("h4", { text: "Autres hypothèses" }), rows,
    el("p", { class: "muted", text: prediction.disclaimer })));
}

export function initAnalyze() {
  const input = document.getElementById("file-input");
  const zone = document.getElementById("dropzone");
  const preview = document.getElementById("preview");
  const hint = document.getElementById("dropzone-hint");
  const button = document.getElementById("analyze-btn");
  const errorBox = document.getElementById("form-error");
  const result = document.getElementById("result");
  let file = null;
  let previewUrl = null;

  const showError = (message) => { errorBox.textContent = message; errorBox.hidden = !message; };

  function select(candidate) {
    showError("");
    if (!candidate) return;
    if (!ALLOWED.includes(candidate.type)) return showError("Format non accepté : utilisez JPG, JPEG ou PNG.");
    if (candidate.size > MAX_BYTES) return showError("Fichier trop volumineux (5 Mo maximum).");
    file = candidate;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(file);
    preview.src = previewUrl;
    preview.hidden = false;
    hint.hidden = true;
    button.disabled = false;
    clear(result);
  }

  input.addEventListener("change", () => select(input.files[0]));
  zone.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
  zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("drag"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("drag"));
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("drag");
    select(e.dataTransfer.files[0]);
  });

  button.addEventListener("click", async () => {
    if (!file) return;
    button.disabled = true;
    showError("");
    clear(result);
    result.append(el("p", {}, el("span", { class: "spinner" }), "Analyse en cours…"));
    try {
      const prediction = await api.predict(file);
      renderResult(result, prediction, previewUrl, await infoFor(prediction.class_name));
    } catch (error) {
      clear(result);
      showError(error.message);
    } finally {
      button.disabled = false;
    }
  });
}
