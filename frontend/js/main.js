import { api } from "./api.js";
import { initAnalyze } from "./analyze.js";
import { initHistory, loadHistory } from "./history.js";
import { loadDashboard } from "./dashboard.js";
import { initDiseases, loadDiseases } from "./diseases.js";

const loaders = { analyze: () => {}, history: loadHistory, dashboard: loadDashboard, diseases: loadDiseases };

function route() {
  const view = location.hash.slice(1) in loaders ? location.hash.slice(1) : "analyze";
  for (const section of document.querySelectorAll(".view")) section.hidden = section.id !== `view-${view}`;
  for (const link of document.querySelectorAll("nav a")) link.classList.toggle("active", link.dataset.view === view);
  loaders[view]();
}

async function checkApi() {
  const banner = document.getElementById("api-status");
  try {
    const health = await api.health();
    if (!health.model_loaded) {
      banner.textContent = "Le serveur est joignable mais le modèle n'est pas chargé : les analyses sont indisponibles.";
      banner.hidden = false;
    }
  } catch (error) {
    banner.textContent = error.message;
    banner.hidden = false;
  }
}

initAnalyze();
initHistory();
initDiseases();
window.addEventListener("hashchange", route);
route();
checkApi();
