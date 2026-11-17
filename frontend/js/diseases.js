import { api } from "./api.js";
import { clear, el } from "./dom.js";

let all = null;

function render(query) {
  const list = document.getElementById("disease-list");
  clear(list);
  const q = query.trim().toLowerCase();
  const shown = all.filter((d) => `${d.plant} ${d.disease}`.toLowerCase().includes(q));
  if (shown.length === 0) list.append(el("p", { class: "muted", text: "Aucun résultat." }));
  for (const d of shown) {
    list.append(el("div", { class: "card" },
      el("strong", { text: d.plant }), " — ", d.disease,
      el("p", { class: "muted", text: d.description || "Fiche non encore rédigée pour cette classe." }),
      d.advice && el("p", {}, el("strong", { text: "Conseil : " }), d.advice)));
  }
}

export async function loadDiseases() {
  const list = document.getElementById("disease-list");
  try {
    all = await api.diseases();
    if (all.length === 0) {
      clear(list);
      list.append(el("p", { class: "muted", text: "Liste indisponible : le modèle n'est pas chargé côté serveur." }));
      return;
    }
    render(document.getElementById("disease-search").value);
  } catch (error) {
    clear(list);
    list.append(el("p", { class: "error", text: error.message }));
  }
}

export function initDiseases() {
  document.getElementById("disease-search").addEventListener("input", (e) => { if (all) render(e.target.value); });
}
