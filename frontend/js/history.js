import { api } from "./api.js";
import { clear, el, formatDate, percent, statusBadge } from "./dom.js";

const PAGE_SIZE = 10;
let offset = 0;

async function showDetail(id) {
  const dialog = document.getElementById("detail-dialog");
  const body = document.getElementById("detail-body");
  clear(body);
  try {
    const item = await api.analysis(id);
    body.append(
      el("img", { src: api.imageUrl(item.image_url), alt: "Image analysée" }),
      el("h3", { text: `${item.plant} — ${item.disease}` }),
      el("p", {}, statusBadge(item)),
      el("p", { text: `Confiance : ${percent(item.confidence)}` }),
      el("p", { class: "muted", text: `Analyse n°${item.id} du ${formatDate(item.created_at)}` }));
  } catch (error) {
    body.append(el("p", { class: "error", text: error.message }));
  }
  dialog.showModal();
}

export async function loadHistory() {
  const list = document.getElementById("history-list");
  const info = document.getElementById("page-info");
  const prev = document.getElementById("prev-page");
  const next = document.getElementById("next-page");
  clear(list);
  try {
    const page = await api.analyses(PAGE_SIZE, offset);
    if (page.items.length === 0 && offset > 0) { offset = Math.max(0, offset - PAGE_SIZE); return loadHistory(); }
    if (page.total === 0) list.append(el("p", { class: "muted", text: "Aucune analyse pour le moment." }));
    for (const item of page.items) {
      list.append(el("div", { class: "history-item" },
        el("img", { src: api.imageUrl(item.image_url), alt: "", loading: "lazy" }),
        el("div", {},
          el("strong", { text: `${item.plant} — ${item.disease}` }), " ", statusBadge(item),
          el("div", { class: "muted", text: `${percent(item.confidence)} · ${formatDate(item.created_at)}` })),
        el("div", { class: "actions" },
          el("button", { class: "btn btn-secondary", type: "button", text: "Détails", onclick: () => showDetail(item.id) }),
          el("button", { class: "btn btn-danger", type: "button", text: "Supprimer", onclick: async () => {
            if (!confirm("Supprimer cette analyse et son image ?")) return;
            try { await api.remove(item.id); } catch (error) { alert(error.message); }
            loadHistory();
          } }))));
    }
    const pages = Math.max(1, Math.ceil(page.total / PAGE_SIZE));
    info.textContent = `Page ${offset / PAGE_SIZE + 1} / ${pages} (${page.total} analyses)`;
    prev.disabled = offset === 0;
    next.disabled = offset + PAGE_SIZE >= page.total;
  } catch (error) {
    list.append(el("p", { class: "error", text: error.message }));
  }
}

export function initHistory() {
  document.getElementById("prev-page").addEventListener("click", () => { offset = Math.max(0, offset - PAGE_SIZE); loadHistory(); });
  document.getElementById("next-page").addEventListener("click", () => { offset += PAGE_SIZE; loadHistory(); });
  document.getElementById("detail-close").addEventListener("click", () => document.getElementById("detail-dialog").close());
}
