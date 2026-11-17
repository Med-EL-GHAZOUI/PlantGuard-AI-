import { api } from "./api.js";
import { bar, clear, el } from "./dom.js";

const SVG_NS = "http://www.w3.org/2000/svg";

function barList(items, labelKey, emptyText) {
  if (items.length === 0) return el("p", { class: "muted", text: emptyText });
  const max = Math.max(...items.map((i) => i.count));
  return el("div", {}, items.map((item) =>
    el("div", { class: "row" },
      el("span", { class: "label", title: item[labelKey], text: item[labelKey].replaceAll("_", " ") }),
      bar(item.count / max),
      el("span", { text: item.count }))));
}

function timeline(points) {
  if (points.length === 0) return el("p", { class: "muted", text: "Pas encore de données." });
  const max = Math.max(...points.map((p) => p.count));
  const width = 100 / points.length;
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("viewBox", "0 0 100 50");
  svg.setAttribute("preserveAspectRatio", "none");
  svg.setAttribute("class", "timeline");
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "Nombre d'analyses par jour");
  points.forEach((p, i) => {
    const h = (p.count / max) * 44;
    const rect = document.createElementNS(SVG_NS, "rect");
    rect.setAttribute("x", i * width + width * 0.1);
    rect.setAttribute("y", 48 - h);
    rect.setAttribute("width", width * 0.8);
    rect.setAttribute("height", h);
    rect.setAttribute("fill", "#2f7d4f");
    const title = document.createElementNS(SVG_NS, "title");
    title.textContent = `${p.date} : ${p.count}`;
    rect.append(title);
    svg.append(rect);
  });
  return el("div", {}, svg, el("div", { class: "muted", text: `${points[0].date} → ${points[points.length - 1].date}` }));
}

export async function loadDashboard() {
  const root = document.getElementById("dashboard");
  clear(root);
  try {
    const s = await api.stats();
    root.append(
      el("div", { class: "tiles" },
        el("div", { class: "tile" }, el("strong", { text: s.total_analyses }), "Analyses"),
        el("div", { class: "tile" }, el("strong", { text: s.healthy_count }), "Plantes saines"),
        el("div", { class: "tile" }, el("strong", { text: s.diseased_count }), "Maladies détectées")),
      el("div", { class: "dash-grid" },
        el("div", { class: "card" }, el("h3", { text: "Maladies détectées" }),
          barList(s.by_disease.slice(0, 10), "class_name", "Aucune maladie détectée.")),
        el("div", { class: "card" }, el("h3", { text: "Plantes les plus analysées" }),
          barList(s.top_plants, "plant", "Aucune analyse.")),
        el("div", { class: "card", style: "grid-column: 1 / -1" }, el("h3", { text: "Analyses par jour" }),
          timeline(s.per_day))));
  } catch (error) {
    root.append(el("p", { class: "error", text: error.message }));
  }
}
