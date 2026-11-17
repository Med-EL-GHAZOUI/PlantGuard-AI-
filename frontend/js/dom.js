// Aides DOM. Tout le texte passe par textContent : aucune donnée venant du serveur n'est injectée comme HTML.
export function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

export const clear = (node) => node.replaceChildren();
export const percent = (x) => `${(x * 100).toFixed(1)} %`;

export function bar(fraction) {
  const fill = el("span");
  fill.style.width = `${Math.max(0, Math.min(1, fraction)) * 100}%`;
  return el("div", { class: "bar" }, fill);
}

export function statusBadge(item) {
  if (item.low_confidence) return el("span", { class: "badge badge-warn", text: "Confiance faible" });
  return item.is_healthy
    ? el("span", { class: "badge badge-ok", text: "Plante saine" })
    : el("span", { class: "badge badge-bad", text: "Maladie détectée" });
}

export function formatDate(iso) {
  // l'API renvoie de l'UTC sans fuseau : on l'indique pour un affichage correct à l'heure locale
  const date = new Date(/[zZ]|[+-]\d\d:?\d\d$/.test(iso) ? iso : iso + "Z");
  return date.toLocaleString("fr-FR", { dateStyle: "short", timeStyle: "short" });
}
