// Client de l'API. Adresse modifiable : <script>window.PLANTGUARD_API = "https://..."</script> avant main.js.
export const API_BASE = (window.PLANTGUARD_API || "http://127.0.0.1:8000").replace(/\/$/, "");

async function request(path, options) {
  let response;
  try {
    response = await fetch(API_BASE + path, options);
  } catch {
    throw new Error("Impossible de joindre le serveur. Vérifiez qu'il est démarré.");
  }
  if (response.status === 204) return null;
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = typeof body.detail === "string" ? body.detail : "Requête invalide.";
    throw new Error(detail);
  }
  return body;
}

export const api = {
  health: () => request("/api/health"),
  predict(file) {
    const form = new FormData();
    form.append("file", file);
    return request("/api/predict", { method: "POST", body: form });
  },
  analyses: (limit, offset) => request(`/api/analyses?limit=${limit}&offset=${offset}`),
  analysis: (id) => request(`/api/analyses/${id}`),
  remove: (id) => request(`/api/analyses/${id}`, { method: "DELETE" }),
  stats: () => request("/api/stats"),
  diseases: () => request("/api/diseases"),
  imageUrl: (path) => API_BASE + path,
};
