let authPromptInFlight = false;

export function syncTokenCookie(token) {
  if (!token) return;
  const secureFlag = location.protocol === "https:" ? "; Secure" : "";
  document.cookie = `hear_token=${encodeURIComponent(token)}; path=/; max-age=31536000; SameSite=Strict${secureFlag}`;
}

export function syncClientIdCookie(clientId) {
  if (!clientId) return;
  const secureFlag = location.protocol === "https:" ? "; Secure" : "";
  document.cookie = `hear_client_id=${encodeURIComponent(clientId)}; path=/; max-age=31536000; SameSite=Strict${secureFlag}`;
}

export function getAuthToken() {
  return localStorage.getItem("hear_token") || "";
}

export function getClientId() {
  let cid = localStorage.getItem("hear_client_id");
  if (!cid) {
    cid =
      typeof crypto !== "undefined" && crypto.randomUUID
        ? crypto.randomUUID().slice(0, 12)
        : Math.random().toString(36).slice(2, 14);
    localStorage.setItem("hear_client_id", cid);
  }
  syncClientIdCookie(cid);
  return cid;
}

export async function apiFetch(url, options = {}) {
  const token = getAuthToken();
  if (token) syncTokenCookie(token);

  const headers = { ...(options.headers || {}), "x-hear-client-id": getClientId() };
  if (token) headers["x-hear-token"] = token;

  let res = await fetch(url, { ...options, headers });
  if (res.status === 401 && !authPromptInFlight) {
    authPromptInFlight = true;
    try {
      const entered = prompt("Enter Hear access token (HEAR_AUTH_TOKEN):")?.trim() || "";
      if (entered) {
        localStorage.setItem("hear_token", entered);
        syncTokenCookie(entered);
        headers["x-hear-token"] = entered;
        res = await fetch(url, { ...options, headers });
      }
    } finally {
      authPromptInFlight = false;
    }
  }
  return res;
}
