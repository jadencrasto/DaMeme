import { useEffect, useState } from "react";

type Status = "checking" | "online" | "offline";

const apiBaseUrl = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export function HealthStatus() {
  const [status, setStatus] = useState<Status>("checking");

  useEffect(() => {
    const controller = new AbortController();
    fetch(`${apiBaseUrl}/api/v1/health`, { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error("Health request failed");
        setStatus("online");
      })
      .catch(() => setStatus("offline"));
    return () => controller.abort();
  }, []);

  const label = status === "checking" ? "Checking API" : status === "online" ? "API online" : "API offline";
  return <p className={`health health--${status}`} aria-live="polite">{label}</p>;
}
