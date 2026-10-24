import { useEffect, useState } from "react";

import { apiBaseUrl } from "../lib/api";

type Status = "checking" | "online" | "offline";

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
