"use client";

import { AlertTriangle } from "lucide-react";
import { useEffect } from "react";

export default function ErrorBoundary({ error, reset }: { error: Error & { digest?: string }; reset: () => void }) {
  useEffect(() => { console.error(error); }, [error]);
  return <div className="page"><section className="state-message error" role="alert"><AlertTriangle /><div><strong>Scientific instrument unavailable</strong><p>The interface encountered an unexpected rendering error. No dataset or model was modified.</p><button className="button secondary" onClick={reset}>Retry interface</button></div></section></div>;
}
