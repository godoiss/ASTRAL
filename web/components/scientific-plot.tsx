"use client";

import { useEffect, useRef } from "react";

interface PlotProps { data: unknown[]; layout?: Record<string, unknown>; title: string; className?: string }

export function ScientificPlot({ data, layout = {}, title, className = "" }: PlotProps) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    let active = true;
    let module: typeof import("plotly.js-basic-dist-min") | null = null;
    let resizeObserver: ResizeObserver | null = null;
    const render = async () => {
      module = await import("plotly.js-basic-dist-min");
      if (!active || !ref.current) return;
      await module.default.react(ref.current, data, {
        paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
        font: { color: "#B5C4D0", family: "Segoe UI, sans-serif", size: 12 },
        margin: { l: 52, r: 24, t: 28, b: 48 }, autosize: true,
        xaxis: { gridcolor: "rgba(112,219,247,.08)", zerolinecolor: "rgba(112,219,247,.18)" },
        yaxis: { gridcolor: "rgba(112,219,247,.08)", zerolinecolor: "rgba(112,219,247,.18)" },
        hoverlabel: { bgcolor: "#111E2D", bordercolor: "#70DBF7", font: { color: "#F0F5F8" } },
        ...layout,
      }, { responsive: true, displaylogo: false, modeBarButtonsToRemove: ["lasso2d", "select2d"] });
      resizeObserver = new ResizeObserver(() => {
        if (active && module && ref.current) void module.default.Plots.resize(ref.current);
      });
      resizeObserver.observe(ref.current.parentElement ?? ref.current);
    };
    void render();
    return () => {
      active = false;
      resizeObserver?.disconnect();
      if (module && ref.current) module.default.purge(ref.current);
    };
  }, [data, layout]);
  return <div className={`scientific-plot ${className}`} ref={ref} role="img" aria-label={title} />;
}
