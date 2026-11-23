import { ImageResponse } from "next/og";

export const size = { width: 1200, height: 630 };
export const contentType = "image/png";
export const alt = "ASTRAL — explore real stellar signals with scientific intelligence";

export default function OpenGraphImage() {
  return new ImageResponse(<div style={{ width: "100%", height: "100%", display: "flex", position: "relative", overflow: "hidden", alignItems: "center", padding: "76px", color: "#f0f5f8", background: "linear-gradient(135deg,#030812 0%,#081828 58%,#10132a 100%)", fontFamily: "sans-serif" }}>
    <div style={{ position: "absolute", width: 520, height: 520, right: -70, top: 55, border: "1px solid rgba(112,219,247,.22)", borderRadius: "50%" }} />
    <div style={{ position: "absolute", width: 350, height: 160, right: 30, top: 220, border: "1px solid rgba(191,167,239,.28)", borderRadius: "50%", transform: "rotate(-18deg)" }} />
    <div style={{ display: "flex", flexDirection: "column", width: 820 }}><div style={{ display: "flex", color: "#70dbf7", fontSize: 20, letterSpacing: 7, marginBottom: 28 }}>ASTRAL / ATLAS</div><div style={{ display: "flex", fontFamily: "serif", fontSize: 74, lineHeight: .95, letterSpacing: -3 }}>Astronomical signals.<br />Transparent intelligence.</div><div style={{ display: "flex", color: "#b5c4d0", fontSize: 23, marginTop: 34 }}>Real Kepler observations · BLS discovery engine · ANN-01 · ANN-02</div></div>
  </div>, size);
}
