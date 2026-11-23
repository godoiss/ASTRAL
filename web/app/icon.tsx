import { ImageResponse } from "next/og";

export const size = { width: 64, height: 64 };
export const contentType = "image/png";

export default function Icon() {
  return new ImageResponse(<div style={{ width: "64px", height: "64px", display: "flex", alignItems: "center", justifyContent: "center", borderRadius: "50%", background: "#07121f", border: "2px solid #70dbf7", color: "#f0f5f8", fontSize: 30, fontFamily: "serif", boxShadow: "inset 0 0 20px rgba(112,219,247,.22)" }}>A</div>, size);
}
