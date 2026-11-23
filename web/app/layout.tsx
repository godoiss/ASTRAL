import type { Metadata, Viewport } from "next";
import { PlatformShell } from "@/components/platform-shell";
import "@fontsource-variable/manrope";
import "@fontsource-variable/newsreader";
import "./globals.css";
import "./experience.css";

export const metadata: Metadata = {
  metadataBase: new URL(process.env.ASTRAL_PUBLIC_URL ?? "http://127.0.0.1:3000"),
  title: { default: "ASTRAL — Astronomical Intelligence", template: "%s · ASTRAL" },
  description: "Immersive scientific exploration of real Kepler observations and ASTRAL neural models.",
  applicationName: "ASTRAL",
  authors: [{ name: "ASTRAL Project", url: "https://github.com/godoiss/ASTRAL" }],
  creator: "ASTRAL Project",
  keywords: ["astronomy", "Kepler", "exoplanets", "neural networks", "light curves", "scientific visualization"],
  alternates: { canonical: "/" },
  openGraph: { type: "website", title: "ASTRAL — Astronomical Intelligence", description: "Explore real Kepler observations, transit-like signals, scientific dossiers, and transparent neural models.", url: "/", siteName: "ASTRAL" },
  twitter: { card: "summary_large_image", title: "ASTRAL — Astronomical Intelligence", description: "A scientific platform for real Kepler signal exploration and transparent AI." },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  colorScheme: "dark",
  themeColor: "#050a12",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body><PlatformShell>{children}</PlatformShell></body></html>;
}
