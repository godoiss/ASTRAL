import type { Metadata, Viewport } from "next";
import { PlatformShell } from "@/components/platform-shell";
import "@fontsource-variable/manrope";
import "@fontsource-variable/newsreader";
import "./globals.css";
import "./experience.css";

export const metadata: Metadata = {
  title: { default: "ASTRAL Atlas", template: "%s · ASTRAL" },
  description: "Immersive scientific exploration of real Kepler observations and ASTRAL neural models.",
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
