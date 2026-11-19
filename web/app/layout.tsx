import type { Metadata, Viewport } from "next";
import { PlatformShell } from "@/components/platform-shell";
import "./globals.css";

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
