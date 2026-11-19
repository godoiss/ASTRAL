"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Atom, Binoculars, FlaskConical, Menu, Orbit, Radar, X } from "lucide-react";
import { useState } from "react";

const routes = [
  { href: "/discover", label: "Discover", note: "Stellar atlas", icon: Radar },
  { href: "/observatory", label: "Observatory", note: "Signal console", icon: Binoculars },
  { href: "/candidates", label: "Candidates", note: "Investigation dossiers", icon: Orbit },
  { href: "/neural", label: "Neural Lab", note: "Model observatory", icon: Atom },
  { href: "/research", label: "Research", note: "Reproducible record", icon: FlaskConical },
];

export function PlatformShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);
  const current = routes.find((route) => pathname.startsWith(route.href)) ?? routes[0];
  return (
    <div className="platform-shell">
      <a className="skip-link" href="#main-content">Skip to scientific content</a>
      <header className="mobile-header">
        <Brand />
        <button className="icon-button" aria-label="Toggle navigation" onClick={() => setOpen((value) => !value)}>{open ? <X /> : <Menu />}</button>
      </header>
      <aside className={`mission-rail ${open ? "is-open" : ""}`} aria-label="Primary navigation">
        <Brand />
        <div className="rail-context"><span className="live-mark" /> Kepler science cache <small>Read-only scientific surface</small></div>
        <nav>
          {routes.map(({ href, label, note, icon: Icon }) => (
            <Link key={href} href={href} className={pathname.startsWith(href) ? "active" : ""} onClick={() => setOpen(false)}>
              <Icon aria-hidden="true" /><span>{label}<small>{note}</small></span>
            </Link>
          ))}
        </nav>
        <footer><span>ASTRAL / ATLAS</span><small>Scientific exploration interface</small><small>No score is a discovery claim.</small></footer>
      </aside>
      <div className="workspace">
        <header className="context-bar">
          <div><span className="context-kicker">Active instrument</span><strong>{current.label}</strong></div>
          <div className="context-meta"><span>FastAPI linked</span><span>CPU compatible</span><span className="status-good">Scientific guardrails active</span></div>
        </header>
        <main id="main-content">{children}</main>
      </div>
    </div>
  );
}

function Brand() {
  return <Link className="brand" href="/discover" aria-label="ASTRAL home"><span className="brand-mark"><i /><i /><i /></span><span><strong>ASTRAL</strong><small>Astronomical Neural Network</small></span></Link>;
}
