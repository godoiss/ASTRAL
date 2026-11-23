"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Atom, Binoculars, FlaskConical, Menu, Orbit, Radar, X } from "lucide-react";
import { useEffect, useState } from "react";

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
  const current = routes.find((route) => pathname.startsWith(route.href)) ?? { label: "Overview", href: "/" };
  useEffect(() => {
    document.body.classList.toggle("navigation-open", open);
    const close = (event: KeyboardEvent) => { if (event.key === "Escape") setOpen(false); };
    window.addEventListener("keydown", close);
    return () => { document.body.classList.remove("navigation-open"); window.removeEventListener("keydown", close); };
  }, [open]);
  return (
    <div className="platform-shell">
      <div className="cosmic-field" aria-hidden="true"><i /><i /><i /><span /></div>
      <a className="skip-link" href="#main-content">Skip to scientific content</a>
      <header className="mobile-header">
        <Brand />
        <button className="icon-button" aria-label="Toggle navigation" aria-expanded={open} aria-controls="primary-navigation" onClick={() => setOpen((value) => !value)}>{open ? <X /> : <Menu />}</button>
      </header>
      {open && <button className="navigation-scrim" aria-label="Close navigation" onClick={() => setOpen(false)} />}
      <aside id="primary-navigation" className={`mission-rail ${open ? "is-open" : ""}`} aria-label="Primary navigation">
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
      <nav className="mobile-dock" aria-label="Mobile instruments">{routes.map(({ href, label, icon: Icon }) => <Link key={href} href={href} className={pathname.startsWith(href) ? "active" : ""}><Icon /><span>{label === "Neural Lab" ? "Neural" : label}</span></Link>)}</nav>
      <div className="workspace" data-route={current.href.slice(1)}>
        <header className="context-bar">
          <div className="instrument-title"><span className="context-kicker">ASTRAL / Active instrument</span><strong>{current.label}</strong></div>
          <div className="context-meta"><span className="context-coordinate">Kepler field · J2000</span><span>FastAPI linked</span><span className="status-good">Guardrails active</span></div>
        </header>
        <main id="main-content">{children}</main>
      </div>
    </div>
  );
}

function Brand() {
  return <Link className="brand" href="/" aria-label="ASTRAL home"><span className="brand-mark"><i /><i /><i /></span><span><strong>ASTRAL</strong><small>Astronomical Neural Network</small></span></Link>;
}
