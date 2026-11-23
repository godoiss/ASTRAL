import type { Metadata } from "next";
import Link from "next/link";
import { ArrowUpRight, Atom, Binoculars, BookOpenCheck, Database, Orbit, Radar, ShieldCheck, Sparkles } from "lucide-react";

export const metadata: Metadata = {
  title: "Explore real stellar signals",
  description: "ASTRAL combines real Kepler observations, catalog-independent transit screening, scientific dossiers, and two transparent neural engines.",
};

const instruments = [
  { href: "/discover", icon: Radar, label: "Discover", note: "Navigate 200 catalog-positioned Kepler objects and launch bounded research expeditions." },
  { href: "/observatory", icon: Binoculars, label: "Observatory", note: "Inspect real luminosity, detrending, orbital phase, transit windows, and provenance." },
  { href: "/candidates", icon: Orbit, label: "Candidates", note: "Interrogate eight measured screening signals through traceable scientific dossiers." },
  { href: "/neural", icon: Atom, label: "Neural Lab", note: "Open ANN-01 and ANN-02 checkpoints, activations, feature maps, and explanations." },
  { href: "/research", icon: BookOpenCheck, label: "Research", note: "Follow campaign events, immutable artifacts, experiments, metrics, and limitations." },
];

export default function HomePage() {
  return <div className="page landing-page">
    <section className="landing-hero">
      <div className="hero-copy"><span className="eyebrow"><Sparkles /> Scientific exploration system</span><h1>Find the signal.<br /><em>Question everything.</em></h1><p>ASTRAL is a local-first astronomical intelligence platform built to explore real Kepler light curves, screen periodic transit-like events, investigate evidence, and expose how two neural networks reach their decisions.</p><div className="hero-actions"><Link className="button primary" href="/discover"><Radar />Enter Mission Control</Link><Link className="button secondary" href="/candidates"><Orbit />Open scientific dossiers</Link></div></div>
      <div className="hero-instrument" aria-label="ASTRAL validated scientific record"><div className="hero-orbit"><i /><i /><i /><span /></div><div className="hero-record"><span>Validated reference record</span><strong>Kepler / Quarter 3</strong><small>12 stars investigated · 8 signals screened</small></div><div className="hero-signal"><span>Independent detector</span><strong>Astropy BLS</strong><small>3 catalog relations · 5 inconclusive</small></div><div className="hero-boundary"><ShieldCheck /><span><strong>Scientific boundaries active</strong><small>Zero discovery claims · frozen test isolated</small></span></div></div>
    </section>

    <section className="landing-facts" aria-label="Persisted ASTRAL facts"><article><span>01</span><strong>ANN-01</strong><p>Proprietary 97-parameter NumPy MLP with inspectable weights, biases, activations, gradients, and local feature explanations.</p></article><article><span>02</span><strong>ANN-02</strong><p>PyTorch dual-view convolutional engine with real global/local signal tensors, learned filters, feature maps, and sensitivity regions.</p></article><article><span>03</span><strong>Discovery Engine</strong><p>Catalog-independent Box Least Squares screening over cached Kepler cadences, with quality masks, event coverage, aliases, and provenance.</p></article></section>

    <section className="instrument-journey"><header><div><span className="eyebrow">One connected scientific workflow</span><h2>Five instruments. One evidence chain.</h2></div><p>Every handoff preserves the active object, catalog identity, campaign context, and interpretation limits.</p></header><div>{instruments.map(({ href, icon: Icon, label, note }, index) => <Link href={href} key={href}><span>{String(index + 1).padStart(2, "0")}</span><Icon /><strong>{label}</strong><p>{note}</p><ArrowUpRight /></Link>)}</div></section>

    <section className="method-boundary"><div><Database /><span><small>Evidence source</small><strong>NASA Kepler catalog + persisted light-curve products</strong></span></div><div><ShieldCheck /><span><small>Interpretation rule</small><strong>A model score is never a discovery probability</strong></span></div><Link href="/research">Inspect provenance <ArrowUpRight /></Link></section>
  </div>;
}
