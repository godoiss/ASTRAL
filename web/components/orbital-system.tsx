import type { ObjectDetail } from "@/lib/types";
import { formatNumber } from "@/lib/format";

export function OrbitalSystem({ object }: { object: ObjectDetail }) {
  const radius = object.planet_radius_earth ? Math.min(9, 3 + Math.log2(Math.max(object.planet_radius_earth, .5)) * 1.6) : 4;
  return <div className="orbital-system">
    <svg viewBox="0 0 640 300" role="img" aria-label={`Illustrative orbital projection for ${object.display_name}`}>
      <defs><radialGradient id="star"><stop offset="0" stopColor="#fff" /><stop offset=".35" stopColor="#EFC77C" /><stop offset="1" stopColor="#EFB681" stopOpacity="0" /></radialGradient><filter id="glow"><feGaussianBlur stdDeviation="5" /></filter></defs>
      {[90, 150, 220].map((rx) => <ellipse key={rx} cx="320" cy="150" rx={rx} ry={rx * .32} fill="none" stroke="rgba(112,219,247,.09)" strokeDasharray="3 7" />)}
      <ellipse cx="320" cy="150" rx="190" ry="62" fill="none" stroke="#70DBF7" strokeOpacity=".62" />
      <circle cx="320" cy="150" r="38" fill="url(#star)" filter="url(#glow)" /><circle cx="320" cy="150" r="15" fill="#EFC77C" />
      <circle cx="486" cy="120" r={radius + 7} fill="none" stroke="#BFA7EF" strokeOpacity=".25" /><circle cx="486" cy="120" r={radius} fill="#80A8F5" />
      <line x1="486" y1="120" x2="535" y2="80" stroke="#8295A5" /><text x="528" y="76" textAnchor="end" fill="#F0F5F8">{object.display_name}</text><text x="528" y="94" textAnchor="end" fill="#8295A5">phase position illustrative</text>
      <line x1="130" y1="225" x2="510" y2="225" stroke="#EFB681" strokeOpacity=".5" strokeDasharray="4 5" /><text x="132" y="248" fill="#8295A5">observer transit axis (schematic)</text>
    </svg>
    <div className="orbit-readout"><span><small>Catalog semi-major axis</small><strong>{formatNumber(object.semi_major_axis_au, 4)} AU</strong></span><span><small>Orbital period</small><strong>{formatNumber(object.period_days, 5)} d</strong></span><span><small>Catalog radius</small><strong>{formatNumber(object.planet_radius_earth, 2)} R⊕</strong></span></div>
    <p className="caption">Illustrative orbital projection. The catalog semi-major axis and period are real; viewing inclination, phase position, body scale, and perspective are not constrained by this view.</p>
  </div>;
}
