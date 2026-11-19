import Link from "next/link";
import { Radar } from "lucide-react";

export default function NotFound() {
  return <div className="page"><section className="empty-state"><Radar /><strong>Coordinate outside the Atlas</strong><p>The requested product route does not exist. No astronomical object was inferred from this address.</p><Link className="button secondary" href="/discover">Return to Discover</Link></section></div>;
}
