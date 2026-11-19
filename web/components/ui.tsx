import { AlertTriangle, Database, LoaderCircle } from "lucide-react";
import { formatNumber } from "@/lib/format";

export function PageIntro({ eyebrow, title, description, children }: { eyebrow: string; title: string; description: string; children?: React.ReactNode }) {
  return <section className="page-intro"><div><span className="eyebrow">{eyebrow}</span><h1>{title}</h1><p>{description}</p></div>{children && <div className="intro-actions">{children}</div>}</section>;
}

export function Panel({ title, eyebrow, className = "", children, action }: { title: string; eyebrow?: string; className?: string; children: React.ReactNode; action?: React.ReactNode }) {
  return <section className={`panel ${className}`}><header>{eyebrow && <span className="eyebrow">{eyebrow}</span>}<h2>{title}</h2>{action && <div className="panel-action">{action}</div>}</header><div className="panel-body">{children}</div></section>;
}

export function Metric({ label, value, unit, tone = "default" }: { label: string; value: number | string | null | undefined; unit?: string; tone?: string }) {
  return <div className={`metric tone-${tone}`}><span>{label}</span><strong>{typeof value === "number" ? formatNumber(value) : value ?? "Unavailable"}</strong>{unit && <small>{unit}</small>}</div>;
}

export function Loading({ label = "Calibrating instrument…" }: { label?: string }) {
  return <div className="state-message"><LoaderCircle className="spin" /><span>{label}</span></div>;
}

export function ErrorState({ message }: { message: string }) {
  return <div className="state-message error"><AlertTriangle /><span>{message}</span></div>;
}

export function EmptyState({ title, message }: { title: string; message: string }) {
  return <div className="empty-state"><Database /><strong>{title}</strong><p>{message}</p></div>;
}

export function ScientificNotice({ children }: { children: React.ReactNode }) {
  return <div className="scientific-notice"><AlertTriangle aria-hidden="true" /><div>{children}</div></div>;
}

export function AuditTable({ headers, rows }: { headers: string[]; rows: Array<Array<React.ReactNode>> }) {
  return <div className="audit-table-wrap"><table className="audit-table"><thead><tr>{headers.map((header) => <th key={header}>{header}</th>)}</tr></thead><tbody>{rows.map((row, index) => <tr key={index}>{row.map((cell, cellIndex) => <td key={cellIndex}>{cell}</td>)}</tr>)}</tbody></table></div>;
}
