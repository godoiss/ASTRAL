export const formatNumber = (value: number | null | undefined, digits = 2) =>
  value == null || !Number.isFinite(value) ? "Unavailable" : new Intl.NumberFormat("en-US", { maximumFractionDigits: digits }).format(value);

export const formatDate = (value: string | null | undefined) => {
  if (!value) return "Timestamp unavailable";
  return new Intl.DateTimeFormat("en-US", { dateStyle: "medium", timeStyle: "short", timeZone: "UTC" }).format(new Date(value)) + " UTC";
};

export const dispositionLabel = (value: string) => value.replaceAll("_", " ");
