export class ApiError extends Error {
  constructor(message: string, public readonly status: number, public readonly code?: string) {
    super(message);
  }
}

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    cache: "no-store",
    headers: { Accept: "application/json", "Content-Type": "application/json", ...init?.headers },
  });
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const error = payload?.error;
    throw new ApiError(error?.message ?? `ASTRAL API returned ${response.status}.`, response.status, error?.code);
  }
  return payload as T;
}

export async function getAllObjects(query = ""): Promise<ObjectPage<ObjectSummary>> {
  const suffix = query ? `&query=${encodeURIComponent(query)}` : "";
  const first = await api<ObjectPage<ObjectSummary>>(`/api/v1/objects?limit=100&offset=0${suffix}`);
  if (first.pagination.total <= 100) return first;
  const second = await api<ObjectPage<ObjectSummary>>(`/api/v1/objects?limit=100&offset=100${suffix}`);
  return { items: [...first.items, ...second.items], pagination: { ...first.pagination, returned: first.items.length + second.items.length } };
}

import type { ObjectPage, ObjectSummary } from "@/lib/types";
