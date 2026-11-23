import type { MetadataRoute } from "next";

export default function sitemap(): MetadataRoute.Sitemap {
  const origin = process.env.ASTRAL_PUBLIC_URL ?? "http://127.0.0.1:3000";
  return ["", "/discover", "/observatory", "/candidates", "/neural", "/research"].map((route, index) => ({ url: `${origin}${route}`, changeFrequency: index ? "weekly" : "monthly", priority: index ? .8 : 1 }));
}
