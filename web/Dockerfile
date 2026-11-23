FROM node:24-alpine AS dependencies
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

FROM node:24-alpine AS builder
WORKDIR /app
ARG ASTRAL_API_ORIGIN=http://api:8000
ARG ASTRAL_PUBLIC_URL=http://127.0.0.1:3000
ENV ASTRAL_API_ORIGIN=$ASTRAL_API_ORIGIN \
    ASTRAL_PUBLIC_URL=$ASTRAL_PUBLIC_URL \
    NEXT_TELEMETRY_DISABLED=1
COPY --from=dependencies /app/node_modules ./node_modules
COPY . .
RUN npm run build

FROM node:24-alpine AS runtime
WORKDIR /app
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    HOSTNAME=0.0.0.0 \
    PORT=3000
RUN addgroup --system --gid 1001 astral && adduser --system --uid 1001 --ingroup astral astral
COPY --from=builder --chown=astral:astral /app/public ./public
COPY --from=builder --chown=astral:astral /app/.next/standalone ./
COPY --from=builder --chown=astral:astral /app/.next/static ./.next/static
USER astral
EXPOSE 3000
CMD ["node", "server.js"]
