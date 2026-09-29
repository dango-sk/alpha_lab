# ── Stage 1: Build Next.js frontend ──
FROM node:20-alpine AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci
COPY frontend/ ./
ENV NEXT_PUBLIC_API_URL=""
RUN npm run build

# ── Stage 2: Python + Node runtime ──
FROM python:3.11-slim
WORKDIR /app

# Install Node.js for Next.js server
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn[standard]

# App code
COPY lib/ ./lib/
COPY backend/ ./backend/
COPY cache/ ./cache/
COPY config/ ./config/
COPY scripts/ ./scripts/
COPY analysis/regime_agent_results.json ./analysis/
COPY analysis/regime_agent_multimodel_results_gemini.json ./analysis/
# HSMM 레짐 익스포저 경로 — /api/hsmm-exposure 가 직접 읽는다.
# analysis/ 전체를 복사하지 않으므로 필요한 파일은 여기에 명시해야 한다.
COPY analysis/hsmm_final_path.csv ./analysis/

# Next.js build
COPY --from=frontend-build /app/frontend/.next ./.next
COPY --from=frontend-build /app/frontend/public ./public
COPY --from=frontend-build /app/frontend/node_modules ./node_modules
COPY --from=frontend-build /app/frontend/package.json ./package.json
COPY --from=frontend-build /app/frontend/next.config.ts ./next.config.ts

# Start script
COPY start.sh ./
RUN chmod +x start.sh

CMD ["./start.sh"]
