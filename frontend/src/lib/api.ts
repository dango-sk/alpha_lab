// In production, use relative paths (Next.js rewrites proxy to FastAPI)
// In dev, also uses rewrites from next.config.ts
const API_BASE = '';

export async function fetchApi(path: string, options?: RequestInit) {
  const res = await fetch(`${API_BASE}${path}`, options);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function getConfig() {
  return fetchApi('/api/config');
}

export async function getResults(params: {
  start?: string;
  end?: string;
  universe?: string;
  rebal_type?: string;
}) {
  const qs = new URLSearchParams(params as Record<string, string>).toString();
  return fetchApi(`/api/results?${qs}`);
}

export async function getHoldings(
  strategy: string,
  date: string,
  universe?: string,
  rebal_type?: string
) {
  const params: Record<string, string> = { strategy, date };
  if (universe) params.universe = universe;
  if (rebal_type) params.rebal_type = rebal_type;
  const qs = new URLSearchParams(params).toString();
  return fetchApi(`/api/holdings?${qs}`);
}

export async function getAttribution(
  strategy: string,
  start_date: string,
  end_date: string,
  universe?: string,
  rebal_type?: string
) {
  const params: Record<string, string> = { strategy, start_date, end_date };
  if (universe) params.universe = universe;
  if (rebal_type) params.rebal_type = rebal_type;
  const qs = new URLSearchParams(params).toString();
  return fetchApi(`/api/attribution?${qs}`);
}

export async function getCharacteristics(
  strategy: string,
  date: string,
  universe?: string,
  rebal_type?: string
) {
  const params: Record<string, string> = { strategy, date };
  if (universe) params.universe = universe;
  if (rebal_type) params.rebal_type = rebal_type;
  const qs = new URLSearchParams(params).toString();
  return fetchApi(`/api/characteristics?${qs}`);
}

export async function getTurnover(
  strategy: string,
  date: string,
  prev_date: string,
  universe?: string,
  rebal_type?: string
) {
  const params: Record<string, string> = { strategy, date, prev_date };
  if (universe) params.universe = universe;
  if (rebal_type) params.rebal_type = rebal_type;
  const qs = new URLSearchParams(params).toString();
  return fetchApi(`/api/turnover?${qs}`);
}

export async function getRobustness(params: Record<string, string>) {
  const qs = new URLSearchParams(params).toString();
  return fetchApi(`/api/robustness?${qs}`);
}

export async function getRegimeAnalysis(params: Record<string, string>) {
  const qs = new URLSearchParams(params).toString();
  return fetchApi(`/api/regime?${qs}`);
}

export async function getStrategies() {
  return fetchApi('/api/strategies');
}

export async function getLatestPriceDate() {
  return fetchApi('/api/latest_price_date');
}

export async function runBacktest(code: string, params: Record<string, unknown>) {
  // Start backtest job
  const { job_id } = await fetchApi('/api/backtest', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ strategy_code: code, ...params }),
  });

  // Poll for result
  const maxWait = 900_000; // 15 min
  const interval = 2_000;
  const start = Date.now();

  while (Date.now() - start < maxWait) {
    await new Promise((r) => setTimeout(r, interval));
    const res = await fetchApi(`/api/backtest/${job_id}`);
    if (res.status === 'done') return res.result;
    if (res.status === 'error') throw new Error(res.detail || '백테스트 실패');
    // still running, continue polling
  }
  throw new Error('백테스트 시간 초과 (5분)');
}

export async function sendChat(
  messages: Array<{ role: string; content: string }>,
  context?: Record<string, unknown>,
  onChunk?: (text: string) => void,
  onThinking?: (label: string | null) => void
) {
  const res = await fetch(`${API_BASE}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages, context }),
  });

  if (!res.ok) throw new Error(`API error: ${res.status}`);
  if (!res.body) throw new Error('No response body');

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let fullText = '';
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // Parse SSE lines
    const lines = buffer.split('\n');
    buffer = lines.pop() || ''; // keep incomplete line in buffer

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed || trimmed === 'data: [DONE]') continue;
      if (trimmed.startsWith('data: ')) {
        try {
          const json = JSON.parse(trimmed.slice(6));
          if (json.thinking && onThinking) {
            onThinking(json.thinking);
          } else if (json.thinking_done && onThinking) {
            onThinking(null);
          } else if (json.text) {
            fullText += json.text;
            if (onChunk) onChunk(fullText);
          }
        } catch {
          // Not JSON, treat as plain text
          fullText += trimmed.slice(6);
          if (onChunk) onChunk(fullText);
        }
      }
    }
  }

  return fullText;
}

export async function executeSql(query: string) {
  return fetchApi('/api/chat/sql', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query }),
  });
}
