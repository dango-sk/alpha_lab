'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { Send, Bot, User, Trash2, Copy, Check, Download, Database, FlaskConical, Sparkles } from 'lucide-react';
import { cn } from '@/lib/utils';
import { sendChat, executeSql } from '@/lib/api';
import SectionHeader from '@/components/SectionHeader';
import MarkdownRenderer from '@/components/MarkdownRenderer';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  sqlResults?: Array<{ query: string; columns: string[]; data: Record<string, unknown>[]; row_count: number }>;
}

const STORAGE_KEY = 'alpha-lab-chat-history';

const SUGGESTIONS = [
  'A0 전략의 최근 성과를 분석해줘',
  '현재 포트폴리오의 섹터 집중도는?',
  'MDD가 큰 시기와 원인을 알려줘',
  'KOSPI 대비 초과수익률 추이는?',
  '최근 리밸런싱에서 편입/편출된 종목은?',
  '전략별 Sharpe Ratio를 비교해줘',
];

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  const handleCopy = () => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };
  return (
    <button
      onClick={handleCopy}
      className="p-1 rounded text-muted hover:text-foreground hover:bg-card-hover transition-colors"
      title="복사"
    >
      {copied ? <Check size={13} /> : <Copy size={13} />}
    </button>
  );
}

function SqlResultTable({ columns, data }: { columns: string[]; data: Record<string, unknown>[] }) {
  if (!data || data.length === 0) return <p className="text-xs text-muted">결과 없음</p>;
  return (
    <div className="overflow-x-auto mt-2 rounded-lg border border-border">
      <table className="w-full text-xs">
        <thead>
          <tr className="bg-surface">
            {columns.map((col) => (
              <th key={col} className="px-3 py-2 text-left font-medium text-muted border-b border-border whitespace-nowrap">
                {col}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.slice(0, 50).map((row, i) => (
            <tr key={i} className="hover:bg-card-hover transition-colors">
              {columns.map((col) => (
                <td key={col} className="px-3 py-1.5 border-b border-border font-num whitespace-nowrap">
                  {String(row[col] ?? '')}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {data.length > 50 && (
        <p className="text-xs text-muted text-center py-2">... {data.length - 50}개 행 더 있음</p>
      )}
    </div>
  );
}

function extractPythonCode(content: string): string | null {
  const match = content.match(/```python\n([\s\S]*?)```/);
  if (!match) return null;
  const code = match[1].trim();
  // Check if it looks like factor_engine code
  if (code.includes('WEIGHTS') || code.includes('def score') || code.includes('factor') || code.includes('def strategy')) {
    return code;
  }
  return null;
}

export default function ChatPage() {
  const router = useRouter();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [streaming, setStreaming] = useState(false);
  const [thinkingLabel, setThinkingLabel] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const [initialized, setInitialized] = useState(false);

  // Load from localStorage
  useEffect(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) setMessages(JSON.parse(saved));
    } catch { /* ignore */ }
    setInitialized(true);
  }, []);

  // Save to localStorage
  useEffect(() => {
    if (!initialized) return;
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
    } catch { /* ignore */ }
  }, [messages, initialized]);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const processSqlBlocks = useCallback(async (content: string): Promise<Message['sqlResults']> => {
    const sqlRegex = /```sql\n([\s\S]*?)```/g;
    const matches = [...content.matchAll(sqlRegex)];
    if (matches.length === 0) return undefined;

    const results: NonNullable<Message['sqlResults']> = [];
    for (const match of matches) {
      const query = match[1].trim();
      if (!query.toLowerCase().startsWith('select')) continue;
      try {
        const result = await executeSql(query);
        results.push({ query, ...result });
      } catch {
        results.push({ query, columns: ['error'], data: [{ error: 'SQL 실행 실패' }], row_count: 0 });
      }
    }
    return results.length > 0 ? results : undefined;
  }, []);

  const handleSend = async (text?: string) => {
    const msg = (text || input).trim();
    if (!msg || streaming) return;

    const userMsg: Message = { role: 'user', content: msg };
    const updated = [...messages, userMsg];
    setMessages(updated);
    setInput('');
    setStreaming(true);

    const assistantMsg: Message = { role: 'assistant', content: '' };
    setMessages([...updated, assistantMsg]);

    try {
      const fullText = await sendChat(
        updated.map((m) => ({ role: m.role, content: m.content })),
        undefined,
        (chunk) => {
          setMessages((prev) => {
            const copy = [...prev];
            copy[copy.length - 1] = { role: 'assistant', content: chunk };
            return copy;
          });
        },
        (label) => setThinkingLabel(label)
      );

      const sqlResults = await processSqlBlocks(fullText);
      if (sqlResults) {
        setMessages((prev) => {
          const copy = [...prev];
          copy[copy.length - 1] = { ...copy[copy.length - 1], sqlResults };
          return copy;
        });
      }
    } catch {
      setMessages((prev) => {
        const copy = [...prev];
        copy[copy.length - 1] = {
          role: 'assistant',
          content: '응답을 가져올 수 없습니다. 다시 시도해주세요.',
        };
        return copy;
      });
    } finally {
      setStreaming(false);
      setThinkingLabel(null);
      inputRef.current?.focus();
    }
  };

  const handleExport = () => {
    const text = messages
      .map((m) => `[${m.role === 'user' ? '사용자' : 'AI'}]\n${m.content}`)
      .join('\n\n---\n\n');
    const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `alpha-lab-chat-${new Date().toISOString().slice(0, 10)}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex flex-col h-[calc(100vh-8rem)] animate-fade-in">
      <div className="flex items-center justify-between">
        <SectionHeader title="AI 어시스턴트" subtitle="전략 분석 및 데이터 질의" />
        <div className="flex items-center gap-2">
          {messages.length > 0 && (
            <>
              <button
                onClick={handleExport}
                className="flex items-center gap-1.5 px-3 py-1.5 text-xs text-muted hover:text-foreground rounded-lg hover:bg-card-hover transition-colors"
              >
                <Download size={13} />
                내보내기
              </button>
              <button
                onClick={() => setMessages([])}
                className="flex items-center gap-1.5 px-3 py-1.5 text-xs text-muted hover:text-accent-red rounded-lg hover:bg-accent-red/10 transition-colors"
              >
                <Trash2 size={13} />
                초기화
              </button>
            </>
          )}
        </div>
      </div>

      <div ref={scrollRef} className="flex-1 overflow-y-auto space-y-4 py-4">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center space-y-6">
            <div className="w-16 h-16 rounded-2xl bg-primary-dim flex items-center justify-center">
              <Bot size={32} className="text-primary" />
            </div>
            <div className="space-y-2">
              <h3 className="text-lg font-semibold text-foreground">무엇을 도와드릴까요?</h3>
              <p className="text-sm text-muted max-w-md">
                포트폴리오 성과, 팩터 분석, 종목 특성 등 퀀트 전략에 대해 질문하세요.
                PG 데이터 기반으로 답변하며, SQL 쿼리도 실행할 수 있습니다.
              </p>
            </div>
            <div className="grid grid-cols-2 gap-2 max-w-lg">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  onClick={() => handleSend(s)}
                  className="px-4 py-3 text-xs text-left text-muted bg-surface border border-border rounded-xl hover:border-primary hover:text-foreground transition-colors"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg, i) => (
          <div
            key={i}
            className={cn(
              'group flex gap-3 max-w-3xl',
              msg.role === 'user' ? 'ml-auto flex-row-reverse' : ''
            )}
          >
            <div
              className={cn(
                'w-8 h-8 rounded-full flex items-center justify-center shrink-0 mt-1',
                msg.role === 'assistant' ? 'bg-primary-dim' : 'bg-surface-light'
              )}
            >
              {msg.role === 'assistant' ? (
                <Bot size={16} className="text-primary" />
              ) : (
                <User size={16} className="text-muted" />
              )}
            </div>
            <div className="max-w-[80%] space-y-2">
              <div
                className={cn(
                  'px-4 py-3 rounded-2xl text-sm leading-relaxed',
                  msg.role === 'user'
                    ? 'bg-primary text-background rounded-br-md'
                    : 'bg-card border border-border rounded-bl-md'
                )}
              >
                {msg.role === 'assistant' ? (
                  <>
                    {streaming && i === messages.length - 1 && thinkingLabel && !msg.content && (
                      <div className="flex items-center gap-2 text-sm text-muted py-1">
                        <Sparkles size={14} className="text-primary animate-spin" style={{ animationDuration: '3s' }} />
                        <span className="animate-pulse">{thinkingLabel}...</span>
                      </div>
                    )}
                    <MarkdownRenderer content={msg.content} />
                    {streaming && i === messages.length - 1 && msg.content && (
                      <span className="inline-block w-1.5 h-4 bg-primary animate-pulse ml-0.5 align-middle" />
                    )}
                  </>
                ) : (
                  <p className="whitespace-pre-wrap break-words">{msg.content}</p>
                )}
              </div>

              {msg.sqlResults?.map((sr, j) => (
                <div key={j} className="bg-card border border-border rounded-xl p-3 space-y-2">
                  <div className="flex items-center gap-2 text-xs text-muted">
                    <Database size={12} />
                    <span>SQL 쿼리 결과 ({sr.row_count}행)</span>
                  </div>
                  <SqlResultTable columns={sr.columns} data={sr.data} />
                </div>
              ))}

              {/* Open in Lab button */}
              {msg.role === 'assistant' && extractPythonCode(msg.content) && !(streaming && i === messages.length - 1) && (
                <button
                  onClick={() => {
                    const code = extractPythonCode(msg.content);
                    if (code) {
                      localStorage.setItem('alpha-lab-strategy-code', code);
                      router.push('/lab');
                    }
                  }}
                  className="flex items-center gap-2 px-4 py-2 text-xs font-medium rounded-lg bg-primary/10 text-primary border border-primary/30 hover:bg-primary/20 transition-colors"
                >
                  <FlaskConical size={14} />
                  전략 실험실에서 열기
                </button>
              )}

              {msg.role === 'assistant' && msg.content && !(streaming && i === messages.length - 1) && (
                <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                  <CopyButton text={msg.content} />
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      <div className="shrink-0 pt-4 pb-2">
        <div className="flex items-center gap-3 bg-surface border border-border rounded-xl px-4 py-3 focus-within:border-primary transition-colors max-w-3xl mx-auto">
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && handleSend()}
            placeholder="메시지를 입력하세요..."
            className="flex-1 bg-transparent text-sm text-foreground placeholder-muted outline-none"
            disabled={streaming}
          />
          <button
            onClick={() => handleSend()}
            disabled={streaming || !input.trim()}
            className="p-2 rounded-lg bg-primary text-background hover:opacity-90 transition-opacity disabled:opacity-30 disabled:cursor-not-allowed"
          >
            <Send size={16} />
          </button>
        </div>
        <p className="text-center text-[10px] text-muted mt-2">
          AI 응답은 참고용이며, PG 데이터 기반으로 분석합니다.
        </p>
      </div>
    </div>
  );
}
