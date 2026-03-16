'use client';

import { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Trash2 } from 'lucide-react';
import { cn } from '@/lib/utils';
import { sendChat } from '@/lib/api';
import SectionHeader from '@/components/SectionHeader';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

const SUGGESTIONS = [
  'A0 전략의 최근 성과를 분석해줘',
  '현재 포트폴리오의 섹터 집중도는?',
  'MDD가 큰 시기와 원인을 알려줘',
  'KOSPI 대비 초과수익률 추이는?',
];

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [streaming, setStreaming] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

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
      await sendChat(updated, undefined, (chunk) => {
        setMessages((prev) => {
          const copy = [...prev];
          copy[copy.length - 1] = { role: 'assistant', content: chunk };
          return copy;
        });
      });
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
      inputRef.current?.focus();
    }
  };

  return (
    <div className="flex flex-col h-[calc(100vh-8rem)] animate-fade-in">
      <div className="flex items-center justify-between">
        <SectionHeader title="AI 어시스턴트" subtitle="전략 분석 및 데이터 질의" />
        {messages.length > 0 && (
          <button
            onClick={() => setMessages([])}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs text-muted hover:text-accent-red rounded-lg hover:bg-accent-red/10 transition-colors"
          >
            <Trash2 size={13} />
            대화 초기화
          </button>
        )}
      </div>

      {/* Messages */}
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
                PG 데이터를 기반으로 답변합니다.
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
              'flex gap-3 max-w-3xl',
              msg.role === 'user' ? 'ml-auto flex-row-reverse' : ''
            )}
          >
            <div
              className={cn(
                'w-8 h-8 rounded-full flex items-center justify-center shrink-0 mt-1',
                msg.role === 'assistant'
                  ? 'bg-primary-dim'
                  : 'bg-surface-light'
              )}
            >
              {msg.role === 'assistant' ? (
                <Bot size={16} className="text-primary" />
              ) : (
                <User size={16} className="text-muted" />
              )}
            </div>
            <div
              className={cn(
                'max-w-[80%] px-4 py-3 rounded-2xl text-sm leading-relaxed',
                msg.role === 'user'
                  ? 'bg-primary text-background rounded-br-md'
                  : 'bg-card border border-border rounded-bl-md'
              )}
            >
              <p className="whitespace-pre-wrap break-words">{msg.content}</p>
              {streaming && i === messages.length - 1 && msg.role === 'assistant' && (
                <span className="inline-block w-1.5 h-4 bg-primary animate-pulse ml-0.5 align-middle" />
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Input */}
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
