'use client';

import { useState, useRef, useEffect } from 'react';
import { MessageSquare, X, Send, Bot, User } from 'lucide-react';
import { cn } from '@/lib/utils';
import { sendChat } from '@/lib/api';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

interface ChatPanelProps {
  open: boolean;
  onToggle: () => void;
}

export default function ChatPanel({ open, onToggle }: ChatPanelProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [streaming, setStreaming] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSend = async () => {
    const text = input.trim();
    if (!text || streaming) return;

    const userMsg: Message = { role: 'user', content: text };
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
    }
  };

  if (!open) {
    return (
      <button
        onClick={onToggle}
        className="fixed right-4 bottom-4 z-30 p-3 rounded-full bg-primary text-background shadow-lg hover:scale-105 transition-transform pulse-glow"
      >
        <MessageSquare size={20} />
      </button>
    );
  }

  return (
    <aside className="w-80 h-screen sticky top-0 flex flex-col bg-surface border-l border-border z-20">
      <div className="flex items-center justify-between h-14 px-4 border-b border-border shrink-0">
        <div className="flex items-center gap-2">
          <Bot size={16} className="text-primary" />
          <span className="text-sm font-semibold">AI 어시스턴트</span>
        </div>
        <button
          onClick={onToggle}
          className="p-1.5 rounded-md text-muted hover:text-foreground hover:bg-card-hover transition-colors"
        >
          <X size={16} />
        </button>
      </div>

      <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
          <div className="text-center text-muted text-xs mt-8 space-y-2">
            <Bot size={32} className="mx-auto text-border" />
            <p>전략 분석에 대해 질문하세요.</p>
            <p className="text-[10px]">예: &quot;최근 수익률 하락 원인은?&quot;</p>
          </div>
        )}
        {messages.map((msg, i) => (
          <div
            key={i}
            className={cn(
              'flex gap-2',
              msg.role === 'user' ? 'justify-end' : 'justify-start'
            )}
          >
            {msg.role === 'assistant' && (
              <div className="w-6 h-6 rounded-full bg-primary-dim flex items-center justify-center shrink-0 mt-0.5">
                <Bot size={12} className="text-primary" />
              </div>
            )}
            <div
              className={cn(
                'max-w-[85%] px-3 py-2 rounded-xl text-sm leading-relaxed',
                msg.role === 'user'
                  ? 'bg-primary text-background rounded-br-sm'
                  : 'bg-card border border-border rounded-bl-sm'
              )}
            >
              <p className="whitespace-pre-wrap break-words">{msg.content}</p>
              {streaming && i === messages.length - 1 && msg.role === 'assistant' && (
                <span className="inline-block w-1.5 h-4 bg-primary animate-pulse ml-0.5 align-middle" />
              )}
            </div>
            {msg.role === 'user' && (
              <div className="w-6 h-6 rounded-full bg-surface-light flex items-center justify-center shrink-0 mt-0.5">
                <User size={12} className="text-muted" />
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="p-3 border-t border-border shrink-0">
        <div className="flex items-center gap-2 bg-card rounded-lg border border-border px-3 py-2 focus-within:border-primary transition-colors">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && handleSend()}
            placeholder="메시지를 입력하세요..."
            className="flex-1 bg-transparent text-sm text-foreground placeholder-muted outline-none"
            disabled={streaming}
          />
          <button
            onClick={handleSend}
            disabled={streaming || !input.trim()}
            className="p-1.5 rounded-md text-primary hover:bg-primary-dim transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
          >
            <Send size={14} />
          </button>
        </div>
      </div>
    </aside>
  );
}
