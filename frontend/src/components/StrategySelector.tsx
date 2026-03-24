'use client';

import { useState, useRef, useEffect } from 'react';

interface StrategySelectorProps {
  allKeys: string[];
  selected: string[];
  labels: Record<string, string>;
  colors: Record<string, string>;
  onChange: (keys: string[]) => void;
}

export default function StrategySelector({ allKeys, selected, labels, colors, onChange }: StrategySelectorProps) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, []);

  function toggle(key: string) {
    if (selected.includes(key)) {
      onChange(selected.filter((k) => k !== key));
    } else {
      onChange([...selected, key]);
    }
  }

  const selectedCount = selected.length;

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium border border-border bg-surface text-foreground hover:border-primary transition-all"
      >
        <span>전략 선택</span>
        {selectedCount > 0 && (
          <span className="bg-primary text-white rounded-full px-1.5 py-0.5 text-[10px] leading-none">
            {selectedCount}
          </span>
        )}
        <svg className={`w-3 h-3 text-muted transition-transform ${open ? 'rotate-180' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {open && (
        <div className="absolute top-full mt-1 right-0 z-50 bg-surface border border-border rounded-lg shadow-lg min-w-[220px] py-1">
          {/* 전체 선택/해제 */}
          <div className="px-3 py-1.5 border-b border-border flex items-center justify-between">
            <span className="text-[10px] text-muted font-medium uppercase tracking-wide">전략</span>
            <div className="flex gap-2">
              <button
                onClick={() => onChange(allKeys)}
                className="text-[10px] text-primary hover:underline"
              >전체</button>
              <button
                onClick={() => onChange([])}
                className="text-[10px] text-muted hover:text-foreground hover:underline"
              >해제</button>
            </div>
          </div>
          {allKeys.map((key) => {
            const isSelected = selected.includes(key);
            const color = colors[key] || '#6366f1';
            const label = labels[key] || key;
            return (
              <button
                key={key}
                onClick={() => toggle(key)}
                className="w-full flex items-center gap-2.5 px-3 py-2 text-xs hover:bg-white/5 transition-colors text-left"
              >
                <span
                  className="w-3 h-3 rounded-sm flex-shrink-0 border transition-all"
                  style={{
                    backgroundColor: isSelected ? color : 'transparent',
                    borderColor: isSelected ? color : 'var(--color-border)',
                  }}
                />
                <span
                  className="flex-1 truncate"
                  style={{ color: isSelected ? color : 'var(--color-muted)' }}
                >
                  {label}
                </span>
                {isSelected && (
                  <span className="w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ backgroundColor: color }} />
                )}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
