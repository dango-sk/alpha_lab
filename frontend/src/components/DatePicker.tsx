'use client';

import { useState, useRef, useEffect } from 'react';
import { createPortal } from 'react-dom';

interface DatePickerProps {
  value: string;
  options: string[];
  onChange: (date: string) => void;
}

export default function DatePicker({ value, options, onChange }: DatePickerProps) {
  const [open, setOpen] = useState(false);
  const [rect, setRect] = useState<DOMRect | null>(null);
  const btnRef = useRef<HTMLButtonElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const selectedRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (
        btnRef.current?.contains(e.target as Node) ||
        dropdownRef.current?.contains(e.target as Node)
      ) return;
      setOpen(false);
    }
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, []);

  useEffect(() => {
    if (open && selectedRef.current) {
      selectedRef.current.scrollIntoView({ block: 'nearest' });
    }
  }, [open]);

  function handleOpen() {
    if (btnRef.current) setRect(btnRef.current.getBoundingClientRect());
    setOpen((v) => !v);
  }

  const dropdown = open && rect && (
    <div
      ref={dropdownRef}
      style={{
        position: 'fixed',
        top: rect.bottom + 4,
        right: window.innerWidth - rect.right,
        zIndex: 99999,
      }}
      className="bg-surface border border-border rounded-lg shadow-xl py-1 min-w-[160px] max-h-72 overflow-y-auto"
    >
      {[...options].reverse().map((d) => {
        const isSelected = d === value;
        return (
          <button
            key={d}
            ref={isSelected ? selectedRef : undefined}
            onClick={() => { onChange(d); setOpen(false); }}
            className={`w-full text-left px-4 py-2 text-xs transition-colors ${
              isSelected ? 'bg-primary text-white font-medium' : 'text-foreground hover:bg-white/5'
            }`}
          >
            {d}
          </button>
        );
      })}
    </div>
  );

  return (
    <>
      <button
        ref={btnRef}
        onClick={handleOpen}
        className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium border border-border bg-surface text-foreground hover:border-primary transition-all"
      >
        <svg className="w-3.5 h-3.5 text-muted" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
        </svg>
        <span>{value || '날짜 선택'}</span>
        <svg className={`w-3 h-3 text-muted transition-transform ${open ? 'rotate-180' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>
      {typeof window !== 'undefined' && createPortal(dropdown, document.body)}
    </>
  );
}
