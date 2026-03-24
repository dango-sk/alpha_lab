'use client';

import { useState, useRef, useEffect } from 'react';

interface DateRangePanelProps {
  startDate: string;
  endDate: string;
  isOosSplit: string;
  onApply: (start: string, end: string, split: string) => void;
  className?: string;
}

export default function DateRangePanel({ startDate, endDate, isOosSplit, onApply }: DateRangePanelProps) {
  const [open, setOpen] = useState(false);
  const [dStart, setDStart] = useState(startDate);
  const [dEnd, setDEnd] = useState(endDate);
  const [dSplit, setDSplit] = useState(isOosSplit);
  const [dropdownStyle, setDropdownStyle] = useState<{ top: number; right: number }>({ top: 0, right: 0 });
  const btnRef = useRef<HTMLButtonElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => { setDStart(startDate); }, [startDate]);
  useEffect(() => { setDEnd(endDate); }, [endDate]);
  useEffect(() => { setDSplit(isOosSplit); }, [isOosSplit]);

  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (
        btnRef.current && !btnRef.current.contains(e.target as Node) &&
        dropdownRef.current && !dropdownRef.current.contains(e.target as Node)
      ) setOpen(false);
    }
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, []);

  function handleOpen() {
    if (!open && btnRef.current) {
      const rect = btnRef.current.getBoundingClientRect();
      setDropdownStyle({ top: rect.bottom + 4, right: window.innerWidth - rect.right });
    }
    setOpen((v) => !v);
  }

  function handleApply() {
    onApply(dStart, dEnd, dSplit);
    setOpen(false);
  }

  const label = startDate && endDate ? `${startDate} ~ ${endDate}` : '기간 설정';

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
        <span>{label}</span>
        <svg className={`w-3 h-3 text-muted transition-transform ${open ? 'rotate-180' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {open && (
        <div
          ref={dropdownRef}
          className="fixed z-[9999] bg-surface border border-border rounded-lg shadow-xl p-4 w-[420px]"
          style={{ top: dropdownStyle.top, right: dropdownStyle.right }}
        >
          <div className="grid grid-cols-3 gap-3 mb-3">
            <div>
              <label className="block text-[10px] text-muted mb-1 font-medium">시작일</label>
              <input
                type="date"
                value={dStart}
                onChange={(e) => setDStart(e.target.value)}
                className="w-full bg-background border border-border rounded-lg px-2 py-1.5 text-xs text-foreground focus:outline-none focus:ring-1 focus:ring-primary"
              />
            </div>
            <div>
              <label className="block text-[10px] text-muted mb-1 font-medium">종료일</label>
              <input
                type="date"
                value={dEnd}
                onChange={(e) => setDEnd(e.target.value)}
                className="w-full bg-background border border-border rounded-lg px-2 py-1.5 text-xs text-foreground focus:outline-none focus:ring-1 focus:ring-primary"
              />
            </div>
            <div>
              <label className="block text-[10px] text-muted mb-1 font-medium">IS/OOS 분할점</label>
              <input
                type="date"
                value={dSplit}
                onChange={(e) => setDSplit(e.target.value)}
                className="w-full bg-background border border-border rounded-lg px-2 py-1.5 text-xs text-foreground focus:outline-none focus:ring-1 focus:ring-primary"
              />
            </div>
          </div>
          <div className="flex justify-end">
            <button
              onClick={handleApply}
              className="px-4 py-1.5 text-xs font-medium rounded-lg bg-primary text-white hover:opacity-90 transition-opacity"
            >
              적용
            </button>
          </div>
        </div>
      )}
    </>
  );
}
