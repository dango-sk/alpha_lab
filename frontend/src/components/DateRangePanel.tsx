'use client';

import { useState, useEffect } from 'react';
import { ChevronDown, ChevronUp } from 'lucide-react';
import { cn } from '@/lib/utils';

interface DateRangePanelProps {
  startDate: string;
  endDate: string;
  isOosSplit: string;
  onApply: (start: string, end: string, split: string) => void;
  className?: string;
}

export default function DateRangePanel({
  startDate,
  endDate,
  isOosSplit,
  onApply,
  className,
}: DateRangePanelProps) {
  const [open, setOpen] = useState(false);
  // Internal draft state — changes here do NOT re-render parent
  const [dStart, setDStart] = useState(startDate);
  const [dEnd, setDEnd] = useState(endDate);
  const [dSplit, setDSplit] = useState(isOosSplit);

  // Sync from parent when props change (e.g. config loaded)
  useEffect(() => { setDStart(startDate); }, [startDate]);
  useEffect(() => { setDEnd(endDate); }, [endDate]);
  useEffect(() => { setDSplit(isOosSplit); }, [isOosSplit]);

  return (
    <div className={cn('rounded-lg border border-border bg-surface overflow-hidden', className)}>
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-4 py-2.5 text-sm font-medium text-foreground hover:bg-card-hover transition-colors"
      >
        <span>기간 설정</span>
        {open ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
      </button>

      {open && (
        <div className="px-4 pb-4 pt-1 border-t border-border">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div>
              <label className="block text-xs text-muted mb-1.5">시작일</label>
              <input
                type="date"
                value={dStart}
                onChange={(e) => setDStart(e.target.value)}
                className="w-full bg-background border border-border rounded-lg px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-1 focus:ring-primary"
              />
            </div>
            <div>
              <label className="block text-xs text-muted mb-1.5">종료일</label>
              <input
                type="date"
                value={dEnd}
                onChange={(e) => setDEnd(e.target.value)}
                className="w-full bg-background border border-border rounded-lg px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-1 focus:ring-primary"
              />
            </div>
            <div>
              <label className="block text-xs text-muted mb-1.5">IS/OOS 분할점</label>
              <input
                type="date"
                value={dSplit}
                onChange={(e) => setDSplit(e.target.value)}
                className="w-full bg-background border border-border rounded-lg px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-1 focus:ring-primary"
              />
            </div>
          </div>
          <div className="mt-3 flex justify-end">
            <button
              onClick={() => onApply(dStart, dEnd, dSplit)}
              className="px-4 py-1.5 text-xs font-medium rounded-lg bg-primary text-background hover:opacity-90 transition-opacity"
            >
              적용
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
