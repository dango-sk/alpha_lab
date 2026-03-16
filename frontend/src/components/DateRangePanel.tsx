'use client';

import { useState } from 'react';
import { ChevronDown, ChevronUp } from 'lucide-react';
import { cn } from '@/lib/utils';

interface DateRangePanelProps {
  startDate: string;
  endDate: string;
  isOosSplit: string;
  onStartDateChange: (v: string) => void;
  onEndDateChange: (v: string) => void;
  onIsOosSplitChange: (v: string) => void;
  className?: string;
}

export default function DateRangePanel({
  startDate,
  endDate,
  isOosSplit,
  onStartDateChange,
  onEndDateChange,
  onIsOosSplitChange,
  className,
}: DateRangePanelProps) {
  const [open, setOpen] = useState(false);

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
                value={startDate}
                onChange={(e) => onStartDateChange(e.target.value)}
                className="w-full bg-background border border-border rounded-lg px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-1 focus:ring-primary"
              />
            </div>
            <div>
              <label className="block text-xs text-muted mb-1.5">종료일</label>
              <input
                type="date"
                value={endDate}
                onChange={(e) => onEndDateChange(e.target.value)}
                className="w-full bg-background border border-border rounded-lg px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-1 focus:ring-primary"
              />
            </div>
            <div>
              <label className="block text-xs text-muted mb-1.5">IS/OOS 분할점</label>
              <input
                type="date"
                value={isOosSplit}
                onChange={(e) => onIsOosSplitChange(e.target.value)}
                className="w-full bg-background border border-border rounded-lg px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-1 focus:ring-primary"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
