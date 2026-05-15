'use client';

import { cn } from '@/lib/utils';

interface FilterBarProps {
  universe: string;
  onUniverseChange: (v: 'KOSPI' | 'KOSPI+KOSDAQ') => void;
  rebalType: string;
  onRebalTypeChange: (v: 'monthly' | 'biweekly') => void;
  className?: string;
  children?: React.ReactNode;
}

function ToggleGroup({
  value,
  options,
  onChange
}: {
  value: string;
  options: { value: string; label: string }[];
  onChange: (v: string) => void;
}) {
  if (options.length <= 1) return null;
  return (
    <div className="flex rounded-lg overflow-hidden border border-border">
      {options.map((opt) => (
        <button
          key={opt.value}
          onClick={() => onChange(opt.value)}
          className={cn(
            'px-3 py-1.5 text-xs font-medium transition-all',
            value === opt.value
              ? 'bg-primary text-background'
              : 'bg-surface text-muted hover:text-foreground hover:bg-card-hover'
          )}
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
}

export default function FilterBar({
  universe,
  onUniverseChange,
  rebalType,
  onRebalTypeChange,
  className,
  children,
}: FilterBarProps) {
  return (
    <div className={cn('flex items-center gap-3 flex-wrap', className)}>
      <ToggleGroup
        value={universe}
        options={[
          { value: 'KOSPI', label: 'KOSPI' },
        ]}
        onChange={(v) => onUniverseChange(v as 'KOSPI' | 'KOSPI+KOSDAQ')}
      />
      <ToggleGroup
        value={rebalType}
        options={[
          { value: 'monthly', label: '월간' },
        ]}
        onChange={(v) => onRebalTypeChange(v as 'monthly' | 'biweekly')}
      />
      {children}
    </div>
  );
}
