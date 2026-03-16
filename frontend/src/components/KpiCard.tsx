'use client';

import { cn } from '@/lib/utils';

interface KpiSubItem {
  label: string;
  value: string;
  color?: string;
}

interface KpiCardProps {
  label: string;
  value: string;
  borderColor?: string;
  valueColor?: string;
  subItems?: KpiSubItem[];
  className?: string;
}

export default function KpiCard({
  label,
  value,
  borderColor = 'border-t-primary',
  valueColor,
  subItems,
  className,
}: KpiCardProps) {
  return (
    <div
      className={cn(
        'glass-card p-5 border-t-2 animate-fade-in',
        borderColor,
        className
      )}
    >
      <p className="text-xs text-muted uppercase tracking-wider mb-2 font-medium">
        {label}
      </p>
      <p
        className={cn(
          'text-2xl font-semibold font-num tracking-tight',
          valueColor || 'text-foreground'
        )}
      >
        {value}
      </p>
      {subItems && subItems.length > 0 && (
        <div className="mt-3 pt-3 border-t border-border-light space-y-1.5">
          {subItems.map((item, i) => (
            <div key={i} className="flex justify-between items-center text-xs">
              <span className="text-muted">{item.label}</span>
              <span className={cn('font-num font-medium', item.color || 'text-foreground')}>
                {item.value}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
