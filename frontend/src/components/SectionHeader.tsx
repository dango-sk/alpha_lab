'use client';

import { cn } from '@/lib/utils';

interface SectionHeaderProps {
  title: string;
  subtitle?: string;
  className?: string;
  children?: React.ReactNode;
}

export default function SectionHeader({
  title,
  subtitle,
  className,
  children,
}: SectionHeaderProps) {
  return (
    <div
      className={cn(
        'flex items-end justify-between pb-3 mb-6 border-b border-border',
        className
      )}
    >
      <div>
        <h2 className="text-lg font-semibold text-foreground">{title}</h2>
        {subtitle && (
          <p className="text-xs text-muted mt-0.5">{subtitle}</p>
        )}
      </div>
      {children && <div className="flex items-center gap-2">{children}</div>}
    </div>
  );
}
