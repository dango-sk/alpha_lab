'use client';

import { useState, useMemo } from 'react';
import { cn } from '@/lib/utils';

interface Column {
  key: string;
  label: string;
  align?: 'left' | 'center' | 'right';
  format?: (value: unknown, row: Record<string, unknown>) => string;
  colorFn?: (value: unknown) => string;
  mono?: boolean;
  width?: string;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Row = any;

interface DataTableProps {
  columns: Column[];
  data: Row[];
  className?: string;
  maxHeight?: string;
  onRowClick?: (row: Row, index: number) => void;
  onCellClick?: (row: Row, colKey: string, value: unknown) => void;
}

export default function DataTable({
  columns,
  data,
  className,
  maxHeight = '400px',
  onRowClick,
  onCellClick,
}: DataTableProps) {
  const [sortKey, setSortKey] = useState<string | null>(null);
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc');

  function handleSort(key: string) {
    if (sortKey === key) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortKey(key);
      setSortDir('desc');
    }
  }

  const sortedData = useMemo(() => {
    if (!sortKey) return data;
    return [...data].sort((a, b) => {
      const av = a[sortKey];
      const bv = b[sortKey];
      if (av == null && bv == null) return 0;
      if (av == null) return 1;
      if (bv == null) return -1;
      const n = typeof av === 'number' && typeof bv === 'number';
      const cmp = n ? av - bv : String(av).localeCompare(String(bv), 'ko');
      return sortDir === 'asc' ? cmp : -cmp;
    });
  }, [data, sortKey, sortDir]);

  return (
    <div className={cn('glass-card overflow-hidden', className)}>
      <div className="overflow-auto" style={{ maxHeight }}>
        <table className="w-full text-sm">
          <thead className="sticky top-0 z-10">
            <tr className="bg-surface border-b border-border">
              {columns.map((col) => (
                <th
                  key={col.key}
                  onClick={() => handleSort(col.key)}
                  className={cn(
                    'px-4 py-3 text-xs font-medium text-muted uppercase tracking-wider whitespace-nowrap cursor-pointer select-none hover:text-foreground transition-colors',
                    col.align === 'right' ? 'text-right' : col.align === 'center' ? 'text-center' : 'text-left'
                  )}
                  style={col.width ? { width: col.width } : undefined}
                >
                  <span className="inline-flex items-center gap-1">
                    {col.label}
                    {sortKey === col.key ? (
                      <span className="text-primary">{sortDir === 'asc' ? '↑' : '↓'}</span>
                    ) : (
                      <span className="opacity-20">↕</span>
                    )}
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sortedData.map((row, i) => (
              <tr
                key={i}
                className={cn(
                  'border-b border-border-light transition-colors',
                  'hover:bg-card-hover',
                  onRowClick && 'cursor-pointer'
                )}
                onClick={() => onRowClick?.(row, i)}
              >
                {columns.map((col) => {
                  const raw = row[col.key];
                  const display = col.format ? col.format(raw, row) : String(raw ?? '');
                  const colorClass = col.colorFn ? col.colorFn(raw) : '';
                  return (
                    <td
                      key={col.key}
                      className={cn(
                        'px-4 py-2.5 whitespace-nowrap',
                        col.align === 'right' ? 'text-right' : col.align === 'center' ? 'text-center' : 'text-left',
                        col.mono && 'font-num',
                        colorClass,
                        onCellClick && raw != null && 'cursor-pointer hover:underline'
                      )}
                      onClick={(e) => {
                        if (onCellClick && raw != null) {
                          e.stopPropagation();
                          onCellClick(row, col.key, raw);
                        }
                      }}
                    >
                      {display}
                    </td>
                  );
                })}
              </tr>
            ))}
            {data.length === 0 && (
              <tr>
                <td colSpan={columns.length} className="px-4 py-8 text-center text-muted text-sm">
                  데이터가 없습니다
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
