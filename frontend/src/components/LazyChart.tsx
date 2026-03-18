'use client';

import { useRef, useState, useEffect, ReactNode } from 'react';

interface LazyChartProps {
  children: ReactNode;
  height?: number;
  className?: string;
}

export default function LazyChart({ children, height = 350, className }: LazyChartProps) {
  const ref = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setVisible(true);
          observer.disconnect();
        }
      },
      { rootMargin: '200px' }
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  return (
    <div ref={ref} className={className} style={{ minHeight: height }}>
      {visible ? children : (
        <div className="flex flex-col items-center justify-center gap-2" style={{ height }}>
          <div className="w-6 h-6 border-2 border-primary/30 border-t-primary rounded-full animate-spin" />
          <span className="text-xs text-muted">차트 로딩 중...</span>
        </div>
      )}
    </div>
  );
}
