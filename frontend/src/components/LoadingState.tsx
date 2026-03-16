'use client';

import { Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';

interface LoadingStateProps {
  message?: string;
  className?: string;
}

export default function LoadingState({ message = '데이터를 불러오는 중...', className }: LoadingStateProps) {
  return (
    <div className={cn('flex flex-col items-center justify-center py-20 gap-4', className)}>
      <div className="relative">
        <div className="w-10 h-10 rounded-full border-2 border-border" />
        <Loader2 size={40} className="absolute inset-0 text-primary animate-spin" />
      </div>
      <p className="text-sm text-muted">{message}</p>
    </div>
  );
}
