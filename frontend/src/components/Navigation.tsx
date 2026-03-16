'use client';

import { usePathname } from 'next/navigation';
import Link from 'next/link';
import {
  BarChart3,
  Calendar,
  PieChart,
  FlaskConical,
  TestTube2,
  MessageSquare,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react';
import { cn } from '@/lib/utils';

const navItems = [
  { href: '/performance', label: '성과 비교', icon: BarChart3 },
  { href: '/monthly', label: '월별 분석', icon: Calendar },
  { href: '/portfolio', label: '포트폴리오', icon: PieChart },
  { href: '/statistics', label: '통계 검증', icon: FlaskConical },
  { href: '/lab', label: '전략 실험실', icon: TestTube2 },
  { href: '/chat', label: 'AI 챗', icon: MessageSquare },
];

interface NavigationProps {
  collapsed: boolean;
  onToggle: () => void;
}

export default function Navigation({ collapsed, onToggle }: NavigationProps) {
  const pathname = usePathname();

  return (
    <aside
      className={cn(
        'h-screen sticky top-0 flex flex-col bg-surface border-r border-border transition-all duration-300 z-20',
        collapsed ? 'w-16' : 'w-56'
      )}
    >
      <div className="flex items-center h-14 px-4 border-b border-border shrink-0">
        {!collapsed && (
          <span className="text-sm font-semibold text-primary tracking-wide">
            Alpha Lab
          </span>
        )}
        <button
          onClick={onToggle}
          className={cn(
            'p-1.5 rounded-md text-muted hover:text-foreground hover:bg-card-hover transition-colors',
            collapsed ? 'mx-auto' : 'ml-auto'
          )}
        >
          {collapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
        </button>
      </div>

      <nav className="flex-1 py-3 px-2 space-y-1">
        {navItems.map((item) => {
          const isActive = pathname === item.href || pathname.startsWith(item.href + '/');
          const Icon = item.icon;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                'flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-all duration-200',
                isActive
                  ? 'bg-primary-dim text-primary font-medium'
                  : 'text-muted hover:text-foreground hover:bg-card-hover'
              )}
              title={collapsed ? item.label : undefined}
            >
              <Icon size={18} className="shrink-0" />
              {!collapsed && <span>{item.label}</span>}
              {isActive && !collapsed && (
                <div className="ml-auto w-1.5 h-1.5 rounded-full bg-primary" />
              )}
            </Link>
          );
        })}
      </nav>

      <div className="px-3 py-3 border-t border-border text-[10px] text-muted text-center">
        {!collapsed && 'v0.1.0'}
      </div>
    </aside>
  );
}
