'use client';

import { useState } from 'react';
import './globals.css';
import Navigation from '@/components/Navigation';
import ChatPanel from '@/components/ChatPanel';

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const [navCollapsed, setNavCollapsed] = useState(false);
  const [chatOpen, setChatOpen] = useState(false);

  return (
    <html lang="ko">
      <head>
        <title>Alpha Lab</title>
        <meta name="description" content="Alpha Lab Quant Dashboard" />
      </head>
      <body className="antialiased">
        <div className="flex min-h-screen">
          <Navigation
            collapsed={navCollapsed}
            onToggle={() => setNavCollapsed((v) => !v)}
          />
          <main className="flex-1 min-w-0 overflow-auto">
            <header className="sticky top-0 z-10 h-14 flex items-center justify-between px-6 bg-background/80 backdrop-blur-md border-b border-border">
              <h1 className="text-sm font-semibold tracking-wide">
                <span className="text-primary">Alpha</span>{' '}
                <span className="text-foreground">Lab</span>
              </h1>
              <div className="flex items-center gap-3 text-xs text-muted">
                <span className="font-num">Quant Dashboard</span>
              </div>
            </header>
            <div className="p-6">{children}</div>
          </main>
          <ChatPanel
            open={chatOpen}
            onToggle={() => setChatOpen((v) => !v)}
          />
        </div>
      </body>
    </html>
  );
}
