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
            <div className="p-5">{children}</div>
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
