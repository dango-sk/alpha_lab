'use client';

import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { cn } from '@/lib/utils';
import PlotlyChart from '@/components/PlotlyChart';
import type { Components } from 'react-markdown';

interface MarkdownRendererProps {
  content: string;
  className?: string;
}

function ChartBlock({ json }: { json: string }) {
  try {
    const parsed = JSON.parse(json) as {
      type?: string;
      data?: Plotly.Data[];
      layout?: Partial<Plotly.Layout>;
    };
    const data = parsed.data ?? [];
    return (
      <div className="my-4">
        <PlotlyChart data={data} layout={parsed.layout} />
      </div>
    );
  } catch {
    return (
      <pre className="rounded-lg bg-red-900/30 border border-red-500/40 p-4 text-sm text-red-300 overflow-x-auto">
        Invalid chart JSON
      </pre>
    );
  }
}

const components: Components = {
  h1: ({ children }) => (
    <h1 className="text-2xl font-bold text-foreground mt-6 mb-3">{children}</h1>
  ),
  h2: ({ children }) => (
    <h2 className="text-xl font-semibold text-foreground mt-5 mb-2">{children}</h2>
  ),
  h3: ({ children }) => (
    <h3 className="text-lg font-semibold text-foreground mt-4 mb-2">{children}</h3>
  ),
  h4: ({ children }) => (
    <h4 className="text-base font-semibold text-foreground mt-3 mb-1">{children}</h4>
  ),
  p: ({ children }) => (
    <p className="text-foreground leading-relaxed mb-3">{children}</p>
  ),
  strong: ({ children }) => (
    <strong className="font-semibold text-foreground">{children}</strong>
  ),
  em: ({ children }) => (
    <em className="italic text-muted">{children}</em>
  ),
  a: ({ href, children }) => (
    <a href={href} className="text-primary underline hover:opacity-80" target="_blank" rel="noopener noreferrer">
      {children}
    </a>
  ),
  ul: ({ children }) => (
    <ul className="list-disc list-inside mb-3 space-y-1 text-foreground">{children}</ul>
  ),
  ol: ({ children }) => (
    <ol className="list-decimal list-inside mb-3 space-y-1 text-foreground">{children}</ol>
  ),
  li: ({ children }) => (
    <li className="text-foreground">{children}</li>
  ),
  blockquote: ({ children }) => (
    <blockquote className="border-l-4 border-primary pl-4 my-3 text-muted italic">
      {children}
    </blockquote>
  ),
  hr: () => <hr className="border-border my-6" />,
  table: ({ children }) => (
    <div className="overflow-x-auto my-4 rounded-lg border border-border">
      <table className="w-full text-sm">{children}</table>
    </div>
  ),
  thead: ({ children }) => (
    <thead className="bg-surface border-b border-border">{children}</thead>
  ),
  tbody: ({ children }) => (
    <tbody className="divide-y divide-border">{children}</tbody>
  ),
  tr: ({ children }) => (
    <tr className="hover:bg-white/[0.02] transition-colors">{children}</tr>
  ),
  th: ({ children }) => (
    <th className="px-3 py-2 text-left text-xs font-medium text-muted uppercase tracking-wider whitespace-nowrap">
      {children}
    </th>
  ),
  td: ({ children }) => (
    <td className="px-3 py-2 text-foreground font-mono text-xs whitespace-nowrap">
      {children}
    </td>
  ),
  code: ({ className, children, ...props }) => {
    const match = /language-(\w+)/.exec(className || '');
    const lang = match?.[1];
    const codeString = String(children).replace(/\n$/, '');

    // Chart block: render Plotly
    if (lang === 'chart') {
      return <ChartBlock json={codeString} />;
    }

    // Inline code (no language class and inside a <p> or similar)
    const isInline = !className;
    if (isInline) {
      return (
        <code className="bg-card text-primary px-1.5 py-0.5 rounded text-sm font-mono" {...props}>
          {children}
        </code>
      );
    }

    // Fenced code block
    return (
      <code className={cn('block', className)} {...props}>
        {children}
      </code>
    );
  },
  pre: ({ children }) => (
    <pre className="rounded-lg bg-card border border-border p-4 my-3 overflow-x-auto text-sm font-mono text-foreground leading-relaxed">
      {children}
    </pre>
  ),
};

export default function MarkdownRenderer({ content, className }: MarkdownRendererProps) {
  return (
    <div className={cn('text-foreground', className)}>
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
        {content}
      </ReactMarkdown>
    </div>
  );
}
