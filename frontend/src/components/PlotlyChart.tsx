'use client';

import dynamic from 'next/dynamic';
import { cn } from '@/lib/utils';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface PlotlyChartProps {
  data: Plotly.Data[];
  layout?: Partial<Plotly.Layout>;
  config?: Partial<Plotly.Config>;
  className?: string;
  height?: number;
}

const darkLayout: Partial<Plotly.Layout> = {
  paper_bgcolor: 'transparent',
  plot_bgcolor: 'transparent',
  font: {
    family: "'Inter', system-ui, sans-serif",
    color: '#a1a1aa',
    size: 11,
  },
  margin: { l: 50, r: 20, t: 30, b: 40 },
  xaxis: {
    gridcolor: 'rgba(255,255,255,0.05)',
    zerolinecolor: 'rgba(255,255,255,0.08)',
    tickfont: { size: 10 },
  },
  yaxis: {
    gridcolor: 'rgba(255,255,255,0.05)',
    zerolinecolor: 'rgba(255,255,255,0.08)',
    tickfont: { size: 10 },
  },
  legend: {
    font: { size: 10, color: '#a1a1aa' },
    bgcolor: 'transparent',
    orientation: 'h',
    y: -0.15,
  },
  hoverlabel: {
    bgcolor: '#1a1a24',
    bordercolor: 'rgba(255,255,255,0.1)',
    font: { color: '#e4e4e7', size: 11 },
  },
};

const defaultConfig: Partial<Plotly.Config> = {
  displayModeBar: false,
  responsive: true,
};

export default function PlotlyChart({
  data,
  layout,
  config,
  className,
  height = 350,
}: PlotlyChartProps) {
  return (
    <div className={cn('glass-card p-4 animate-fade-in', className)}>
      <Plot
        data={data}
        layout={{
          ...darkLayout,
          ...layout,
          height,
          xaxis: { ...darkLayout.xaxis, ...layout?.xaxis },
          yaxis: { ...darkLayout.yaxis, ...layout?.yaxis },
        }}
        config={{ ...defaultConfig, ...config }}
        useResizeHandler
        style={{ width: '100%' }}
      />
    </div>
  );
}
