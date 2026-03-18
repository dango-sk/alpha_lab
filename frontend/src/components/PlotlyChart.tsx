'use client';

import { memo } from 'react';
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
    color: '#6b7280',
    size: 11,
  },
  margin: { l: 50, r: 20, t: 30, b: 40 },
  xaxis: {
    gridcolor: 'rgba(0,0,0,0.06)',
    zerolinecolor: 'rgba(0,0,0,0.1)',
    tickfont: { size: 10, color: '#6b7280' },
  },
  yaxis: {
    gridcolor: 'rgba(0,0,0,0.06)',
    zerolinecolor: 'rgba(0,0,0,0.1)',
    tickfont: { size: 10, color: '#6b7280' },
  },
  legend: {
    font: { size: 10, color: '#6b7280' },
    bgcolor: 'transparent',
    orientation: 'h',
    y: -0.15,
  },
  hoverlabel: {
    bgcolor: '#ffffff',
    bordercolor: 'rgba(0,0,0,0.15)',
    font: { color: '#1a1a2e', size: 12 },
  },
};

const defaultConfig: Partial<Plotly.Config> = {
  displayModeBar: false,
  responsive: true,
};

export default memo(function PlotlyChart({
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
          hoverlabel: { ...darkLayout.hoverlabel, ...layout?.hoverlabel },
        }}
        config={{ ...defaultConfig, ...config }}
        useResizeHandler
        style={{ width: '100%' }}
      />
    </div>
  );
});
