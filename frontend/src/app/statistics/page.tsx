'use client';

import { useState, useEffect } from 'react';
import { getConfig, getRobustness } from '@/lib/api';
import { Config, valueColor, fmtPct, fmtNum } from '@/lib/hooks';
import SectionHeader from '@/components/SectionHeader';
import DataTable from '@/components/DataTable';
import PlotlyChart from '@/components/PlotlyChart';
import FilterBar from '@/components/FilterBar';
import LoadingState from '@/components/LoadingState';

interface IsOosResult {
  total_return: number;
  cagr: number;
  mdd: number;
  sharpe: number;
  monthly_returns?: number[];
}

interface BmSignificance {
  mean_diff: number;
  t_stat: number;
  p_value: number;
  ci_lower: number;
  ci_upper: number;
  win_rate: number;
  significant: boolean;
}

interface RollingEntry {
  start_date: string;
  excess_return: number;
}

interface RollingResult {
  total_windows: number;
  positive_windows: number;
  win_rate: number;
  rolling_data: RollingEntry[];
}

interface RobustnessData {
  is_oos: {
    is_results: Record<string, IsOosResult>;
    oos_results: Record<string, IsOosResult>;
    benchmarks: {
      is: Record<string, { total_return: number; cagr: number }>;
      oos: Record<string, { total_return: number; cagr: number }>;
    };
  };
  stat: {
    bm_significance: Record<string, BmSignificance>;
  };
  rolling: Record<string, RollingResult>;
}

export default function StatisticsPage() {
  const [universe, setUniverse] = useState<'KOSPI' | 'KOSPI+KOSDAQ'>('KOSPI');
  const [rebalType, setRebalType] = useState<'monthly' | 'biweekly'>('monthly');
  const [config, setConfig] = useState<Config | null>(null);
  const [data, setData] = useState<RobustnessData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedStrategies, setSelectedStrategies] = useState<string[]>([]);

  useEffect(() => {
    setLoading(true);
    setError(null);

    Promise.all([
      getConfig(),
      getRobustness({ universe, rebal_type: rebalType }),
    ])
      .then(([cfg, rob]) => {
        setConfig(cfg);
        setData(rob);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [universe, rebalType]);

  if (loading) return <LoadingState message="강건성 데이터를 불러오는 중..." />;
  if (error) {
    return (
      <div className="flex items-center justify-center py-20 text-accent-red text-sm">
        오류: {error}
      </div>
    );
  }
  if (!data || !config) return null;

  const labels: Record<string, string> = { ...(config.strategy_labels ?? {}), KOSPI: universe === 'KOSPI+KOSDAQ' ? 'KRX 300' : 'KODEX 200' };
  const colors = config.strategy_colors ?? {};

  // All strategy keys available in robustness data
  const allStrategyKeys = Object.keys(data.is_oos?.is_results ?? {});
  // Auto-select all on first load
  if (selectedStrategies.length === 0 && allStrategyKeys.length > 0) {
    setSelectedStrategies(allStrategyKeys);
  }
  const strategyKeys = allStrategyKeys.filter((k) => selectedStrategies.includes(k));
  const label = (k: string) => labels[k] || k;

  // ──────── IS vs OOS Section ────────
  const isResults = data.is_oos?.is_results ?? {};
  const oosResults = data.is_oos?.oos_results ?? {};

  const isOosBarData: Plotly.Data[] = [
    {
      x: strategyKeys.map(label),
      y: strategyKeys.map((k) => isResults[k]?.sharpe ?? 0),
      name: 'IS Sharpe',
      type: 'bar' as const,
      marker: { color: '#2196F3' },
    },
    {
      x: strategyKeys.map(label),
      y: strategyKeys.map((k) => oosResults[k]?.sharpe ?? 0),
      name: 'OOS Sharpe',
      type: 'bar' as const,
      marker: { color: '#E91E63' },
    },
  ];

  const isOosTableData = strategyKeys.map((k) => {
    const is = isResults[k] ?? {} as IsOosResult;
    const oos = oosResults[k] ?? {} as IsOosResult;
    return {
      strategy: label(k),
      is_return: is.total_return ?? 0,
      is_cagr: is.cagr ?? 0,
      is_sharpe: is.sharpe ?? 0,
      is_mdd: is.mdd ?? 0,
      oos_return: oos.total_return ?? 0,
      oos_cagr: oos.cagr ?? 0,
      oos_sharpe: oos.sharpe ?? 0,
      oos_mdd: oos.mdd ?? 0,
    };
  });

  const isOosColumns = [
    { key: 'strategy', label: '전략', width: '120px' },
    { key: 'is_return', label: 'IS 수익률', align: 'right' as const, mono: true, format: (v: unknown) => fmtPct(v as number), colorFn: (v: unknown) => valueColor(v as number) },
    { key: 'is_cagr', label: 'IS CAGR', align: 'right' as const, mono: true, format: (v: unknown) => fmtPct(v as number), colorFn: (v: unknown) => valueColor(v as number) },
    { key: 'is_sharpe', label: 'IS Sharpe', align: 'right' as const, mono: true, format: (v: unknown) => fmtNum(v as number) },
    { key: 'is_mdd', label: 'IS MDD', align: 'right' as const, mono: true, format: (v: unknown) => fmtPct(v as number), colorFn: () => 'text-accent-red' },
    { key: 'oos_return', label: 'OOS 수익률', align: 'right' as const, mono: true, format: (v: unknown) => fmtPct(v as number), colorFn: (v: unknown) => valueColor(v as number) },
    { key: 'oos_cagr', label: 'OOS CAGR', align: 'right' as const, mono: true, format: (v: unknown) => fmtPct(v as number), colorFn: (v: unknown) => valueColor(v as number) },
    { key: 'oos_sharpe', label: 'OOS Sharpe', align: 'right' as const, mono: true, format: (v: unknown) => fmtNum(v as number) },
    { key: 'oos_mdd', label: 'OOS MDD', align: 'right' as const, mono: true, format: (v: unknown) => fmtPct(v as number), colorFn: () => 'text-accent-red' },
  ];

  // ──────── Bootstrap Significance Section ────────
  const bmSig = data.stat?.bm_significance ?? {};
  const sigKeys = Object.keys(bmSig);

  const sigTableData = sigKeys.map((k) => {
    const s = bmSig[k];
    return {
      strategy: label(k),
      mean_diff: s.mean_diff,
      t_stat: s.t_stat,
      p_value: s.p_value,
      ci: `[${fmtPct(s.ci_lower)}, ${fmtPct(s.ci_upper)}]`,
      win_rate: s.win_rate,
      significant: s.significant,
    };
  });

  const sigColumns = [
    { key: 'strategy', label: '전략', width: '120px' },
    { key: 'mean_diff', label: '월평균초과수익', align: 'right' as const, mono: true, format: (v: unknown) => fmtPct(v as number, 2), colorFn: (v: unknown) => valueColor(v as number) },
    { key: 't_stat', label: 't-stat', align: 'right' as const, mono: true, format: (v: unknown) => fmtNum(v as number, 2) },
    { key: 'p_value', label: 'p-value', align: 'right' as const, mono: true, format: (v: unknown) => fmtNum(v as number, 4) },
    { key: 'ci', label: '95% CI', align: 'center' as const, mono: true },
    { key: 'win_rate', label: 'Bootstrap 승률', align: 'right' as const, mono: true, format: (v: unknown) => `${((v as number) * 100).toFixed(1)}%` },
    {
      key: 'significant',
      label: '유의여부',
      align: 'center' as const,
      format: (v: unknown) => (v ? 'Yes' : 'No'),
      colorFn: (v: unknown) => (v ? 'text-accent-green' : 'text-accent-red'),
    },
  ];

  // Summary bar chart for significance (t-stat per strategy)
  const sigBarData: Plotly.Data[] = [
    {
      x: sigKeys.map(label),
      y: sigKeys.map((k) => bmSig[k].t_stat),
      type: 'bar' as const,
      marker: {
        color: sigKeys.map((k) => (bmSig[k].significant ? '#22c55e' : '#ef4444')),
      },
      name: 't-stat',
    },
  ];

  // ──────── Rolling Section ────────
  const rollingData = data.rolling ?? {};
  const rollingKeys = Object.keys(rollingData);

  const rollingChartTraces: Plotly.Data[] = rollingKeys.map((k) => {
    const rd = rollingData[k].rolling_data ?? [];
    return {
      x: rd.map((d) => d.start_date),
      y: rd.map((d) => d.excess_return * 100),
      type: 'scatter' as const,
      mode: 'lines' as const,
      name: label(k),
      line: { color: colors[k] || undefined, width: 1.5 },
    };
  });

  // Zero line shape
  const zeroLineShape: Partial<Plotly.Shape> = {
    type: 'line',
    x0: 0,
    x1: 1,
    xref: 'paper',
    y0: 0,
    y1: 0,
    yref: 'y',
    line: { color: 'rgba(255,255,255,0.3)', width: 1, dash: 'dash' },
  };

  const rollingTableData = rollingKeys.map((k) => {
    const r = rollingData[k];
    return {
      strategy: label(k),
      total_windows: r.total_windows,
      positive_windows: r.positive_windows,
      win_rate: r.win_rate,
    };
  });

  const rollingColumns = [
    { key: 'strategy', label: '전략', width: '120px' },
    { key: 'total_windows', label: '총 윈도우', align: 'right' as const, mono: true },
    { key: 'positive_windows', label: '양의 알파', align: 'right' as const, mono: true },
    { key: 'win_rate', label: '승률', align: 'right' as const, mono: true, format: (v: unknown) => `${((v as number) * 100).toFixed(1)}%`, colorFn: (v: unknown) => ((v as number) >= 0.5 ? 'text-accent-green' : 'text-accent-red') },
  ];

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Filter Bar */}
      <div className="flex items-center justify-between">
        <SectionHeader title="통계 검증" subtitle="강건성 테스트 및 통계적 유의성 분석" className="border-0 mb-0 pb-0" />
        <FilterBar
          universe={universe}
          onUniverseChange={setUniverse}
          rebalType={rebalType}
          onRebalTypeChange={setRebalType}
        />
      </div>

      {/* Strategy selector */}
      <div className="space-y-2">
        <label className="text-xs text-muted font-medium">전략 선택</label>
        <div className="flex flex-wrap gap-2">
          {allStrategyKeys.map((key) => {
            const selected = selectedStrategies.includes(key);
            const lbl = label(key);
            const color = colors[key] || '#6366f1';
            return (
              <button
                key={key}
                onClick={() => {
                  if (selected) {
                    setSelectedStrategies((prev) => prev.filter((k) => k !== key));
                  } else {
                    setSelectedStrategies((prev) => [...prev, key]);
                  }
                }}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium transition-all border"
                style={{
                  backgroundColor: selected ? color + '20' : 'transparent',
                  borderColor: selected ? color : 'var(--color-border)',
                  color: selected ? color : 'var(--color-muted)',
                }}
              >
                {lbl}
                {selected && <span className="ml-1 opacity-60">&times;</span>}
              </button>
            );
          })}
        </div>
      </div>

      {/* ─── IS vs OOS Section ─── */}
      <section className="space-y-4">
        <h3 className="text-sm font-semibold text-muted uppercase tracking-wider">
          IS vs OOS 비교
        </h3>

        <PlotlyChart
          data={isOosBarData}
          layout={{
            title: { text: 'IS vs OOS Sharpe Ratio', font: { size: 13, color: '#a1a1aa' } },
            barmode: 'group',
            yaxis: { title: { text: 'Sharpe Ratio', font: { size: 10 } } },
          }}
          height={320}
        />

        <DataTable columns={isOosColumns} data={isOosTableData} maxHeight="360px" />
      </section>

      {/* ─── Bootstrap Significance Section ─── */}
      <section className="space-y-4">
        <h3 className="text-sm font-semibold text-muted uppercase tracking-wider">
          Bootstrap 유의성 검정
        </h3>

        <PlotlyChart
          data={sigBarData}
          layout={{
            title: { text: 't-statistic by Strategy', font: { size: 13, color: '#a1a1aa' } },
            yaxis: { title: { text: 't-stat', font: { size: 10 } } },
            shapes: [
              {
                type: 'line',
                x0: 0,
                x1: 1,
                xref: 'paper',
                y0: 1.96,
                y1: 1.96,
                yref: 'y',
                line: { color: 'rgba(255,255,255,0.4)', width: 1, dash: 'dot' },
              },
              {
                type: 'line',
                x0: 0,
                x1: 1,
                xref: 'paper',
                y0: -1.96,
                y1: -1.96,
                yref: 'y',
                line: { color: 'rgba(255,255,255,0.4)', width: 1, dash: 'dot' },
              },
            ],
          }}
          height={300}
        />

        <DataTable columns={sigColumns} data={sigTableData} maxHeight="360px" />
      </section>

      {/* ─── Rolling 24-Month Window Section ─── */}
      <section className="space-y-4">
        <h3 className="text-sm font-semibold text-muted uppercase tracking-wider">
          Rolling 24개월 초과수익
        </h3>

        <PlotlyChart
          data={rollingChartTraces}
          layout={{
            title: { text: `Rolling 24M Excess Return vs ${universe === 'KOSPI' ? 'KOSPI' : 'KOSPI+KOSDAQ'}`, font: { size: 13, color: '#a1a1aa' } },
            yaxis: { title: { text: '초과수익률 (%)', font: { size: 10 } } },
            xaxis: { title: { text: '시작일', font: { size: 10 } } },
            shapes: [zeroLineShape],
          }}
          height={380}
        />

        <DataTable columns={rollingColumns} data={rollingTableData} maxHeight="360px" />
      </section>
    </div>
  );
}
