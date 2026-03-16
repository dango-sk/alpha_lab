'use client';

import { useState, useEffect, useMemo, useCallback } from 'react';
import dynamic from 'next/dynamic';
import { getResults, getConfig, getAttribution } from '@/lib/api';
import { StrategyResult, Config, valueColor, fmtPct, fmtPctRaw } from '@/lib/hooks';
import SectionHeader from '@/components/SectionHeader';
import DataTable from '@/components/DataTable';
import PlotlyChart from '@/components/PlotlyChart';
import FilterBar from '@/components/FilterBar';
import LoadingState from '@/components/LoadingState';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface AttributionRow {
  종목명: string;
  섹터: string;
  '비중(%)': number;
  '종목수익률(%)': number;
  '기여도(%)': number;
}

interface DrillDown {
  strategyKey: string;
  strategyLabel: string;
  month: string;
  data: AttributionRow[];
}

// ─── Helpers ───

/** Given rebalance_dates and monthly_returns, group returns by YYYY-MM */
function buildMonthlyMap(dates: string[], returns: number[]): Record<string, number> {
  const map: Record<string, number> = {};
  for (let i = 0; i < dates.length; i++) {
    const ym = dates[i].slice(0, 7);
    if (map[ym] !== undefined) {
      map[ym] = (1 + map[ym]) * (1 + returns[i]) - 1;
    } else {
      map[ym] = returns[i];
    }
  }
  return map;
}

/** Get sorted unique YYYY-MM keys across all strategies */
function getAllMonths(results: Record<string, StrategyResult>): string[] {
  const set = new Set<string>();
  for (const r of Object.values(results)) {
    for (const d of r.rebalance_dates) {
      set.add(d.slice(0, 7));
    }
  }
  return Array.from(set).sort();
}

/** Get start/end date for a given YYYY-MM string */
function monthRange(ym: string): { start: string; end: string } {
  const [y, m] = ym.split('-').map(Number);
  const start = `${ym}-01`;
  const lastDay = new Date(y, m, 0).getDate();
  const end = `${ym}-${String(lastDay).padStart(2, '0')}`;
  return { start, end };
}

// ─── Dark layout for inline Plot usage ───
const darkBase: Partial<Plotly.Layout> = {
  paper_bgcolor: 'transparent',
  plot_bgcolor: 'transparent',
  font: { family: "'Inter', system-ui, sans-serif", color: '#a1a1aa', size: 11 },
  hoverlabel: {
    bgcolor: '#1a1a24',
    bordercolor: 'rgba(255,255,255,0.1)',
    font: { color: '#e4e4e7', size: 11 },
  },
};

// ─── Page Component ───

export default function MonthlyPage() {
  // Filters
  const [universe, setUniverse] = useState<'KOSPI' | 'KOSPI+KOSDAQ'>('KOSPI');
  const [rebalType, setRebalType] = useState<'monthly' | 'biweekly'>('monthly');

  // Data
  const [results, setResults] = useState<Record<string, StrategyResult>>({});
  const [config, setConfig] = useState<Config | null>(null);
  const [loading, setLoading] = useState(true);

  // Drill-down
  const [drillDown, setDrillDown] = useState<DrillDown | null>(null);
  const [drillLoading, setDrillLoading] = useState(false);

  // Fetch results + config
  useEffect(() => {
    setLoading(true);
    setDrillDown(null);
    Promise.all([
      getResults({ universe, rebal_type: rebalType }),
      getConfig(),
    ])
      .then(([res, cfg]) => {
        setResults(res);
        setConfig(cfg);
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [universe, rebalType]);

  // ─── Derived data ───
  const labels = config?.strategy_labels ?? {};
  const colors = config?.strategy_colors ?? {};
  const allKeys = config?.all_keys ?? Object.keys(results).filter((k) => k !== 'KOSPI');

  const strategyKeys = useMemo(
    () => allKeys.filter((k) => k !== 'KOSPI' && results[k]),
    [allKeys, results]
  );

  const months = useMemo(() => getAllMonths(results), [results]);

  const monthlyMaps = useMemo(() => {
    const maps: Record<string, Record<string, number>> = {};
    for (const key of Object.keys(results)) {
      const r = results[key];
      maps[key] = buildMonthlyMap(r.rebalance_dates, r.monthly_returns);
    }
    return maps;
  }, [results]);

  // ─── Heatmap data ───
  const heatmapData = useMemo(() => {
    if (strategyKeys.length === 0 || months.length === 0) return null;

    const z: number[][] = [];
    const text: string[][] = [];
    const yLabels: string[] = [];

    for (const key of strategyKeys) {
      const row: number[] = [];
      const textRow: string[] = [];
      yLabels.push(labels[key] || key);
      for (const m of months) {
        const val = (monthlyMaps[key]?.[m] ?? 0) * 100;
        row.push(val);
        textRow.push(`${val >= 0 ? '+' : ''}${val.toFixed(1)}%`);
      }
      z.push(row);
      text.push(textRow);
    }

    return { z, text, yLabels };
  }, [strategyKeys, months, monthlyMaps, labels]);

  // ─── Heatmap click handler ───
  const handleHeatmapClick = useCallback(
    (event: Plotly.PlotMouseEvent) => {
      if (!event.points || event.points.length === 0) return;
      const pt = event.points[0];

      // For heatmap, pointIndex is [row, col]
      const pointIndex = (pt as unknown as { pointIndex: [number, number] }).pointIndex;

      let sKey: string;
      let month: string;

      if (pointIndex) {
        sKey = strategyKeys[pointIndex[0]];
        month = months[pointIndex[1]];
      } else {
        // fallback: match by label text
        const yLabel = String(pt.y);
        sKey = strategyKeys.find((k) => (labels[k] || k) === yLabel) || strategyKeys[0];
        month = String(pt.x);
      }

      if (!sKey || !month) return;

      const { start, end } = monthRange(month);
      setDrillLoading(true);
      setDrillDown(null);

      getAttribution(sKey, start, end, universe, rebalType)
        .then((data: AttributionRow[]) => {
          setDrillDown({
            strategyKey: sKey,
            strategyLabel: labels[sKey] || sKey,
            month,
            data,
          });
        })
        .catch(console.error)
        .finally(() => setDrillLoading(false));
    },
    [strategyKeys, months, labels, universe, rebalType]
  );

  // ─── Rolling 12m excess return ───
  const rollingExcessTraces = useMemo(() => {
    const kospiMap = monthlyMaps['KOSPI'];
    if (!kospiMap || months.length < 12) return [];

    const traces: Plotly.Data[] = [];

    for (const key of strategyKeys) {
      const sMap = monthlyMaps[key];
      if (!sMap) continue;

      const xVals: string[] = [];
      const yVals: number[] = [];

      for (let i = 11; i < months.length; i++) {
        const window = months.slice(i - 11, i + 1);
        let cumS = 1;
        let cumB = 1;
        for (const m of window) {
          cumS *= 1 + (sMap[m] ?? 0);
          cumB *= 1 + (kospiMap[m] ?? 0);
        }
        const excess = (cumS - cumB) * 100;
        xVals.push(months[i]);
        yVals.push(excess);
      }

      traces.push({
        type: 'scatter',
        mode: 'lines',
        name: labels[key] || key,
        x: xVals,
        y: yVals,
        line: { color: colors[key], width: 2 },
        hovertemplate: '%{x}<br>%{y:+.1f}%p<extra>%{fullData.name}</extra>',
      });
    }

    return traces;
  }, [strategyKeys, months, monthlyMaps, labels, colors]);

  // ─── Drill-down bar chart data ───
  const drillBarData = useMemo(() => {
    if (!drillDown || drillDown.data.length === 0) return null;

    const sorted = [...drillDown.data].sort(
      (a, b) => a['기여도(%)'] - b['기여도(%)']
    );

    const bottom5 = sorted.slice(0, 5);
    const top5 = sorted.slice(-5).reverse();
    const combined = [...top5, ...bottom5.reverse()];

    return {
      combined,
      trace: {
        type: 'bar' as const,
        orientation: 'h' as const,
        y: combined.map((r) => r.종목명),
        x: combined.map((r) => r['기여도(%)']),
        marker: {
          color: combined.map((r) =>
            r['기여도(%)'] >= 0 ? '#22c55e' : '#ef4444'
          ),
        },
        text: combined.map(
          (r) => `${r['기여도(%)'] >= 0 ? '+' : ''}${r['기여도(%)'].toFixed(2)}%`
        ),
        textposition: 'auto' as const,
        hovertemplate:
          '%{y}<br>기여도: %{x:.2f}%<extra></extra>',
      },
    };
  }, [drillDown]);

  // ─── Drill-down summary ───
  const drillSummary = useMemo(() => {
    if (!drillDown || drillDown.data.length === 0) return null;

    const totalReturn = drillDown.data.reduce(
      (sum, r) => sum + r['기여도(%)'],
      0
    );

    const sorted = [...drillDown.data].sort(
      (a, b) => b['기여도(%)'] - a['기여도(%)']
    );

    return {
      totalReturn,
      top5: sorted.slice(0, 5),
      bottom5: sorted.slice(-5).reverse(),
    };
  }, [drillDown]);

  // ─── Render ───

  if (loading) {
    return (
      <div className="space-y-6 animate-fade-in">
        <SectionHeader title="월별 분석" subtitle="전략별 월간 수익률 히트맵 및 기여도 분석" />
        <LoadingState />
      </div>
    );
  }

  return (
    <div className="space-y-6 animate-fade-in">
      <SectionHeader title="월별 분석" subtitle="전략별 월간 수익률 히트맵 및 기여도 분석">
        <FilterBar
          universe={universe}
          onUniverseChange={setUniverse}
          rebalType={rebalType}
          onRebalTypeChange={setRebalType}
        />
      </SectionHeader>

      {/* ── Monthly Heatmap ── */}
      {heatmapData && (
        <div className="glass-card p-4 animate-fade-in">
          <h3 className="text-sm font-medium text-muted mb-2">월간 수익률 히트맵</h3>
          <p className="text-xs text-muted mb-4">
            셀을 클릭하면 해당 월의 종목별 기여도를 확인할 수 있습니다
          </p>
          <Plot
            data={[
              {
                type: 'heatmap' as const,
                z: heatmapData.z,
                x: months,
                y: heatmapData.yLabels,
                text: heatmapData.text as unknown as string[],
                texttemplate: '%{text}',
                colorscale: 'RdYlGn',
                colorbar: {
                  title: { text: '%', font: { size: 10, color: '#a1a1aa' } },
                  tickfont: { size: 10, color: '#a1a1aa' },
                  ticksuffix: '%',
                  len: 0.8,
                },
                hovertemplate: '%{y}<br>%{x}: %{text}<extra></extra>',
              } as Plotly.Data,
            ]}
            layout={{
              ...darkBase,
              height: Math.max(300, strategyKeys.length * 40 + 100),
              margin: { l: 140, r: 80, t: 10, b: 50 },
              xaxis: {
                gridcolor: 'rgba(255,255,255,0.05)',
                tickfont: { size: 9, color: '#a1a1aa' },
                tickangle: -45,
                dtick: 3,
              },
              yaxis: {
                gridcolor: 'rgba(255,255,255,0.05)',
                tickfont: { size: 10, color: '#a1a1aa' },
                autorange: 'reversed' as const,
              },
            }}
            config={{ displayModeBar: false, responsive: true }}
            onClick={handleHeatmapClick}
            useResizeHandler
            style={{ width: '100%' }}
          />
        </div>
      )}

      {/* ── Drill-down: Attribution ── */}
      {drillLoading && <LoadingState message="기여도 데이터를 불러오는 중..." />}

      {drillDown && drillDown.data.length > 0 && (
        <div className="space-y-4 animate-fade-in">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-medium text-foreground">
              {drillDown.strategyLabel} — {drillDown.month} 기여도 분석
            </h3>
            <button
              onClick={() => setDrillDown(null)}
              className="text-xs text-muted hover:text-foreground transition-colors px-2 py-1 rounded border border-border"
            >
              닫기
            </button>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Bar chart */}
            {drillBarData && (
              <div className="glass-card p-4">
                <h4 className="text-xs font-medium text-muted mb-2">
                  상위 5 / 하위 5 종목 기여도
                </h4>
                <Plot
                  data={[drillBarData.trace as Plotly.Data]}
                  layout={{
                    ...darkBase,
                    height: 350,
                    margin: { l: 100, r: 30, t: 10, b: 30 },
                    xaxis: {
                      gridcolor: 'rgba(255,255,255,0.05)',
                      ticksuffix: '%',
                      tickfont: { size: 10, color: '#a1a1aa' },
                    },
                    yaxis: {
                      gridcolor: 'rgba(255,255,255,0.05)',
                      tickfont: { size: 10, color: '#a1a1aa' },
                      autorange: 'reversed' as const,
                    },
                  }}
                  config={{ displayModeBar: false, responsive: true }}
                  useResizeHandler
                  style={{ width: '100%' }}
                />
              </div>
            )}

            {/* Summary tables */}
            {drillSummary && (
              <div className="space-y-4">
                <div className="glass-card p-4">
                  <p className="text-xs text-muted mb-1">포트폴리오 월간 수익률</p>
                  <p
                    className={`text-xl font-semibold font-num ${
                      drillSummary.totalReturn >= 0
                        ? 'text-accent-green'
                        : 'text-accent-red'
                    }`}
                  >
                    {drillSummary.totalReturn >= 0 ? '+' : ''}
                    {drillSummary.totalReturn.toFixed(2)}%
                  </p>
                </div>

                <DataTable
                  columns={[
                    { key: '종목명', label: '종목명' },
                    { key: '섹터', label: '섹터' },
                    {
                      key: '비중(%)',
                      label: '비중',
                      align: 'right',
                      mono: true,
                      format: (v) => `${Number(v).toFixed(1)}%`,
                    },
                    {
                      key: '종목수익률(%)',
                      label: '수익률',
                      align: 'right',
                      mono: true,
                      format: (v) =>
                        `${Number(v) >= 0 ? '+' : ''}${Number(v).toFixed(1)}%`,
                      colorFn: (v) => valueColor(Number(v)),
                    },
                    {
                      key: '기여도(%)',
                      label: '기여도',
                      align: 'right',
                      mono: true,
                      format: (v) =>
                        `${Number(v) >= 0 ? '+' : ''}${Number(v).toFixed(2)}%`,
                      colorFn: (v) => valueColor(Number(v)),
                    },
                  ]}
                  data={[
                    ...drillSummary.top5.map((r) => ({ ...r })),
                    ...drillSummary.bottom5.map((r) => ({ ...r })),
                  ]}
                  maxHeight="320px"
                />
              </div>
            )}
          </div>
        </div>
      )}

      {drillDown && drillDown.data.length === 0 && !drillLoading && (
        <div className="glass-card p-6 text-center text-muted text-sm">
          해당 월의 기여도 데이터가 없습니다
        </div>
      )}

      {/* ── Rolling 12-month Excess Return ── */}
      {rollingExcessTraces.length > 0 && (
        <div>
          <h3 className="text-sm font-medium text-muted mb-2">
            롤링 12개월 누적 초과수익률 (vs KOSPI)
          </h3>
          <PlotlyChart
            data={rollingExcessTraces}
            layout={{
              height: 350,
              margin: { l: 50, r: 20, t: 10, b: 40 },
              xaxis: {
                tickfont: { size: 10 },
                dtick: 6,
              },
              yaxis: {
                ticksuffix: '%p',
                tickfont: { size: 10 },
                title: {
                  text: '초과수익률 (%p)',
                  font: { size: 10, color: '#a1a1aa' },
                },
              },
              legend: {
                font: { size: 10, color: '#a1a1aa' },
                bgcolor: 'transparent',
                orientation: 'h',
                y: -0.2,
              },
              shapes: [
                {
                  type: 'line',
                  x0: 0,
                  x1: 1,
                  xref: 'paper',
                  y0: 0,
                  y1: 0,
                  line: {
                    color: 'rgba(255,255,255,0.2)',
                    width: 1,
                    dash: 'dash',
                  },
                },
              ],
            }}
            height={350}
          />
        </div>
      )}
    </div>
  );
}
