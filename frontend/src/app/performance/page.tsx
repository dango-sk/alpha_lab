'use client';

import { useState, useEffect, useMemo, useCallback } from 'react';
import { getResults, getConfig, getRegimeAnalysis } from '@/lib/api';
import { StrategyResult, Config, valueColor, fmtPct, fmtNum } from '@/lib/hooks';
import SectionHeader from '@/components/SectionHeader';
import KpiCard from '@/components/KpiCard';
import DataTable from '@/components/DataTable';
import PlotlyChart from '@/components/PlotlyChart';
import FilterBar from '@/components/FilterBar';
import StrategySelector from '@/components/StrategySelector';
import LoadingState from '@/components/LoadingState';
import DateRangePanel from '@/components/DateRangePanel';
import LazyChart from '@/components/LazyChart';

// ─── Helpers ───

function calcDrawdown(values: number[]): number[] {
  const dd: number[] = [];
  let peak = values[0];
  for (const v of values) {
    if (v > peak) peak = v;
    dd.push((v / peak - 1) * 100);
  }
  return dd;
}

function calcSharpe(returns: number[]): number {
  if (returns.length < 2) return 0;
  const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
  const variance = returns.reduce((a, b) => a + (b - mean) ** 2, 0) / (returns.length - 1);
  const std = Math.sqrt(variance);
  if (std === 0) return 0;
  return (mean / std) * Math.sqrt(12);
}

function calcCumulativeReturn(returns: number[]): number {
  return returns.reduce((acc, r) => acc * (1 + r), 1) - 1;
}

interface MonthRow {
  year: number;
  [key: string]: unknown;
}

function buildMonthlyTable(
  dates: string[],
  returns: number[]
): MonthRow[] {
  const byYear: Record<number, Record<number, number>> = {};
  for (let i = 0; i < dates.length; i++) {
    const r = returns[i];
    if (r == null || isNaN(r) || !isFinite(r)) continue;
    const d = new Date(dates[i]);
    const y = d.getFullYear();
    const m = d.getMonth() + 1;
    if (!byYear[y]) byYear[y] = {};
    // If multiple returns in same month, compound them
    byYear[y][m] = byYear[y][m] !== undefined
      ? (1 + byYear[y][m]) * (1 + r) - 1
      : r;
  }

  return Object.keys(byYear)
    .map(Number)
    .sort()
    .map((year) => {
      const row: MonthRow = { year };
      const monthRets: number[] = [];
      for (let m = 1; m <= 12; m++) {
        const v = byYear[year][m];
        row[`m${m}`] = v !== undefined ? v : null;
        if (v !== undefined) monthRets.push(v);
      }
      row.ytd = monthRets.length > 0 ? calcCumulativeReturn(monthRets) : null;
      return row;
    });
}

interface YearlyStatsRow {
  year: number;
  ret: number;
  sharpe: number;
  mdd: number;
  months: number;
}

function buildYearlyStats(
  dates: string[],
  returns: number[],
  values: number[]
): YearlyStatsRow[] {
  const byYear: Record<number, { rets: number[]; vals: number[] }> = {};
  for (let i = 0; i < dates.length; i++) {
    const r = returns[i];
    const v = values[i];
    if (r == null || isNaN(r) || !isFinite(r)) continue;
    if (v == null || isNaN(v) || !isFinite(v)) continue;
    const y = new Date(dates[i]).getFullYear();
    if (!byYear[y]) byYear[y] = { rets: [], vals: [] };
    byYear[y].rets.push(r);
    byYear[y].vals.push(v);
  }

  return Object.keys(byYear)
    .map(Number)
    .sort()
    .map((year) => {
      const { rets, vals } = byYear[year];
      const ret = calcCumulativeReturn(rets);
      const sharpe = calcSharpe(rets);
      // MDD within year
      let peak = vals[0];
      let mdd = 0;
      for (const v of vals) {
        if (v > peak) peak = v;
        const dd = v / peak - 1;
        if (dd < mdd) mdd = dd;
      }
      return { year, ret, sharpe, mdd, months: rets.length };
    });
}

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

function getAllMonths(results: Record<string, StrategyResult>): string[] {
  const set = new Set<string>();
  for (const r of Object.values(results)) {
    for (const d of r.rebalance_dates) {
      set.add(d.slice(0, 7));
    }
  }
  return Array.from(set).sort();
}

// ─── Regime Section (extracted to avoid IIFE re-computation) ───

const REGIME_COLORS: Record<string, string> = {
  Bull: '#22c55e',
  Bear: '#ef4444',
};
const REGIME_ORDER = ['Bull', 'Bear'];

const manualBears = [
  { start: '2018-01-01', end: '2019-01-31', label: '미중 무역전쟁' },
  { start: '2020-01-01', end: '2020-03-31', label: '코로나 쇼크' },
  { start: '2021-06-01', end: '2022-10-31', label: '글로벌 금리 인상' },
  { start: '2024-07-01', end: '2024-10-31', label: 'AI 랠리 조정' },
];

const regimePerfColumns = [
  { key: 'strategy', label: '전략', align: 'left' as const },
  {
    key: 'regime',
    label: 'Regime',
    align: 'left' as const,
    format: (v: unknown) => v as string,
    colorFn: (v: unknown) => {
      if (v === 'Bull') return 'text-green-400';
      if (v === 'Bear') return 'text-red-400';
      return 'text-gray-400';
    },
  },
  { key: 'count', label: '개월수', align: 'right' as const, mono: true, format: (v: unknown) => fmtNum(v as number, 0) },
  {
    key: 'avg_monthly_return',
    label: '평균월수익률',
    align: 'right' as const,
    mono: true,
    format: (v: unknown) => fmtPct(v as number),
    colorFn: (v: unknown) => valueColor(v as number),
  },
  {
    key: 'total_return',
    label: '누적수익률',
    align: 'right' as const,
    mono: true,
    format: (v: unknown) => fmtPct(v as number),
    colorFn: (v: unknown) => valueColor(v as number),
  },
  {
    key: 'sharpe',
    label: 'Sharpe',
    align: 'right' as const,
    mono: true,
    format: (v: unknown) => fmtNum(v as number),
  },
  {
    key: 'win_rate',
    label: '승률',
    align: 'right' as const,
    mono: true,
    format: (v: unknown) => v == null ? '-' : fmtPct(v as number),
    colorFn: (v: unknown) => v == null ? 'text-muted' : valueColor((v as number) - 0.5),
  },
  {
    key: 'avg_excess',
    label: 'BM 대비 초과',
    align: 'right' as const,
    mono: true,
    format: (v: unknown) => v == null ? '-' : fmtPct(v as number),
    colorFn: (v: unknown) => v == null ? 'text-muted' : valueColor(v as number),
  },
];

function RegimeSection({ regimeData, regimeLoading, strategyKeys, labels, maWindow, setMaWindow, setRegimeLoading }: {
  regimeData: { regimes: Record<string, string>; summary: Record<string, Record<string, { count: number; avg_monthly_return: number; total_return: number; sharpe: number; win_rate: number; avg_excess: number }>>; regime_counts: Record<string, number> } | null;
  regimeLoading: boolean;
  strategyKeys: string[];
  labels: Record<string, string>;
  colors: Record<string, string>;
  maWindow: number;
  setMaWindow: (v: number) => void;
  setRegimeLoading: (v: boolean) => void;
}) {
  const [activeRegime, setActiveRegime] = useState<string>('Bull');
  const [localInput, setLocalInput] = useState(String(maWindow));

  const regimeStats = useMemo(() => {
    if (!regimeData) return null;
    const sorted = Object.keys(regimeData?.regimes ?? {}).sort().map((d) => regimeData?.regimes[d]);
    if (sorted.length === 0) return null;
    let transitions = 0;
    const bullDurations: number[] = [], bearDurations: number[] = [];
    let cur = sorted[0], len = 1;
    for (let i = 1; i < sorted.length; i++) {
      if (sorted[i] === cur) { len++; }
      else {
        transitions++;
        (cur === 'Bull' ? bullDurations : bearDurations).push(len);
        cur = sorted[i]; len = 1;
      }
    }
    (cur === 'Bull' ? bullDurations : bearDurations).push(len);
    const avg = (arr: number[]) => arr.length ? (arr.reduce((a, b) => a + b, 0) / arr.length).toFixed(1) : '-';
    return { transitions, avgBull: avg(bullDurations), avgBear: avg(bearDurations) };
  }, [regimeData?.regimes]);

  const regimeTimelineTraces = useMemo<Plotly.Data[]>(() => {
    const regimeDates = Object.keys(regimeData?.regimes ?? {}).sort();
    return REGIME_ORDER.map((regime) => ({
      type: 'bar' as const,
      name: regime,
      x: regimeDates.filter((d) => regimeData?.regimes[d] === regime),
      y: regimeDates.filter((d) => regimeData?.regimes[d] === regime).map(() => 1),
      marker: { color: REGIME_COLORS[regime] },
      hovertemplate: `%{x}<br>${regime}<extra></extra>`,
    }));
  }, [regimeData?.regimes]);

  const regimeTimelineLayout = useMemo<Partial<Plotly.Layout>>(() => ({
    title: { text: '시장 국면 (Regime) 분류', font: { size: 13, color: '#e4e4e7' } },
    barmode: 'stack' as const,
    bargap: 0,
    yaxis: { showticklabels: false, fixedrange: true },
    xaxis: { tickfont: { size: 10 } },
    legend: { font: { size: 10, color: '#a1a1aa' }, bgcolor: 'transparent', orientation: 'h', y: -0.25 },
    shapes: manualBears.map((b) => ({
      type: 'rect' as const,
      xref: 'x' as const,
      yref: 'paper' as const,
      x0: b.start,
      x1: b.end,
      y0: 0,
      y1: 1,
      fillcolor: 'rgba(239, 68, 68, 0.15)',
      line: { color: 'rgba(239, 68, 68, 0.4)', width: 1, dash: 'dot' as const },
    })),
    annotations: manualBears.map((b) => ({
      x: b.start,
      y: 1,
      xref: 'x' as const,
      yref: 'paper' as const,
      text: b.label,
      showarrow: false,
      font: { size: 9, color: '#fca5a5' },
      xanchor: 'left' as const,
      yanchor: 'bottom' as const,
      yshift: 2,
    })),
  }), []);

  const regimePerfRows = useMemo(() => {
    const rows: Record<string, unknown>[] = [];
    for (const key of strategyKeys) {
      const stratSummary = regimeData?.summary?.[key];
      if (!stratSummary) continue;
      const m = stratSummary[activeRegime];
      if (!m) continue;
      const isBm = key === 'KOSPI';
      rows.push({
        strategy: labels[key] || key,
        regime: activeRegime,
        count: m.count,
        avg_monthly_return: m.avg_monthly_return,
        total_return: m.total_return,
        sharpe: m.sharpe,
        win_rate: isBm ? null : m.win_rate,
        avg_excess: isBm ? null : m.avg_excess,
        _key: key,
      });
    }
    // Sort by Sharpe descending
    rows.sort((a, b) => ((b.sharpe as number) ?? -Infinity) - ((a.sharpe as number) ?? -Infinity));
    // Mark rank
    return rows.map((r, i) => ({ ...r, _rank: i + 1 }));
  }, [regimeData?.summary, strategyKeys, labels, activeRegime]);

  const regimeBarTraces = useMemo<Plotly.Data[]>(() => REGIME_ORDER.map((regime) => ({
    type: 'bar' as const,
    name: regime,
    x: strategyKeys.map((k) => labels[k] || k),
    y: strategyKeys.map((k) => regimeData?.summary?.[k]?.[regime]?.avg_monthly_return ?? null),
    marker: { color: REGIME_COLORS[regime] },
    hovertemplate: `%{x}<br>${regime}: %{y:.2%}<extra></extra>`,
  })), [regimeData?.summary, strategyKeys, labels]);

  const regimeBarLayout = useMemo<Partial<Plotly.Layout>>(() => ({
    title: { text: '국면별 평균 월수익률', font: { size: 13, color: '#e4e4e7' } },
    barmode: 'group' as const,
    yaxis: { tickformat: '.1%', tickfont: { size: 10 } },
    xaxis: { tickfont: { size: 10 } },
    legend: { font: { size: 10, color: '#a1a1aa' }, bgcolor: 'transparent', orientation: 'h', y: -0.25 },
  }), []);

  return (
    <div className="space-y-4 relative">
      {regimeLoading && (
        <div className="absolute inset-0 bg-background/60 flex items-center justify-center z-10 rounded-xl">
          <div className="flex items-center gap-2 text-sm text-muted">
            <span className="inline-block w-2 h-2 rounded-full bg-primary animate-bounce" style={{ animationDelay: '0ms' }} />
            <span className="inline-block w-2 h-2 rounded-full bg-primary animate-bounce" style={{ animationDelay: '150ms' }} />
            <span className="inline-block w-2 h-2 rounded-full bg-primary animate-bounce" style={{ animationDelay: '300ms' }} />
            <span>레짐 분석 중...</span>
          </div>
        </div>
      )}
      <div className="flex items-start justify-between">
        <SectionHeader
          title="시장 국면 분석 (Regime Analysis)"
          subtitle={`KOSPI 200 ${maWindow}일 MA 기준 | Bull: ${regimeData?.regime_counts?.Bull ?? 0}개월 | Bear: ${regimeData?.regime_counts?.Bear ?? 0}개월`}
        />
        <div className="flex items-center gap-1.5 mt-1">
          <label className="text-xs text-muted whitespace-nowrap">MA 기간</label>
          <input
            type="number"
            min={5}
            max={500}
            value={localInput}
            onChange={(e) => setLocalInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                const v = parseInt(localInput);
                if (!isNaN(v) && v >= 5 && v <= 500) { setRegimeLoading(true); setMaWindow(v); }
                else setLocalInput(String(maWindow));
              }
            }}
            className="w-16 px-2 py-1 rounded border border-border bg-surface text-center text-sm"
          />
          <span className="text-xs text-muted">일</span>
          <button
            onClick={() => {
              const v = parseInt(localInput);
              if (!isNaN(v) && v >= 5 && v <= 500) { setRegimeLoading(true); setMaWindow(v); }
              else setLocalInput(String(maWindow));
            }}
            className="px-2 py-1 text-xs rounded bg-primary/20 text-primary hover:bg-primary/30"
          >
            적용
          </button>
        </div>
      </div>
      <div className="flex gap-6 text-xs text-muted bg-surface/50 rounded-lg px-4 py-2.5 border border-border">
        <span><span className="text-green-400 font-medium">Bull (강세장)</span> — KOSPI 200 ≥ {maWindow}일 이동평균</span>
        <span><span className="text-red-400 font-medium">Bear (약세장)</span> — KOSPI 200 &lt; {maWindow}일 이동평균</span>
      </div>
      <LazyChart height={120}>
        <PlotlyChart data={regimeTimelineTraces} layout={regimeTimelineLayout} height={120} />
      </LazyChart>
      {regimeStats && (
        <div className="grid grid-cols-3 gap-3 text-center text-sm">
          <div className="bg-surface/50 border border-border rounded-lg py-2.5">
            <p className="text-xs text-muted mb-0.5">레짐 전환 횟수</p>
            <p className={`text-lg font-semibold ${regimeStats.transitions > 20 ? 'text-red-400' : regimeStats.transitions > 10 ? 'text-yellow-400' : 'text-green-400'}`}>{regimeStats.transitions}회</p>
          </div>
          <div className="bg-surface/50 border border-border rounded-lg py-2.5">
            <p className="text-xs text-muted mb-0.5">평균 Bull 지속</p>
            <p className="text-lg font-semibold text-green-400">{regimeStats.avgBull}개월</p>
          </div>
          <div className="bg-surface/50 border border-border rounded-lg py-2.5">
            <p className="text-xs text-muted mb-0.5">평균 Bear 지속</p>
            <p className="text-lg font-semibold text-red-400">{regimeStats.avgBear}개월</p>
          </div>
        </div>
      )}
      {regimePerfRows.length > 0 && (
        <div className="space-y-2">
          <div className="flex gap-1">
            {REGIME_ORDER.map((r) => (
              <button
                key={r}
                onClick={() => setActiveRegime(r)}
                className={`px-4 py-1.5 rounded text-sm font-medium transition-colors ${
                  activeRegime === r
                    ? r === 'Bull' ? 'bg-green-500/20 text-green-400 border border-green-500/40'
                      : 'bg-red-500/20 text-red-400 border border-red-500/40'
                    : 'text-muted hover:text-foreground border border-transparent'
                }`}
              >
                {r === 'Bull' ? '강세장 (Bull)' : '약세장 (Bear)'}
                <span className="ml-1.5 text-xs opacity-60">{regimeData?.regime_counts?.[r] ?? 0}개월</span>
              </button>
            ))}
          </div>
          <DataTable
            columns={regimePerfColumns.filter((c) => c.key !== 'regime')}
            data={regimePerfRows}
            maxHeight="none"
          />
        </div>
      )}
      {regimeBarTraces.length > 0 && (
        <LazyChart height={300}>
          <PlotlyChart data={regimeBarTraces} layout={regimeBarLayout} height={300} />
        </LazyChart>
      )}
      <div className="flex justify-end">
        <a
          href="/lab#regime-combo"
          className="text-sm text-primary hover:underline flex items-center gap-1"
        >
          전략 실험실에서 레짐 조합 실험 →
        </a>
      </div>
    </div>
  );
}

// ─── Page Component ───

export default function PerformancePage() {
  const [universe, setUniverse] = useState<'KOSPI' | 'KOSPI+KOSDAQ'>('KOSPI');
  const [rebalType, setRebalType] = useState<'monthly' | 'biweekly'>('monthly');
  // Applied dates (trigger API). Draft dates are managed inside DateRangePanel.
  const [startDate, setStartDate] = useState('2018-04-01');
  const [endDate, setEndDate] = useState('2026-04-01');
  const [isOosSplit, setIsOosSplit] = useState('2024-07-01');
  const [results, setResults] = useState<Record<string, StrategyResult>>({});
  const [config, setConfig] = useState<Config | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedStrategies, setSelectedStrategies] = useState<string[]>([]);
  const [maWindow, setMaWindow] = useState(50);
  const [regimeLoading, setRegimeLoading] = useState(false);
  const [regimeData, setRegimeData] = useState<{
    regimes: Record<string, string>;
    summary: Record<string, Record<string, { count: number; avg_monthly_return: number; total_return: number; sharpe: number; win_rate: number; avg_excess: number }>>;
    regime_counts: Record<string, number>;
  } | null>(null);

  // Load config first to get default dates
  useEffect(() => {
    getConfig().then((cfg) => {
      setConfig(cfg);
      const bc = cfg?.backtest_config;
      if (bc) {
        const s = bc.start || '2018-04-01';
        const e = bc.end || '2026-04-01';
        const sp = bc.oos_start || '2024-07-01';
        setStartDate(s);
        setEndDate(e);
        setIsOosSplit(sp);
      }
    }).catch(console.error);
  }, []);

  const applyDates = useCallback((start: string, end: string, split: string) => {
    if (start.length === 10 && end.length === 10) {
      setStartDate(start);
      setEndDate(end);
      setIsOosSplit(split);
    }
  }, []);

  useEffect(() => {
    setLoading(true);
    getResults({ start: startDate, end: endDate, universe, rebal_type: rebalType })
      .then((res) => {
        setResults(res);
        // Auto-select all strategies on first load
        setSelectedStrategies((prev) => prev.length > 0 ? prev.filter((k) => res[k]) : Object.keys(res));
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [universe, rebalType, startDate, endDate]);

  useEffect(() => {
    setRegimeLoading(true);
    getRegimeAnalysis({ start: startDate, end: endDate, universe, rebal_type: rebalType }, maWindow)
      .then(setRegimeData)
      .catch(() => setRegimeData(null))
      .finally(() => setRegimeLoading(false));
  }, [universe, rebalType, startDate, endDate, maWindow]);

  // All keys available for selection
  const allStrategyKeys = useMemo(() => Object.keys(results), [results]);

  // Filtered by selection
  const strategyKeys = useMemo(
    () => selectedStrategies.filter((k) => results[k]),
    [selectedStrategies, results]
  );

  const bmName = universe === 'KOSPI+KOSDAQ' ? 'KRX 300' : 'KODEX 200';
  const labels: Record<string, string> = { ...(config?.strategy_labels ?? {}), KOSPI: `벤치마크 (${bmName})` };
  const colors = config?.strategy_colors ?? {};
  const bc = config?.backtest_config;

  // ─── Monthly maps for heatmap & rolling ───
  const nonBmKeys = useMemo(
    () => strategyKeys.filter((k) => k !== 'KOSPI' && k !== 'KOSDAQ'),
    [strategyKeys]
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

  const rollingExcessTraces = useMemo(() => {
    const kospiMap = monthlyMaps['KOSPI'];
    if (!kospiMap || months.length < 12) return [];
    const traces: Plotly.Data[] = [];
    for (const key of nonBmKeys) {
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
        xVals.push(months[i]);
        yVals.push((cumS - cumB) * 100);
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
  }, [nonBmKeys, months, monthlyMaps, labels, colors]);

  // ─── IS/OOS split ───
  const isOosData = useMemo(() => {
    if (strategyKeys.length === 0) return [];
    return strategyKeys.map((key) => {
      const r = results[key];
      const oosIdx = r.rebalance_dates.findIndex((d) => d >= isOosSplit);
      const splitIdx = oosIdx > 0 ? oosIdx : r.rebalance_dates.length;
      const isRets = r.monthly_returns.slice(0, splitIdx);
      const oosRets = r.monthly_returns.slice(splitIdx);
      return {
        strategy: labels[key] || key,
        is_return: calcCumulativeReturn(isRets),
        is_sharpe: calcSharpe(isRets),
        oos_return: oosRets.length > 0 ? calcCumulativeReturn(oosRets) : null,
        oos_sharpe: oosRets.length > 0 ? calcSharpe(oosRets) : null,
      };
    });
  }, [results, isOosSplit, strategyKeys, labels]);

  // ─── Selected strategy for yearly detail (first strategy = A0) ───
  const [selectedStrategy, setSelectedStrategy] = useState<string>('');
  const [detailTab, setDetailTab] = useState<'chart' | 'summary' | 'yearly' | 'isoos' | 'rolling' | 'regime'>('chart');

  useEffect(() => {
    if (strategyKeys.length > 0 && !strategyKeys.includes(selectedStrategy)) {
      setSelectedStrategy(strategyKeys[0]);
    }
  }, [strategyKeys, selectedStrategy]);

  if (loading || !config) {
    return <LoadingState />;
  }

  const primaryKey = strategyKeys[0] || '';
  const primary = results[primaryKey];

  // ─── Comparison table data ───
  const comparisonData = strategyKeys.map((key) => {
    const r = results[key];
    return {
      strategy: labels[key] || key,
      total_return: r.total_return,
      cagr: r.cagr,
      mdd: r.mdd,
      sharpe: r.sharpe,
      avg_turnover: r.avg_turnover ?? 0,
      avg_size: r.avg_portfolio_size ?? 0,
    };
  });

  // ─── Cumulative return chart ───
  const cumRetTraces: Plotly.Data[] = strategyKeys.map((key) => {
    const r = results[key];
    return {
      x: r.rebalance_dates,
      y: r.portfolio_values.map((v) => (v - 1) * 100),
      name: labels[key] || key,
      type: 'scatter' as const,
      mode: 'lines' as const,
      line: { color: colors[key] || '#42A5F5', width: 2 },
      hovertemplate: '%{y:+.1f}%<extra></extra>',
    };
  });

  const cumRetLayout: Partial<Plotly.Layout> = {
    title: { text: '누적 수익률', font: { size: 13, color: '#e4e4e7' } },
    yaxis: { ticksuffix: '%' },
    shapes: [
      {
        type: 'line',
        x0: isOosSplit,
        x1: isOosSplit,
        y0: 0,
        y1: 1,
        yref: 'paper',
        line: { color: 'rgba(255,255,255,0.3)', width: 1, dash: 'dash' },
      },
    ],
    annotations: [
      {
        x: isOosSplit,
        y: 1,
        yref: 'paper',
        text: 'OOS Start',
        showarrow: false,
        font: { size: 9, color: '#a1a1aa' },
        yanchor: 'bottom',
      },
    ],
  };

  // ─── Drawdown chart ───
  const ddTraces: Plotly.Data[] = strategyKeys.map((key) => {
    const r = results[key];
    const dd = calcDrawdown(r.portfolio_values);
    return {
      x: r.rebalance_dates,
      y: dd,
      name: labels[key] || key,
      type: 'scatter' as const,
      mode: 'lines' as const,
      fill: 'tozeroy' as const,
      line: { color: colors[key] || '#42A5F5', width: 1 },
      fillcolor: (colors[key] || '#42A5F5') + '33',
      hovertemplate: '%{y:.1f}%<extra></extra>',
    };
  });

  const ddLayout: Partial<Plotly.Layout> = {
    title: { text: 'Drawdown', font: { size: 13, color: '#e4e4e7' } },
    yaxis: { ticksuffix: '%' },
  };

  // ─── Yearly monthly return table ───
  const selResult = results[selectedStrategy];
  const monthlyRows = selResult
    ? buildMonthlyTable(selResult.rebalance_dates, selResult.monthly_returns)
    : [];
  const yearlyStats = selResult
    ? buildYearlyStats(
        selResult.rebalance_dates,
        selResult.monthly_returns,
        selResult.portfolio_values
      )
    : [];

  const monthNames = ['1월', '2월', '3월', '4월', '5월', '6월', '7월', '8월', '9월', '10월', '11월', '12월'];

  const monthlyColumns = [
    { key: 'year', label: '연도', align: 'left' as const, width: '60px' },
    ...monthNames.map((name, i) => ({
      key: `m${i + 1}`,
      label: name,
      align: 'right' as const,
      mono: true,
      format: (v: unknown) => (v !== null && v !== undefined ? fmtPct(v as number) : ''),
      colorFn: (v: unknown) => (v !== null && v !== undefined ? valueColor(v as number) : ''),
    })),
    {
      key: 'ytd',
      label: 'YTD',
      align: 'right' as const,
      mono: true,
      format: (v: unknown) => (v !== null && v !== undefined ? fmtPct(v as number) : ''),
      colorFn: (v: unknown) => (v !== null && v !== undefined ? valueColor(v as number) : ''),
    },
  ];

  const yearlyStatsColumns = [
    { key: 'year', label: '연도', align: 'left' as const, width: '60px' },
    {
      key: 'ret',
      label: '수익률',
      align: 'right' as const,
      mono: true,
      format: (v: unknown) => fmtPct(v as number),
      colorFn: (v: unknown) => valueColor(v as number),
    },
    {
      key: 'sharpe',
      label: 'Sharpe',
      align: 'right' as const,
      mono: true,
      format: (v: unknown) => fmtNum(v as number),
    },
    {
      key: 'mdd',
      label: 'MDD',
      align: 'right' as const,
      mono: true,
      format: (v: unknown) => fmtPct(v as number),
      colorFn: () => 'text-accent-red',
    },
  ];

  // ─── Comparison table columns ───
  const comparisonColumns = [
    { key: 'strategy', label: '전략', align: 'left' as const },
    {
      key: 'total_return',
      label: '총수익률',
      align: 'right' as const,
      mono: true,
      format: (v: unknown) => fmtPct(v as number),
      colorFn: (v: unknown) => valueColor(v as number),
    },
    {
      key: 'cagr',
      label: 'CAGR',
      align: 'right' as const,
      mono: true,
      format: (v: unknown) => fmtPct(v as number),
      colorFn: (v: unknown) => valueColor(v as number),
    },
    {
      key: 'mdd',
      label: 'MDD',
      align: 'right' as const,
      mono: true,
      format: (v: unknown) => fmtPct(v as number),
      colorFn: () => 'text-accent-red',
    },
    {
      key: 'sharpe',
      label: 'Sharpe',
      align: 'right' as const,
      mono: true,
      format: (v: unknown) => fmtNum(v as number),
    },
    {
      key: 'avg_turnover',
      label: '평균회전율',
      align: 'right' as const,
      mono: true,
      format: (v: unknown) => fmtPct(v as number),
    },
    {
      key: 'avg_size',
      label: '평균종목수',
      align: 'right' as const,
      mono: true,
      format: (v: unknown) => fmtNum(v as number, 0),
    },
  ];

  // ─── IS/OOS columns ───
  const isOosColumns = [
    { key: 'strategy', label: '전략', align: 'left' as const },
    {
      key: 'is_return',
      label: 'IS 수익률',
      align: 'right' as const,
      mono: true,
      format: (v: unknown) => fmtPct(v as number),
      colorFn: (v: unknown) => valueColor(v as number),
    },
    {
      key: 'is_sharpe',
      label: 'IS Sharpe',
      align: 'right' as const,
      mono: true,
      format: (v: unknown) => fmtNum(v as number),
    },
    {
      key: 'oos_return',
      label: 'OOS 수익률',
      align: 'right' as const,
      mono: true,
      format: (v: unknown) =>
        v !== null && v !== undefined ? fmtPct(v as number) : '-',
      colorFn: (v: unknown) =>
        v !== null && v !== undefined ? valueColor(v as number) : '',
    },
    {
      key: 'oos_sharpe',
      label: 'OOS Sharpe',
      align: 'right' as const,
      mono: true,
      format: (v: unknown) =>
        v !== null && v !== undefined ? fmtNum(v as number) : '-',
    },
  ];

  return (
    <div className="animate-fade-in -mt-2">
      {/* ─── 상단: 필터 + 전략 선택 (compact) ─── */}
      <div className="glass-card p-3 mb-3">
        <div className="flex items-center justify-between flex-wrap gap-3">
          <FilterBar
            universe={universe}
            onUniverseChange={setUniverse}
            rebalType={rebalType}
            onRebalTypeChange={setRebalType}
          />
          <StrategySelector
            allKeys={allStrategyKeys}
            selected={selectedStrategies}
            labels={labels}
            colors={colors}
            onChange={setSelectedStrategies}
          />
          <DateRangePanel
            startDate={startDate}
            endDate={endDate}
            isOosSplit={isOosSplit}
            onApply={applyDates}
          />
        </div>
        {bc && (
          <p className="text-xs text-muted mt-2">
            기간: {startDate} ~ {endDate} | 리밸런싱: {rebalType === 'monthly' ? '월간' : '격주'}, 상위 {bc.top_n_stocks}종목 | 비중: 시총비례 + {bc.weight_cap_pct}% 캡 | 거래비용: 편도 {bc.transaction_cost_bp}bp | 유니버스: {universe} (BM: {universe === 'KOSPI+KOSDAQ' ? 'KRX 300' : 'KODEX 200'})
          </p>
        )}
      </div>

      {/* ─── KPI Cards ─── */}
      {primary && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
          {strategyKeys.map((key) => {
            const r = results[key];
            const borderColors: Record<string, string> = {
              0: 'border-t-accent-blue',
              1: 'border-t-[#90A4AE]',
              2: 'border-t-accent-green',
              3: 'border-t-accent-yellow',
            };
            const idx = strategyKeys.indexOf(key);
            return (
              <KpiCard
                key={key}
                label={labels[key] || key}
                value={fmtPct(r.total_return)}
                borderColor={borderColors[idx] || 'border-t-primary'}
                valueColor={valueColor(r.total_return)}
                subItems={[
                  { label: 'MDD', value: fmtPct(r.mdd), color: 'text-accent-red' },
                  { label: 'Sharpe', value: fmtNum(r.sharpe) },
                ]}
              />
            );
          })}
        </div>
      )}

      {/* ─── 탭 ─── */}
      <div className="glass-card p-1 mb-4 flex gap-0 rounded-lg sticky top-0 z-10">
        {(['chart', 'summary', 'yearly', 'isoos', 'rolling', 'regime'] as const).map((tab) => {
          const tabLabels = { chart: '차트', summary: '성과 요약', yearly: '연도별 성과', isoos: 'IS/OOS', rolling: '초과수익률', regime: '레짐 분석' };
          return (
            <button key={tab} onClick={() => setDetailTab(tab)}
              className={`flex-1 px-4 py-2 text-sm font-medium rounded-md transition-all ${
                detailTab === tab ? 'bg-primary text-white shadow-sm' : 'text-muted hover:text-foreground hover:bg-surface'
              }`}>
              {tabLabels[tab]}
            </button>
          );
        })}
      </div>

      {detailTab === 'chart' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">
          <div className="glass-card p-4">
            <h3 className="text-sm font-semibold text-foreground mb-2">누적 수익률</h3>
            <PlotlyChart data={cumRetTraces} layout={cumRetLayout} height={320} />
          </div>
          <div className="glass-card p-4">
            <h3 className="text-sm font-semibold text-foreground mb-2">Drawdown</h3>
            <LazyChart height={320}>
              <PlotlyChart data={ddTraces} layout={ddLayout} height={320} />
            </LazyChart>
          </div>
        </div>
      )}

      {detailTab === 'summary' && (
        <DataTable columns={comparisonColumns} data={comparisonData} maxHeight="none" />
      )}

      {detailTab === 'yearly' && (
        <div className="space-y-4">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-sm font-semibold text-foreground">연도별 성과</h3>
            <select
              value={selectedStrategy}
              onChange={(e) => setSelectedStrategy(e.target.value)}
              className="px-3 py-1.5 text-xs bg-surface border border-border rounded-lg text-foreground"
            >
              {strategyKeys.map((key) => (
                <option key={key} value={key}>
                  {labels[key] || key}
                </option>
              ))}
            </select>
          </div>
          <DataTable columns={monthlyColumns} data={monthlyRows} maxHeight="none" />
          <DataTable columns={yearlyStatsColumns} data={yearlyStats} maxHeight="none" />
        </div>
      )}

      {detailTab === 'isoos' && (
        <div>
          <h3 className="text-sm font-semibold text-foreground mb-2">
            In-Sample / Out-of-Sample 비교
            <span className="text-xs text-muted ml-2">IS: {startDate} ~ {isOosSplit} | OOS: {isOosSplit} ~ {endDate}</span>
          </h3>
          <DataTable columns={isOosColumns} data={isOosData} maxHeight="none" />
        </div>
      )}

      {detailTab === 'rolling' && rollingExcessTraces.length > 0 && (
        <div>
          <h3 className="text-sm font-semibold text-foreground mb-2">롤링 12개월 누적 초과수익률 (vs 벤치마크)</h3>
          <LazyChart height={350}>
            <PlotlyChart
              data={rollingExcessTraces}
              layout={{
                height: 350,
                margin: { l: 50, r: 20, t: 10, b: 50 },
                xaxis: { tickfont: { size: 10 }, dtick: 12, tickangle: 0 },
                yaxis: {
                  ticksuffix: '%p',
                  tickfont: { size: 10 },
                  title: { text: '초과수익률 (%p)', font: { size: 10, color: '#a1a1aa' } },
                },
                legend: { font: { size: 10, color: '#a1a1aa' }, bgcolor: 'transparent', orientation: 'h', y: -0.2 },
                shapes: [{
                  type: 'line', x0: 0, x1: 1, xref: 'paper', y0: 0, y1: 0,
                  line: { color: 'rgba(255,255,255,0.2)', width: 1, dash: 'dash' },
                }],
              }}
              height={350}
            />
          </LazyChart>
        </div>
      )}

      {detailTab === 'regime' && (regimeData || regimeLoading) && (
        <RegimeSection
          regimeData={regimeData}
          regimeLoading={regimeLoading}
          strategyKeys={strategyKeys}
          labels={labels}
          colors={colors}
          maWindow={maWindow}
          setMaWindow={setMaWindow}
          setRegimeLoading={setRegimeLoading}
        />
      )}
    </div>
  );
}

