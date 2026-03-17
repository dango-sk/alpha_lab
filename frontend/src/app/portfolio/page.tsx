'use client';

import { useState, useEffect, useMemo } from 'react';
import { getResults, getConfig, getHoldings, getCharacteristics, getTurnover } from '@/lib/api';
import { StrategyResult, Config, valueColor, fmtPct, fmtNum } from '@/lib/hooks';
import SectionHeader from '@/components/SectionHeader';
import KpiCard from '@/components/KpiCard';
import DataTable from '@/components/DataTable';
import PlotlyChart from '@/components/PlotlyChart';
import FilterBar from '@/components/FilterBar';
import LoadingState from '@/components/LoadingState';

interface Holding {
  종목코드: string;
  종목명: string;
  섹터: string;
  '비중(%)': number;
  점수: number;
  PER: number;
  PBR: number;
  'EV/EBITDA': number;
  시가총액: number;
}

interface Characteristics {
  PER: number;
  PBR: number;
  'EV/EBITDA': number;
  PER_simple: number;
  PBR_simple: number;
  'EV/EBITDA_simple': number;
}

interface TurnoverData {
  added: Array<Record<string, unknown>>;
  removed: Array<Record<string, unknown>>;
  added_count: number;
  removed_count: number;
  retained_count: number;
  turnover_rate: number;
}

type CapCategory = '초대형' | '대형' | '중형' | '소형';

function classifyCap(marketCap: number): CapCategory {
  const trillion = 1_000_000_000_000;
  const billion = 100_000_000;
  if (marketCap >= 10 * trillion) return '초대형';
  if (marketCap >= 1 * trillion) return '대형';
  if (marketCap >= 3000 * billion) return '중형';
  return '소형';
}

function computeHHI(holdings: Holding[]): number {
  return holdings.reduce((sum, h) => sum + (h['비중(%)'] / 100) ** 2, 0) * 10000;
}

function computeTop5Weight(holdings: Holding[]): number {
  const sorted = [...holdings].sort((a, b) => b['비중(%)'] - a['비중(%)']);
  return sorted.slice(0, 5).reduce((sum, h) => sum + h['비중(%)'], 0);
}

function computeWeightedAvgMarketCap(holdings: Holding[]): number {
  const totalWeight = holdings.reduce((s, h) => s + h['비중(%)'], 0);
  if (totalWeight === 0) return 0;
  return holdings.reduce((s, h) => s + (h['비중(%)'] / totalWeight) * (h.시가총액 || 0), 0);
}

function formatMarketCap(value: number): string {
  const trillion = 1_000_000_000_000;
  const billion = 100_000_000;
  if (value >= trillion) return `${(value / trillion).toFixed(1)}조`;
  if (value >= billion) return `${(value / billion).toFixed(0)}억`;
  return `${value}`;
}

export default function PortfolioPage() {
  const [universe, setUniverse] = useState<'KOSPI' | 'KOSPI+KOSDAQ'>('KOSPI');
  const [rebalType, setRebalType] = useState<'monthly' | 'biweekly'>('monthly');
  const [config, setConfig] = useState<Config | null>(null);
  const [results, setResults] = useState<Record<string, StrategyResult>>({});
  const [selectedDate, setSelectedDate] = useState<string>('');
  const [holdingsMap, setHoldingsMap] = useState<Record<string, Holding[]>>({});
  const [charsMap, setCharsMap] = useState<Record<string, Characteristics>>({});
  const [turnoverMap, setTurnoverMap] = useState<Record<string, TurnoverData>>({});
  const [selectedStrategies, setSelectedStrategies] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [detailLoading, setDetailLoading] = useState(false);

  // Load config and results
  useEffect(() => {
    setLoading(true);
    Promise.all([
      getConfig(),
      getResults({ universe, rebal_type: rebalType }),
    ])
      .then(([cfg, res]) => {
        setConfig(cfg);
        setResults(res);
        const keys = (cfg.all_keys || Object.keys(res)).filter((k: string) => res[k]);
        const nonBm = keys.filter((k: string) => k !== 'KOSPI' && k !== 'KOSDAQ');
        setSelectedStrategies(nonBm.slice(0, 3));
        for (const key of keys) {
          if (res[key]?.rebalance_dates?.length) {
            const dates = res[key].rebalance_dates;
            setSelectedDate(dates[dates.length - 1]);
            break;
          }
        }
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [universe, rebalType]);

  // Available dates — use A0 (base strategy) dates as canonical schedule
  const availableDates = useMemo(() => {
    const base = results['A0']?.rebalance_dates;
    if (base?.length) return [...base].sort();
    // fallback: intersection of all strategies
    const dateSet = new Set<string>();
    Object.values(results).forEach((r) => {
      r.rebalance_dates?.forEach((d) => dateSet.add(d));
    });
    return [...dateSet].sort();
  }, [results]);

  // All available strategy keys (for selector)
  const allStrategyKeys = useMemo(() => {
    if (!config) return Object.keys(results);
    return (config.all_keys || Object.keys(results)).filter((k) => results[k] && k !== 'KOSPI' && k !== 'KOSDAQ');
  }, [config, results]);

  // Active strategy keys (filtered by selection)
  const strategyKeys = useMemo(
    () => selectedStrategies.filter((k) => results[k]),
    [selectedStrategies, results]
  );

  // Previous date for turnover
  const prevDate = useMemo(() => {
    const idx = availableDates.indexOf(selectedDate);
    return idx > 0 ? availableDates[idx - 1] : '';
  }, [availableDates, selectedDate]);

  // Stable key for useEffect dependency
  const strategyKeysStr = strategyKeys.join(',');

  // Load holdings, characteristics, and turnover when date changes
  useEffect(() => {
    if (!selectedDate || strategyKeys.length === 0) return;
    setDetailLoading(true);

    const holdingsPromises = strategyKeys.map((key) =>
      getHoldings(key, selectedDate, universe, rebalType)
        .then((data) => ({ key, data: data as Holding[] }))
        .catch(() => ({ key, data: [] as Holding[] }))
    );
    const charsPromises = strategyKeys.map((key) =>
      getCharacteristics(key, selectedDate, universe, rebalType)
        .then((data) => ({ key, data: data as Characteristics }))
        .catch(() => ({ key, data: null }))
    );
    const turnoverPromises = prevDate
      ? strategyKeys.map((key) =>
          getTurnover(key, selectedDate, prevDate, universe, rebalType)
            .then((data) => ({ key, data: data as TurnoverData }))
            .catch(() => ({ key, data: null }))
        )
      : [];

    Promise.all([
      Promise.all(holdingsPromises),
      Promise.all(charsPromises),
      Promise.all(turnoverPromises),
    ])
      .then(([holdingsResults, charsResults, turnoverResults]) => {
        const hMap: Record<string, Holding[]> = {};
        holdingsResults.forEach(({ key, data }) => { hMap[key] = data; });
        setHoldingsMap(hMap);

        const cMap: Record<string, Characteristics> = {};
        charsResults.forEach(({ key, data }) => { if (data) cMap[key] = data; });
        setCharsMap(cMap);

        const tMap: Record<string, TurnoverData> = {};
        turnoverResults.forEach(({ key, data }) => { if (data) tMap[key] = data; });
        setTurnoverMap(tMap);
      })
      .catch(console.error)
      .finally(() => setDetailLoading(false));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedDate, strategyKeysStr, universe, rebalType, prevDate]);

  const bmName = universe === 'KOSPI+KOSDAQ' ? 'KRX 300' : 'KODEX 200';
  const labels: Record<string, string> = { ...(config?.strategy_labels || {}), KOSPI: `벤치마크 (${bmName})` };
  const colors = config?.strategy_colors || {};

  // Top holdings comparison chart (dot chart)
  const topHoldingsChartData = useMemo(() => {
    // Collect top 5 stocks across all strategies
    const stockSet = new Set<string>();
    strategyKeys.forEach((key) => {
      const holdings = holdingsMap[key] || [];
      const sorted = [...holdings].sort((a, b) => b['비중(%)'] - a['비중(%)']);
      sorted.slice(0, 5).forEach((h) => stockSet.add(h.종목명));
    });
    const stocks = [...stockSet];

    return strategyKeys.map((key) => {
      const holdings = holdingsMap[key] || [];
      const byName: Record<string, number> = {};
      holdings.forEach((h) => { byName[h.종목명] = h['비중(%)']; });
      return {
        type: 'scatter' as const,
        orientation: 'h' as const,
        name: labels[key] || key,
        y: stocks,
        x: stocks.map((s) => byName[s] || 0),
        mode: 'markers' as const,
        marker: { color: colors[key] || '#6366f1', size: 12 },
        hovertemplate: '%{y}: %{x:.1f}%<extra>' + (labels[key] || key) + '</extra>',
      };
    });
  }, [holdingsMap, strategyKeys, labels, colors]);

  // Sector data for chart
  const sectorChartData = useMemo(() => {
    const sectorSet = new Set<string>();
    const sectorWeights: Record<string, Record<string, number>> = {};

    strategyKeys.forEach((key) => {
      const holdings = holdingsMap[key] || [];
      const bySector: Record<string, number> = {};
      holdings.forEach((h) => {
        const sector = h.섹터 || '기타';
        sectorSet.add(sector);
        bySector[sector] = (bySector[sector] || 0) + h['비중(%)'];
      });
      sectorWeights[key] = bySector;
    });

    const sectors = [...sectorSet].sort();

    return strategyKeys.map((key) => ({
      type: 'bar' as const,
      orientation: 'h' as const,
      name: labels[key] || key,
      y: sectors,
      x: sectors.map((s) => sectorWeights[key]?.[s] || 0),
      marker: { color: colors[key] || '#6366f1' },
      hovertemplate: '%{y}: %{x:.1f}%<extra>' + (labels[key] || key) + '</extra>',
    }));
  }, [holdingsMap, strategyKeys, labels, colors]);

  // Market cap distribution data
  const capCategories: CapCategory[] = ['초대형', '대형', '중형', '소형'];
  const capData = useMemo(() => {
    const capWeights: Record<string, Record<CapCategory, { weight: number; count: number }>> = {};

    strategyKeys.forEach((key) => {
      const holdings = holdingsMap[key] || [];
      const byCap: Record<CapCategory, { weight: number; count: number }> = {
        '초대형': { weight: 0, count: 0 },
        '대형': { weight: 0, count: 0 },
        '중형': { weight: 0, count: 0 },
        '소형': { weight: 0, count: 0 },
      };
      holdings.forEach((h) => {
        const cat = classifyCap(h.시가총액 || 0);
        byCap[cat].weight += h['비중(%)'];
        byCap[cat].count += 1;
      });
      capWeights[key] = byCap;
    });

    return capWeights;
  }, [holdingsMap, strategyKeys]);

  const capChartData = useMemo(() => {
    return strategyKeys.map((key) => ({
      type: 'bar' as const,
      orientation: 'h' as const,
      name: labels[key] || key,
      y: capCategories,
      x: capCategories.map((c) => capData[key]?.[c]?.weight || 0),
      marker: { color: colors[key] || '#6366f1' },
      hovertemplate: '%{y}: %{x:.1f}%<extra>' + (labels[key] || key) + '</extra>',
    }));
  }, [capData, strategyKeys, labels, colors]);

  // Cap distribution table
  const capTableData = useMemo(() => {
    return capCategories
      .filter((c) => strategyKeys.some((k) => (capData[k]?.[c]?.count || 0) > 0))
      .map((c) => {
        const row: Record<string, unknown> = { 구분: c };
        strategyKeys.forEach((key) => {
          const label = labels[key] || key;
          row[`${label} 비중`] = `${(capData[key]?.[c]?.weight || 0).toFixed(1)}%`;
          row[`${label} 종목수`] = capData[key]?.[c]?.count || 0;
        });
        return row;
      });
  }, [capData, strategyKeys, labels]);

  const capTableColumns = useMemo(() => {
    const cols: Array<{ key: string; label: string; align?: 'left' | 'right' | 'center'; mono?: boolean }> = [
      { key: '구분', label: '구분' },
    ];
    strategyKeys.forEach((key) => {
      const label = labels[key] || key;
      cols.push({ key: `${label} 비중`, label: `${label} 비중`, align: 'right', mono: true });
      cols.push({ key: `${label} 종목수`, label: `${label} 종목수`, align: 'right', mono: true });
    });
    return cols;
  }, [strategyKeys, labels]);

  // Characteristics table data — row-based (지표 = rows, 전략 = columns)
  const charsTableData = useMemo(() => {
    const activeKeys = strategyKeys.filter((k) => charsMap[k]);
    const metrics = [
      { key: 'PER', label: '가중평균 PER', field: 'PER', fmt: 1 },
      { key: 'PER_simple', label: '단순평균 PER', field: 'PER_simple', fmt: 1 },
      { key: 'PBR', label: '가중평균 PBR', field: 'PBR', fmt: 2 },
      { key: 'PBR_simple', label: '단순평균 PBR', field: 'PBR_simple', fmt: 2 },
      { key: 'EV_EBITDA', label: '가중평균 EV/EBITDA', field: 'EV/EBITDA', fmt: 2 },
      { key: 'EV_EBITDA_simple', label: '단순평균 EV/EBITDA', field: 'EV/EBITDA_simple', fmt: 2 },
    ];
    return metrics.map((m) => {
      const row: Record<string, unknown> = { 지표: m.label };
      activeKeys.forEach((k) => {
        const val = charsMap[k]?.[m.field as keyof Characteristics];
        row[labels[k] || k] = typeof val === 'number' ? val.toFixed(m.fmt) : '-';
      });
      return row;
    });
  }, [charsMap, strategyKeys, labels]);

  // Holdings table columns
  const holdingsColumns = [
    { key: '종목코드', label: '종목코드', align: 'left' as const, width: '90px' },
    { key: '종목명', label: '종목명', align: 'left' as const },
    { key: '섹터', label: '섹터', align: 'left' as const },
    {
      key: '비중(%)',
      label: '비중(%)',
      align: 'right' as const,
      mono: true,
      format: (v: unknown) => typeof v === 'number' ? v.toFixed(2) : String(v ?? ''),
    },
    {
      key: '점수',
      label: '점수',
      align: 'right' as const,
      mono: true,
      format: (v: unknown) => typeof v === 'number' ? v.toFixed(2) : String(v ?? ''),
    },
    {
      key: 'PER',
      label: 'PER',
      align: 'right' as const,
      mono: true,
      format: (v: unknown) => typeof v === 'number' ? v.toFixed(1) : String(v ?? ''),
      colorFn: (v: unknown) => {
        if (typeof v !== 'number') return '';
        return v < 10 ? 'text-accent-green' : v > 30 ? 'text-accent-red' : '';
      },
    },
    {
      key: 'PBR',
      label: 'PBR',
      align: 'right' as const,
      mono: true,
      format: (v: unknown) => typeof v === 'number' ? v.toFixed(2) : String(v ?? ''),
      colorFn: (v: unknown) => {
        if (typeof v !== 'number') return '';
        return v < 1 ? 'text-accent-green' : v > 3 ? 'text-accent-red' : '';
      },
    },
    {
      key: 'EV/EBITDA',
      label: 'EV/EBITDA',
      align: 'right' as const,
      mono: true,
      format: (v: unknown) => typeof v === 'number' ? v.toFixed(1) : String(v ?? ''),
    },
  ];

  const charsColumns = useMemo(() => {
    const cols: Array<{ key: string; label: string; align?: 'left' | 'right'; mono?: boolean }> = [
      { key: '지표', label: '지표', align: 'left' },
    ];
    strategyKeys.filter((k) => charsMap[k]).forEach((k) => {
      cols.push({ key: labels[k] || k, label: labels[k] || k, align: 'right', mono: true });
    });
    return cols;
  }, [strategyKeys, charsMap, labels]);

  const turnoverColumns = [
    { key: '종목코드', label: '종목코드', align: 'left' as const, width: '90px' },
    { key: '종목명', label: '종목명', align: 'left' as const },
    { key: '섹터', label: '섹터', align: 'left' as const },
    {
      key: '비중(%)',
      label: '비중(%)',
      align: 'right' as const,
      mono: true,
      format: (v: unknown) => typeof v === 'number' ? v.toFixed(2) : String(v ?? ''),
    },
  ];

  if (loading || !config) return <LoadingState />;

  return (
    <div className="space-y-6 animate-fade-in">
      <SectionHeader title="포트폴리오" subtitle="보유 종목 및 섹터 구성">
        <select
          value={selectedDate}
          onChange={(e) => setSelectedDate(e.target.value)}
          className="bg-surface border border-border rounded-lg px-3 py-1.5 text-xs text-foreground focus:outline-none focus:ring-1 focus:ring-primary"
        >
          {availableDates.map((d) => (
            <option key={d} value={d}>
              {d}
            </option>
          ))}
        </select>
      </SectionHeader>

      <FilterBar
        universe={universe}
        onUniverseChange={setUniverse}
        rebalType={rebalType}
        onRebalTypeChange={setRebalType}
      />

      {/* Strategy multi-selector */}
      <div className="space-y-2">
        <label className="text-xs text-muted font-medium">전략 선택</label>
        <div className="flex flex-wrap gap-2">
          {allStrategyKeys.map((key) => {
            const selected = selectedStrategies.includes(key);
            const label = config?.strategy_labels?.[key] || key;
            const color = config?.strategy_colors?.[key] || '#6366f1';
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
                {label}
                {selected && <span className="ml-1 opacity-60">&times;</span>}
              </button>
            );
          })}
        </div>
      </div>

      {detailLoading ? (
        <LoadingState message="포트폴리오 데이터를 불러오는 중..." />
      ) : (
        <>
          {/* Portfolio Characteristics */}
          <section>
            <h3 className="text-sm font-medium text-muted mb-3">포트폴리오 특성</h3>
            <DataTable
              columns={charsColumns}
              data={charsTableData}
              maxHeight="300px"
            />
          </section>

          {/* Concentration Analysis */}
          <section>
            <h3 className="text-sm font-medium text-muted mb-3">비중 집중도 분석</h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              {strategyKeys.map((key) => {
                const holdings = holdingsMap[key] || [];
                if (holdings.length === 0) return null;
                const hhi = computeHHI(holdings);
                const top5 = computeTop5Weight(holdings);
                return (
                  <KpiCard
                    key={key}
                    label={labels[key] || key}
                    value={`HHI ${fmtNum(hhi, 0)}`}
                    borderColor={`border-t-[${colors[key] || '#6366f1'}]`}
                    subItems={[
                      { label: 'Top 5 비중', value: `${top5.toFixed(1)}%` },
                      { label: '종목 수', value: `${holdings.length}` },
                    ]}
                  />
                );
              })}
            </div>
          </section>

          {/* Top Holdings Comparison */}
          {topHoldingsChartData.length > 0 && topHoldingsChartData[0]?.y?.length > 0 && (
            <section>
              <h3 className="text-sm font-medium text-muted mb-3">주요 보유 종목 비중 비교</h3>
              <PlotlyChart
                data={topHoldingsChartData}
                layout={{
                  xaxis: { title: { text: '비중 (%)' }, ticksuffix: '%' },
                  yaxis: { automargin: true },
                  margin: { l: 100, r: 20, t: 10, b: 50 },
                  legend: { orientation: 'h', y: -0.3 },
                }}
                height={Math.max(250, (topHoldingsChartData[0]?.y?.length || 5) * 40)}
              />
            </section>
          )}

          {/* Sector Comparison Chart */}
          <section>
            <h3 className="text-sm font-medium text-muted mb-3">섹터 비중 비교</h3>
            <PlotlyChart
              data={sectorChartData}
              layout={{
                barmode: 'group',
                xaxis: { title: { text: '비중(%)' }, ticksuffix: '%' },
                yaxis: { automargin: true, dtick: 1 },
                margin: { l: 120, r: 20, t: 10, b: 50 },
                legend: { orientation: 'h', y: -0.2 },
              }}
              height={Math.max(350, (sectorChartData[0]?.y?.length || 10) * 28)}
            />
          </section>

          {/* Market Cap Distribution */}
          <section>
            <h3 className="text-sm font-medium text-muted mb-3">시가총액 분포</h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
              {strategyKeys.map((key) => {
                const holdings = holdingsMap[key] || [];
                if (holdings.length === 0) return null;
                const avgCap = computeWeightedAvgMarketCap(holdings);
                return (
                  <KpiCard
                    key={key}
                    label={labels[key] || key}
                    value={`가중평균 시총 ${formatMarketCap(avgCap)}`}
                    borderColor={`border-t-[${colors[key] || '#6366f1'}]`}
                    subItems={[
                      { label: '종목수', value: `${holdings.length}` },
                    ]}
                  />
                );
              })}
            </div>
            <PlotlyChart
              data={capChartData}
              layout={{
                barmode: 'group',
                xaxis: { title: { text: '비중(%)' }, ticksuffix: '%' },
                yaxis: { automargin: true, categoryorder: 'array', categoryarray: ['소형', '중형', '대형', '초대형'] },
                margin: { l: 80, r: 20, t: 10, b: 50 },
                legend: { orientation: 'h', y: -0.3 },
              }}
              height={250}
            />
            {capTableData.length > 0 && (
              <div className="mt-4">
                <DataTable
                  columns={capTableColumns}
                  data={capTableData}
                  maxHeight="300px"
                />
              </div>
            )}
            <p className="text-xs text-muted mt-2">
              초대형: 시총 10조+ | 대형: 1조~10조 | 중형: 3000억~1조 | 소형: 3000억 미만
            </p>
          </section>

          {/* Rebalancing Turnover */}
          {prevDate && Object.keys(turnoverMap).length > 0 && (
            <section>
              <h3 className="text-sm font-medium text-muted mb-1">리밸런싱 변화</h3>
              <p className="text-xs text-muted mb-4">비교: {prevDate} → {selectedDate}</p>
              <div className="space-y-6">
                {strategyKeys.map((key) => {
                  const t = turnoverMap[key];
                  if (!t) return null;
                  return (
                    <div key={key} className="space-y-3">
                      <h4 className="text-sm font-medium text-foreground flex items-center gap-2">
                        <span
                          className="inline-block w-3 h-3 rounded-full"
                          style={{ backgroundColor: colors[key] || '#6366f1' }}
                        />
                        {labels[key] || key}
                      </h4>
                      <div className="grid grid-cols-4 gap-4">
                        <div className="text-center">
                          <p className="text-xs text-muted">신규 편입</p>
                          <p className="text-2xl font-bold text-foreground">{t.added_count}</p>
                        </div>
                        <div className="text-center">
                          <p className="text-xs text-muted">편출</p>
                          <p className="text-2xl font-bold text-foreground">{t.removed_count}</p>
                        </div>
                        <div className="text-center">
                          <p className="text-xs text-muted">유지</p>
                          <p className="text-2xl font-bold text-foreground">{t.retained_count}</p>
                        </div>
                        <div className="text-center">
                          <p className="text-xs text-muted">회전율</p>
                          <p className="text-2xl font-bold text-foreground">{(t.turnover_rate * 100).toFixed(0)}%</p>
                        </div>
                      </div>
                      {t.added.length > 0 && (
                        <div>
                          <p className="text-xs font-medium text-accent-green mb-1">신규 편입</p>
                          <DataTable
                            columns={turnoverColumns}
                            data={t.added}
                            maxHeight="250px"
                          />
                        </div>
                      )}
                      {t.removed.length > 0 && (
                        <div>
                          <p className="text-xs font-medium text-accent-red mb-1">편출</p>
                          <DataTable
                            columns={turnoverColumns}
                            data={t.removed}
                            maxHeight="250px"
                          />
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </section>
          )}

          {/* Holdings Detail Tables */}
          <section>
            <h3 className="text-sm font-medium text-muted mb-3">종목 상세</h3>
            <div className="space-y-6">
              {strategyKeys.map((key) => {
                const holdings = holdingsMap[key] || [];
                if (holdings.length === 0) return null;
                const sorted = [...holdings].sort((a, b) => b['비중(%)'] - a['비중(%)']);
                return (
                  <div key={key}>
                    <h4 className="text-sm font-medium text-foreground mb-2 flex items-center gap-2">
                      <span
                        className="inline-block w-3 h-3 rounded-full"
                        style={{ backgroundColor: colors[key] || '#6366f1' }}
                      />
                      {labels[key] || key}
                      <span className="text-xs text-muted font-normal">({sorted.length}종목)</span>
                    </h4>
                    <DataTable
                      columns={holdingsColumns}
                      data={sorted as unknown as Record<string, unknown>[]}
                      maxHeight="500px"
                    />
                  </div>
                );
              })}
            </div>
          </section>
        </>
      )}
    </div>
  );
}
