'use client';

import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { getResults, getConfig, getStrategies, runBacktest, fetchApi } from '@/lib/api';
import { StrategyResult, Config, valueColor, fmtPct, fmtNum } from '@/lib/hooks';
import SectionHeader from '@/components/SectionHeader';
import KpiCard from '@/components/KpiCard';
import DataTable from '@/components/DataTable';
import PlotlyChart from '@/components/PlotlyChart';
import LoadingState from '@/components/LoadingState';

// ─── Factor definitions ───

const FACTOR_CATEGORIES: Record<string, { factors: string[]; color: string }> = {
  '밸류에이션': {
    factors: ['T_PER', 'F_PER', 'T_EVEBITDA', 'F_EVEBITDA', 'T_PBR', 'F_PBR', 'T_PCF'],
    color: '#2196F3',
  },
  '회귀 매력도': {
    factors: ['ATT_PBR', 'ATT_EVIC', 'ATT_PER', 'ATT_EVEBIT'],
    color: '#E91E63',
  },
  '성장성': {
    factors: ['T_SPSG', 'F_SPSG'],
    color: '#4CAF50',
  },
  '차별화': {
    factors: ['F_EPS_M', 'PRICE_M', 'NDEBT_EBITDA', 'CURRENT'],
    color: '#FF9800',
  },
};

const FACTOR_LABELS: Record<string, string> = {
  T_PER: 'Trailing PER',
  F_PER: 'Forward PER',
  T_EVEBITDA: 'Trailing EV/EBITDA',
  F_EVEBITDA: 'Forward EV/EBITDA',
  T_PBR: 'Trailing PBR',
  F_PBR: 'Forward PBR',
  T_PCF: 'Trailing PCF',
  ATT_PBR: 'ATT PBR',
  ATT_EVIC: 'ATT EV/IC',
  ATT_PER: 'ATT PER',
  ATT_EVEBIT: 'ATT EV/EBIT',
  T_SPSG: 'Trailing 매출성장',
  F_SPSG: 'Forward 매출성장',
  F_EPS_M: 'EPS 모멘텀',
  PRICE_M: '가격 모멘텀',
  NDEBT_EBITDA: '순부채/EBITDA',
  CURRENT: '유동비율',
};

// ─── Parse weights from code ───

function parseWeights(code: string): Record<string, number> {
  const weights: Record<string, number> = {};
  // Match WEIGHTS_LARGE = { ... } block
  const match = code.match(/WEIGHTS_LARGE\s*=\s*\{([^}]+)\}/);

  if (!match) return weights;
  const block = match[1];
  // Match each key: value pair like "T_PER": -1.0 or 'T_PER': -1.0
  const entries = block.matchAll(/["'](\w+)["']\s*:\s*([+-]?\d+(?:\.\d+)?)/g);
  for (const entry of entries) {
    weights[entry[1]] = parseFloat(entry[2]);
  }
  return weights;
}

// ─── Saved strategy type ───

interface SavedStrategy {
  name: string;
  summary?: string;
  code?: string;
  description?: string;
  universe?: string;
  rebal_type?: string;
}

export default function LabPage() {
  // ─── Loading state ───
  const [initialLoading, setInitialLoading] = useState(true);

  // ─── Config & strategies ───
  const [config, setConfig] = useState<Config | null>(null);
  const [defaultCode, setDefaultCode] = useState('');
  const [savedStrategies, setSavedStrategies] = useState<SavedStrategy[]>([]);
  const [selectedStrategy, setSelectedStrategy] = useState('');

  // ─── Strategy settings ───
  const [universe, setUniverse] = useState<'KOSPI' | 'KOSPI+KOSDAQ'>('KOSPI');
  const [weightCapPct, setWeightCapPct] = useState(5);
  const [topN, setTopN] = useState(30);
  const [txCostBp, setTxCostBp] = useState(30);
  const [rebalType, setRebalType] = useState<'monthly' | 'biweekly'>('monthly');

  // ─── Code ───
  const [code, setCode] = useState('');
  const [codeExpanded, setCodeExpanded] = useState(false);

  // ─── Backtest ───
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressMsg, setProgressMsg] = useState('');
  const progressRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [backtestResults, setBacktestResults] = useState<Record<string, StrategyResult> | null>(null);
  const [baseResults, setBaseResults] = useState<Record<string, StrategyResult> | null>(null);

  // ─── Save ───
  const [saveName, setSaveName] = useState('');
  const [saveDesc, setSaveDesc] = useState('');
  const [saving, setSaving] = useState(false);
  const [saveMsg, setSaveMsg] = useState('');

  // ─── Auto-generate strategy name ───
  const autoName = useMemo(() => {
    const uni = universe === 'KOSPI' ? '코스피' : '코스피+코스닥';
    const rebal = rebalType === 'monthly' ? '월간' : '격주';
    return `수정전략_${uni}_cap${weightCapPct}%_top${topN}_tx${txCostBp}bp_${rebal}`;
  }, [universe, weightCapPct, topN, txCostBp, rebalType]);

  // ─── Init ───
  useEffect(() => {
    Promise.all([getConfig(), getStrategies(), getResults({ universe: 'KOSPI', rebal_type: 'monthly' })])
      .then(([cfg, strats, results]) => {
        setConfig(cfg);
        setDefaultCode(cfg.default_strategy_code || '');

        // Check if AI chat injected a strategy code
        const injectedCode = localStorage.getItem('alpha-lab-strategy-code');
        if (injectedCode) {
          setCode(injectedCode);
          setCodeExpanded(true);
          localStorage.removeItem('alpha-lab-strategy-code');
        } else {
          setCode(cfg.default_strategy_code || '');
        }

        setSavedStrategies(strats || []);
        setBaseResults(results || null);
        if (cfg.backtest_config) {
          setWeightCapPct(cfg.backtest_config.weight_cap_pct ?? 5);
          setTopN(cfg.backtest_config.top_n_stocks ?? 30);
          setTxCostBp(cfg.backtest_config.transaction_cost_bp ?? 30);
          setUniverse((cfg.backtest_config.universe as 'KOSPI' | 'KOSPI+KOSDAQ') ?? 'KOSPI');
        }
      })
      .catch(console.error)
      .finally(() => setInitialLoading(false));
  }, []);

  // Reload base results when universe/rebalType changes
  useEffect(() => {
    if (initialLoading) return;
    getResults({ universe, rebal_type: rebalType })
      .then(setBaseResults)
      .catch(console.error);
  }, [universe, rebalType, initialLoading]);

  // ─── Load strategy ───
  const loadStrategy = useCallback((name: string) => {
    setSelectedStrategy(name);
    if (!name) {
      setCode(defaultCode);
      return;
    }
    const found = savedStrategies.find((s) => s.name === name);
    if (found?.code) {
      setCode(found.code);
    }
  }, [savedStrategies, defaultCode]);

  // ─── Delete strategy ───
  const deleteStrategy = useCallback(async () => {
    if (!selectedStrategy) return;
    if (!confirm(`"${selectedStrategy}" 전략을 삭제하시겠습니까?`)) return;
    try {
      await fetchApi(`/api/strategies/${encodeURIComponent(selectedStrategy)}`, { method: 'DELETE' });
      setSavedStrategies((prev) => prev.filter((s) => s.name !== selectedStrategy));
      setSelectedStrategy('');
      setCode(defaultCode);
    } catch (e) {
      alert(`삭제 실패: ${(e as Error).message}`);
    }
  }, [selectedStrategy, defaultCode]);

  // ─── Reset ───
  const resetCode = useCallback(() => {
    setSelectedStrategy('');
    setCode(defaultCode);
    setBacktestResults(null);
  }, [defaultCode]);

  // ─── Run backtest ───
  const PROGRESS_STEPS = [
    { at: 5, msg: '유니버스 로딩...' },
    { at: 15, msg: '팩터 데이터 수집...' },
    { at: 30, msg: '종목 스코어링...' },
    { at: 50, msg: '포트폴리오 구성...' },
    { at: 70, msg: '수익률 계산...' },
    { at: 85, msg: '성과 지표 산출...' },
    { at: 95, msg: '마무리...' },
  ];

  const handleBacktest = useCallback(async () => {
    setRunning(true);
    setBacktestResults(null);
    setProgress(0);
    setProgressMsg('백테스트 시작...');

    // Simulate progress
    let step = 0;
    progressRef.current = setInterval(() => {
      if (step < PROGRESS_STEPS.length) {
        setProgress(PROGRESS_STEPS[step].at);
        setProgressMsg(PROGRESS_STEPS[step].msg);
        step++;
      }
    }, 2500);

    try {
      const result = await runBacktest(code, {
        universe,
        rebal_type: rebalType,
        weight_cap_pct: weightCapPct,
        top_n_stocks: topN,
        tx_cost_bp: txCostBp,
      });
      setProgress(100);
      setProgressMsg('완료!');
      setBacktestResults(result);
    } catch (e) {
      alert(`백테스트 실패: ${(e as Error).message}`);
    } finally {
      if (progressRef.current) clearInterval(progressRef.current);
      setTimeout(() => setRunning(false), 500);
    }
  }, [code, universe, rebalType, weightCapPct, topN, txCostBp]);

  // ─── Save strategy ───
  const handleSave = useCallback(async () => {
    if (!saveName.trim()) {
      alert('전략 이름을 입력해주세요.');
      return;
    }
    setSaving(true);
    setSaveMsg('');
    try {
      await fetchApi('/api/strategies', {
        method: 'POST',
        body: JSON.stringify({
          name: saveName.trim(),
          code,
          description: saveDesc,
          results: backtestResults,
          universe,
          rebal_type: rebalType,
        }),
        headers: { 'Content-Type': 'application/json' },
      });
      setSaveMsg('저장 완료');
      setSaveName('');
      setSaveDesc('');
      const strats = await getStrategies();
      setSavedStrategies(strats || []);
    } catch (e) {
      setSaveMsg(`저장 실패: ${(e as Error).message}`);
    } finally {
      setSaving(false);
    }
  }, [saveName, saveDesc, code, backtestResults, universe, rebalType]);

  // ─── Parse weights ───
  const weights = useMemo(() => parseWeights(code), [code]);

  // ─── Build results table data ───
  const tableData = useMemo(() => {
    const rows: Record<string, unknown>[] = [];
    // Base A0
    if (baseResults?.['A0']) {
      const r = baseResults['A0'];
      rows.push({
        strategy: '기존전략 (A0)',
        total_return: r.total_return,
        cagr: r.cagr,
        mdd: r.mdd,
        sharpe: r.sharpe,
      });
    }
    // Custom
    if (backtestResults?.['CUSTOM']) {
      const r = backtestResults['CUSTOM'];
      rows.push({
        strategy: '수정전략 (CUSTOM)',
        total_return: r.total_return,
        cagr: r.cagr,
        mdd: r.mdd,
        sharpe: r.sharpe,
      });
    }
    // BM
    const bm = backtestResults?.['KOSPI'] || baseResults?.['KOSPI'];
    if (bm) {
      rows.push({
        strategy: 'BM (KOSPI)',
        total_return: bm.total_return,
        cagr: bm.cagr,
        mdd: bm.mdd,
        sharpe: bm.sharpe,
      });
    }
    return rows;
  }, [baseResults, backtestResults]);

  const tableColumns = [
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
      colorFn: (v: unknown) => valueColor(v as number),
    },
  ];

  // ─── Chart data ───
  const chartData = useMemo(() => {
    const traces: Plotly.Data[] = [];
    // A0 from base results
    if (baseResults?.['A0']) {
      const r = baseResults['A0'];
      traces.push({
        x: r.rebalance_dates,
        y: r.portfolio_values,
        name: '기존전략 (A0)',
        type: 'scatter',
        mode: 'lines',
        line: { color: '#6366f1', width: 2 },
      });
    }
    // CUSTOM from backtest
    if (backtestResults?.['CUSTOM']) {
      const r = backtestResults['CUSTOM'];
      traces.push({
        x: r.rebalance_dates,
        y: r.portfolio_values,
        name: '수정전략 (CUSTOM)',
        type: 'scatter',
        mode: 'lines',
        line: { color: '#22c55e', width: 2 },
      });
    }
    // BM
    const bm = backtestResults?.['KOSPI'] || baseResults?.['KOSPI'];
    if (bm) {
      traces.push({
        x: bm.rebalance_dates,
        y: bm.portfolio_values,
        name: 'BM (KOSPI)',
        type: 'scatter',
        mode: 'lines',
        line: { color: '#a1a1aa', width: 1.5, dash: 'dot' },
      });
    }
    return traces;
  }, [baseResults, backtestResults]);

  // ─── Render ───

  if (initialLoading) {
    return <LoadingState message="전략 실험실 로딩 중..." />;
  }

  const maxAbsWeight = Math.max(...Object.values(weights).map(Math.abs), 1);

  return (
    <div className="space-y-6 animate-fade-in">
      <SectionHeader title="전략 실험실" subtitle="커스텀 백테스트 및 전략 실험" />

      {/* ─── Saved Strategy Selector ─── */}
      <div className="glass-card p-5">
        <div className="flex flex-wrap items-center gap-3">
          <label className="text-xs text-muted font-medium">저장된 전략</label>
          <select
            className="bg-surface border border-border rounded-lg px-3 py-2 text-sm text-foreground min-w-[200px] focus:outline-none focus:ring-1 focus:ring-primary"
            value={selectedStrategy}
            onChange={(e) => loadStrategy(e.target.value)}
          >
            <option value="">기본 전략</option>
            {savedStrategies.map((s) => (
              <option key={s.name} value={s.name}>
                {s.name}
              </option>
            ))}
          </select>
          {selectedStrategy && (
            <button
              onClick={deleteStrategy}
              className="px-3 py-2 text-xs rounded-lg bg-accent-red/10 text-accent-red hover:bg-accent-red/20 transition-colors"
            >
              삭제
            </button>
          )}
          <button
            onClick={resetCode}
            className="px-3 py-2 text-xs rounded-lg bg-surface border border-border text-muted hover:text-foreground transition-colors"
          >
            초기화
          </button>
        </div>
      </div>

      {/* ─── Strategy Settings ─── */}
      <div className="glass-card p-5">
        <h3 className="text-sm font-semibold text-foreground mb-4">전략 설정</h3>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
          <div>
            <label className="text-xs text-muted block mb-1">유니버스</label>
            <select
              className="w-full bg-surface border border-border rounded-lg px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-1 focus:ring-primary"
              value={universe}
              onChange={(e) => setUniverse(e.target.value as 'KOSPI' | 'KOSPI+KOSDAQ')}
            >
              <option value="KOSPI">KOSPI</option>
              <option value="KOSPI+KOSDAQ">KOSPI+KOSDAQ</option>
            </select>
          </div>
          <div>
            <label className="text-xs text-muted block mb-1">Weight Cap %</label>
            <input
              type="number"
              className="w-full bg-surface border border-border rounded-lg px-3 py-2 text-sm text-foreground font-num focus:outline-none focus:ring-1 focus:ring-primary"
              value={weightCapPct}
              onChange={(e) => setWeightCapPct(Number(e.target.value))}
              min={1}
              max={100}
            />
          </div>
          <div>
            <label className="text-xs text-muted block mb-1">Top N 종목</label>
            <input
              type="number"
              className="w-full bg-surface border border-border rounded-lg px-3 py-2 text-sm text-foreground font-num focus:outline-none focus:ring-1 focus:ring-primary"
              value={topN}
              onChange={(e) => setTopN(Number(e.target.value))}
              min={5}
              max={100}
            />
          </div>
          <div>
            <label className="text-xs text-muted block mb-1">거래비용 (bp)</label>
            <input
              type="number"
              className="w-full bg-surface border border-border rounded-lg px-3 py-2 text-sm text-foreground font-num focus:outline-none focus:ring-1 focus:ring-primary"
              value={txCostBp}
              onChange={(e) => setTxCostBp(Number(e.target.value))}
              min={0}
              max={100}
            />
          </div>
          <div>
            <label className="text-xs text-muted block mb-1">리밸런싱</label>
            <select
              className="w-full bg-surface border border-border rounded-lg px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-1 focus:ring-primary"
              value={rebalType}
              onChange={(e) => setRebalType(e.target.value as 'monthly' | 'biweekly')}
            >
              <option value="monthly">월간</option>
              <option value="biweekly">격주</option>
            </select>
          </div>
        </div>
      </div>

      {/* ─── Factor Weights Visualization ─── */}
      {Object.keys(weights).length > 0 && (
        <div className="glass-card p-5">
          <h3 className="text-sm font-semibold text-foreground mb-4">팩터 가중치</h3>
          <div className="space-y-5">
            {Object.entries(FACTOR_CATEGORIES).map(([category, { factors, color }]) => {
              const activeFactors = factors.filter((f) => weights[f] !== undefined);
              if (activeFactors.length === 0) return null;
              return (
                <div key={category}>
                  <div className="flex items-center gap-2 mb-2">
                    <span
                      className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                      style={{ backgroundColor: color }}
                    />
                    <span className="text-xs font-medium text-muted">{category}</span>
                  </div>
                  <div className="space-y-1.5">
                    {activeFactors.map((factor) => {
                      const w = weights[factor] ?? 0;
                      const pct = (Math.abs(w) / maxAbsWeight) * 100;
                      return (
                        <div key={factor} className="flex items-center gap-3">
                          <span className="text-xs text-muted w-36 flex-shrink-0 truncate">
                            {FACTOR_LABELS[factor] || factor}
                          </span>
                          <div className="flex-1 h-5 relative flex items-center">
                            {/* center line */}
                            <div className="absolute left-1/2 top-0 bottom-0 w-px bg-border" />
                            {/* bar */}
                            <div
                              className="absolute h-4 rounded-sm transition-all"
                              style={{
                                backgroundColor: color,
                                opacity: 0.7,
                                width: `${pct / 2}%`,
                                ...(w >= 0
                                  ? { left: '50%' }
                                  : { right: '50%' }),
                              }}
                            />
                          </div>
                          <span className="text-xs font-num text-foreground w-12 text-right flex-shrink-0">
                            {w.toFixed(1)}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* ─── Code Viewer ─── */}
      <div className="glass-card p-5">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-semibold text-foreground">전략 코드</h3>
          <button
            onClick={() => setCodeExpanded(!codeExpanded)}
            className="text-xs text-primary hover:underline"
          >
            {codeExpanded ? '접기' : '펼치기'}
          </button>
        </div>
        {codeExpanded && (
          <textarea
            className="w-full bg-surface border border-border rounded-lg p-4 text-xs text-foreground font-mono leading-relaxed focus:outline-none focus:ring-1 focus:ring-primary resize-y"
            rows={24}
            value={code}
            onChange={(e) => setCode(e.target.value)}
            spellCheck={false}
          />
        )}
        {!codeExpanded && (
          <div className="bg-surface border border-border rounded-lg p-4 max-h-24 overflow-hidden relative">
            <pre className="text-xs text-muted font-mono whitespace-pre-wrap leading-relaxed">
              {code.slice(0, 500)}
              {code.length > 500 && '...'}
            </pre>
            <div className="absolute bottom-0 left-0 right-0 h-8 bg-gradient-to-t from-surface to-transparent" />
          </div>
        )}
      </div>

      {/* ─── Backtest & Save ─── */}
      <div className="glass-card p-5 space-y-4">
        <div className="flex flex-wrap items-end gap-4">
          <button
            onClick={handleBacktest}
            disabled={running || !code.trim()}
            className="px-6 py-2.5 rounded-lg bg-primary text-background text-sm font-medium hover:opacity-90 transition-opacity disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {running ? '실행 중...' : '백테스트 실행'}
          </button>

          {/* Save inline */}
          <div className="flex items-end gap-2 ml-auto">
            <div>
              <label className="text-xs text-muted block mb-1">전략 이름</label>
              <div className="flex items-center gap-1">
                <input
                  type="text"
                  className="w-48 bg-surface border border-border rounded-lg px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-1 focus:ring-primary"
                  value={saveName}
                  onChange={(e) => setSaveName(e.target.value)}
                  placeholder={autoName}
                />
                <button
                  onClick={() => setSaveName(autoName)}
                  className="px-2 py-2 text-xs rounded-lg bg-surface border border-border text-muted hover:text-foreground transition-colors whitespace-nowrap"
                  title="자동 이름 채우기"
                >
                  자동
                </button>
              </div>
            </div>
            <div>
              <label className="text-xs text-muted block mb-1">설명</label>
              <input
                type="text"
                className="w-56 bg-surface border border-border rounded-lg px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-1 focus:ring-primary"
                value={saveDesc}
                onChange={(e) => setSaveDesc(e.target.value)}
                placeholder="변경 사항 메모"
              />
            </div>
            <button
              onClick={handleSave}
              disabled={saving || !saveName.trim()}
              className="px-4 py-2 rounded-lg bg-accent-green/20 text-accent-green text-sm font-medium hover:bg-accent-green/30 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {saving ? '저장 중...' : '저장'}
            </button>
          </div>
        </div>

        {running && (
          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="text-muted">{progressMsg}</span>
              <span className="text-foreground font-num">{progress}%</span>
            </div>
            <div className="w-full h-2 bg-surface rounded-full overflow-hidden">
              <div
                className="h-full bg-primary rounded-full transition-all duration-700 ease-out"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>
        )}
        {saveMsg && (
          <p className={`text-xs ${saveMsg.includes('실패') ? 'text-accent-red' : 'text-accent-green'}`}>
            {saveMsg}
          </p>
        )}

        {/* Results table */}
        {tableData.length > 0 && (
          <>
            <DataTable columns={tableColumns} data={tableData} maxHeight="300px" />

            {/* Cumulative return chart */}
            {chartData.length > 0 && (
              <PlotlyChart
                data={chartData}
                layout={{
                  title: { text: '누적 수익률 비교', font: { size: 13 } },
                  yaxis: { title: { text: '포트폴리오 가치' }, tickformat: ',.0f' },
                  xaxis: { title: { text: '' } },
                  legend: { orientation: 'h', y: -0.15 },
                }}
                height={400}
              />
            )}
          </>
        )}
      </div>
    </div>
  );
}
