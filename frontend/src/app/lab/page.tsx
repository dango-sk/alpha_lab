'use client';

import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { getResults, getConfig, getStrategies, getStrategy, runBacktest, fetchApi, getRegimeCombo } from '@/lib/api';
import { StrategyResult, Config, valueColor, fmtPct, fmtNum } from '@/lib/hooks';
import SectionHeader from '@/components/SectionHeader';
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
    factors: ['F_EPS_M', 'PRICE_M', 'PRICE_MA_REV', 'NDEBT_EBITDA', 'CURRENT'],
    color: '#f59e0b',
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
  PRICE_MA_REV: 'MA 평균회귀',
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
  const entries = block.matchAll(/["'](\w+)["']\s*:\s*([+-]?(?:\d+\.?\d*|\.\d+))/g);
  for (const entry of entries) {
    weights[entry[1]] = parseFloat(entry[2]);
  }
  return weights;
}

// ─── Parse strategy params from code ───
function parseStrategyParams(code: string): {
  weightCapPct?: number;
  topN?: number;
  txCostBp?: number;
  universe?: 'KOSPI' | 'KOSPI+KOSDAQ';
  rebalType?: 'monthly' | 'biweekly';
  stopLossEnabled?: boolean;
  stopLossPct?: number;
  stopLossMode?: string;
  stopLossBasis?: string;
  maRevWindow?: number;
} {
  const result: ReturnType<typeof parseStrategyParams> = {};
  const cap = code.match(/weight_cap_pct["']?\s*[=:]\s*(\d+)/);
  if (cap) result.weightCapPct = parseInt(cap[1]);
  const top = code.match(/top_n(?:_stocks)?["']?\s*[=:]\s*(\d+)/);
  if (top) result.topN = parseInt(top[1]);
  const tx = code.match(/(?:transaction_cost_bp|tx_cost_bp)["']?\s*[=:]\s*(\d+)/);
  if (tx) result.txCostBp = parseInt(tx[1]);
  if (/KOSPI\+KOSDAQ|kospi.*kosdaq/i.test(code)) result.universe = 'KOSPI+KOSDAQ';
  else if (/KOSPI/i.test(code)) result.universe = 'KOSPI';
  if (/biweekly|격주/i.test(code)) result.rebalType = 'biweekly';
  else if (/monthly|월간/i.test(code)) result.rebalType = 'monthly';
  const slEnabled = code.match(/stop_loss_enabled["']?\s*[=:]\s*(True|False|true|false)/);
  if (slEnabled) result.stopLossEnabled = slEnabled[1].toLowerCase() === 'true';
  const slPct = code.match(/stop_loss_pct["']?\s*[=:]\s*(\d+)/);
  if (slPct) result.stopLossPct = parseInt(slPct[1]);
  const slMode = code.match(/stop_loss_mode["']?\s*[=:]\s*["'](\w+)["']/);
  if (slMode) result.stopLossMode = slMode[1];
  const slBasis = code.match(/stop_loss_basis["']?\s*[=:]\s*["'](\w+)["']/);
  if (slBasis) result.stopLossBasis = slBasis[1];
  const maRev = code.match(/ma_reversion_window["']?\s*[=:]\s*(\d+)/);
  if (maRev) result.maRevWindow = parseInt(maRev[1]);
  return result;
}

function updateWeightInCode(code: string, factor: string, newValue: number): string {
  // Update a single weight value in WEIGHTS_LARGE block
  const regex = new RegExp(
    `(WEIGHTS_LARGE\\s*=\\s*\\{[^}]*["']${factor}["']\\s*:\\s*)([+-]?(?:\\d+\\.?\\d*|\\.\\d+))`,
    's'
  );
  if (regex.test(code)) {
    return code.replace(regex, `$1${newValue.toFixed(2)}`);
  }
  return code;
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

  // ─── MA Reversion Window ───
  const [maRevWindow, setMaRevWindow] = useState<120 | 250>(120);

  // ─── Stop Loss ───
  const [stopLossEnabled, setStopLossEnabled] = useState(false);
  const [stopLossPct, setStopLossPct] = useState(15);
  const [stopLossMode, setStopLossMode] = useState<'sell' | 'reduce' | 'redistribute'>('sell');
  const [stopLossBasis, setStopLossBasis] = useState<'entry' | 'peak'>('entry');

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

  // ─── Tab ───
  const [activeTab, setActiveTab] = useState<'single' | 'regime'>('single');

  // ─── Regime Combo ───
  const [regimeBullKey, setRegimeBullKey] = useState('');
  const [regimeBearKey, setRegimeBearKey] = useState('');
  const [regimeResult, setRegimeResult] = useState<Record<string, unknown> | null>(null);
  const [regimeLoading, setRegimeLoading] = useState(false);
  const [regimeSaveMsg, setRegimeSaveMsg] = useState('');
  const [regimeMaWindow, setRegimeMaWindow] = useState(50);
  const [regimeMaWindowInput, setRegimeMaWindowInput] = useState('50');
  const [regimeMode, setRegimeMode] = useState<'ma' | 'cycle' | 'ai'>('ma');

  // ─── Auto-generate strategy name ───
  const autoName = useMemo(() => {
    const rebal = rebalType === 'monthly' ? '월간' : '격주';
    const parts: string[] = [];
    if (stopLossEnabled) parts.push(`손절${stopLossPct}%${stopLossBasis === 'peak' ? '고점' : ''}`);
    if (topN !== 30) parts.push(`top${topN}`);
    if (weightCapPct !== 5) parts.push(`cap${weightCapPct}%`);
    const detail = parts.length > 0 ? `_${parts.join('_')}` : '';
    return `전략_${rebal}${detail}`;
  }, [rebalType, weightCapPct, topN, stopLossEnabled, stopLossPct]);

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

  // Reload base results + strategies when universe/rebalType changes
  useEffect(() => {
    if (initialLoading) return;
    Promise.all([
      getResults({ universe, rebal_type: rebalType }),
      getStrategies(universe),
    ])
      .then(([results, strats]) => {
        setBaseResults(results);
        setSavedStrategies(strats || []);
      })
      .catch(console.error);
  }, [universe, rebalType, initialLoading]);

  // ─── Load strategy ───
  const loadStrategy = useCallback(async (name: string) => {
    if (!name) {
      setCode(defaultCode);
      return;
    }
    try {
      const data = await getStrategy(name, universe, rebalType);
      if (data?.code) {
        setCode(data.code);
        setSaveName(name);
        // 코드에서 파싱
        const parsed = parseStrategyParams(data.code);
        // 전략 이름에서 파싱 (코드보다 더 신뢰성 높음 - 실제 저장 시점의 값)
        const capFromName = name.match(/cap(\d+)%/);
        const topFromName = name.match(/top(\d+)/);
        const txFromName = name.match(/tx(\d+)bp/);
        const capVal = capFromName ? parseInt(capFromName[1]) : parsed.weightCapPct;
        const topVal = topFromName ? parseInt(topFromName[1]) : parsed.topN;
        const txVal = txFromName ? parseInt(txFromName[1]) : parsed.txCostBp;

        if (capVal !== undefined) setWeightCapPct(capVal);
        if (topVal !== undefined) setTopN(topVal);
        if (txVal !== undefined) setTxCostBp(txVal);
        if (parsed.stopLossEnabled !== undefined) setStopLossEnabled(parsed.stopLossEnabled);
        if (parsed.stopLossPct !== undefined) setStopLossPct(parsed.stopLossPct);
        if (parsed.stopLossMode !== undefined) setStopLossMode(parsed.stopLossMode as 'sell' | 'reduce');
        if (parsed.stopLossBasis !== undefined) setStopLossBasis(parsed.stopLossBasis as 'entry' | 'peak');
        if (parsed.maRevWindow !== undefined) setMaRevWindow(parsed.maRevWindow as 120 | 250);
        if (parsed.universe) setUniverse(parsed.universe);
        if (parsed.rebalType) setRebalType(parsed.rebalType);
        // 유니버스/리밸런싱은 전략 이름에서도 파싱
        if (/코스피\+코스닥|KOSPI\+KOSDAQ/i.test(name)) setUniverse('KOSPI+KOSDAQ');
        else if (/코스피|KOSPI/i.test(name)) setUniverse('KOSPI');
        if (/격주|biweekly/i.test(name)) setRebalType('biweekly');
        else if (/월간|monthly/i.test(name)) setRebalType('monthly');
      }
    } catch {
      // 코드 없는 전략 (레짐 조합 등)은 무시
    }
  }, [defaultCode]);

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

  // ─── Inject actual params into code ───
  const codeWithParams = useMemo(() => {
    let c = code;
    c = c.replace(/(["']?top_n["']?\s*:\s*)\d+/, `$1${topN}`);
    c = c.replace(/(["']?tx_cost_bp["']?\s*:\s*)\d+/, `$1${txCostBp}`);
    c = c.replace(/(["']?weight_cap_pct["']?\s*:\s*)\d+/, `$1${weightCapPct}`);
    c = c.replace(/(["']?stop_loss_enabled["']?\s*[=:]\s*)(True|False|true|false)/, `$1${stopLossEnabled ? 'True' : 'False'}`);
    c = c.replace(/(["']?stop_loss_pct["']?\s*[=:]\s*)\d+/, `$1${stopLossPct}`);
    c = c.replace(/(["']?stop_loss_mode["']?\s*[=:]\s*["'])\w+(["'])/, `$1${stopLossMode}$2`);
    c = c.replace(/(["']?stop_loss_basis["']?\s*[=:]\s*["'])\w+(["'])/, `$1${stopLossBasis}$2`);
    if (/ma_reversion_window/.test(c)) {
      c = c.replace(/(["']?ma_reversion_window["']?\s*[=:]\s*)\d+/, `$1${maRevWindow}`);
    } else {
      c = c.replace(/(PARAMS\s*=\s*\{)/, `$1\n    "ma_reversion_window": ${maRevWindow},`);
    }
    return c;
  }, [code, topN, txCostBp, weightCapPct, stopLossEnabled, stopLossPct, stopLossMode, stopLossBasis, maRevWindow]);

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
      const result = await runBacktest(codeWithParams, {
        universe,
        rebal_type: rebalType,
        weight_cap_pct: weightCapPct,
        top_n_stocks: topN,
        tx_cost_bp: txCostBp,
        stop_loss_enabled: stopLossEnabled,
        stop_loss_pct: stopLossPct,
        stop_loss_mode: stopLossMode,
        stop_loss_basis: stopLossBasis,
      });
      setProgress(100);
      setProgressMsg('완료!');
      setBacktestResults(result);
      if (!saveName.trim()) setSaveName(autoName);
    } catch (e) {
      alert(`백테스트 실패: ${(e as Error).message}`);
    } finally {
      if (progressRef.current) clearInterval(progressRef.current);
      setTimeout(() => setRunning(false), 500);
    }
  }, [codeWithParams, universe, rebalType, weightCapPct, topN, txCostBp, stopLossEnabled, stopLossPct, stopLossMode, stopLossBasis]);

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
          code: codeWithParams,
          description: saveDesc,
          results: backtestResults,
          universe,
          rebal_type: rebalType,
        }),
        headers: { 'Content-Type': 'application/json' },
      });
      setSaveMsg('저장 완료');
      setSaveDesc('');
      const strats = await getStrategies();
      setSavedStrategies(strats || []);
    } catch (e) {
      setSaveMsg(`저장 실패: ${(e as Error).message}`);
    } finally {
      setSaving(false);
    }
  }, [saveName, saveDesc, codeWithParams, backtestResults, universe, rebalType]);

  // ─── Parse weights ───
  const weights = useMemo(() => parseWeights(code), [code]);

  // ─── Build results table data ───
  const tableData = useMemo(() => {
    const rows: Record<string, unknown>[] = [];
    // Base A0
    if (baseResults?.['A0']) {
      const r = baseResults['A0'];
      rows.push({
        strategy: '기존전략',
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
        name: '기존전략',
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


  return (
    <div className="animate-fade-in -mt-2">

      {/* ─── Saved Strategy Selector ─── */}
      <div className="glass-card p-3 mb-3">
        <div className="flex flex-wrap items-center gap-3">
          <label className="text-xs text-muted font-medium">저장된 전략</label>
          <select
            className="bg-surface border border-border rounded-lg px-3 py-2 text-sm text-foreground min-w-[200px] focus:outline-none focus:ring-1 focus:ring-primary"
            value={selectedStrategy}
            onChange={(e) => setSelectedStrategy(e.target.value)}
          >
            <option value="">-- 전략 선택 --</option>
            {savedStrategies.map((s) => (
              <option key={s.name} value={s.name}>
                {s.name}
              </option>
            ))}
          </select>
          <button
            onClick={() => loadStrategy(selectedStrategy)}
            disabled={!selectedStrategy}
            className="px-3 py-2 text-xs rounded-lg bg-primary/10 text-primary hover:bg-primary/20 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
          >
            불러오기
          </button>
          {selectedStrategy && (
            <button
              onClick={deleteStrategy}
              className="px-3 py-2 text-xs rounded-lg bg-accent-red/10 text-accent-red hover:bg-accent-red/20 transition-colors"
            >
              삭제
            </button>
          )}
        </div>
      </div>

      {/* ─── Tab Bar ─── */}
      <div className="flex gap-0 mb-3 border-b border-border">
        <button onClick={() => setActiveTab('single')} className={`px-5 py-2.5 text-sm font-medium border-b-2 -mb-px transition-colors ${activeTab === 'single' ? 'border-primary text-primary' : 'border-transparent text-muted hover:text-foreground'}`}>
          전략 백테스트
        </button>
        <button onClick={() => setActiveTab('regime')} className={`px-5 py-2.5 text-sm font-medium border-b-2 -mb-px transition-colors ${activeTab === 'regime' ? 'border-primary text-primary' : 'border-transparent text-muted hover:text-foreground'}`}>
          레짐 조합
        </button>
      </div>

      {/* ─── Single Strategy Tab ─── */}
      {activeTab === 'single' && (
        <div className="flex gap-4">
          {/* LEFT PANEL */}
          <div className="w-[400px] min-w-[400px] flex flex-col" style={{ maxHeight: 'calc(100vh - 180px)' }}>
          <div className="flex-1 space-y-3 overflow-y-auto pr-1">
            {/* ─── Strategy Settings ─── */}
            <div className="glass-card p-4">
              <h3 className="text-sm font-semibold text-foreground mb-3">전략 설정</h3>
              <div className="grid grid-cols-2 gap-4">
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
                    onFocus={(e) => e.target.select()}
                    onChange={(e) => setWeightCapPct(parseInt(e.target.value) || 0)}
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
                    onFocus={(e) => e.target.select()}
                    onChange={(e) => setTopN(parseInt(e.target.value) || 0)}
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
                    onFocus={(e) => e.target.select()}
                    onChange={(e) => setTxCostBp(parseInt(e.target.value) || 0)}
                    min={0}
                    max={100}
                  />
                </div>
                <div className="col-span-2">
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

              {/* ─── Stop Loss ─── */}
              <div className="mt-3 flex flex-wrap items-center gap-4">
                <label className="flex items-center gap-2 text-sm text-foreground cursor-pointer">
                  <input
                    type="checkbox"
                    checked={stopLossEnabled}
                    onChange={(e) => setStopLossEnabled(e.target.checked)}
                    className="accent-primary"
                  />
                  손절 (Stop Loss)
                </label>
                <div className="flex items-center gap-2">
                  <label className="text-xs text-muted">하락률 %</label>
                  <input
                    type="number"
                    className="w-16 bg-surface border border-border rounded-lg px-2 py-1 text-sm text-foreground font-num focus:outline-none focus:ring-1 focus:ring-primary disabled:opacity-40"
                    value={stopLossPct}
                    onFocus={(e) => e.target.select()}
                    onChange={(e) => setStopLossPct(parseInt(e.target.value) || 0)}
                    min={1}
                    max={50}
                    disabled={!stopLossEnabled}
                  />
                </div>
                <div className="flex gap-1">
                  <button
                    className={`px-2 py-1 text-xs rounded-md border ${stopLossMode === 'sell' ? 'bg-primary text-white border-primary' : 'bg-surface border-border text-muted'} disabled:opacity-40`}
                    onClick={() => setStopLossMode('sell')}
                    disabled={!stopLossEnabled}
                  >
                    전량 매도
                  </button>
                  <button
                    className={`px-2 py-1 text-xs rounded-md border ${stopLossMode === 'reduce' ? 'bg-primary text-white border-primary' : 'bg-surface border-border text-muted'} disabled:opacity-40`}
                    onClick={() => setStopLossMode('reduce')}
                    disabled={!stopLossEnabled}
                  >
                    비중 50% 축소
                  </button>
                </div>
                <div className="flex gap-1">
                  <button
                    className={`px-2 py-1 text-xs rounded-md border ${stopLossBasis === 'entry' ? 'bg-primary text-white border-primary' : 'bg-surface border-border text-muted'} disabled:opacity-40`}
                    onClick={() => setStopLossBasis('entry')}
                    disabled={!stopLossEnabled}
                  >
                    매입가 대비
                  </button>
                  <button
                    className={`px-2 py-1 text-xs rounded-md border ${stopLossBasis === 'peak' ? 'bg-primary text-white border-primary' : 'bg-surface border-border text-muted'} disabled:opacity-40`}
                    onClick={() => setStopLossBasis('peak')}
                    disabled={!stopLossEnabled}
                  >
                    고점 대비
                  </button>
                </div>
              </div>

              {/* ─── MA Reversion Window ─── */}
              <div className="mt-3 flex items-center gap-3">
                <label className="text-xs text-muted">MA 평균회귀 기간</label>
                <div className="flex gap-1">
                  <button
                    className={`px-3 py-1 text-xs rounded-md border ${maRevWindow === 120 ? 'bg-primary text-white border-primary' : 'bg-surface border-border text-muted'}`}
                    onClick={() => setMaRevWindow(120)}
                  >
                    120일
                  </button>
                  <button
                    className={`px-3 py-1 text-xs rounded-md border ${maRevWindow === 250 ? 'bg-primary text-white border-primary' : 'bg-surface border-border text-muted'}`}
                    onClick={() => setMaRevWindow(250)}
                  >
                    250일
                  </button>
                </div>
              </div>
            </div>

            {/* ─── Factor Weights Editor ─── */}
            {Object.keys(weights).length > 0 && (() => {
              const totalWeight = Object.values(weights).reduce((s, v) => s + v, 0);
              const categoryData = Object.entries(FACTOR_CATEGORIES).map(([cat, { factors, color }]) => {
                const active = factors.filter((f) => weights[f] !== undefined);
                const catSum = active.reduce((s, f) => s + (weights[f] ?? 0), 0);
                return { cat, color, active, catSum };
              }).filter((d) => d.active.length > 0);

              return (
                <div className="glass-card p-4">
                  <h3 className="text-sm font-semibold text-foreground mb-3">팩터 가중치</h3>

                  {/* Category summary bar */}
                  <div className="flex rounded-lg overflow-hidden h-10 mb-6">
                    {categoryData.map(({ cat, color, catSum }) => {
                      const pct = totalWeight > 0 ? (catSum / totalWeight) * 100 : 0;
                      if (pct <= 0) return null;
                      return (
                        <div
                          key={cat}
                          className="flex items-center justify-center text-xs font-semibold text-white"
                          style={{ backgroundColor: color, width: `${pct}%`, minWidth: pct > 5 ? undefined : '2rem' }}
                        >
                          {pct >= 10 && <span>{cat}<br />{Math.round(pct)}%</span>}
                          {pct < 10 && <span>{Math.round(pct)}%</span>}
                        </div>
                      );
                    })}
                  </div>

                  {/* Total */}
                  <div className="flex justify-end mb-4 text-xs text-muted">
                    전체 합계: <span className={`ml-1 font-semibold ${Math.abs(totalWeight - 1) < 0.01 ? 'text-accent-green' : 'text-accent-red'}`}>
                      {(totalWeight * 100).toFixed(0)}%
                    </span>
                  </div>

                  {/* Category sections */}
                  <div className="space-y-6">
                    {categoryData.map(({ cat, color, active, catSum }) => (
                      <div key={cat} className="rounded-lg border border-border overflow-hidden">
                        {/* Category header */}
                        <div className="flex items-center justify-between px-4 py-3" style={{ borderLeft: `3px solid ${color}` }}>
                          <span className="text-sm font-semibold text-foreground">{cat}</span>
                          <span className="text-sm font-semibold" style={{ color }}>
                            {(catSum * 100).toFixed(0)}%
                          </span>
                        </div>

                        {/* Factor rows */}
                        <div className="px-4 pb-3 space-y-2">
                          {active.map((factor) => {
                            const w = weights[factor] ?? 0;
                            const barPct = totalWeight > 0 ? (Math.abs(w) / totalWeight) * 100 : 0;
                            return (
                              <div key={factor} className="flex items-center gap-3">
                                <span className="text-xs text-muted w-44 flex-shrink-0 truncate">
                                  {FACTOR_LABELS[factor] || factor}
                                </span>
                                <div className="flex-1 h-4 bg-background rounded-full overflow-hidden">
                                  <div
                                    className="h-full rounded-full transition-all"
                                    style={{ backgroundColor: color, width: `${Math.min(barPct * 2, 100)}%`, opacity: 0.8 }}
                                  />
                                </div>
                                <div className="flex items-center gap-1 flex-shrink-0">
                                  <input
                                    type="text"
                                    inputMode="numeric"
                                    value={Math.round(w * 100)}
                                    onFocus={(e) => e.target.select()}
                                    onChange={(e) => {
                                      const raw = e.target.value.replace(/[^0-9-]/g, '');
                                      if (raw === '' || raw === '-') return;
                                      const v = parseInt(raw);
                                      if (!isNaN(v)) {
                                        setCode((prev) => updateWeightInCode(prev, factor, v / 100));
                                      }
                                    }}
                                    className="w-12 text-xs font-num text-foreground text-right bg-surface border border-border rounded px-1.5 py-1 focus:outline-none focus:ring-1 focus:ring-primary"
                                  />
                                  <span className="text-xs text-muted">%</span>
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              );
            })()}

            {/* ─── Code Viewer ─── */}
            <div className="glass-card p-4">
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
          </div>{/* end scrollable area */}
          {/* ─── Sticky Footer: Run Button ─── */}
          <div className="pt-3 border-t border-border bg-background">
            <button
              onClick={handleBacktest}
              disabled={running || !code.trim()}
              className={`w-full px-6 py-2.5 rounded-lg text-sm font-medium transition-all disabled:cursor-not-allowed ${
                running
                  ? 'bg-primary/30 text-primary border border-primary/40 animate-pulse cursor-not-allowed'
                  : 'bg-primary text-background hover:opacity-90 disabled:opacity-40'
              }`}
            >
              {running ? (
                <span className="flex items-center justify-center gap-2">
                  <span className="inline-block w-1.5 h-1.5 rounded-full bg-primary animate-bounce" style={{ animationDelay: '0ms' }} />
                  <span className="inline-block w-1.5 h-1.5 rounded-full bg-primary animate-bounce" style={{ animationDelay: '150ms' }} />
                  <span className="inline-block w-1.5 h-1.5 rounded-full bg-primary animate-bounce" style={{ animationDelay: '300ms' }} />
                  <span>{progressMsg || '실행 중...'}</span>
                </span>
              ) : '백테스트 실행'}
            </button>
            {running && (
              <div className="space-y-3 mt-3">
                <div className="flex items-center gap-2 text-sm text-muted">
                  <span className="inline-block w-2 h-2 rounded-full bg-primary animate-pulse" />
                  <span className="animate-typing">{progressMsg}</span>
                  <span className="animate-blink text-primary font-light">|</span>
                </div>
                <div className="w-full h-1.5 bg-surface rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-primary/80 to-primary rounded-full transition-all duration-700 ease-out"
                    style={{ width: `${progress}%` }}
                  />
                </div>
                <span className="text-xs text-muted/60 font-num">{progress}%</span>
              </div>
            )}
          </div>
          </div>{/* end left panel column */}

          {/* RIGHT PANEL */}
          <div className="flex-1 space-y-4 overflow-y-auto" style={{ maxHeight: 'calc(100vh - 220px)' }}>
            {/* ─── Save Bar ─── */}
            <div className="glass-card p-5">
              <div className="flex flex-wrap items-end gap-3">
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
            {saveMsg && (
              <p className={`text-xs ${saveMsg.includes('실패') ? 'text-accent-red' : 'text-accent-green'}`}>
                {saveMsg}
              </p>
            )}

            {/* ─── Results ─── */}
            {tableData.length > 0 ? (
              <div className="space-y-4">
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
              </div>
            ) : (
              <div className="flex items-center justify-center h-64 text-muted text-sm">
                백테스트를 실행하면 결과가 여기에 표시됩니다
              </div>
            )}
          </div>
        </div>
      )}

      {/* ─── Regime Combo Tab ─── */}
      {activeTab === 'regime' && (
      <div id="regime-combo" className="space-y-4 border border-border rounded-xl p-5">
        <div>
          <h2 className="text-base font-semibold text-foreground">레짐 조합 백테스트</h2>
          <p className="text-xs text-muted mt-0.5">KOSPI 200 50일 이동평균 기준 — Bull(≥ MA) vs Bear(&lt; MA) 이진 분류, 실제 종목 선택 기반 완전 재백테스트</p>
        </div>

        {/* 전략 선택 */}
        {(() => {
          // 사전 정의 전략 + 저장된 전략 모두 표시 (KOSPI 벤치마크 제외)
          const stratKeys = baseResults
            ? Object.keys(baseResults).filter((k) => k !== 'KOSPI' && k !== 'KOSDAQ' && k !== 'CUSTOM')
            : [];
          const cfgLabels: Record<string, string> = config?.strategy_labels || {};
          const savedNameMap = Object.fromEntries(savedStrategies.map((s) => [s.name, s.name]));
          const labels: Record<string, string> = Object.fromEntries(
            stratKeys.map((k) => [k, savedNameMap[k] || cfgLabels[k] || k])
          );

          function RegimeProgressBar() {
            const [progress, setProgress] = useState(0);
            useEffect(() => {
              const total = 270; // 4.5분 기준 (초)
              const interval = setInterval(() => {
                setProgress((p) => {
                  if (p >= 95) { clearInterval(interval); return p; }
                  // 초반 빠르게, 후반 느리게
                  const step = p < 50 ? 0.6 : p < 80 ? 0.3 : 0.1;
                  return Math.min(p + step, 95);
                });
              }, total * 10); // ~270초에 걸쳐 95%까지
              return () => clearInterval(interval);
            }, []);

            const stages = [
              { label: '데이터 로드', pct: 20 },
              { label: '레짐 분류', pct: 40 },
              { label: '종목 선택', pct: 70 },
              { label: '수익률 계산', pct: 90 },
            ];
            const currentStage = stages.findLast((s) => progress >= s.pct)?.label ?? '초기화 중';

            return (
              <div className="space-y-1.5 py-1">
                <div className="flex justify-between text-xs text-muted">
                  <span>{currentStage}...</span>
                  <span>{Math.round(progress)}%</span>
                </div>
                <div className="w-full h-1.5 bg-surface rounded-full overflow-hidden">
                  <div
                    className="h-full bg-primary rounded-full transition-all duration-500"
                    style={{ width: `${progress}%` }}
                  />
                </div>
              </div>
            );
          }

          const handleRun = async () => {
            if (!regimeBullKey || !regimeBearKey) return;
            setRegimeLoading(true);
            setRegimeResult(null);
            setRegimeSaveMsg('');
            try {
              const res = await getRegimeCombo(regimeBullKey, regimeBearKey, universe, rebalType, regimeMaWindow, regimeMode);
              setRegimeResult(res);
            } catch (e) {
              console.error(e);
            } finally {
              setRegimeLoading(false);
            }
          };

          const handleSaveCombo = async () => {
            if (!regimeResult || !regimeBullKey || !regimeBearKey) return;
            const allRes = regimeResult as Record<string, unknown>;
            const combo = allRes['REGIME_COMBO'] as Record<string, unknown> | undefined;
            if (!combo) return;
            const name = `레짐조합_${labels[regimeBullKey] || regimeBullKey}↑_${labels[regimeBearKey] || regimeBearKey}↓`;
            try {
              await fetchApi('/api/strategies', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                  name,
                  code: '',
                  description: `레짐 조합: 상승(${labels[regimeBullKey] || regimeBullKey}) × 하락(${labels[regimeBearKey] || regimeBearKey})`,
                  results: combo,
                  universe,
                  rebal_type: rebalType,
                }),
              });
              setRegimeSaveMsg(`'${name}' 저장 완료 — 성과 비교 탭에서 확인 가능합니다.`);
            } catch {
              setRegimeSaveMsg('저장 실패');
            }
          };

          return (
            <>
              <div className="flex items-center gap-4 text-sm flex-wrap">
                <div className="flex items-center gap-2">
                  <span className="text-xs text-muted font-medium">레짐 기준</span>
                  <div className="flex rounded-lg border border-border overflow-hidden text-xs">
                    <button
                      onClick={() => setRegimeMode('ma')}
                      className={`px-3 py-1.5 transition-colors ${regimeMode === 'ma' ? 'bg-primary text-white' : 'bg-surface text-muted hover:text-foreground'}`}
                    >MA 기준</button>
                    <button
                      onClick={() => setRegimeMode('cycle')}
                      className={`px-3 py-1.5 transition-colors ${regimeMode === 'cycle' ? 'bg-primary text-white' : 'bg-surface text-muted hover:text-foreground'}`}
                    >사이클 기준</button>
                    <button
                      onClick={() => setRegimeMode('ai')}
                      className={`px-3 py-1.5 transition-colors ${regimeMode === 'ai' ? 'bg-primary text-white' : 'bg-surface text-muted hover:text-foreground'}`}
                    >AI 기준</button>
                  </div>
                </div>
                {regimeMode === 'ma' && (
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-muted font-medium">MA 기간 (KOSPI 200 기준)</span>
                    <input
                      type="number"
                      min={5}
                      max={500}
                      value={regimeMaWindowInput}
                      onChange={(e) => setRegimeMaWindowInput(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') {
                          const v = parseInt(regimeMaWindowInput);
                          if (!isNaN(v) && v >= 5 && v <= 500) setRegimeMaWindow(v);
                        }
                      }}
                      onBlur={() => {
                        const v = parseInt(regimeMaWindowInput);
                        if (!isNaN(v) && v >= 5 && v <= 500) setRegimeMaWindow(v);
                        else setRegimeMaWindowInput(String(regimeMaWindow));
                      }}
                      className="w-20 px-2 py-1 rounded border border-border bg-background text-center text-sm"
                    />
                    <span className="text-xs text-muted">일</span>
                  </div>
                )}
                {regimeMode === 'cycle' && (
                  <span className="text-xs text-muted">고점 대비 -20% 하락 사이클 기준 (2018년 이후)</span>
                )}
                {regimeMode === 'ai' && (
                  <span className="text-xs text-muted">GPT-4o 멀티에이전트 월간 방향 예측</span>
                )}
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1">
                  <label className="text-xs text-muted font-medium">
                    {regimeMode === 'ma' ? `강세장 전략 (Bull: KOSPI 200 ≥ ${regimeMaWindow}일 MA)` : '강세장 전략 (Bull)'}
                  </label>
                  <select
                    value={regimeBullKey}
                    onChange={(e) => setRegimeBullKey(e.target.value)}
                    className="w-full bg-surface border border-border rounded-lg px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-1 focus:ring-primary"
                  >
                    <option value="">선택</option>
                    {stratKeys.map((k) => (
                      <option key={k} value={k}>{labels[k] || k}</option>
                    ))}
                  </select>
                </div>
                <div className="space-y-1">
                  <label className="text-xs text-muted font-medium">
                    {regimeMode === 'ma' ? `약세장 전략 (Bear: KOSPI 200 < ${regimeMaWindow}일 MA)` : '약세장 전략 (Bear)'}
                  </label>
                  <select
                    value={regimeBearKey}
                    onChange={(e) => setRegimeBearKey(e.target.value)}
                    className="w-full bg-surface border border-border rounded-lg px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-1 focus:ring-primary"
                  >
                    <option value="">선택</option>
                    {stratKeys.map((k) => (
                      <option key={k} value={k}>{labels[k] || k}</option>
                    ))}
                  </select>
                </div>
              </div>

              <button
                onClick={handleRun}
                disabled={!regimeBullKey || !regimeBearKey || regimeLoading}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all disabled:cursor-not-allowed ${
                  regimeLoading
                    ? 'bg-primary/10 text-primary border border-primary/30 animate-pulse'
                    : 'bg-primary/20 text-primary hover:bg-primary/30 disabled:opacity-40'
                }`}
              >
                {regimeLoading ? (
                  <span className="flex items-center gap-2">
                    <span className="inline-block w-1.5 h-1.5 rounded-full bg-primary animate-bounce" style={{ animationDelay: '0ms' }} />
                    <span className="inline-block w-1.5 h-1.5 rounded-full bg-primary animate-bounce" style={{ animationDelay: '150ms' }} />
                    <span className="inline-block w-1.5 h-1.5 rounded-full bg-primary animate-bounce" style={{ animationDelay: '300ms' }} />
                    <span>백테스트 실행 중... (3~5분 소요)</span>
                  </span>
                ) : '레짐 조합 실행'}
              </button>

              {regimeLoading && (
                <RegimeProgressBar />
              )}

              {regimeResult && (() => {
                const allRes = regimeResult as Record<string, unknown>;
                const combo = allRes['REGIME_COMBO'] as Record<string, unknown> | undefined;
                if (!combo) return null;

                // 원전략 결과: 새 백테스트 결과 우선, 없으면 캐시
                const bullRes = (allRes[regimeBullKey] as Record<string, unknown>) || baseResults?.[regimeBullKey];
                const bearRes = (allRes[regimeBearKey] as Record<string, unknown>) || baseResults?.[regimeBearKey];
                const bmRes = (allRes['KOSPI'] as Record<string, unknown>) || baseResults?.['KOSPI'];

                const statsRow = (res: Record<string, unknown> | undefined, name: string) => ({
                  전략: name,
                  수익률: res ? `${((res.total_return as number || 0) * 100).toFixed(1)}%` : '-',
                  CAGR: res ? `${((res.cagr as number || 0) * 100).toFixed(1)}%` : '-',
                  Sharpe: res ? (res.sharpe as number || 0).toFixed(2) : '-',
                  MDD: res ? `${((res.mdd as number || 0) * 100).toFixed(1)}%` : '-',
                });

                const tableData = [
                  statsRow(bullRes as Record<string, unknown>, labels[regimeBullKey] || regimeBullKey),
                  statsRow(bearRes as Record<string, unknown>, labels[regimeBearKey] || regimeBearKey),
                  statsRow(combo, '레짐 조합 (실제 재백테스트)'),
                  statsRow(bmRes as Record<string, unknown>, 'BM (KOSPI)'),
                ];

                const makeTrace = (res: Record<string, unknown> | undefined, name: string, color: string, dash?: string) => {
                  if (!res) return null;
                  return {
                    type: 'scatter' as const, mode: 'lines' as const, name,
                    x: (res.rebalance_dates as string[])?.slice(0, (res.portfolio_values as number[])?.length),
                    y: (res.portfolio_values as number[])?.map((v) => v * 100),
                    line: { width: dash ? 1.5 : 2.5, ...(dash ? { dash: dash as 'dot' } : {}), color },
                  };
                };

                const comboChartData = [
                  makeTrace(bullRes as Record<string, unknown>, labels[regimeBullKey] || regimeBullKey, '#6366f1', 'dot'),
                  ...(regimeBearKey !== regimeBullKey ? [makeTrace(bearRes as Record<string, unknown>, labels[regimeBearKey] || regimeBearKey, '#E91E63', 'dot')] : []),
                  makeTrace(combo, '레짐 조합', '#43A047'),
                  makeTrace(bmRes as Record<string, unknown>, 'BM (KOSPI)', '#9E9E9E', 'dot'),
                ].filter((x): x is NonNullable<typeof x> => x !== null);

                // 레짐 전환 선 + 구간별 겹침율 계산
                const regimeByDate = (combo.regime_by_date || {}) as Record<string, string>;
                const overlapByDate = (combo.overlap_by_date || {}) as Record<string, number>;
                const sortedDates = Object.keys(regimeByDate).sort();
                const transitionDates: string[] = [];
                for (let i = 1; i < sortedDates.length; i++) {
                  if (regimeByDate[sortedDates[i]] !== regimeByDate[sortedDates[i - 1]]) {
                    transitionDates.push(sortedDates[i]);
                  }
                }

                // 각 구간의 중간 날짜 & 평균 겹침율
                const segmentBoundaries = [sortedDates[0], ...transitionDates, sortedDates[sortedDates.length - 1]].filter(Boolean);
                const segmentAnnotations = [];
                for (let i = 0; i < segmentBoundaries.length - 1; i++) {
                  const segStart = segmentBoundaries[i];
                  const segEnd = segmentBoundaries[i + 1];
                  const segDates = sortedDates.filter((d) => d >= segStart && d < segEnd);
                  const segOverlaps = segDates.map((d) => overlapByDate[d]).filter((v) => v !== undefined);
                  const avgOverlap = segOverlaps.length ? Math.round(segOverlaps.reduce((a, b) => a + b, 0) / segOverlaps.length) : null;
                  const midDate = segDates[Math.floor(segDates.length / 2)] || segStart;
                  const regime = regimeByDate[segStart] || regimeByDate[segDates[0]];
                  if (avgOverlap !== null) {
                    segmentAnnotations.push({ x: midDate, regime, avgOverlap, segLen: segDates.length });
                  }
                }

                const chartShapes = transitionDates.map((d) => ({
                  type: 'line' as const,
                  x0: d, x1: d, y0: 0, y1: 1,
                  xref: 'x' as const, yref: 'paper' as const,
                  line: { color: 'rgba(150,150,150,0.5)', width: 1, dash: 'dash' as const },
                }));

                const chartAnnotations = segmentAnnotations
                  .filter(({ segLen }) => (segLen ?? 0) >= 3) // 3개월 이상 구간만 표시
                  .map(({ x, regime, avgOverlap }) => ({
                    x, y: 0.97,
                    xref: 'x' as const, yref: 'paper' as const,
                    text: `${avgOverlap}%`,
                    showarrow: false,
                    font: { size: 10, color: regime === 'Bull' ? '#4ade80' : '#f87171' },
                    bgcolor: 'rgba(0,0,0,0.5)',
                    borderpad: 2,
                  }));

                return (
                  <div className="space-y-4">
                    <p className="text-xs text-muted">실제 종목 선택 기반 완전 재백테스트 — 레짐 전환 시 거래비용 자동 반영</p>

                    <DataTable
                      columns={[
                        { key: '전략', label: '전략', align: 'left' },
                        { key: '수익률', label: '수익률', align: 'right', mono: true },
                        { key: 'CAGR', label: 'CAGR', align: 'right', mono: true },
                        { key: 'Sharpe', label: 'Sharpe', align: 'right', mono: true },
                        { key: 'MDD', label: 'MDD', align: 'right', mono: true },
                      ]}
                      data={tableData}
                      maxHeight="200px"
                    />

                    {/* 레짐 조합 지표 */}
                    {(() => {
                      const tc = combo.transition_count as number | undefined;
                      const ab = combo.avg_bull_duration as number | undefined;
                      const abr = combo.avg_bear_duration as number | undefined;
                      const op = combo.overlap_pct as number | null | undefined;
                      if (tc == null) return null;
                      return (
                        <div className="rounded-lg border border-border bg-surface/50 px-4 py-3 text-sm">
                          <p className="text-xs text-muted font-medium mb-2">레짐 조합 지표</p>
                          <div className={`grid gap-3 ${op != null ? 'grid-cols-3' : 'grid-cols-2'}`}>
                            {op != null && (
                              <div className="text-center">
                                <p className="text-xs text-muted">종목 겹침율</p>
                                <p className={`text-lg font-semibold ${op < 30 ? 'text-red-400' : op < 60 ? 'text-yellow-400' : 'text-green-400'}`}>{op}%</p>
                                <p className="text-[10px] text-muted">{op < 30 ? '낮음 → 전환 비용 주의' : op < 60 ? '보통' : '높음 → 전환 비용 적음'}</p>
                              </div>
                            )}
                            <div className="text-center">
                              <p className="text-xs text-muted">레짐 전환 횟수</p>
                              <p className={`text-lg font-semibold ${(tc ?? 0) > 15 ? 'text-red-400' : (tc ?? 0) > 8 ? 'text-yellow-400' : 'text-green-400'}`}>{tc}회</p>
                              <p className="text-[10px] text-muted">{(tc ?? 0) > 15 ? '잦음 → 비용 누적' : (tc ?? 0) > 8 ? '보통' : '적음 → 유리'}</p>
                            </div>
                            <div className="text-center">
                              <p className="text-xs text-muted">평균 레짐 지속</p>
                              <p className="text-lg font-semibold text-foreground">Bull {ab}개월</p>
                              <p className="text-[10px] text-muted">Bear {abr}개월</p>
                            </div>
                          </div>
                        </div>
                      );
                    })()}

                    <PlotlyChart
                      data={comboChartData}
                      layout={{
                        yaxis: { title: { text: '누적수익률 (%)' }, ticksuffix: '%' },
                        xaxis: { title: { text: '' } },
                        legend: { orientation: 'h', y: -0.15 },
                        shapes: chartShapes,
                        annotations: chartAnnotations,
                      }}
                      height={360}
                    />

                    <div className="flex items-center gap-3">
                      <button
                        onClick={handleSaveCombo}
                        className="px-4 py-2 rounded-lg bg-accent-green/20 text-accent-green text-sm font-medium hover:bg-accent-green/30 transition-colors"
                      >
                        조합 전략 저장
                      </button>
                      {regimeSaveMsg && (
                        <p className={`text-xs ${regimeSaveMsg.includes('실패') ? 'text-accent-red' : 'text-accent-green'}`}>
                          {regimeSaveMsg}
                        </p>
                      )}
                    </div>
                  </div>
                );
              })()}
            </>
          );
        })()}
      </div>
      )}
    </div>
  );
}
