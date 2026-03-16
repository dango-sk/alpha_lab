'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { getResults, getConfig } from './api';

// ─── Types ───
export interface StrategyResult {
  rebalance_dates: string[];
  portfolio_values: number[];
  monthly_returns: number[];
  total_return: number;
  cagr: number;
  mdd: number;
  sharpe: number;
  avg_turnover?: number;
  avg_portfolio_size?: number;
  strategy?: string;
  portfolio_sizes?: number[];
}

export interface Config {
  backtest_config: {
    start: string;
    end: string;
    insample_end: string;
    oos_start: string;
    top_n_stocks: number;
    weight_cap_pct: number;
    transaction_cost_bp: number;
    universe: string;
  };
  strategy_labels: Record<string, string>;
  strategy_colors: Record<string, string>;
  all_keys: string[];
}

// ─── Dashboard params hook ───
export function useDashboardParams() {
  const [universe, setUniverse] = useState<'KOSPI' | 'KOSPI+KOSDAQ'>('KOSPI');
  const [rebalType, setRebalType] = useState<'monthly' | 'biweekly'>('monthly');

  return { universe, setUniverse, rebalType, setRebalType };
}

// ─── Config hook ───
export function useConfig() {
  const [config, setConfig] = useState<Config | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getConfig().then(setConfig).catch(console.error).finally(() => setLoading(false));
  }, []);

  return { config, loading };
}

// ─── Results hook ───
export function useResults(universe: string, rebalType: string) {
  const [results, setResults] = useState<Record<string, StrategyResult>>({});
  const [labels, setLabels] = useState<Record<string, string>>({});
  const [colors, setColors] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    getResults({ universe, rebal_type: rebalType })
      .then((data) => {
        setResults(data);
        // Also fetch config for labels/colors
        getConfig().then((cfg) => {
          setLabels(cfg.strategy_labels);
          setColors(cfg.strategy_colors);
        });
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [universe, rebalType]);

  return { results, labels, colors, loading, error };
}

// ─── Color helpers ───
export function valueColor(v: number): string {
  if (v > 0) return 'text-accent-green';
  if (v < 0) return 'text-accent-red';
  return 'text-foreground';
}

export function valueBg(v: number): string {
  if (v > 0) return 'bg-accent-green/10';
  if (v < 0) return 'bg-accent-red/10';
  return '';
}

export function fmtPct(v: number, decimals = 1): string {
  if (v == null || isNaN(v) || !isFinite(v)) return '-';
  return `${v >= 0 ? '+' : ''}${(v * 100).toFixed(decimals)}%`;
}

export function fmtPctRaw(v: number, decimals = 1): string {
  if (v == null || isNaN(v) || !isFinite(v)) return '-';
  return `${v >= 0 ? '+' : ''}${v.toFixed(decimals)}%`;
}

export function fmtNum(v: number, decimals = 2): string {
  if (v == null || isNaN(v) || !isFinite(v)) return '-';
  return v.toFixed(decimals);
}
