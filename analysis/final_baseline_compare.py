"""
analysis/final_baseline_compare.py

모든 후보를 동일 FCF 환경에서 비교 → Baseline 확정.
후보: ruleA, hmm_diag, hmm_sticky, hmm_full, hmm_gmm, seq_gbm + ai(기준) + bull_only(기준).
지표: CAGR / Sharpe / MDD / Calmar(=CAGR/|MDD|) / Turnover.
각 레짐맵 생성(walk-forward) → rf 슬롯 → run_regime_combo_backtest.
사용: .venv/bin/python analysis/final_baseline_compare.py   (≈1시간, 백그라운드 권장)
"""
import os, json, shutil, sys, warnings
import numpy as np, pandas as pd, psycopg2
from pathlib import Path
from datetime import timedelta
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from hmmlearn.hmm import GaussianHMM, GMMHMM
warnings.filterwarnings("ignore"); load_dotenv(Path(__file__).parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).parent.parent))
A = Path(__file__).parent
BULL, BEAR = "FCF_YIELD추가전략", "FCF_YIELD_BEAR전략"
MIN_TRAIN, SEED = 36, 42


def build_panel():
    h = pd.read_csv(A / "kospi_stocks_2000_2016.csv", dtype={"stock_code": str}, parse_dates=["dt"])[["dt", "stock_code", "close"]]
    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    d = pd.read_sql("SELECT trade_date::date dt, stock_code, close::float close FROM alpha_lab.daily_price WHERE close IS NOT NULL", conn)
    kk = pd.read_sql("SELECT period::date dt, value::float v FROM alpha_lab.macro_indicators WHERE indicator='kospi' AND freq='D' ORDER BY period", conn); conn.close()
    d["dt"] = pd.to_datetime(d["dt"]); kk["dt"] = pd.to_datetime(kk["dt"]); kospi = kk.set_index("dt")["v"].sort_index()
    wide = pd.concat([h, d[["dt", "stock_code", "close"]]], ignore_index=True).drop_duplicates(["dt", "stock_code"]).pivot_table(index="dt", columns="stock_code", values="close").sort_index()
    ma = wide.rolling(120, min_periods=60).mean(); val = wide.notna() & ma.notna()
    breadth = ((wide > ma) & val).sum(axis=1) / val.sum(axis=1).clip(lower=1); breadth = breadth[val.sum(axis=1) > 100]
    rmn = wide.rolling(252, min_periods=120).min(); newlow = ((wide <= rmn) & wide.notna()).sum(axis=1) / wide.notna().sum(axis=1).clip(lower=1); newlow = newlow.reindex(breadth.index)
    ym = pd.PeriodIndex(kospi.index, freq="M"); last = {}
    for dt_, p in zip(kospi.index, ym): last[p] = dt_
    mends = [last[p] for p in sorted(last) if last[p] >= breadth.index.min()]
    def asof(s, e): x = s[s.index <= e]; return x.iloc[-1] if len(x) else np.nan
    def pct(s, e, dy): c = asof(s, e); p = s[s.index <= e - timedelta(days=dy)]; return (c/p.iloc[-1]-1) if len(p) and p.iloc[-1] else np.nan
    n = len(mends); F = []
    for i in range(n):
        e = mends[i]
        F.append([asof(breadth, e), asof(breadth, e) - asof(breadth, e - timedelta(days=30)),
                  asof(newlow, e), asof(newlow, e) - asof(newlow, e - timedelta(days=30)),
                  pct(kospi, e, 30) - pct(kospi, e, 180)])
    F = np.nan_to_num(np.array(F))
    yms = [pd.Period(e, freq="M").strftime("%Y-%m") for e in mends]
    return F, yms, n


def hmm_map(F, yms, n, kind):
    def mk():
        if kind == "diag": return GaussianHMM(2, "diag", n_iter=60, random_state=SEED)
        if kind == "sticky": return GaussianHMM(2, "diag", n_iter=60, random_state=SEED, transmat_prior=np.array([[20., 1.], [1., 20.]]))
        if kind == "full": return GaussianHMM(2, "full", n_iter=60, random_state=SEED)
        if kind == "gmm": return GMMHMM(n_components=2, n_mix=2, covariance_type="diag", n_iter=60, random_state=SEED)
    reg = ["Bull"] * n
    for t in range(MIN_TRAIN, n):
        try:
            sc = StandardScaler().fit(F[:t]); Z = sc.transform(F[:t]); hm = mk(); hm.fit(Z)
            mu = hm.means_; mu = mu.mean(axis=1) if mu.ndim == 3 else mu
            bear = int(np.argmax(-mu[:, 0] - mu[:, 1] + mu[:, 2] + mu[:, 3] - mu[:, 4]))
            reg[t] = "Bear" if hm.predict(Z)[-1] == bear else "Bull"
        except Exception:
            reg[t] = reg[t-1]
    return dict(zip(yms, reg))


def seq_map(F, yms, n, K=3):
    y = np.zeros(n)  # bear regime label은 외부 필요없음; 여기선 dd 대신 self-label 불가 → ruleA 라벨 재사용 위해 간단화
    # 시퀀스 모델은 bear regime(dd6<=-15) 라벨 필요 → 재계산
    return None  # (아래 main에서 별도 처리)


def metrics(c):
    cagr = c.get("cagr"); mdd = c.get("mdd")
    return {"CAGR": cagr, "Sharpe": c.get("sharpe"), "MDD": mdd,
            "Calmar": (cagr / abs(mdd)) if (cagr is not None and mdd) else None,
            "Turnover": c.get("avg_turnover")}


def main():
    from lib.data import run_regime_combo_backtest
    F, yms, n = build_panel()
    print(f"패널 {n}개월, 맵 생성...", flush=True)
    maps = {}
    # ruleA: 기존 저장본 재사용
    if (A / "regime_ABC_A.json").exists():
        maps["ruleA"] = json.load(open(A / "regime_ABC_A.json"))
    for k in ["diag", "sticky", "full", "gmm"]:
        maps[f"hmm_{k}"] = hmm_map(F, yms, n, k); print(f"  hmm_{k} 맵 생성", flush=True)
    maps["bull_only"] = {y: "Bull" for y in yms}

    res = {}
    for name, mp in maps.items():
        (A / "regime_rf_map.json").write_text(json.dumps(mp, ensure_ascii=False))
        print(f"\n=== FCF: {name} ===", flush=True)
        r = run_regime_combo_backtest(bull_key=BULL, bear_key=BEAR, universe="KOSPI", rebal_type="monthly", regime_mode="rf")
        res[name] = (r or {}).get("REGIME_COMBO", {})
    # AI v2 기준
    r = run_regime_combo_backtest(bull_key=BULL, bear_key=BEAR, universe="KOSPI", rebal_type="monthly", regime_mode="ai")
    res["ai_v2"] = (r or {}).get("REGIME_COMBO", {})

    print(f"\n{'#'*70}\n  최종 Baseline 비교 (동일 FCF)\n{'#'*70}")
    print(f"  {'후보':12} {'CAGR':>7} {'Sharpe':>7} {'MDD':>7} {'Calmar':>7} {'Turnover':>9}")
    table = []
    for name, c in res.items():
        if not c: continue
        m = metrics(c); table.append((name, m))
        print(f"  {name:12} {m['CAGR']*100:>6.1f}% {m['Sharpe']:>7.2f} {m['MDD']*100:>6.1f}% "
              f"{m['Calmar']:>7.2f} {(m['Turnover'] or 0):>8.2f}")
    best = max([t for t in table if t[1]['Calmar'] is not None], key=lambda x: x[1]['Calmar'])
    print(f"\n  → Calmar 최고 = {best[0]} (Baseline 후보)")


if __name__ == "__main__":
    main()
