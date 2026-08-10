"""
analysis/eval_pnl_regime.py

Step 3: 레짐 신호를 *실제 전략 수익(PnL)* 으로 채점.
"매달 우산 들지 말지" = 매월 레짐(Bull/Bear) 정하고 trailing-stop 돌려서
BM(무손절) vs AI v2 vs HMM 누적수익·CAGR·MDD·Sharpe·손절횟수 비교.

전략 (단순화):
  - KOSPI 지수 보유.
  - Bear 월에만 고점(진입 후 최고가)대비 -STOP% 빠지면 현금화(exit).
  - Bull 월 되면 재매수(peak 리셋).
  (님 전략: bull/bear 동일 비중, Bear일 때만 trailing stop)

레짐 신호:
  - BM      : 항상 보유 (벤치마크)
  - AI v2   : gemini judgment='약세' → Bear (sticky: 변동성/중립은 직전 유지)
  - HMM     : p_bear >= 임계 → Bear. 임계는 *전반부로 정하고 후반부 검증*(하드코딩 X).

입력:  Railway PG (kospi freq='D')
       analysis/hmm_regime_features.csv (p_bear_2s)
       analysis/regime_agent_multimodel_results_gemini.json (AI v2)
출력:  콘솔 비교표 + analysis/pnl_equity_curves.csv

사용:  .venv/bin/python analysis/eval_pnl_regime.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import psycopg2
from pathlib import Path
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv(Path(__file__).parent.parent / ".env")

HMM_CSV = Path(__file__).parent / "hmm_regime_features.csv"
AI_JSON = Path(__file__).parent / "regime_agent_multimodel_results_gemini.json"
OUT_CSV = Path(__file__).parent / "pnl_equity_curves.csv"

STOP = 0.15          # trailing stop -15%
ANN = 252            # 연율화


def load_kospi(conn):
    df = pd.read_sql(
        "SELECT period::date AS dt, value::float AS close FROM alpha_lab.macro_indicators "
        "WHERE indicator='kospi' AND freq='D' ORDER BY period", conn)
    df["dt"] = pd.to_datetime(df["dt"])
    return df.dropna().drop_duplicates(subset="dt").set_index("dt")["close"].sort_index()


def simulate(close, regime_by_month, stop=STOP, cost=0.0):
    """일별 시뮬. regime_by_month: {'YYYY-MM': 'Bull'/'Bear'}. cost=편도 거래비용(비율).
    반환: equity 시리즈, 손절횟수."""
    ret = close.pct_change().fillna(0.0)
    equity = pd.Series(index=close.index, dtype=float)
    eq = 1.0
    invested = True
    peak = close.iloc[0]
    stops = 0
    for i, (dt, px) in enumerate(close.items()):
        ym = dt.strftime("%Y-%m")
        reg = regime_by_month.get(ym, "Bull")
        # 그날 수익 반영 (보유 중일 때만)
        if i > 0 and invested:
            eq *= (1 + ret.iloc[i])
        equity.iloc[i] = eq
        if invested:
            peak = max(peak, px)
            if reg == "Bear" and px / peak - 1 <= -stop:
                invested = False           # 손절 → 현금
                eq *= (1 - cost)
                stops += 1
        else:
            if reg == "Bull":
                invested = True            # 재매수
                eq *= (1 - cost)
                peak = px
    return equity, stops


def metrics(equity):
    total = equity.iloc[-1] / equity.iloc[0] - 1
    yrs = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / yrs) - 1
    dret = equity.pct_change().dropna()
    sharpe = dret.mean() / dret.std() * np.sqrt(ANN) if dret.std() > 0 else 0
    mdd = (equity / equity.cummax() - 1).min()
    return total, cagr, sharpe, mdd


def ai_regime(close_index):
    """AI v2 judgment → 월별 Bull/Bear (sticky)."""
    if not AI_JSON.exists():
        return {}
    recs = sorted(json.load(open(AI_JSON)), key=lambda r: r["as_of"])
    out, prev = {}, "Bull"
    for r in recs:
        ym = r["as_of"][:7]
        j = r.get("judgment")
        cur = "Bear" if j == "약세" else ("Bull" if j == "강세" else prev)
        out[ym] = cur
        prev = cur
    return out


def hmm_regime(threshold):
    """HMM p_bear_2s >= threshold → Bear."""
    if not HMM_CSV.exists():
        return {}
    h = pd.read_csv(HMM_CSV)
    return {row["pred_month"]: ("Bear" if row["p_bear_2s"] >= threshold else "Bull")
            for _, row in h.iterrows() if pd.notna(row.get("p_bear_2s"))}


def pick_hmm_threshold(close, hmm_df, split_date):
    """전반부(<split)에서 누적수익 최대 임계 선택 → 후반부 검증 (하드코딩 회피)."""
    best_t, best_v = 0.5, -1e9
    train = close[close.index < split_date]
    for t in np.arange(0.20, 0.71, 0.05):
        reg = {row["pred_month"]: ("Bear" if row["p_bear_2s"] >= t else "Bull")
               for _, row in hmm_df.iterrows() if pd.notna(row.get("p_bear_2s"))}
        eq, _ = simulate(train, reg)
        v = metrics(eq)[2]   # Sharpe (위험조정) — 상승장 "방어 안함" 함정 회피
        if v > best_v:
            best_v, best_t = v, t
    return round(best_t, 2)


def hmm_reg_from_df(hmm_df, thr):
    return {row["pred_month"]: ("Bear" if row["p_bear_2s"] >= thr else "Bull")
            for _, row in hmm_df.iterrows() if pd.notna(row.get("p_bear_2s"))}


def sweep(close_window, hmm_df, label):
    """임계 0.2~0.8 전수 비교 + BM 기준선. BM Sharpe/MDD를 이기는 임계 있나?"""
    print(f"\n{'='*78}\n  임계 스윕: {label}  ({close_window.index[0].date()} ~ {close_window.index[-1].date()})")
    print(f"  {'임계':>8} {'누적수익':>10} {'CAGR':>8} {'Sharpe':>7} {'MDD':>8} {'손절':>5}")
    bm, _ = simulate(close_window, {})
    t, c, s, m = metrics(bm)
    print(f"  {'BM':>8} {t*100:>9.1f}% {c*100:>7.1f}% {s:>7.2f} {m*100:>7.1f}% {0:>5}")
    bm_sharpe, bm_mdd = s, m
    for thr in np.arange(0.20, 0.81, 0.10):
        reg = hmm_reg_from_df(hmm_df, round(thr, 2))
        eq, stops = simulate(close_window, reg)
        t, c, s, m = metrics(eq)
        flag = ""
        if s > bm_sharpe:
            flag += " ★Sharpe>BM"
        if m > bm_mdd and s >= bm_sharpe * 0.95:
            flag += " ◎MDD<BM&Sharpe유지"
        print(f"  p>={thr:>4.1f} {t*100:>9.1f}% {c*100:>7.1f}% {s:>7.2f} {m*100:>7.1f}% {stops:>5}{flag}")


def simulate_bounce(close, bear_months, bounce_pct, stop=STOP, cost=0.0):
    """HMM Bear월 손절 exit + *저점대비 bounce_pct 반등하면 즉시 재진입* (방향 기반).
    재진입이 변동성과 무관해 V자 반등을 일찍 잡음. cost=편도 거래비용."""
    ret = close.pct_change().fillna(0.0)
    equity = pd.Series(index=close.index, dtype=float)
    eq, invested, peak, low, stops = 1.0, True, close.iloc[0], None, 0
    for i, (dt, px) in enumerate(close.items()):
        reg = bear_months.get(dt.strftime("%Y-%m"), "Bull")
        if i > 0 and invested:
            eq *= (1 + ret.iloc[i])
        equity.iloc[i] = eq
        if invested:
            peak = max(peak, px)
            if reg == "Bear" and px / peak - 1 <= -stop:
                invested, low, stops = False, px, stops + 1
                eq *= (1 - cost)
        else:
            low = min(low, px)
            if px >= low * (1 + bounce_pct):   # 저점대비 반등 확인 → 재진입
                invested, peak = True, px
                eq *= (1 - cost)
    return equity, stops


def simulate_trend(close, window, cost=0.0):
    """순수 추세추종: 가격 > window일 이동평균이면 보유, 아니면 현금 (양방향). cost=편도."""
    ma = close.rolling(window).mean()
    ret = close.pct_change().fillna(0.0)
    equity = pd.Series(index=close.index, dtype=float)
    eq = 1.0
    switches = 0
    prev_in = True
    for i, (dt, px) in enumerate(close.items()):
        in_mkt = bool(px > ma.iloc[i]) if pd.notna(ma.iloc[i]) else True
        if i > 0 and prev_in:
            eq *= (1 + ret.iloc[i])
        equity.iloc[i] = eq
        if in_mkt != prev_in:
            switches += 1
            eq *= (1 - cost)
        prev_in = in_mkt
    return equity, switches


def run_window(close, regimes, label):
    print(f"\n{'='*78}\n  구간: {label}  ({close.index[0].date()} ~ {close.index[-1].date()})")
    print(f"  {'전략':22} {'누적수익':>10} {'CAGR':>8} {'Sharpe':>7} {'MDD':>8} {'손절':>5}")
    curves = {}
    for name, reg in regimes.items():
        eq, stops = simulate(close, reg)
        t, c, s, m = metrics(eq)
        curves[name] = eq
        print(f"  {name:22} {t*100:>9.1f}% {c*100:>7.1f}% {s:>7.2f} {m*100:>7.1f}% {stops:>5}")
    return curves


def main():
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    close = load_kospi(conn)
    conn.close()
    print(f"KOSPI {close.index[0].date()} ~ {close.index[-1].date()} ({len(close)})")

    hmm_df = pd.read_csv(HMM_CSV)
    ai = ai_regime(close.index)

    # HMM 임계: 전반부(2012년 이전)로 정하고 전체/겹침에서 검증
    split = pd.Timestamp("2012-01-01")
    thr = pick_hmm_threshold(close, hmm_df, split)
    print(f"HMM 임계 (전반부<2012로 선정): p_bear >= {thr}")
    hmm = hmm_regime(thr)

    bm = {}  # 빈 dict → 항상 Bull(보유)

    # 전체 구간 (HMM 데이터 시작 ~)
    hmm_start = pd.Timestamp(min(hmm.keys()) + "-01")
    full = close[close.index >= hmm_start]
    curves_full = run_window(full, {"BM(무손절)": bm, f"HMM(p>={thr})": hmm}, "전체 1999~")

    # AI v2 겹침 구간 (2018-04~)
    ov = None
    if ai:
        ov_start = pd.Timestamp(min(ai.keys()) + "-01")
        ov = close[close.index >= ov_start]
        run_window(ov, {"BM(무손절)": bm, "AI v2": ai, f"HMM(p>={thr})": hmm}, "AI v2 겹침 2018~")

    # 임계 전수 스윕 — BM 이기는 임계가 하나라도 있나?
    sweep(full, hmm_df, "전체 1999~")
    if ov is not None:
        sweep(ov, hmm_df, "AI v2 겹침 2018~")

    # ── 반등 잡기: 비대칭(HMM exit + 반등 재진입) & 순수 추세추종 ──
    train = close[close.index < split]
    # bounce_pct 튜닝 (train Sharpe 최대)
    best_b, best_bv = 0.05, -1e9
    for b in np.arange(0.02, 0.13, 0.02):
        eq, _ = simulate_bounce(train, hmm, round(b, 2))
        v = metrics(eq)[2]
        if v > best_bv:
            best_bv, best_b = v, round(b, 2)
    # MA window 튜닝 (train Sharpe 최대)
    best_w, best_wv = 200, -1e9
    for w in [50, 100, 150, 200]:
        eq, _ = simulate_trend(train, w)
        v = metrics(eq)[2]
        if v > best_wv:
            best_wv, best_w = v, w
    print(f"\n튜닝(전반부<2012): 반등 재진입 = 저점+{best_b*100:.0f}% / 추세 MA = {best_w}일")

    for win, lbl in [(full, "전체 1999~"), (ov, "AI v2 겹침 2018~")]:
        if win is None:
            continue
        print(f"\n{'='*78}\n  반등잡기 비교: {lbl}  ({win.index[0].date()} ~ {win.index[-1].date()})")
        print(f"  {'전략':28} {'누적수익':>10} {'CAGR':>8} {'Sharpe':>7} {'MDD':>8} {'전환':>5}")
        rows = [("BM(무손절)", *simulate(win, bm)),
                (f"HMM이진(p>={thr})", *simulate(win, hmm)),
                (f"HMM exit+반등{best_b*100:.0f}%", *simulate_bounce(win, hmm, best_b)),
                (f"추세추종(MA{best_w})", *simulate_trend(win, best_w))]
        bm_sh = metrics(simulate(win, bm)[0])[2]
        for name, eq, ev in rows:
            t, c, s, m = metrics(eq)
            star = " ★>BM" if name != "BM(무손절)" and s > bm_sh else ""
            print(f"  {name:28} {t*100:>9.1f}% {c*100:>7.1f}% {s:>7.2f} {m*100:>7.1f}% {ev:>5}{star}")

    # ── 거래비용 민감도 + MA 길이 robustness (진짜 검증) ──
    for win, lbl in [(full, "전체 1999~ (튜닝구간 포함=in-sample 주의)"),
                     (ov, "AI v2 겹침 2018~ (OOS, 신뢰구간)")]:
        if win is None:
            continue
        bm_eq, _ = simulate(win, bm)
        bm_t, bm_c, bm_sh, bm_m = metrics(bm_eq)
        print(f"\n{'='*78}\n  거래비용·MA robustness: {lbl}")
        print(f"  BM 기준: 누적 {bm_t*100:.0f}%  Sharpe {bm_sh:.2f}  MDD {bm_m*100:.0f}%")
        print(f"  {'전략 / 편도비용':22} {'0.0%':>14} {'0.1%':>14} {'0.3%':>14}")
        # 추세추종 여러 MA + HMM+반등, 비용 0/0.1/0.3%
        configs = [(f"추세 MA{w}", "trend", w) for w in [100, 150, 200]]
        configs.append((f"HMM+반등{best_b*100:.0f}%", "bounce", best_b))
        for name, kind, param in configs:
            cells = []
            for cost in [0.0, 0.001, 0.003]:
                if kind == "trend":
                    eq, _ = simulate_trend(win, param, cost=cost)
                else:
                    eq, _ = simulate_bounce(win, hmm, param, cost=cost)
                t, _, s, m = metrics(eq)
                flag = "★" if s > bm_sh else " "
                cells.append(f"{t*100:>6.0f}%/{s:.2f}{flag}")
            print(f"  {name:22} {cells[0]:>14} {cells[1]:>14} {cells[2]:>14}")
        print("  (셀=누적%/Sharpe, ★=BM Sharpe 초과)")

    pd.DataFrame(curves_full).to_csv(OUT_CSV)
    print(f"\nSaved equity curves → {OUT_CSV}")
    print("판정: HMM이 BM 대비 MDD 줄이면서 누적수익/Sharpe 지키면 → 실전 가치 有.")


if __name__ == "__main__":
    main()
