"""
analysis/test_early_signals.py

조기경보 신호 직접 검증: 각 신호가 미래 수익과 *올바른 방향*인지 (패러독스 없는지).
신호↑ → 미래수익↓ (음의 상관, 단조) 면 진짜 조기경보. AUC 아니라 수익-tilt로 판정.

안 해본 신호:
  mom_decel  : 모멘텀 감속 (1m - 6m, 음=둔화)
  sox_div    : SOX 선행 다이버전스 (sox_1m - kospi_1m, 음=반도체 먼저 약함)
  chgpt      : 변화점 (최근21d 평균 - 252d 평균)/std, 음=하락전환
  overext    : 과확장 (pos12, 고=과열)
  riskoff    : 크로스에셋 (dxy_1m - us10y_chg, 고=리스크오프)
  vix_accel  : VIX 가속 (이번달 변화 - 지난달 변화)
대상수익: 향후 1/3/6개월. Spearman 상관 + 신호 tercile별 미래수익.
사용: .venv/bin/python analysis/test_early_signals.py
"""
import os, warnings
import numpy as np, pandas as pd, psycopg2
from pathlib import Path
from datetime import timedelta
from scipy.stats import spearmanr
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv(Path(__file__).parent.parent / ".env")
START = pd.Timestamp("2004-07-01")


def db(conn, ind):
    df = pd.read_sql("SELECT period::date dt, value::float v FROM alpha_lab.macro_indicators "
                     "WHERE indicator=%s AND freq='D' ORDER BY period", conn, params=(ind,))
    df["dt"] = pd.to_datetime(df["dt"])
    return df.dropna().drop_duplicates("dt").set_index("dt")["v"].sort_index()


def asof(s, e):
    sub = s[s.index <= e]; return sub.iloc[-1] if len(sub) else np.nan


def pct(s, e, d):
    cur = asof(s, e); past = s[s.index <= e - timedelta(days=d)]
    return (cur / past.iloc[-1] - 1) if len(past) and past.iloc[-1] else np.nan


def chg(s, e, d):
    cur = asof(s, e); past = s[s.index <= e - timedelta(days=d)]
    return (cur - past.iloc[-1]) if len(past) else np.nan


def main():
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    kospi = db(conn, "kospi"); sox = db(conn, "sox")
    vix = db(conn, "vix"); dxy = db(conn, "dxy"); us10y = db(conn, "us10y")
    conn.close()
    kret = kospi.pct_change()
    ym = pd.PeriodIndex(kospi.index, freq="M"); lastd = {}
    for d, p in zip(kospi.index, ym): lastd[p] = d
    mends = [lastd[p] for p in sorted(lastd)]

    rows = []
    for i in range(len(mends) - 1):
        e = mends[i]
        if e < START or i + 6 >= len(mends):
            continue
        s = kospi[kospi.index <= e]; hi, lo = s.iloc[-252:].max(), s.iloc[-252:].min()
        r21 = kret[kret.index <= e].iloc[-21:].mean(); r252 = kret[kret.index <= e].iloc[-252:]
        chgpt = (r21 - r252.mean()) / r252.std() if r252.std() else np.nan
        sig = {
            "mom_decel": pct(kospi, e, 30) - pct(kospi, e, 180),
            "sox_div": pct(sox, e, 30) - pct(kospi, e, 30),
            "chgpt": chgpt,
            "overext": (asof(kospi, e) - lo) / (hi - lo) if hi > lo else np.nan,
            "riskoff": pct(dxy, e, 30) - chg(us10y, e, 30) / 100,
            "vix_accel": chg(vix, e, 30) - chg(vix, e, 60) / 2,
            "vix_z": (asof(vix, e) - vix[vix.index <= e].iloc[-63:].mean()) / (vix[vix.index <= e].iloc[-63:].std() or 1),
        }
        sig["r1"] = (kospi.loc[mends[i + 1]] / kospi.loc[e] - 1) * 100
        sig["r3"] = (kospi.loc[mends[i + 3]] / kospi.loc[e] - 1) * 100
        sig["r6"] = (kospi.loc[mends[i + 6]] / kospi.loc[e] - 1) * 100
        rows.append(sig)

    df = pd.DataFrame(rows).dropna().reset_index(drop=True)
    sigs = ["mom_decel", "sox_div", "chgpt", "overext", "riskoff", "vix_accel", "vix_z"]
    print(f"표본 {len(df)}개월\n")
    print("신호별 미래수익 Spearman 상관 (음수=신호↑→수익↓=조기경보로 유용)")
    print(f"  {'신호':10} {'vs r1':>8} {'vs r3':>8} {'vs r6':>8}")
    for sg in sigs:
        c1 = spearmanr(df[sg], df["r1"]).correlation
        c3 = spearmanr(df[sg], df["r3"]).correlation
        c6 = spearmanr(df[sg], df["r6"]).correlation
        flag = " ◀경보가능" if (c6 < -0.12 and c3 < -0.08) else ""
        print(f"  {sg:10} {c1:>+8.3f} {c3:>+8.3f} {c6:>+8.3f}{flag}")

    print("\n신호 고/저 tercile별 향후 6개월 수익 (패러독스 체크):")
    for sg in sigs:
        d = df.copy(); d["t"] = pd.qcut(d[sg], 3, labels=["저", "중", "고"], duplicates="drop")
        lo_r = d[d["t"] == "저"]["r6"].mean(); hi_r = d[d["t"] == "고"]["r6"].mean()
        verdict = "✓경보(고<저)" if hi_r < lo_r - 1 else ("✗패러독스(고>저)" if hi_r > lo_r + 1 else "~중립")
        print(f"  {sg:10} 저={lo_r:+.1f}%  고={hi_r:+.1f}%   {verdict}")

    print("\n판정: '음의 상관 + 고tercile 수익<저tercile' 신호가 있으면 진짜 조기경보(패러독스 탈출).")


if __name__ == "__main__":
    main()
