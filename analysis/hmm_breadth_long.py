"""
analysis/hmm_breadth_long.py

장기 breadth/newlow(2000~) + HMM 레짐.
입력: analysis/kospi_stocks_2000_2016.csv (yfinance 수집분) + alpha_lab.daily_price(2017~).
breadth=MA120 상회%, newlow=52주 신저가%. 둘을 2000~로 계산.
HMM emission=[breadth, breadth_chg, newlow, mom_decel], 2·3-state walk-forward.
평가: 전 구간 Bull/Bear 수익격차 + 전환수 + 위기 타임라인. 맵 저장.

사용: .venv/bin/python analysis/hmm_breadth_long.py
(1단계 collect_kospi_stocks_yf.py 먼저 실행 필요)
"""
import os, json, warnings
import numpy as np, pandas as pd, psycopg2
from pathlib import Path
from datetime import timedelta
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
warnings.filterwarnings("ignore"); load_dotenv(Path(__file__).parent.parent / ".env")
A = Path(__file__).parent
HIST = A / "kospi_stocks_2000_2016.csv"
MIN_TRAIN = 36
SEED = 42
N_STATES = [2, 3]


def load_prices():
    """2000~16(yfinance csv) + 2017~(daily_price) 합쳐 wide(date×stock) close."""
    frames = []
    if HIST.exists():
        h = pd.read_csv(HIST, dtype={"stock_code": str}, parse_dates=["dt"])
        frames.append(h[["dt", "stock_code", "close"]])
        print(f"  과거(csv): {len(h):,}행 {h['dt'].min().date()}~{h['dt'].max().date()}", flush=True)
    else:
        print("  ⚠️ kospi_stocks_2000_2016.csv 없음 — 1단계(collect_kospi_stocks_yf.py) 먼저 실행", flush=True)
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    d = pd.read_sql("SELECT trade_date::date dt, stock_code, close::float close FROM alpha_lab.daily_price WHERE close IS NOT NULL", conn)
    conn.close()
    d["dt"] = pd.to_datetime(d["dt"])
    frames.append(d[["dt", "stock_code", "close"]])
    allp = pd.concat(frames, ignore_index=True).drop_duplicates(["dt", "stock_code"])
    wide = allp.pivot_table(index="dt", columns="stock_code", values="close").sort_index()
    print(f"  통합 wide: {wide.shape[0]}일 × {wide.shape[1]}종목 ({wide.index.min().date()}~{wide.index.max().date()})", flush=True)
    return wide


def kospi_index(conn=None):
    c = psycopg2.connect(os.environ["DATABASE_URL"])
    k = pd.read_sql("SELECT period::date dt, value::float v FROM alpha_lab.macro_indicators WHERE indicator='kospi' AND freq='D' ORDER BY period", c)
    c.close()
    k["dt"] = pd.to_datetime(k["dt"]); return k.set_index("dt")["v"].sort_index()


def main():
    print("가격 로드+통합...", flush=True)
    wide = load_prices()
    ma120 = wide.rolling(120, min_periods=60).mean()
    valid = wide.notna() & ma120.notna()
    breadth = ((wide > ma120) & valid).sum(axis=1) / valid.sum(axis=1).clip(lower=1)
    breadth = breadth[valid.sum(axis=1) > 100]
    rollmin = wide.rolling(252, min_periods=120).min()
    newlow = ((wide <= rollmin) & wide.notna()).sum(axis=1) / wide.notna().sum(axis=1).clip(lower=1)
    newlow = newlow.reindex(breadth.index)
    print(f"  breadth/newlow: {breadth.index.min().date()}~{breadth.index.max().date()} ({len(breadth)}일)", flush=True)

    kospi = kospi_index()
    ym = pd.PeriodIndex(kospi.index, freq="M"); lastd = {}
    for d, p in zip(kospi.index, ym): lastd[p] = d
    mends = [lastd[p] for p in sorted(lastd) if lastd[p] >= breadth.index.min()]
    def asof(s, e): x = s[s.index <= e]; return x.iloc[-1] if len(x) else np.nan
    def pct(s, e, d): c = asof(s, e); p = s[s.index <= e - timedelta(days=d)]; return (c/p.iloc[-1]-1) if len(p) and p.iloc[-1] else np.nan

    rows = []
    for i in range(len(mends) - 1):
        e = mends[i]
        rows.append({"ym": pd.Period(mends[i+1], freq="M").strftime("%Y-%m"),
                     "breadth": asof(breadth, e), "br_chg": asof(breadth, e) - asof(breadth, e - timedelta(days=30)),
                     "newlow": asof(newlow, e), "mom_decel": pct(kospi, e, 30) - pct(kospi, e, 180),
                     "ret": (kospi.loc[mends[i+1]] / kospi.loc[e] - 1) * 100})
    df = pd.DataFrame(rows).dropna().reset_index(drop=True); n = len(df)
    feats = ["breadth", "br_chg", "newlow", "mom_decel"]
    X = df[feats].values
    print(f"\n월별 패널 {n}개월 ({df['ym'].iloc[0]}~{df['ym'].iloc[-1]})\n", flush=True)

    for ns in N_STATES:
        pbear = np.full(n, np.nan)
        for t in range(MIN_TRAIN, n):
            try:
                sc = StandardScaler().fit(X[:t]); Z = sc.transform(X[:t])
                hm = GaussianHMM(n_components=ns, covariance_type="diag", n_iter=60, random_state=SEED)
                hm.fit(Z)
                bear = int(np.argmin(hm.means_[:, 0]))   # breadth 평균 최저 = 약세
                pbear[t] = hm.predict_proba(Z)[-1, bear]
            except Exception:
                continue
        col = f"pbear_{ns}s"; df[col] = pbear
        ev = df.dropna(subset=[col]).copy()
        # raw (pbear>=0.5)
        ev["reg"] = np.where(ev[col] >= 0.5, "Bear", "Bull")
        tr = (ev["reg"].values[1:] != ev["reg"].values[:-1]).sum()
        br = ev[ev["reg"] == "Bull"]["ret"].mean(); ber = ev[ev["reg"] == "Bear"]["ret"].mean()
        print(f"=== HMM {ns}-state (raw) ===  평가 {len(ev)}개월, Bear {(ev['reg']=='Bear').sum()}, 전환 {tr}회")
        print(f"  Bull {br:+.2f}% vs Bear {ber:+.2f}% → 격차 {br-ber:+.2f}%p")
        # min-hold 안정화 (3개월): pbear>=0.5 진입, 단 최소 3개월 보유 후 전환
        pv = ev[col].values; mh = []; state = "Bull"; held = 99
        for x in pv:
            sw = False
            if held >= 3:
                if state == "Bull" and x >= 0.5: state = "Bear"; sw = True
                elif state == "Bear" and x < 0.5: state = "Bull"; sw = True
            mh.append(state); held = 0 if sw else held + 1
        ev["reg_mh"] = mh
        trm = (ev["reg_mh"].values[1:] != ev["reg_mh"].values[:-1]).sum()
        brm = ev[ev["reg_mh"] == "Bull"]["ret"].mean(); berm = ev[ev["reg_mh"] == "Bear"]["ret"].mean()
        print(f"  [min-hold3] Bear {(ev['reg_mh']=='Bear').sum()}, 전환 {trm}회 → "
              f"Bull {brm:+.2f}% vs Bear {berm:+.2f}% → 격차 {brm-berm:+.2f}%p\n")
        if ns == 2:
            json.dump(dict(zip(ev["ym"], ev["reg_mh"])), open(A / "regime_hmm_long_map.json", "w"), ensure_ascii=False)

    print("저장: regime_hmm_long_map.json (2-state). 다음: run_fcf_hmm 식으로 rf 슬롯 넣어 FCF vs AI v2.")


if __name__ == "__main__":
    main()
