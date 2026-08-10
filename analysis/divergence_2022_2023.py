"""
analysis/divergence_2022_2023.py
2022/2023 hsmm 4/5 vs ai_v2 레짐 분기 원인 진단.
각 달 hsmm 입력(breadth/mom/FX) + 후보 피처(VIX/US10Y Δ/KR10Y/장단기/외국인/HY OAS/DXY) + 두 레짐 + KOSPI수익.
불일치 달에서 어떤 후보가 ai 방향과 일치하는 극단값이었는지 → 추가 피처 후보 규명. (HMM 재적합 없음)
사용: DATABASE_URL=<ip> .venv/bin/python analysis/divergence_2022_2023.py
"""
import os, json, warnings
import numpy as np, pandas as pd, psycopg2
from datetime import timedelta
from pathlib import Path
from dotenv import load_dotenv
warnings.filterwarnings("ignore"); load_dotenv(Path(__file__).parent.parent / ".env")
A = Path(__file__).parent


def main():
    h = pd.read_csv(A / "kospi_stocks_2000_2016.csv", dtype={"stock_code": str}, parse_dates=["dt"])[["dt", "stock_code", "close"]]
    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    d = pd.read_sql("SELECT trade_date::date dt, stock_code, close::float close FROM alpha_lab.daily_price WHERE close IS NOT NULL", conn)
    def macD(ind):
        x = pd.read_sql(f"SELECT left(period,10) p, value::float v FROM alpha_lab.macro_indicators WHERE indicator='{ind}' AND freq='D'", conn)
        x['p'] = pd.to_datetime(x['p']); return x.set_index('p')['v'].sort_index()
    def macM(ind):
        x = pd.read_sql(f"SELECT left(period,7) p, value::float v FROM alpha_lab.macro_indicators WHERE indicator='{ind}' AND freq='M'", conn)
        return x.set_index('p')['v']
    kospi = macD('kospi'); usdkrw = macD('usd_krw'); vix = macD('vix'); us10y = macD('us10y'); dxy = macD('dxy')
    kr10 = macD('bond_10y'); foreign = macD('investor_foreign_kospi'); yspread = macM('yield_spread'); conn.close()
    # HY OAS (csv, 2023-06+)
    try:
        oas = pd.read_csv(A / "macro_hy_oas.csv"); oas['date'] = pd.to_datetime(oas['date']); oas = oas.set_index('date')['oas'].sort_index()
    except Exception: oas = pd.Series(dtype=float)
    d["dt"] = pd.to_datetime(d["dt"])
    wide = pd.concat([h, d[["dt", "stock_code", "close"]]], ignore_index=True).drop_duplicates(["dt", "stock_code"]).pivot_table(index="dt", columns="stock_code", values="close").sort_index()
    ma = wide.rolling(120, min_periods=60).mean(); val = wide.notna() & ma.notna()
    breadth = ((wide > ma) & val).sum(axis=1) / val.sum(axis=1).clip(lower=1); breadth = breadth[val.sum(axis=1) > 100]
    ym = pd.PeriodIndex(kospi.index, freq="M"); last = {}
    for dt_, p in zip(kospi.index, ym): last[p] = dt_
    mends = [last[p] for p in sorted(last)]
    def asof(s, e): x = s[s.index <= e]; return x.iloc[-1] if len(x) else np.nan
    def pct(s, e, dy): c = asof(s, e); p = s[s.index <= e - timedelta(days=dy)]; return (c/p.iloc[-1]-1)*100 if len(p) and p.iloc[-1] else np.nan
    def chg(s, e, dy): c = asof(s, e); p = s[s.index <= e - timedelta(days=dy)]; return (c - p.iloc[-1]) if len(p) else np.nan
    def msum(s, e):  # 그 달 외국인 순매수 합
        m = s[(s.index.to_period('M') == pd.Period(e, 'M'))]; return m.sum() if len(m) else np.nan

    bc = json.load(open(A / "hsmm_fx_bearcount.json"))
    ers = {r["as_of"][:7]: r.get("expected_return", 0) for r in json.load(open(A / "regime_agent_results.json"))}
    amap = {}; prev = "Bull"
    for y in sorted(ers):
        er = ers[y]; cur = ("Bear" if er <= -2 else "Bull") if prev == "Bull" else ("Bull" if er >= 1 else "Bear"); amap[y] = cur; prev = cur

    rows = []
    for e in mends:
        y = pd.Period(e, 'M').strftime("%Y-%m")
        if y < "2021-10" or y > "2023-12": continue
        nxt = [m for m in mends if m > e]
        kret = (asof(kospi, nxt[0])/asof(kospi, e)-1)*100 if nxt else np.nan
        rows.append(dict(ym=y,
            breadth=asof(breadth, e)*100, mom=pct(kospi, e, 30)-pct(kospi, e, 180), fx3m=pct(usdkrw, e, 90),
            VIX=asof(vix, e), US10Y_d3=chg(us10y, e, 90), KR10Y=asof(kr10, e),
            ysprd=yspread.get(y, np.nan), foreign_b=msum(foreign, e)/1e8 if not foreign.empty else np.nan,
            HYOAS=asof(oas, e) if not oas.empty else np.nan, DXY_d3=pct(dxy, e, 90),
            hsmm=("Bear" if bc.get(y, 0) >= 4 else "Bull"), ai=amap.get(y, "Bull"), kret=kret))
    df = pd.DataFrame(rows)
    pd.set_option('display.width', 220, 'display.max_columns', 30)
    print("=== 2021-10~2023-12 월별 feature + 레짐 (hsmm 입력 | 후보 피처 | 레짐 | 익월KOSPI) ===\n")
    show = df.copy()
    for c in ['breadth','mom','fx3m','VIX','US10Y_d3','KR10Y','ysprd','foreign_b','HYOAS','DXY_d3','kret']:
        show[c] = show[c].map(lambda x: f"{x:+.1f}" if pd.notna(x) else "  .")
    show['flag'] = df.apply(lambda r: "◀불일치" if r.hsmm != r.ai else "", axis=1)
    print(show[['ym','breadth','mom','fx3m','VIX','US10Y_d3','KR10Y','ysprd','foreign_b','HYOAS','DXY_d3','hsmm','ai','kret','flag']].to_string(index=False))

    print("\n=== 불일치 달 그룹별 후보 피처 평균 ===")
    df['pair'] = df.hsmm + "/" + df.ai
    cand = ['breadth','mom','fx3m','VIX','US10Y_d3','KR10Y','ysprd','foreign_b','HYOAS','DXY_d3']
    g = df.groupby('pair')[cand].mean()
    print(g.round(2).to_string())
    print("\n  외국인_b = 그달 외국인 순매수 합(억원), foreign<0=순매도(risk-off)")
    print("  해석: hsmm=Bull/ai=Bear(2022) 행에서 risk-off로 극단(VIX↑/US10Y_d3↑/OAS↑/foreign<0)인 후보 = hsmm에 없어서 놓친 신호")
    df.to_csv(A / "divergence_2022_2023.csv", index=False); print("\n  상세: analysis/divergence_2022_2023.csv")


if __name__ == "__main__":
    main()
