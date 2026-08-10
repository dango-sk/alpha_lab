"""
analysis/slowbear_macro_signal.py  (v2: 역사적 episode 직접 정의)
DXY 모멘텀·금리변화가 Slow Bear에서 *일반적* risk-off 신호인지 검증.
가설: Slow Bear(긴축형)=금리↑+달러↑ / Fast Crash(패닉형)=금리↓+VIX급등.
각 episode 초기 하락구간(peak~+6M)의 DXY Δ3m, US10Y Δ3m, KR10Y Δ3m, VIX 비교.
사용: DATABASE_URL=<ip> .venv/bin/python analysis/slowbear_macro_signal.py
"""
import os, warnings
import numpy as np, pandas as pd, psycopg2
from datetime import timedelta
from pathlib import Path
from dotenv import load_dotenv
warnings.filterwarnings("ignore"); load_dotenv(Path(__file__).parent.parent / ".env")
A = Path(__file__).parent

# (이름, peak_ym, trough_ym, type)  — 역사적 분류
EPISODES = [
    ("닷컴 2000",    "2000-01", "2001-09", "Slow"),
    ("2002 약세",    "2002-04", "2003-03", "Slow"),
    ("2004 급락",    "2004-04", "2004-08", "Fast"),
    ("GFC 2008",     "2007-10", "2008-10", "Fast"),
    ("유럽 2011",    "2011-04", "2011-09", "Fast"),
    ("2015-16 둔화", "2015-04", "2016-01", "Slow"),
    ("2018 긴축",    "2018-01", "2019-01", "Slow"),
    ("코로나 2020",  "2020-01", "2020-03", "Fast"),
    ("2022 긴축",    "2021-12", "2022-10", "Slow"),
]


def main():
    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    def macD(ind):
        x = pd.read_sql(f"SELECT left(period,10) p, value::float v FROM alpha_lab.macro_indicators WHERE indicator='{ind}' AND freq='D'", conn)
        x['p'] = pd.to_datetime(x['p']); return x.set_index('p')['v'].sort_index()
    kospi = macD('kospi'); dxy = macD('dxy'); us10 = macD('us10y'); vix = macD('vix'); kr10 = macD('bond_10y'); conn.close()

    def me(ymv):  # 해당 월말 timestamp
        p = pd.Period(ymv, 'M'); return p.to_timestamp('M')
    def asof(s, e): x = s[s.index <= e]; return x.iloc[-1] if len(x) else np.nan
    def pct(s, e, dy): c = asof(s, e); p = s[s.index <= e - timedelta(days=dy)]; return (c/p.iloc[-1]-1)*100 if len(p) and pd.notna(c) and p.iloc[-1] else np.nan
    def chg(s, e, dy): c = asof(s, e); p = s[s.index <= e - timedelta(days=dy)]; return (c - p.iloc[-1]) if len(p) and pd.notna(c) else np.nan

    rows = []
    for name, pk, tr, typ in EPISODES:
        e0 = me(pk); e1 = me(tr)
        # 초기 하락구간: peak ~ min(peak+6M, trough) 의 월말들
        early_end = min(e1, (pd.Period(pk, 'M') + 6).to_timestamp('M'))
        months = pd.period_range(pk, pd.Period(early_end, 'M'), freq='M')
        ts = [p.to_timestamp('M') for p in months]
        dxy_d3 = np.nanmean([pct(dxy, t, 90) for t in ts])
        us_d3 = np.nanmean([chg(us10, t, 90) for t in ts])
        kr_d3 = np.nanmean([chg(kr10, t, 90) for t in ts])
        vmax = np.nanmax([asof(vix, t) for t in ts])
        ddv = (asof(kospi, e1)/asof(kospi, e0)-1)*100
        rows.append(dict(episode=name, type=typ, peak=pk, trough=tr, dd=ddv,
                         DXY_d3=dxy_d3, US10Y_d3=us_d3, KR10Y_d3=kr_d3, VIXmax=vmax))
    df = pd.DataFrame(rows)
    pd.set_option('display.width', 200)
    print("=== episode 초기하락구간(peak~+6M) 매크로 신호 ===\n")
    print(df.round(2).to_string(index=False))
    print("\n=== type별 평균 ===")
    g = df.groupby('type').agg(개수=('dd','size'), DXY_d3=('DXY_d3','mean'),
        US10Y_d3=('US10Y_d3','mean'), KR10Y_d3=('KR10Y_d3','mean'), VIXmax=('VIXmax','mean'))
    print(g.round(2).to_string())
    print("\n  DXY_d3=달러3m변화(%,+강세) US10Y_d3/KR10Y_d3=금리3m변화(pp,+상승) VIXmax=구간최고VIX")
    print("  가설: Slow가 DXY↑·금리↑ 일관 / Fast는 금리↓·VIX↑. 일관되면 DXY+금리는 Slow Bear 일반신호.")
    df.to_csv(A / "slowbear_macro_signal.csv", index=False)


if __name__ == "__main__":
    main()
