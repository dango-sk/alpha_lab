"""
analysis/news_score_compare.py
LLM 뉴스 3-score가 hsmm 오판(2018·2022·2023)을 설명하는지 검증.
news_scores_2018_2023.csv + hsmm 4/5 + ai_v2 레짐 + 실제 KOSPI 결과 병합.
사용: DATABASE_URL=<ip> .venv/bin/python analysis/news_score_compare.py
"""
import os, json, warnings
import numpy as np, pandas as pd, psycopg2
from datetime import timedelta
from pathlib import Path
warnings.filterwarnings("ignore")
A = Path(__file__).parent


def main():
    sc = pd.read_csv(A / "news_scores_2018_2023.csv")
    bc = json.load(open(A / "hsmm_fx_bearcount.json"))
    hmap = {y: ("Bear" if v >= 4 else "Bull") for y, v in bc.items()}
    ers = {r["as_of"][:7]: r.get("expected_return", 0) for r in json.load(open(A / "regime_agent_results.json"))}
    amap = {}; prev = "Bull"
    for y in sorted(ers):
        er = ers[y]; cur = ("Bear" if er <= -2 else "Bull") if prev == "Bull" else ("Bull" if er >= 1 else "Bear"); amap[y] = cur; prev = cur
    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    kk = pd.read_sql("SELECT left(period,10) p, value::float v FROM alpha_lab.macro_indicators WHERE indicator='kospi' AND freq='D' ORDER BY period", conn); conn.close()
    kk['p'] = pd.to_datetime(kk['p']); kospi = kk.set_index('p')['v'].sort_index()
    ymp = pd.PeriodIndex(kospi.index, freq='M'); last = {}
    for dt_, p in zip(kospi.index, ymp): last[p] = dt_
    mends = {pd.Period(e, 'M').strftime("%Y-%m"): e for e in last.values()}
    def asof(e): x = kospi[kospi.index <= e]; return x.iloc[-1] if len(x) else np.nan
    def nxt_ret(ym):
        e = mends.get(ym);
        if e is None: return np.nan
        fut = sorted(p for p in mends if p > ym)
        return (asof(mends[fut[0]])/asof(e)-1)*100 if fut else np.nan
    def dd6(ym):
        e = mends.get(ym)
        if e is None: return np.nan
        fut = sorted(p for p in mends if p > ym)[:6]
        if len(fut) < 6: return np.nan
        path = pd.concat([pd.Series([asof(e)], index=[e]), kospi[(kospi.index > e) & (kospi.index <= mends[fut[-1]])]])
        return (path/path.cummax()-1).min()*100

    sc['hsmm'] = sc.ym.map(hmap); sc['ai'] = sc.ym.map(amap)
    sc['nxt'] = sc.ym.map(nxt_ret); sc['dd6'] = sc.ym.map(dd6)
    v = sc[sc.tight.notna()].copy()
    print(f"유효 점수 {len(v)}개월 / {len(sc)}\n")

    def block(lo, hi, title):
        b = sc[(sc.ym >= lo) & (sc.ym <= hi)]
        print(f"=== {title} ({lo}~{hi}) ===")
        print(f"  {'월':9}{'T':>4}{'S':>4}{'R':>4}{'hsmm':>6}{'ai':>5}{'익월%':>7}{'fwd6dd':>8}  {'flag':>8}")
        for r in b.itertuples():
            t = f"{r.tight:.0f}" if pd.notna(r.tight) else "·"; s = f"{r.stress:.0f}" if pd.notna(r.stress) else "·"; rc = f"{r.recov:.0f}" if pd.notna(r.recov) else "·"
            nx = f"{r.nxt:+.1f}" if pd.notna(r.nxt) else "·"; dd = f"{r.dd6:+.0f}" if pd.notna(r.dd6) else "·"
            fl = "◀불일치" if (pd.notna(r.hsmm) and r.hsmm != r.ai) else ""
            print(f"  {r.ym:9}{t:>4}{s:>4}{rc:>4}{str(r.hsmm):>6}{str(r.ai):>5}{nx:>7}{dd:>8}  {fl:>8}")

    block("2018-04", "2019-02", "2018 긴축 Bear")
    block("2021-12", "2022-12", "2022 긴축 Bear")
    block("2022-12", "2023-08", "2023 Recovery")

    # 설명력: hsmm=Bull/ai=Bear(긴축 오판) vs hsmm=Bull/ai=Bull(정상Bull) 점수 비교
    print("\n=== 설명력 테스트 ===")
    vv = v[v.hsmm.notna() & v.ai.notna()]
    grp = vv.groupby([vv.hsmm, vv.ai])[['tight','stress','recov']].mean().round(1)
    print("  레짐쌍별 평균 점수(T/S/R):")
    print(grp.to_string())
    hb = vv[(vv.hsmm=="Bull") & (vv.ai=="Bear")]; bb = vv[(vv.hsmm=="Bull") & (vv.ai=="Bull")]
    print(f"\n  ★ hsmm=Bull&ai=Bear(hsmm 긴축오판) {len(hb)}개월: T={hb.tight.mean():.0f} S={hb.stress.mean():.0f} R={hb.recov.mean():.0f}")
    print(f"    vs hsmm=Bull&ai=Bull(정상 Bull)   {len(bb)}개월: T={bb.tight.mean():.0f} S={bb.stress.mean():.0f} R={bb.recov.mean():.0f}")
    be = vv[(vv.hsmm=="Bear") & (vv.ai=="Bull")]
    print(f"  ★ hsmm=Bear&ai=Bull(hsmm 과잉방어) {len(be)}개월: Recovery={be.recov.mean():.0f} (높으면 회복신호를 hsmm이 놓침)")
    print("\n  해석: 긴축오판 달의 T/S가 정상Bull보다 뚜렷이 높으면 → 뉴스score가 hsmm 오판 설명. 과잉방어 달 R 높으면 회복 놓침 설명.")


if __name__ == "__main__":
    main()
