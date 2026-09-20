"""
analysis/fcf_quartile_ic.py   (실험 스크립트, production 미수정)

[액션 5] FCF Yield 단독 선별력 확인. 두 가지 분류 기준을 **각각** 산출한다.

  RAW  : 분석용 raw 사분위 — universe∩퀄리티통과 표본에서 qcut(동일 개수 4등분),
         NaN 제외, 동점은 rank(first)로 임의 분할.
  PROD : production 채점 그대로 — factor_engine.quartile_rule2가 매긴 `fcf_yield_score`
         (0~4)를 재계산 없이 사용. 경계는 '퀄리티통과 대형주 전체'(universe 교집합 前)
         에서 뽑은 고정 컷오프 Q1/Q2/Q3 + `>=` 비교, NaN→0점(5번째 등급),
         동점은 전부 상위 등급, non-NaN<20이면 전원 0점.

  → 두 기준은 정렬 방향만 같고 표본·결측·경계·동점 처리가 다르므로 결과가 갈릴 수 있다.
    RAW 분위 × PROD 점수 교차표와 일치율도 함께 출력한다.

산출:
  - (RAW) Q1~Q4 / (PROD) 0~4점 그룹의 다음 달 수익률(동일가중)
  - 최상-최하 스프레드 + Newey-West t + IR + 승률
  - 월별 Rank IC (Spearman): RAW는 raw fcf_yield, PROD는 fcf_yield_score 기준
  - 구간별: 전체기간 / 2020년 이후 / Bull 기간 / Bear 기간
    (Bull = 리밸일에 KOSPI200 ETF(069500) 종가 > 200일 이동평균 — 단순 구간분할용 라벨)

실행:
    .venv/bin/python analysis/fcf_quartile_ic.py
산출물:
    analysis/results/fcf_quartile_monthly.csv   월별 RAW/PROD 그룹수익·스프레드·IC·레짐
    analysis/results/fcf_quartile_panel.csv     종목×월 패널 (fcf_yield, raw분위, prod점수, 수익)
    analysis/results/fcf_quartile_crosstab.csv  RAW 분위 × PROD 점수 교차표(전체 누적)
"""
import os, sys
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(REPO / ".env")

from lib.data import load_strategy
from lib.factor_engine import code_to_module, score_stocks_from_strategy, prefetch_all_data
from step7_backtest import get_db, get_rebalance_dates, get_universe_stocks
from config.settings import BACKTEST_CONFIG
from analysis.fcf_base_compare import nw_tstat, stock_returns   # 동일 규칙 재사용

BULL = "FCF_YIELD추가전략"
CUTOFF = "2026-07"
NQ = 4
OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)


def ma200_regime(conn, date, window=200):
    rows = conn.execute(
        "SELECT close FROM daily_price WHERE stock_code = '069500' "
        f"AND trade_date <= ? ORDER BY trade_date DESC LIMIT {window + 1}", (date,)
    ).fetchall()
    p = [r[0] for r in rows if r[0]]
    if len(p) < window + 1:
        return "NA"
    return "Bull" if p[0] >= float(np.mean(p[1:window + 1])) else "Bear"


def main():
    sd = load_strategy(BULL, rebal_type="monthly", universe="KOSPI")
    module = code_to_module(sd["code"])

    conn = get_db()
    prefetch_all_data(conn)
    BACKTEST_CONFIG["universe"] = "KOSPI"
    dates = get_rebalance_dates(conn, "monthly")
    nxt = {dates[i]: dates[i + 1] for i in range(len(dates) - 1)}

    panel_rows, mon_rows = [], []
    for d in dates[:-1]:
        if d[:7] > CUTOFF:
            continue
        e = nxt[d]
        uni = get_universe_stocks(conn, d, "monthly")
        if not uni:
            continue
        # 퀄리티 통과 + 대형주 df (score_stocks_from_strategy 내부 파이프라인 그대로)
        # → df["fcf_yield_score"]는 이미 production quartile_rule2가 매긴 0~4점.
        #   경계는 이 df(=universe 교집합 前 전체)에서 뽑혔으므로 production과 정의상 동일.
        res = score_stocks_from_strategy(conn, d, module, return_df=True)
        if not isinstance(res, tuple):
            continue
        _, df = res
        if df is None or df.empty or "fcf_yield" not in df.columns:
            continue
        df = df.copy()
        if "fcf_yield_score" not in df.columns:
            print(f"  {d[:7]}  fcf_yield_score 없음 — 스킵", flush=True)
            continue
        n_pre_uni = int(df["fcf_yield"].notna().sum())          # PROD 경계 산출 표본 크기
        df["code"] = df["stock_code"].str.lstrip("A")
        df = df[df["code"].isin(uni)].copy()
        if df.empty:
            continue

        rets = stock_returns(conn, set(df["code"]), d, e)
        df["ret"] = df["code"].map(rets)
        df = df[df["ret"].notna()].copy()
        if len(df) < 4 * NQ:
            continue

        n_nan = int(df["fcf_yield"].isna().sum())
        n_neg = int((df["fcf_yield"] < 0).sum())

        # ── PROD: production 점수 그대로 (0~4). NaN→0점 포함 ──
        df["prod"] = df["fcf_yield_score"].fillna(0).astype(int)
        gp = df.groupby("prod")["ret"].mean()
        ic_p = df["prod"].corr(df["ret"], method="spearman")

        # ── RAW: NaN 제외 + 동일 개수 4등분 ──
        dr = df[df["fcf_yield"].notna()].copy()
        if len(dr) < 4 * NQ:
            continue
        dr["q"] = pd.qcut(dr["fcf_yield"].rank(method="first"), NQ, labels=[1, 2, 3, 4]).astype(int)
        df["q"] = dr["q"]                                        # NaN 행은 q=NaN
        g = dr.groupby("q")["ret"].mean()
        ic_r = dr["fcf_yield"].corr(dr["ret"], method="spearman")

        # RAW 분위 ↔ PROD 점수 일치 (동일 라벨 q==prod 기준, NaN 행 제외)
        agree = float((dr["q"] == df.loc[dr.index, "prod"]).mean())
        agree_top = float((dr.loc[dr.q == 4, "prod"] == 4).mean()) if (dr.q == 4).any() else np.nan
        agree_bot = float((dr.loc[dr.q == 1, "prod"] == 1).mean()) if (dr.q == 1).any() else np.nan

        mon_rows.append(dict(
            date=d, ym=d[:7], n=len(df), n_raw=len(dr), n_pre_uni=n_pre_uni,
            n_nan=n_nan, n_neg=n_neg, regime=ma200_regime(conn, d),
            # RAW
            q1=g.get(1, np.nan), q2=g.get(2, np.nan), q3=g.get(3, np.nan), q4=g.get(4, np.nan),
            spread=g.get(4, np.nan) - g.get(1, np.nan), ic=ic_r,
            # PROD (p0 = NaN 등급)
            p0=gp.get(0, np.nan), p1=gp.get(1, np.nan), p2=gp.get(2, np.nan),
            p3=gp.get(3, np.nan), p4=gp.get(4, np.nan),
            spread_p=gp.get(4, np.nan) - gp.get(1, np.nan), ic_p=ic_p,
            n_p0=int((df["prod"] == 0).sum()), n_p1=int((df["prod"] == 1).sum()),
            n_p2=int((df["prod"] == 2).sum()), n_p3=int((df["prod"] == 3).sum()),
            n_p4=int((df["prod"] == 4).sum()),
            agree=agree, agree_top=agree_top, agree_bot=agree_bot,
            fcf_q1=dr.loc[dr.q == 1, "fcf_yield"].median(),
            fcf_q4=dr.loc[dr.q == 4, "fcf_yield"].median(),
        ))
        panel_rows.append(df[["code", "fcf_yield", "q", "prod", "ret"]].assign(ym=d[:7]))
        print(f"  {d[:7]}  n={len(df):3d}(경계표본 {n_pre_uni}, NaN {n_nan}, 음수 {n_neg})  "
              f"RAW Q4-Q1 {(g.get(4,np.nan)-g.get(1,np.nan))*100:+6.2f}% IC {ic_r:+.3f} | "
              f"PROD 4-1 {(gp.get(4,np.nan)-gp.get(1,np.nan))*100:+6.2f}% IC {ic_p:+.3f} | "
              f"일치 {agree*100:4.0f}%", flush=True)
    conn.close()

    m = pd.DataFrame(mon_rows)
    if m.empty:
        raise SystemExit("표본 없음")
    m.to_csv(OUT / "fcf_quartile_monthly.csv", index=False, encoding="utf-8-sig")
    panel = pd.concat(panel_rows)
    panel.to_csv(OUT / "fcf_quartile_panel.csv", index=False, encoding="utf-8-sig")

    def report(sub, tag, mode):
        """mode='raw' → Q1~Q4 / mode='prod' → 0~4점 (0=NaN 등급)"""
        cols = ["q1", "q2", "q3", "q4"] if mode == "raw" else ["p0", "p1", "p2", "p3", "p4"]
        labs = ["Q1", "Q2", "Q3", "Q4"] if mode == "raw" else ["0점(NaN)", "1점", "2점", "3점", "4점"]
        spc = "spread" if mode == "raw" else "spread_p"
        icc = "ic" if mode == "raw" else "ic_p"
        if len(sub) < 6:
            print(f"\n[{tag}] 표본 부족 (n={len(sub)})")
            return
        print(f"\n[{tag}]  n={len(sub)}개월  ({sub.ym.iloc[0]}~{sub.ym.iloc[-1]})")
        for c, lab in zip(cols, labs):
            x = sub[c].dropna().values
            if len(x) == 0:
                print(f"   {lab:9s} 표본 없음")
                continue
            nm = (f"  평균종목수 {sub['n_' + c].mean():5.1f}" if mode == "prod" else "")
            print(f"   {lab:9s} 월평균 {x.mean()*100:+6.2f}%  연환산(기하) "
                  f"{(np.prod(1+x)**(12/len(x))-1)*100:+7.2f}%  (n월 {len(x)}){nm}")
        sp = sub[spc].dropna().values
        t, lag = nw_tstat(sp)
        ir = sp.mean() / sp.std(ddof=1) * np.sqrt(12) if sp.std(ddof=1) > 0 else np.nan
        head = "Q4-Q1" if mode == "raw" else "4점-1점"
        print(f"   {head}  월평균 {sp.mean()*100:+.3f}%  연환산 {sp.mean()*12*100:+.2f}%  "
              f"NW t={t:+.2f}(lag{lag})  IR {ir:+.2f}  승률 {float((sp>0).mean())*100:.0f}%")
        ics = sub[icc].dropna().values
        t_ic, lag_ic = nw_tstat(ics)
        print(f"   RankIC 평균 {ics.mean():+.4f}  표준편차 {ics.std(ddof=1):.4f}  "
              f"IC-IR {ics.mean()/ics.std(ddof=1):+.3f}  NW t={t_ic:+.2f}(lag{lag_ic})  "
              f"양(+) {float((ics>0).mean())*100:.0f}%")

    for mode, title in (("raw", "RAW 사분위 (qcut·NaN제외·동일개수)"),
                        ("prod", "PROD 채점 (factor_engine quartile_rule2, NaN=0점)")):
        print("\n" + "═" * 78)
        print(f"  [액션 5-{'A' if mode=='raw' else 'B'}] FCF Yield 단독 선별력 — {title}")
        print("═" * 78)
        report(m, "전체기간", mode)
        report(m[m.ym >= "2020-01"], "2020년 이후", mode)
        report(m[m.regime == "Bull"], "Bull (MA200 상회)", mode)
        report(m[m.regime == "Bear"], "Bear (MA200 하회)", mode)

    # ── 두 기준 일치도 ──
    print("\n" + "═" * 78)
    print("  RAW 분위 vs PROD 점수 일치도")
    print("═" * 78)
    print(f"  월평균 라벨 일치율 {m.agree.mean()*100:.1f}%  "
          f"(최저월 {m.agree.min()*100:.0f}% / 최고월 {m.agree.max()*100:.0f}%)")
    print(f"  최상위 일치(RAW Q4 중 PROD 4점) {m.agree_top.mean()*100:.1f}%   "
          f"최하위 일치(RAW Q1 중 PROD 1점) {m.agree_bot.mean()*100:.1f}%")
    print(f"  PROD 등급별 평균 종목수: 0점 {m.n_p0.mean():.1f} / 1점 {m.n_p1.mean():.1f} / "
          f"2점 {m.n_p2.mean():.1f} / 3점 {m.n_p3.mean():.1f} / 4점 {m.n_p4.mean():.1f}  "
          f"(RAW는 정의상 각 {m.n_raw.mean()/4:.1f} 균등)")
    print(f"  월평균: universe 내 종목 {m.n.mean():.1f} / PROD 경계 산출 표본 {m.n_pre_uni.mean():.1f} "
          f"(경계는 더 넓은 표본에서 산출됨) / NaN {m.n_nan.mean():.1f} / 음수 fcf_yield {m.n_neg.mean():.1f}")

    pan = panel.dropna(subset=["q"])
    ct = pd.crosstab(pan["q"].astype(int), pan["prod"].astype(int))
    ct.index.name = "RAW분위"; ct.columns.name = "PROD점수"
    print("\n  교차표 (전 기간 누적 종목×월 관측치)")
    print(ct.to_string())
    print("\n  행 비율(%) — RAW 각 분위가 PROD 어느 점수로 갔는지")
    print((ct.div(ct.sum(axis=1), axis=0) * 100).round(1).to_string())
    diag = sum(ct.loc[i, i] for i in ct.index if i in ct.columns)
    print(f"\n  전체 라벨 일치율 {diag/ct.values.sum()*100:.1f}%  "
          f"(관측치 {ct.values.sum():,}개)  "
          f"Spearman(RAW분위, PROD점수) = {pan['q'].corr(pan['prod'], method='spearman'):+.3f}")
    ct.to_csv(OUT / "fcf_quartile_crosstab.csv", encoding="utf-8-sig")

    print(f"\n저장: {OUT}/fcf_quartile_monthly.csv, fcf_quartile_panel.csv, fcf_quartile_crosstab.csv")


if __name__ == "__main__":
    main()
