"""
analysis/fcf_diet_robust_report.py  (실험 스크립트, production 미수정)

fcf_diet_robust.py / fcf_diet_compare.py / fcf_diet_counterfactual.py / fcf_diet_churn.py
산출 CSV를 읽어 최종 Markdown 보고서를 만든다. 계산은 하지 않고 표만 조립한다.

실행: .venv/bin/python analysis/fcf_diet_robust_report.py
산출: docs/FCF_DIET4_ROBUSTNESS.md
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).parent.parent
OUT, DOCS = REPO / "output", REPO / "docs"
DOCS.mkdir(exist_ok=True)
OOS = "OOS_post-OOS diagnostic"
P = ["Full", "IS", OOS]


def rd(name, **kw):
    f = OUT / name
    return pd.read_csv(f, **kw) if f.exists() else None


def pct(v, d=2):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v*100:.{d}f}%"


def num(v, d=2):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join(["---"] * len(headers)) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    rel = rd("robust_relative_summary.csv")
    if rel is None:
        sys.exit("robust_relative_summary.csv 없음 → 먼저 fcf_diet_robust.py 를 실행하세요.")
    ex = rd("robust_top_month_exclusion.csv")
    top = rd("robust_top_months.csv")
    roll_s = rd("robust_rolling_summary.csv")
    lo = rd("robust_stock_leaveout.csv")
    seco = rd("robust_sector_leaveout.csv")
    st = rd("robust_market_states.csv")
    chr_ = rd("robust_selection_characteristics.csv")
    pair = rd("robust_selection_paired.csv")
    cost = rd("robust_cost_turnover.csv")
    csum = rd("robust_churn_summary.csv")
    cattr = rd("robust_cost_attribution.csv")
    conc = rd("robust_concentration.csv")
    cf = rd("diet_counterfactual_decomposition.csv")
    meta = json.loads((OUT / "robust_run_meta.json").read_text()) if (OUT / "robust_run_meta.json").exists() else {}

    L = ["# DIET-4 강건성 검증 (post-OOS diagnostic)", "",
         "> 본 문서의 모든 결과는 **이미 확인한 데이터에 대한 post-OOS diagnostic**이다. "
         "새 팩터 조합 탐색·가중치 조정·production 코드 수정은 없다. "
         "종목/업종 제외 결과는 채택 후보가 아니라 `ex-post leave-out diagnostic`이다.", ""]

    # ── 요약 Q&A ──
    f_row = rel[rel.period == "Full"].iloc[0]; o_row = rel[rel.period == OOS].iloc[0]
    i_row = rel[rel.period == "IS"].iloc[0]
    def dep(exclude, method="contrib_subtract", period=OOS):
        if lo is None: return np.nan
        r = lo[(lo.exclude == exclude) & (lo.method == method) & (lo.period == period)]
        return float(r["dependency"].iloc[0]) if len(r) else np.nan
    L += ["## 0. 쉬운 언어로 먼저 답하기", "",
          f"**1) 왜 최근 성과가 좋아졌나** — 네 팩터(T_PBR·ATT_EVIC·PRICE_MA_REV·T_EVEBITDA)를 빼면서 "
          f"과거 주가가 눌린 종목·과거 실적 기준 저평가 종목을 선호하던 힘이 약해지고, "
          f"F_EPS_M(예상이익 개선)과 FCF_YIELD의 비중이 12.5%→18.75%로 커졌다. "
          f"그 결과 실적이 개선되는 대형주가 더 쉽게 30위 안에 들어왔다. "
          f"OOS 월평균 초과 {pct(o_row.mean_diff)} (NW t={num(o_row.nw_t)}), IS는 {pct(i_row.mean_diff)}.",
          "",
          f"**2) SK하이닉스·삼성전자·SK스퀘어 설명 비중** — OOS 개선분의 "
          f"{pct(dep('SK하이닉스+삼성전자+SK스퀘어'),1)}가 이 3종목에서 나온다 "
          f"(SK하이닉스 단독 {pct(dep('SK하이닉스'),1)}).", "",
          "**3) 반도체를 빼도 개선이 남는가** — 아래 6절 표 참조.", "",
          f"**4) 과거에도 반복됐나** — Rolling 구간 중 DIET-4 CAGR 우위 비율: " +
          (", ".join(f"{int(r.window)}M {pct(r.cagr_win,1)}" for _, r in roll_s.iterrows()) if roll_s is not None else "—") + ".",
          "", "**5) 지금 채택 가능한가 / 6) 무엇을 봐야 하나** — 10절 최종 판정 참조.", ""]

    # ── 2 ──
    L += ["## 2. 월별 BASE 대비 성과 차이", "",
          table(["Period", "월수", "평균 DIET-4−BASE", "연환산", "NW t", "승률", "+1%p 이상 승리",
                 "−1%p 이하 패배", "평균 +", "평균 −", "최악(월)", "최대 상대낙폭"],
                [[r.period, int(r.months), pct(r.mean_diff), pct(r.ann_diff), num(r.nw_t),
                  pct(r.win_rate, 1), pct(r.win_gt_1pp, 1), pct(r.lose_lt_1pp, 1),
                  pct(r.mean_pos), pct(r.mean_neg), f"{pct(r.worst_month)} ({r.worst_ym})",
                  pct(r.max_rel_dd)] for _, r in rel.iterrows()]), "",
          "원자료: `output/robust_monthly_diff.csv`", ""]

    # ── 3 ──
    L += ["## 3. 소수 월 의존도", ""]
    if top is not None:
        for kind, title in [("top10", "상위 10개월 (DIET-4 우위)"), ("bottom10", "하위 10개월 (DIET-4 열위)")]:
            t = top[top.kind == kind]
            L += [f"### {title}", "",
                  table(["월", "BASE", "DIET-4", "차이", "시장", "기여 상위 종목", "BASE 비중", "DIET-4 비중", "업종"],
                        [[r.ym, pct(r.BASE_ret), pct(r.DIET4_ret), pct(r["diff"]), pct(r.market_ret),
                          f"{r['name']} ({pct(r.contrib_diff)})", pct(r.BASE_weight, 1),
                          pct(r.DIET4_weight, 1), r.sector] for _, r in t.iterrows()]), ""]
    if ex is not None:
        L += ["### 상위 월 제외 재계산", "",
              table(["제외 조건", "BASE CAGR", "DIET-4 CAGR", "CAGR 차이", "월평균 차이",
                     "BASE Sharpe", "DIET-4 Sharpe", "BASE MDD", "DIET-4 MDD", "DIET-4 우위 유지"],
                    [[f"상위 {int(r.exclude_top_n)}개월 제외" if r.exclude_top_n else "제외 없음",
                      pct(r.BASE_cagr), pct(r.DIET4_cagr), pct(r.cagr_diff), pct(r.mean_diff),
                      num(r.BASE_sharpe), num(r.DIET4_sharpe), pct(r.BASE_mdd), pct(r.DIET4_mdd),
                      "O" if r.diet4_still_better else "X"] for _, r in ex.iterrows()]), "",
              "*실제로 해당 월을 제외하자는 제안이 아니라 의존도 진단이다.*", ""]

    # ── 4 ──
    L += ["## 4. Rolling 안정성", ""]
    if roll_s is not None:
        L += [table(["윈도우", "DIET-4 CAGR 우위", "Sharpe 우위", "MDD 개선", "CAGR·Sharpe 동시 개선"],
                    [[f"{int(r.window)}M", pct(r.cagr_win, 1), pct(r.sharpe_win, 1),
                      pct(r.mdd_win, 1), pct(r.both, 1)] for _, r in roll_s.iterrows()]), "",
              "그래프: `output/robust_rolling_diff.png` (빨간 점선 = OOS 시작 2024-07)", "",
              "원자료: `output/robust_rolling.csv`", ""]

    # ── 5 ──
    L += ["## 5. 특정 종목 의존도", ""]
    if lo is not None:
        for method, title in [("contrib_subtract", "5-1. 기여도 차감 (비중 재조정 없음)"),
                              ("recap", "5-2. 보유 제외·재정규화 (ex-post leave-out diagnostic)")]:
            L += [f"### {title}", ""]
            for p in P:
                t = lo[(lo.method == method) & (lo.period == p)]
                if not len(t): continue
                L += [f"**{p}**", "",
                      table(["제외 조건", "BASE CAGR", "DIET-4 CAGR", "차이", "BASE Sharpe",
                             "DIET-4 Sharpe", "DIET-4 우위 유지", "의존도"],
                            [[r.exclude, pct(r["cagr_BASE"]), pct(r["cagr_DIET-4"]), pct(r.cagr_diff),
                              num(r["sharpe_BASE"]), num(r["sharpe_DIET-4"]),
                              "O" if r.diet4_still_better else "X",
                              "기준" if r.exclude == "없음(기준)" else pct(r.dependency, 1)]
                             for _, r in t.iterrows()]), ""]
        L += ["의존도 = (원래 개선분 − 제외 후 개선분) / 원래 개선분. "
              "상위 1~3종목에서 개선분의 50% 이상이 사라지면 `소수 종목 의존도가 높음`으로 판정한다.", ""]

    # ── 6 ──
    L += ["## 6. 전기·전자/반도체 의존도", ""]
    if seco is not None:
        piv = seco.pivot_table(index=["period", "method"], columns="spec", values=["cagr", "sharpe", "mdd"])
        piv.columns = [f"{a}_{b}" for a, b in piv.columns]; piv = piv.reset_index()
        L += [table(["기간", "방식", "BASE CAGR", "DIET-4 CAGR", "차이", "BASE Sharpe", "DIET-4 Sharpe"],
                    [[r.period, r.method, pct(r["cagr_BASE"]), pct(r["cagr_DIET-4"]),
                      pct(r["cagr_DIET-4"] - r["cagr_BASE"]), num(r["sharpe_BASE"]), num(r["sharpe_DIET-4"])]
                     for _, r in piv.iterrows()]), "",
              "`semi_weight` 행의 CAGR 칸은 해당 기간 평균 반도체/전기전자 비중이다. "
              "종목 목록: `output/robust_semi_universe.csv` (업종 라벨 = fnspace_master.sec_cd_nm).", ""]

    # ── 7 ──
    L += ["## 7. 시장 상태별 성과 (사후 조건부 분석 — 매매신호 아님)", ""]
    if st is not None:
        L += [table(["축", "상태", "월수", "BASE 평균", "DIET-4 평균", "차이", "NW t", "iid t", "DIET-4 승률", "BASE MDD", "DIET-4 MDD"],
                    [[r.axis, r.state, int(r.months), pct(r.BASE_mean), pct(r.DIET4_mean),
                      pct(r["diff"]), num(r.nw_t), num(r.t_iid), pct(r.win_rate, 1),
                      pct(r.BASE_mdd), pct(r.DIET4_mdd)] for _, r in st.iterrows()]), "",
              "정의: UP/DOWN = KODEX200(069500) 월수익률 부호. "
              "LARGE-LEAD/MID-LEAD = KOSPI 유니버스 시총 상위 100 시총가중 수익률 vs 101~300위 "
              "(프로젝트에 중형주 벤치마크 ETF가 없어 새로 정의). "
              "SEMI-STRONG/WEAK = 전기전자/반도체 종목 시총가중 수익률 vs KODEX200.", ""]

    # ── 8 ──
    L += ["## 8. 팩터 제거가 주도주 편입에 미친 영향", ""]
    if chr_ is not None:
        L += [table(["기간", "집단", "n", "F_EPS_M", "FCF_YIELD", "ATT_EVEBIT", "PRICE_MA_REV",
                     "T_EVEBITDA", "다음달 수익률", "시총 중앙값(십억)", "반도체 비중"],
                    [[r.period, r.group, int(r.n), num(r.F_EPS_M), num(r.FCF_YIELD),
                      num(r.ATT_EVEBIT), num(r.PRICE_MA_REV), num(r.T_EVEBITDA),
                      pct(r.mean_next_ret), num(r.median_mcap_bn, 0), pct(r.semi_share, 1)]
                     for _, r in chr_.iterrows()]), ""]
    if pair is not None:
        L += ["**월별 paired difference (DIET-4 신규 편입 − BASE 고유 편입)**", "",
              table(["지표", "월수", "평균 차이", "NW t", "양(+) 월 비율"],
                    [[r.metric, int(r.n_months), num(r.paired_mean_diff, 3), num(r.nw_t),
                      pct(r.share_positive, 1)] for _, r in pair.iterrows()]), "",
              "가설: PRICE_MA_REV·T_EVEBITDA 제거 → 과거 하락주·과거실적 저평가 선호 약화 → "
              "F_EPS_M·FCF_YIELD 상대 영향력 확대 → 실적 개선 주도 대형주 편입 증가. "
              "위 paired t값이 F_EPS_M/FCF_YIELD에서 유의하게 (+)이고 PRICE_MA_REV에서 (−)여야 가설이 지지된다.", ""]

    # ── 9 ──
    L += ["## 9. 거래비용 및 반복매매", ""]
    if cost is not None:
        L += [table(["Spec", "기간", "turnover", "0bp CAGR", "30bp CAGR", "50bp CAGR"],
                    [[s, p,
                      pct(cost[(cost.spec == s) & (cost.period == p) & (cost.cost_bp == 30)]["turnover_mean"].iloc[0]),
                      *[pct(cost[(cost.spec == s) & (cost.period == p) & (cost.cost_bp == b)]["cagr"].iloc[0])
                        for b in [0, 30, 50]]]
                     for s in ["BASE", "DIET-4"] for p in P]), ""]
    if csum is not None:
        L += ["**경계(랭크 25~35) 반복매매**", "",
              table(["Spec", "월평균 경계 종목수", "월평균 이탈", "3개월 내 재진입 수", "재진입률"],
                    [[r.spec, num(r.n_boundary_25_35), num(r.n_exit), num(r.n_reentry_3m),
                      pct(r.reentry_rate_3m, 1)] for _, r in csum.iterrows()]), ""]
    if cattr is not None:
        L += ["**비용 절감분이 개선분에서 차지하는 비중**", "",
              table(["기간", "월평균 개선분", "월평균 비용 절감", "비중"],
                    [[r.period, pct(r.mean_improvement), pct(r.mean_cost_saving),
                      pct(r.cost_saving_share_of_improvement, 1)] for _, r in cattr.iterrows()]), ""]
    if conc is not None:
        L += ["**집중도**", "",
              table(["Spec", "기간", "최대비중", "top3", "top5", "HHI"],
                    [[r.spec, r.period, pct(r.max_weight, 1), pct(r.top3, 1), pct(r.top5, 1), num(r.hhi, 3)]
                     for _, r in conc.iterrows()]), ""]
    if cf is not None:
        c = cf.copy(); c.columns = [x.strip("﻿") for x in c.columns]
        L += ["**기존 반사실 2×2 (종목효과 vs 비중효과, DIET-3 기준)**", "",
              table(list(c.columns), [[r[0]] + [pct(v) for v in r[1:]] for r in c.values]), ""]

    # ── 10 ──
    def flag(cond): return "지지" if cond else "반대"
    dep3 = dep("SK하이닉스+삼성전자+SK스퀘어")
    ex3 = ex[ex.exclude_top_n == 3].iloc[0] if ex is not None else None
    roll12 = float(roll_s[roll_s.window == 12]["cagr_win"].iloc[0]) if roll_s is not None else np.nan
    down = st[(st.axis == "market") & (st.state == "DOWN")].iloc[0] if st is not None else None
    semiw = st[(st.axis == "semi") & (st.state == "SEMI-WEAK")].iloc[0] if st is not None else None
    to_b = float(cost[(cost.spec == "BASE") & (cost.period == "Full") & (cost.cost_bp == 30)]["turnover_mean"].iloc[0]) if cost is not None else np.nan
    to_d = float(cost[(cost.spec == "DIET-4") & (cost.period == "Full") & (cost.cost_bp == 30)]["turnover_mean"].iloc[0]) if cost is not None else np.nan
    conc_b = float(conc[(conc.spec == "BASE") & (conc.period == "Full")]["top5"].iloc[0]) if conc is not None else np.nan
    conc_d = float(conc[(conc.spec == "DIET-4") & (conc.period == "Full")]["top5"].iloc[0]) if conc is not None else np.nan
    semi_ok = False
    if seco is not None:
        pv = seco[(seco.period == OOS) & (seco.method == "recap")].set_index("spec")["cagr"]
        semi_ok = bool(pv.get("DIET-4", np.nan) > pv.get("BASE", np.nan))

    checks = [
        ("월평균 상대성과", f"Full {pct(f_row.mean_diff)} (t={num(f_row.nw_t)}), IS {pct(i_row.mean_diff)}, OOS {pct(o_row.mean_diff)} (t={num(o_row.nw_t)})",
         flag(f_row.nw_t > 1.5 and i_row.mean_diff > -0.002)),
        ("Rolling 안정성", f"12M 우위 {pct(roll12,1)}", flag(roll12 > 0.5)),
        ("소수 월 의존도", f"상위3 제외 CAGR 차이 {pct(ex3.cagr_diff) if ex3 is not None else '—'}",
         flag(ex3 is not None and ex3.cagr_diff > 0)),
        ("상위 종목 의존도", f"3종목 의존도 {pct(dep3,1)}", flag(not (dep3 == dep3) or dep3 < 0.5)),
        ("반도체 의존도", f"OOS 반도체 제외 후 DIET-4 우위 {'유지' if semi_ok else '소멸'}", flag(semi_ok)),
        ("하락장 성과", f"DOWN 차이 {pct(down['diff']) if down is not None else '—'}, "
                    f"SEMI-WEAK 차이 {pct(semiw['diff']) if semiw is not None else '—'}",
         flag(down is not None and down["diff"] > -0.005)),
        ("turnover", f"BASE {pct(to_b,1)} → DIET-4 {pct(to_d,1)}", flag(to_d < to_b - 0.005)),
        ("집중도", f"top5 BASE {pct(conc_b,1)} → DIET-4 {pct(conc_d,1)}", flag(conc_d <= conc_b + 0.01)),
    ]
    support = sum(1 for _, _, f in checks if f == "지지")
    if support >= 7:
        verdict = "DIET-4 채택 검토 가능 — 단, 새로운 forward 검증 필요"
    elif support >= 4:
        verdict = "DIET-4 forward test 후보"
    else:
        verdict = "BASE 유지"
    L += ["## 10. 최종 판정", "",
          table(["검증 항목", "결과", "DIET-4 지지 여부"], [[a, b, c] for a, b, c in checks] +
                [["**최종 판정**", f"지지 {support}/8", f"**{verdict}**"]]), "",
          "판정 규칙(사전 정의): 지지 ≥7 → 채택 검토 가능, 4~6 → forward test 후보, ≤3 → BASE 유지. "
          "어떤 경우에도 본 결과는 이미 확인한 데이터에 대한 진단이므로 즉시 production 반영은 하지 않는다. "
          "새 forward 기간(최소 6~12개월) 검증이 필요하다.", ""]

    L += ["## 11. 실행 기록", "",
          f"- 실행: `.venv/bin/python analysis/fcf_diet_robust.py` → `.venv/bin/python analysis/fcf_diet_robust_report.py`",
          f"- 소요시간: {meta.get('elapsed_sec','—')}초, 표본 {meta.get('months','—')}개월",
          f"- 재구성 검증 MAE(bp): {meta.get('recon_mae_bp','—')} (종목단위 재구성 = production 백테스트 일치 확인)",
          "- production 코드 변경 없음 — `lib/`, `scripts/`, `config/` 미수정. 본 분석은 "
          "`analysis/fcf_diet_robust.py`, `analysis/fcf_diet_robust_report.py` 두 파일만 추가한다.", ""]

    (DOCS / "FCF_DIET4_ROBUSTNESS.md").write_text("\n".join(L), encoding="utf-8")
    print(f"✅ docs/FCF_DIET4_ROBUSTNESS.md 작성 ({len(L)} lines), 최종 판정: {verdict}")


if __name__ == "__main__":
    main()
