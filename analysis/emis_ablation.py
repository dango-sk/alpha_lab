# -*- coding: utf-8 -*-
"""
analysis/emis_ablation.py  (실험, production 무수정)

5-arm emission ablation + 월별 성과 attribution.
목적: EMIS-aug의 MDD개선/CAGR훼손을 피처별·월별로 분해.
  - MDD 개선이 '2022 하락 본체 방어'인가 '2021 조기 현금화'인가?
  - CAGR 훼손이 entry-side(조기진입)인가 exit-side(늦은복귀)인가?
  - d_breadth/d_newlow(Δ3m)의 기계적 회복지연 vs diverg(동시점)의 lag-free 여부.
결과를 미리 가정하지 않음. 월별 데이터로만 판정.

5 arm (전부 emission, transition=base stress 고정):
  BASE      breadth·newlow·trend
  D_BREADTH BASE + d_breadth
  D_NEWLOW  BASE + d_newlow
  DIVERG    BASE + diverg
  FULL_EMIS BASE + d_breadth + d_newlow + diverg

사용: DATABASE_URL 세팅 후  .venv/bin/python analysis/emis_ablation.py
"""
import os, sys
from pathlib import Path
REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
import hsmm_final as H

BASE_EMIS = ["breadth", "newlow", "trend"]
ARMS = [("BASE", BASE_EMIS),
        ("D_BREADTH", BASE_EMIS + ["d_breadth"]),
        ("D_NEWLOW", BASE_EMIS + ["d_newlow"]),
        ("DIVERG", BASE_EMIS + ["diverg"]),
        ("FULL_EMIS", BASE_EMIS + ["d_breadth", "d_newlow", "diverg"])]


# ───────────────── 피처/유틸 ─────────────────
def augment(df):
    d = df.copy()
    d["d_breadth"] = df["breadth"].diff(3).fillna(0.0)
    d["d_newlow"] = df["newlow"].diff(3).fillna(0.0)
    w = 36
    def z(s): return ((s - s.rolling(w, min_periods=12).mean()) / (s.rolling(w, min_periods=12).std() + 1e-9)).fillna(0.0)
    d["diverg"] = (z(df["trend"]) - z(df["breadth"])).clip(-4, 4)
    return d


def base_stress(df):
    """모든 arm 공통 transition stress = Z(fx3m) - Z(fflow) (H.walk_forward와 동일)."""
    Z = H.roll_z(df, ["fx3m", "fflow"])
    return Z["fx3m"].values - Z["fflow"].values


def shift_ym(y, k):
    yy, mm = int(y[:4]), int(y[5:7]); mm += k
    yy += (mm - 1) // 12; mm = (mm - 1) % 12 + 1
    return f"{yy:04d}-{mm:02d}"


# ───────────────── HSMM walk-forward (arm=emission만 다름) ─────────────────
def walk_forward_exp(df, yms, n, emis_cols, stress):
    start = yms.index(H.DECIDE_START) if H.DECIDE_START in yms else 12
    EM = df[emis_cols].values
    pbear = np.full(n, np.nan); params, sc, last_refit = None, None, -10 ** 9
    for t in range(start, n):
        lo = max(0, t + 1 - H.WINDOW_M); Xr = EM[lo:t + 1]; nw = len(Xr)
        w = 0.5 ** ((nw - 1 - np.arange(nw)) / H.HL)
        if params is None or (t - last_refit) >= H.REFIT_EVERY:
            sc = StandardScaler().fit(Xr)
            params = H.fit_emission(sc.transform(Xr), w,
                                    H.cold_emission(sc.transform(Xr)) if params is None else params,
                                    40 if params is None else 10)
            last_refit = t
        Xz = sc.transform(Xr)
        means, covs = params['means'], params['covs']
        logB = H.emis_logB(Xz, means, covs)
        bear = int(np.argmax(H.bear_score(means)))
        sw = stress[lo:t + 1]; sw = (sw - sw.mean()) / (sw.std() + H.EPS)
        if H.USE_DURATION:
            gamma, _ = H.forward_backward(logB, params['Amat'], params['pi'], w)
            durs = H.state_durations(gamma.argmax(1))
            haz = np.vstack([H.to_hazard(H.dur_pmf(durs[0])), H.to_hazard(H.dur_pmf(durs[1]))])
            filt = H.hsmm_filter(logB, sw, haz, bear, params['pi'])
        else:
            filt = H.plain_hmm_filter(logB, sw, params['Amat'], params['pi'], bear)
        pbear[t] = filt[-1, bear]
    return pbear, start


def smooth_regime_exposure(pbear_raw, start, n):
    """H와 동일: EMA → 히스테리시스 이산레짐 + pbear-only 연속 익스포저(리밸밴드)."""
    idx = list(range(start, n)); pb = pbear_raw.copy()
    for t in idx[1:]:
        pb[t] = H.PBEAR_EMA * pbear_raw[t] + (1 - H.PBEAR_EMA) * pb[t - 1]
    reg = ["Bull"] * n; p = "Bull"
    for t in idx:
        p = ("Bear" if pb[t] >= H.T_OUT else "Bull") if p == "Bear" else ("Bear" if pb[t] >= H.T_IN else "Bull")
        reg[t] = p
    exp = np.ones(n); raw = np.clip(1 - pb, H.EXP_FLOOR, 1.0); held = None
    for t in idx:
        if held is None or abs(raw[t] - held) >= H.REBAL_BAND:
            held = round(raw[t] / 0.05) * 0.05
        exp[t] = min(max(held, H.EXP_FLOOR), 1.0)
    return pb, reg, exp, idx


# ───────────────── 이벤트 정의 점검 (dd6 첫크로싱 vs 신고가-경계) ─────────────────
def old_events(dd6, idx, start, yms):
    """현행(버그 의심): dd6<=-15 & 직전월 아니면 첫크로싱."""
    return [yms[i] for i in idx if dd6[i] <= -15 and (i == start or not (dd6[i - 1] <= -15))]


def price_index(ret, n):
    P = np.ones(n)
    for t in range(1, n):
        r = ret[t - 1]; P[t] = P[t - 1] * (1 + (r if not np.isnan(r) else 0.0))
    return P


def true_episodes(P, yms, thr=-0.15):
    """신고가 회복으로 에피소드 경계를 나눔 → 각 하락장의 peak/trough/depth.
       dd6 forward창 오염(2019Q4가 2020코로나를 미리 봄) 없이 실제 저점 기준."""
    n = len(P); out = []
    peak, peak_t, trough_t, peak_ref, peak_ref_t = P[0], 0, None, P[0], 0
    for t in range(1, n):
        if P[t] > peak:
            if trough_t is not None and (P[trough_t] / peak_ref - 1) <= thr:
                out.append((peak_ref_t, trough_t, P[trough_t] / peak_ref - 1))
            peak, peak_t, trough_t = P[t], t, None
        else:
            if trough_t is None or P[t] < P[trough_t]:
                trough_t = t; peak_ref, peak_ref_t = peak, peak_t
    if trough_t is not None and (P[trough_t] / peak_ref - 1) <= thr:
        out.append((peak_ref_t, trough_t, P[trough_t] / peak_ref - 1))
    return out  # list of (peak_t, trough_t, depth)


# ───────────────── 성과/기여도 ─────────────────
def fcf_series():
    s = pd.read_csv(REPO / "analysis" / "fcf_overlay_series.csv")
    return list(s["ym"]), dict(zip(s["ym"], s["bench"]))


def overlay_perf(exp, yms, fmonths, fcf):
    """pbear-only 익스포저를 FCF 전략에 적용: CAGR/MDD/Sharpe/turnover/평균exp."""
    ebym = {yms[t]: exp[t] for t in range(len(yms))}
    ov = np.array([ebym.get(shift_ym(y, -1), 1.0) * fcf[y] for y in fmonths])
    eq = np.cumprod(1 + ov); yr = len(ov) / 12
    cagr = eq[-1] ** (1 / yr) - 1
    mdd = (eq / np.maximum.accumulate(eq) - 1).min()
    shp = (ov.mean() * 12) / (ov.std() * np.sqrt(12) + 1e-12)
    return dict(cagr=cagr, mdd=mdd, sharpe=shp, eq=eq, ov=ov)


def main():
    df0, yms, n, ret, rvol, dvol, dd6 = H.build_features()
    df = augment(df0)
    stress = base_stress(df)
    fmonths, fcf = fcf_series()
    P = price_index(ret, n)
    fcf_next = np.array([fcf.get(shift_ym(yms[t], 1), np.nan) for t in range(n)])  # exp[t]가 먹는 다음달 FCF수익

    # 각 arm 실행
    R = {}
    for name, emis in ARMS:
        print(f">>> {name}: {emis}", flush=True)
        pbr, start = walk_forward_exp(df, yms, n, emis, stress)
        pb, reg, exp, idx = smooth_regime_exposure(pbr, start, n)
        perf = overlay_perf(exp, yms, fmonths, fcf)
        turn = sum(abs(exp[t] - exp[t - 1]) for t in idx[1:])
        R[name] = dict(pb=pb, reg=reg, exp=exp, idx=idx, start=start,
                       bearm=sum(1 for t in idx if reg[t] == "Bear"),
                       avgexp=np.mean([exp[t] for t in idx]), turn=turn, **perf)
    start = R["BASE"]["start"]; idx = R["BASE"]["idx"]
    anames = [a[0] for a in ARMS]

    # ═══════════ 0. 이벤트 정의 점검 ═══════════
    print(f"\n{'='*76}\n[0] 이벤트 정의 점검: 왜 2020 코로나가 누락되나\n{'='*76}")
    oev = old_events(dd6, idx, start, yms)
    print(f"  현행(dd6<=-15 첫크로싱) 이벤트: {oev}")
    print("  dd6<=-15 연속구간(첫크로싱만 이벤트로 잡혀 나머지는 흡수):")
    run = []
    for i in idx:
        if dd6[i] <= -15: run.append(yms[i])
        elif run: print(f"    {run[0]}~{run[-1]} ({len(run)}개월 연속)  → 이벤트: {run[0]}만"); run = []
    if run: print(f"    {run[0]}~{run[-1]} ({len(run)}개월 연속)  → 이벤트: {run[0]}만")
    print("  ※ dd6는 '향후6M' 낙폭이라, 2019말 각 월의 6M창이 2020-03 코로나를 미리 포함 →")
    print("     2019Q4~2020상반기가 한 덩어리로 <=-15 연속 → 2020-02가 '첫크로싱 아님'으로 흡수됨.")
    eps = true_episodes(P, yms)
    print("\n  [수정案] 신고가-회복 경계 에피소드(실제 저점 기준, forward창 오염 없음):")
    print(f"  {'peak':>9} → {'trough':>9}  depth")
    for pk, tr, dep in eps:
        print(f"  {yms[pk]:>9} → {yms[tr]:>9}  {dep*100:>6.1f}%")

    # ═══════════ 1. 기본 성과 ═══════════
    print(f"\n{'='*76}\n[1] 기본 성과 (FCF 오버레이, pbear-only)\n{'='*76}")
    print(f"  {'arm':11}{'CAGR':>8}{'MDD':>9}{'Sharpe':>8}{'Bear월':>8}{'평균exp':>9}{'turnover':>10}")
    for a in anames:
        r = R[a]
        print(f"  {a:11}{r['cagr']*100:>7.1f}%{r['mdd']*100:>8.1f}%{r['sharpe']:>8.2f}"
              f"{r['bearm']:>8}{r['avgexp']:>9.2f}{r['turn']:>10.2f}")

    # ═══════════ 2. 레짐 탐지 ═══════════
    print(f"\n{'='*76}\n[2] 레짐 탐지: 최초 Bear진입 / 에피소드별 lead(vs trough)\n{'='*76}")
    print(f"  {'arm':11}{'최초Bear':>10}   에피소드별 진입(lead=trough-진입, +=저점前 방어)")
    for a in anames:
        reg = R[a]["reg"]
        fb = next((yms[t] for t in idx if reg[t] == "Bear"), "없음")
        parts = []
        for pk, tr, dep in eps:
            ent = next((t for t in range(pk, min(tr + 4, n)) if reg[t] == "Bear"), None)
            parts.append(f"{yms[tr][:7]}:{('X' if ent is None else f'{tr-ent:+d}m')}")
        print(f"  {a:11}{fb:>10}   " + "  ".join(parts))

    # ═══════════ 3. 월별 attribution (vs BASE) ═══════════
    be, breg = R["BASE"]["exp"], R["BASE"]["reg"]
    def classify(t, ea):
        if ea[t] < be[t] - 1e-9:                     # arm이 더 방어
            r = ret[t]
            if not np.isnan(r) and r < 0: return "하락중방어"
            fut = P[t + 1:min(t + 7, n)]
            return "조기진입" if (len(fut) and np.min(fut) < P[t]) else "늦은복귀"
        elif ea[t] > be[t] + 1e-9: return "공격적"
        return None

    print(f"\n{'='*76}\n[3] 월별 attribution: BASE와 포지션 다른 월 (기여도=Δexp×FCF다음달)\n{'='*76}")
    agg = {}
    for a in anames:
        if a == "BASE": continue
        ea, ereg = R[a]["exp"], R[a]["reg"]
        rows = []
        for t in idx:
            cls = classify(t, ea)
            if cls is None: continue
            dexp = ea[t] - be[t]
            c_fcf = dexp * (fcf_next[t] if not np.isnan(fcf_next[t]) else 0.0)
            c_kos = dexp * (ret[t] if not np.isnan(ret[t]) else 0.0)
            rows.append((yms[t], breg[t], ereg[t], be[t], ea[t], ret[t], fcf_next[t], c_fcf, c_kos, cls))
        agg[a] = rows
        print(f"\n  ── {a} (BASE와 다른 {len(rows)}개월) ──")
        print(f"  {'월':8}{'B/arm':>10}{'expB':>6}{'expA':>6}{'KOSPI':>7}{'FCF+1':>7}{'기여FCF':>8}  분류")
        for ym, rb, ra, xb, xa, rk, rf, cf, ck, cls in rows:
            rk_s = f"{rk*100:>6.1f}" if not np.isnan(rk) else "   n/a"
            rf_s = f"{rf*100:>6.1f}" if not np.isnan(rf) else "   n/a"
            print(f"  {ym:8}{rb[:1]+'/'+ra[:1]:>10}{xb:>6.2f}{xa:>6.2f}{rk_s}{rf_s}{cf*100:>7.2f}%  {cls}")

    # 버킷 집계
    print(f"\n{'='*76}\n[3b] arm별 기여도 버킷 집계 (기여FCF 합, %p)\n{'='*76}")
    print(f"  {'arm':11}{'하락중방어':>11}{'조기진입':>10}{'늦은복귀':>10}{'공격적':>9}{'순합':>9}")
    for a in anames:
        if a == "BASE": continue
        b = {"하락중방어": 0.0, "조기진입": 0.0, "늦은복귀": 0.0, "공격적": 0.0}
        for row in agg[a]: b[row[9]] += row[7]
        net = sum(b.values())
        print(f"  {a:11}{b['하락중방어']*100:>10.2f}%{b['조기진입']*100:>9.2f}%"
              f"{b['늦은복귀']*100:>9.2f}%{b['공격적']*100:>8.2f}%{net*100:>8.2f}%")
    print("  ※ +하락중방어=제대로 방어한 이득 / -조기진입=저점前 상승월 방어비용(entry-side)")
    print("     -늦은복귀=저점後 상승월 방어비용(exit-side) / net=BASE 대비 총 초과기여")

    # MDD 기여: BASE의 MDD 구간에서 arm이 방어한 정도 + 2021 vs 2022 분해
    beq = R["BASE"]["eq"]; dd = beq / np.maximum.accumulate(beq) - 1
    tr_i = int(np.argmin(dd)); pk_i = int(np.argmax(beq[:tr_i + 1]))
    win_ym = set(fmonths[pk_i:tr_i + 1])
    print(f"\n{'='*76}\n[3c] MDD 개선 출처: BASE MDD구간 [{fmonths[pk_i]}~{fmonths[tr_i]}] 내 arm 방어이득 (연도별)\n{'='*76}")
    print(f"  {'arm':11}{'2021이전':>10}{'2021':>9}{'2022':>9}{'2023+':>9}{'구간합':>9}")
    for a in anames:
        if a == "BASE": continue
        ea = R[a]["exp"]; yr = {"pre": 0.0, "2021": 0.0, "2022": 0.0, "post": 0.0}
        for t in idx:
            nym = shift_ym(yms[t], 1)
            if nym not in win_ym: continue
            g = (be[t] - ea[t]) * (fcf_next[t] if not np.isnan(fcf_next[t]) else 0.0)  # 방어이득(+)
            k = "2021" if nym[:4] == "2021" else "2022" if nym[:4] == "2022" else ("pre" if nym < "2021" else "post")
            yr[k] += g
        print(f"  {a:11}{yr['pre']*100:>9.2f}%{yr['2021']*100:>8.2f}%{yr['2022']*100:>8.2f}%"
              f"{yr['post']*100:>8.2f}%{sum(yr.values())*100:>8.2f}%")
    print("  ※ 값이 2021에 몰리면 '조기 현금화 효과', 2022에 몰리면 '하락 본체 방어'.")

    # ═══════════ 4. 피처별 기계적 lag 진단 (arm 무관, 원피처 값) ═══════════
    print(f"\n{'='*76}\n[4] 피처 기계적 lag 진단: 각 저점(trough) 전후 원피처 값\n{'='*76}")
    print("  (d_breadth/d_newlow=Δ3m → 저점後에도 음수로 남으면 기계적 회복지연 / diverg=동시점)")
    for pk, tr, dep in eps:
        print(f"\n  ── trough {yms[tr]} (depth {dep*100:.0f}%) ──")
        print(f"  {'월':8}{'breadth':>9}{'d_breadth':>10}{'d_newlow':>10}{'diverg':>9}")
        for t in range(max(tr - 1, 0), min(tr + 4, n)):
            mark = "◀저점" if t == tr else ""
            print(f"  {yms[t]:8}{df['breadth'].iloc[t]:>9.2f}{df['d_breadth'].iloc[t]:>10.3f}"
                  f"{df['d_newlow'].iloc[t]:>10.3f}{df['diverg'].iloc[t]:>9.2f}  {mark}")

    # ═══════════ 5. 최종 판정 (데이터 자동요약) ═══════════
    print(f"\n{'='*76}\n[5] 최종 판정 (데이터 요약)\n{'='*76}")
    base = R["BASE"]
    print(f"  {'arm':11}{'ΔMDD(개선)':>12}{'ΔCAGR(손실)':>13}{'개선/손실비':>12}{'entry비용':>10}{'exit비용':>10}")
    for a in anames:
        if a == "BASE": continue
        dmdd = (R[a]["mdd"] - base["mdd"]) * 100          # +면 MDD 개선(덜 빠짐)
        dcagr = (base["cagr"] - R[a]["cagr"]) * 100        # +면 CAGR 손실
        ratio = dmdd / dcagr if abs(dcagr) > 1e-6 else float('inf')
        ent = sum(row[7] for row in agg[a] if row[9] == "조기진입") * 100
        ext = sum(row[7] for row in agg[a] if row[9] == "늦은복귀") * 100
        print(f"  {a:11}{dmdd:>11.1f}p{dcagr:>12.1f}p{ratio:>12.2f}{ent:>9.2f}%{ext:>9.2f}%")
    print("\n  해석 가이드:")
    print("  Q1 MDD출처 = [3c]에서 2021 vs 2022 어디에 방어이득이 몰렸나")
    print("  Q2 CAGR훼손 = entry비용 vs exit비용 절대크기 (entry↑=조기진입 문제 / exit↑=회복지연 문제)")
    print("  Q3 최소손실 조합 = [5] 개선/손실비 최대 arm")
    print("  Q4 exit수정 근거 = exit비용이 entry비용보다 유의하게 커야 성립")
    print("  Q5 다음실험 = entry지배면 '피처교체(diverg계열)', exit지배면 'exit비대칭화', 둘다크면 'exposure recovery 분리'")

    # 경로 저장
    out = pd.DataFrame({"ym": yms})
    for a in anames:
        out[f"pbear_{a}"] = R[a]["pb"]; out[f"reg_{a}"] = R[a]["reg"]; out[f"exp_{a}"] = R[a]["exp"]
    out.to_csv(REPO / "analysis" / "emis_ablation_path.csv", index=False, encoding="utf-8-sig")
    print(f"\n  경로 저장 → analysis/emis_ablation_path.csv")


if __name__ == "__main__":
    main()
