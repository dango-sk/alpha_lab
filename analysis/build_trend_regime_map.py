"""
analysis/build_trend_regime_map.py

결합 추세 레짐(학습모델 게이트 + MA150)을 월별 Bull/Bear 맵으로 만들어,
기존 백테스트(regime_mode="rf"가 읽는 analysis/regime_rf_map.json)에 꽂는다.
→ production 코드 수정 없이 월단위로 적용해 성능 확인.

3-state → binary 두 가지 (whipsaw 비교):
  raw    : 강세장→Bull, (약세장+변동성장)→Bear
  sticky : 강세장→Bull, 약세장→Bear, 변동성장→직전 유지   ← whipsaw↓ (기존 ai_v2 방식)

lookahead-free: 모델 게이트=walk-forward, MA=각 월말 시점값. map[M]=M월 레짐(M-1말 데이터로 결정).

입력:  Railway PG(kospi) + analysis/trend_ml_features.csv
출력:  analysis/regime_trend_raw.json, analysis/regime_trend_sticky.json
       analysis/regime_rf_map.json (sticky를 복사 — 기존 있으면 .bak 백업)

사용:  .venv/bin/python analysis/build_trend_regime_map.py
이후:  run_regime_combo_backtest(regime_mode="rf", bull_key=..., bear_key=...)
"""

import os
import json
import shutil
import psycopg2
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
A = Path(__file__).parent
GATE_THRESH = 0.4
MA_WINDOW = 150


def main():
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    c = pd.read_sql("SELECT period::date dt, value::float close FROM alpha_lab.macro_indicators "
                    "WHERE indicator='kospi' AND freq='D' ORDER BY period", conn)
    conn.close()
    c["dt"] = pd.to_datetime(c["dt"])
    close = c.set_index("dt")["close"]

    ym = pd.PeriodIndex(close.index, freq="M")
    last = {}
    for d, p in zip(close.index, ym):
        last[p] = d
    mends = [last[p] for p in sorted(last)]

    f = pd.read_csv(A / "trend_ml_features.csv")
    gate = {r["pred_month"]: (r["gb_bull_3m"] >= GATE_THRESH)
            for _, r in f.iterrows() if pd.notna(r["gb_bull_3m"])}

    raw, sticky = {}, {}
    states = {}
    prev = "Bull"
    for i in range(len(mends) - 1):
        e, nxt = mends[i], mends[i + 1]
        pm = pd.Period(nxt, freq="M").strftime("%Y-%m")
        if pm not in gate:
            continue
        ma = close[close.index <= e].iloc[-MA_WINDOW:].mean()
        ma_up = close.loc[e] > ma
        mdl = gate[pm]
        if mdl and ma_up:
            state, rb, sb = "강세장", "Bull", "Bull"
        elif (not mdl) and (not ma_up):
            state, rb, sb = "약세장", "Bear", "Bear"
        else:
            state, rb, sb = "변동성장", "Bear", prev   # raw:Bear / sticky:직전유지
        states[pm] = state
        raw[pm] = rb
        sticky[pm] = sb
        prev = sb

    def trans(m):
        ks = sorted(m)
        return sum(1 for i in range(1, len(ks)) if m[ks[i]] != m[ks[i - 1]])

    print(f"총 {len(raw)}개월")
    for nm, m in [("raw", raw), ("sticky", sticky)]:
        b = sum(1 for v in m.values() if v == "Bull")
        print(f"  {nm:7} Bull={b}  Bear={len(m)-b}  전환={trans(m)}회  "
              f"← whipsaw {'적음' if nm=='sticky' else '많음'}")

    (A / "regime_trend_raw.json").write_text(json.dumps(raw, ensure_ascii=False))
    (A / "regime_trend_sticky.json").write_text(json.dumps(sticky, ensure_ascii=False))

    # regime_rf_map.json에 sticky 복사 (기존 백업)
    rf = A / "regime_rf_map.json"
    if rf.exists():
        shutil.copy(rf, A / "regime_rf_map.json.bak")
        print(f"  기존 regime_rf_map.json → regime_rf_map.json.bak 백업")
    rf.write_text(json.dumps(sticky, ensure_ascii=False))
    print(f"  sticky 맵 → regime_rf_map.json (regime_mode='rf'로 테스트)")

    print("\n최근 12개월 (월: 3-state → raw/sticky):")
    for pm in sorted(states)[-12:]:
        print(f"  {pm}  {states[pm]:6} → raw={raw[pm]:4} sticky={sticky[pm]}")


if __name__ == "__main__":
    main()
