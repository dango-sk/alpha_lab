"""
FCF 전략 팩터 진단 (면접 대비용)

출력:
  1) FCF_YIELD추가전략 / FCF_YIELD_BEAR전략의 팩터 목록·비중·PARAMS (둘이 동일한지 확인)
  2) 지정 종목들의 전체 유니버스 내 순위 + 팩터별 점수 (top30 진입/탈락 이유)

사용:
  .venv/bin/python analysis/fcf_factor_diagnosis.py [YYYY-MM-DD]
  (날짜 생략 시 2026-07-01)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")
from lib.data import load_strategy, code_to_module
from lib.factor_engine import prefetch_all_data, score_stocks_from_strategy
from step7_backtest import get_db

CALC = sys.argv[1] if len(sys.argv) > 1 else "2026-07-01"
TARGETS = ["기아", "삼성전자", "SK하이닉스"]
STRATS = ["FCF_YIELD추가전략", "FCF_YIELD_BEAR전략"]

# ── 1) 전략 비중/PARAMS ──
for key in STRATS:
    d = load_strategy(key, rebal_type="monthly", universe="KOSPI")
    if not d:
        print(f"[{key}] 없음"); continue
    ns = {}
    exec(compile(d["code"], "<s>", "exec"), {"__builtins__": {}}, ns)
    w = {k: v for k, v in ns.get("WEIGHTS_LARGE", {}).items() if v and v > 0}
    p = ns.get("PARAMS", {})
    print(f"\n[{key}]  팩터 {len(w)}개 (합 {sum(w.values()):.2f})")
    for k, v in sorted(w.items(), key=lambda x: -x[1]):
        print(f"    {k:14} {v:.3f}")
    print(f"    PARAMS: top_n={p.get('top_n')} cap={p.get('weight_cap_pct')}% "
          f"stop={p.get('stop_loss_enabled')} {p.get('stop_loss_pct')}% basis={p.get('stop_loss_basis')}")

# ── 2) 종목별 순위/팩터점수 ──
d = load_strategy("FCF_YIELD_BEAR전략", rebal_type="monthly", universe="KOSPI")
mod = code_to_module(d["code"])
w = {k: v for k, v in getattr(mod, "WEIGHTS_LARGE", {}).items() if v and v > 0}
score_map = getattr(mod, "SCORE_MAP", {})
used_cols = {score_map[k]: (k, w[k]) for k in w if k in score_map}

conn = get_db()
prefetch_all_data(conn)
_r, df = score_stocks_from_strategy(conn, CALC, mod, return_df=True)
df = df.sort_values("value_score", ascending=False).reset_index(drop=True)
df["순위"] = df.index + 1
namecol = "종목명" if "종목명" in df.columns else "stock_name"

print(f"\n{'='*60}\n[{CALC}] 전체 {len(df)}종목 채점")
for tgt in TARGETS:
    row = df[df[namecol] == tgt]
    if row.empty:
        print(f"\n★ {tgt}: 퀄리티 필터 제외"); continue
    r = row.iloc[0]
    print(f"\n★ {tgt}: {int(r['순위'])}위 / {len(df)}  (value_score {r['value_score']:.1f})  "
          f"→ top30 {'진입' if r['순위'] <= 30 else '탈락'}")
    # 전략이 실제 쓰는 팩터만, 기여도(점수×비중) 순
    rows = []
    for col, (fac, wt) in used_cols.items():
        if col in df.columns:
            sc = float(r[col])
            rows.append((fac, sc, wt, sc * wt))
    for fac, sc, wt, contrib in sorted(rows, key=lambda x: -x[3]):
        print(f"     {fac:14} 점수 {sc:.0f}/5  비중 {wt:.2f}  기여 {contrib:.3f}")
