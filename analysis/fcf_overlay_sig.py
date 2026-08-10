"""
analysis/fcf_overlay_sig.py

fcf_overlay_series.csv(월수익 시계열)로 오버레이 버전 간 '유의미하게 좋은지' 검정.
방법: stationary block bootstrap(평균 블록 6M) — 자기상관·드로다운 경로 보존, paired(같은 리샘플 인덱스).
지표: CAGR / Sharpe / MDD / Calmar. 우열은 paired P(X>Y)와 차이의 90% CI로 판단.
사용: .venv/bin/python analysis/fcf_overlay_sig.py
"""
import numpy as np, pandas as pd
from pathlib import Path

A = Path(__file__).parent
df = pd.read_csv(A / "fcf_overlay_series.csv")
COLS = ["bench", "ovA", "ovB", "ovP"]
LABEL = {"bench": "FCF불 단독(BM)", "ovA": "A:20일실현", "ovB": "B:60일하방", "ovP": "pbear만(vol無)"}
R = {c: df[c].values.astype(float) for c in COLS}
T = len(df)
NB, L, SEED = 5000, 6, 42
rng = np.random.default_rng(SEED)


def metrics(r):
    eq = np.cumprod(1 + r); yrs = len(r) / 12
    cagr = eq[-1] ** (1 / yrs) - 1
    vol = r.std(ddof=0) * np.sqrt(12)
    sharpe = (r.mean() * 12) / vol if vol > 0 else np.nan
    mdd = (eq / np.maximum.accumulate(eq) - 1).min()
    calmar = cagr / abs(mdd) if mdd < 0 else np.nan
    return dict(CAGR=cagr, Sharpe=sharpe, MDD=mdd, Calmar=calmar)


MET = ["CAGR", "Sharpe", "MDD", "Calmar"]
point = {c: metrics(R[c]) for c in COLS}

# ── paired stationary block bootstrap ──
samp = {c: {m: np.empty(NB) for m in MET} for c in COLS}
for b in range(NB):
    idx = []
    while len(idx) < T:
        s = int(rng.integers(0, T)); blen = int(rng.geometric(1.0 / L))
        idx.extend(((s + k) % T) for k in range(blen))
    ix = np.array(idx[:T])
    for c in COLS:
        m = metrics(R[c][ix])
        for k in MET: samp[c][k][b] = m[k]


def ci(a):
    return np.nanpercentile(a, 5), np.nanpercentile(a, 95)


print(f"기간 {df['ym'].iloc[0]}~{df['ym'].iloc[-1]} ({T}개월) · 부트스트랩 {NB}회 · 블록 {L}M · seed {SEED}\n")
print(f"{'전략':16}{'CAGR':>18}{'Sharpe':>18}{'MDD':>20}{'Calmar':>18}")
for c in COLS:
    def cell(m, pct=False, scale=1):
        lo, hi = ci(samp[c][m]); p = point[c][m]
        f = (lambda x: f"{x*100:.1f}%") if pct else (lambda x: f"{x:.2f}")
        return f"{f(p)} [{f(lo)},{f(hi)}]"
    print(f"{LABEL[c]:16}{cell('CAGR',1):>18}{cell('Sharpe'):>18}{cell('MDD',1):>20}{cell('Calmar'):>18}")
print("  (점추정 [부트스트랩 90% CI])\n")

# ── paired 우열 검정 ──
def compare(x, y):
    print(f"── {LABEL[x]}  vs  {LABEL[y]} ──")
    for m in MET:
        dx = samp[x][m] - samp[y][m]
        better = (dx > 0).mean() if m != "MDD" else (dx > 0).mean()  # MDD는 값이 클수록(덜 음수) 좋음 → dx>0이 x가 나음
        lo, hi = ci(dx)
        sig = "유의" if (lo > 0 or hi < 0) else "무의미"
        arrow = "↑x우세" if better >= 0.5 else "↓y우세"
        f = (lambda v: f"{v*100:+.1f}%p") if m in ("CAGR", "MDD") else (lambda v: f"{v:+.2f}")
        print(f"   {m:7} Δ(x-y)={f(np.median(dx)):>9}  90%CI[{f(lo)},{f(hi)}]  P(x우세)={better:.2f}  {arrow}  → {sig}")
    print()


PAIRS = [("ovA", "bench"), ("ovB", "bench"), ("ovP", "bench"),
         ("ovA", "ovB"), ("ovP", "ovA"), ("ovP", "ovB")]
print("=" * 74)
print("우열 검정 (MDD/Calmar/Sharpe = 방어·위험조정, 값 클수록 x가 좋음)")
print("  '유의' = 차이의 90% CI가 0을 넘지 않음\n")
for x, y in PAIRS:
    compare(x, y)
