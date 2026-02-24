"""
alpha_lab 설정 파일
4개 밸류 전략 (A0, A, A+M, ATT2) 탐구용
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# ─── API 키 ───
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# ─── 경로 ───
BASE_DIR = Path(__file__).parent.parent
ALPHA_RADAR_DIR = Path.home() / "Downloads" / "alpha_radar"
DB_PATH = ALPHA_RADAR_DIR / "db" / "alpha_radar.db"
CACHE_DIR = BASE_DIR / "cache"

# ─── 퀄리티 필터 ───
QUALITY_FILTER = {
    "max_debt_ratio": 200,
    "consecutive_loss_quarters": 1,
    "min_avg_volume_20d": 500_000_000,
    "exclude_admin_issue": True,
}

# ─── 백테스트 설정 ───
BACKTEST_CONFIG = {
    "start": "2021-01-01",
    "end": "2026-03-01",
    "insample_end": "2024-06-30",
    "oos_start": "2024-07-01",
    "rebalance_freq": "monthly",
    "transaction_cost_bp": 30,       # 편도 30bp
    "top_n_stocks": 30,
    "weight_cap_pct": 15,            # 개별종목 비중상한 (%)
}
