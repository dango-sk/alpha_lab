"""
alpha_lab 설정 파일
"""
import os
import gzip
import shutil
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# ─── API 키 ───
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# ─── 경로 ───
BASE_DIR = Path(__file__).parent.parent
ALPHA_RADAR_DIR = Path.home() / "Downloads" / "alpha_radar"
DB_PATH = Path(os.environ.get("DB_PATH", str(ALPHA_RADAR_DIR / "db" / "alpha_radar.db")))
CACHE_DIR = Path(os.environ.get("CACHE_DIR", str(BASE_DIR / "cache")))

# ─── Railway: gz 압축 해제 (PostgreSQL 미사용 시 fallback) ───
if not os.environ.get("DATABASE_URL", ""):
    _deploy_db = BASE_DIR / "data" / "alpha_radar.db"
    _deploy_gz = BASE_DIR / "data" / "alpha_radar.db.gz"
    if not _deploy_db.exists() and _deploy_gz.exists():
        with gzip.open(_deploy_gz, "rb") as f_in, open(_deploy_db, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

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
    "end": "2026-04-01",
    "insample_end": "2024-06-30",
    "oos_start": "2024-07-01",
    "rebalance_freq": "monthly",
    "transaction_cost_bp": 30,       # 편도 30bp
    "top_n_stocks": 30,
    "weight_cap_pct": 10,            # 개별종목 비중상한 (%)
    "min_market_cap": 500_000_000_000,  # 유니버스 시총 하한: 5천억원
}
