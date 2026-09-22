"""daily_price 지연 점검. run_monthly.sh 0단계에서 호출."""
import sys, warnings
from datetime import date
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/namsugyeong/Desktop/alpha_lab")
from lib.db import get_conn

conn = get_conn()
mx = conn.execute("SELECT max(trade_date) FROM daily_price").fetchone()[0]
fw = conn.execute("SELECT max(trade_date) FROM fnspace_forward").fetchone()[0]
conn.close()
lag = (date.today() - date.fromisoformat(str(mx))).days
print(f"  daily_price     최신 {mx}  (오늘 기준 {lag}일 전)")
print(f"  fnspace_forward 최신 {fw}")
if lag > 7:
    print(f"  X 주가가 {lag}일 밀려 있다. scripts/legacy/step1_update_prices.py 로 먼저 업로드할 것.")
    sys.exit(1)
print("  O 주가 최신")
