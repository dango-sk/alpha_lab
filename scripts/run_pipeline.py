"""
Alpha Lab 전체 파이프라인 실행
실행: python scripts/run_pipeline.py

순서:
  1. Step 1: 주가 업데이트 (alpha_radar DB)
  2. Step 3: 밸류 팩터 계산
  3. Step 6: 시그널 생성
  4. Step 7: 백테스트
  5. Step 8: 강건성 검증
  6. Railway 배포

소요 시간: 약 20~40분 (step1 주가 수집이 대부분)
"""
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich import box

console = Console()

# 각 스텝 소요시간 기록
_step_times: list[tuple[str, float]] = []


def run_step(name: str, func, *args, **kwargs):
    """스텝 실행 + 시간 측정"""
    t = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - t
    _step_times.append((name, elapsed))
    return result


def print_summary_table():
    """완료 후 요약 테이블 출력"""
    table = Table(
        title="파이프라인 실행 결과",
        box=box.ROUNDED,
        show_lines=True,
        title_style="bold cyan",
    )
    table.add_column("Step", style="bold", min_width=25)
    table.add_column("소요 시간", justify="right", min_width=10)
    table.add_column("상태", justify="center", min_width=6)

    total = 0
    for name, elapsed in _step_times:
        total += elapsed
        if elapsed < 60:
            time_str = f"{elapsed:.1f}s"
        else:
            time_str = f"{elapsed / 60:.1f}m"
        table.add_row(name, time_str, "[green]OK[/]")

    table.add_section()
    total_str = f"{total / 60:.1f}m" if total >= 60 else f"{total:.1f}s"
    table.add_row("[bold]전체[/]", f"[bold]{total_str}[/]", "")

    console.print()
    console.print(table)


def main():
    t0 = time.time()

    console.print(
        Panel(
            "[bold]Alpha Lab 전체 파이프라인[/]",
            subtitle="Step 1 → 3 → 6 → 7 → 8 → Deploy",
            box=box.DOUBLE,
            style="cyan",
        )
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=30),
        TextColumn("[cyan]{task.fields[status]}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        overall = progress.add_task("파이프라인", total=6, status="시작")

        # ── Step 1: 주가 업데이트 ──
        progress.update(overall, status="[1/6] 주가 업데이트")
        from step1_update_prices import update_prices, show_summary

        def _step1():
            update_prices()
            show_summary()

        run_step("주가 업데이트 (Step 1)", _step1)
        progress.advance(overall)

        # ── Step 3: 밸류 팩터 계산 ──
        progress.update(overall, status="[2/6] 밸류 팩터 계산")
        from step3_calc_value_factors import calc_valuation_for_date
        from config.settings import DB_PATH, BACKTEST_CONFIG
        import sqlite3

        def _step3():
            conn = sqlite3.connect(str(DB_PATH))
            trade_dates = conn.execute("""
                SELECT DISTINCT trade_date FROM daily_price
                WHERE trade_date >= ? AND trade_date <= ?
                ORDER BY trade_date
            """, (BACKTEST_CONFIG["start"], BACKTEST_CONFIG["end"])).fetchall()

            monthly_dates = []
            current_month = ""
            for (td,) in trade_dates:
                month = td[:7]
                if month != current_month:
                    monthly_dates.append(td)
                    current_month = month
            conn.close()

            conn = sqlite3.connect(str(DB_PATH))
            existing_dates = set(
                row[0] for row in conn.execute(
                    "SELECT DISTINCT calc_date FROM valuation_factors"
                ).fetchall()
            )
            conn.close()

            new_dates = [d for d in monthly_dates if d not in existing_dates]
            if new_dates:
                console.print(f"  신규 계산: [yellow]{len(new_dates)}개월[/]")
                sub = progress.add_task("  팩터 계산", total=len(new_dates), status="")
                for date in new_dates:
                    progress.update(sub, status=date)
                    calc_valuation_for_date(target_date=date)
                    progress.advance(sub)
                progress.remove_task(sub)
            else:
                console.print(f"  [dim]{len(monthly_dates)}개월 이미 계산됨 → 스킵[/]")

        run_step("밸류 팩터 계산 (Step 3)", _step3)
        progress.advance(overall)

        # ── Step 6: 시그널 생성 ──
        progress.update(overall, status="[3/6] 시그널 생성")
        from step6_generate_signals import generate_signals_for_date

        def _step6():
            conn = sqlite3.connect(str(DB_PATH))
            trade_dates = conn.execute("""
                SELECT DISTINCT trade_date FROM daily_price
                WHERE trade_date >= ? AND trade_date <= ?
                ORDER BY trade_date
            """, (BACKTEST_CONFIG["start"], BACKTEST_CONFIG["end"])).fetchall()

            monthly_dates = []
            current_month = ""
            for (td,) in trade_dates:
                month = td[:7]
                if month != current_month:
                    monthly_dates.append(td)
                    current_month = month

            existing_signals = set(
                row[0] for row in conn.execute(
                    "SELECT DISTINCT calc_date FROM signals"
                ).fetchall()
            )
            conn.close()

            new_signal_dates = [d for d in monthly_dates if d not in existing_signals]
            if new_signal_dates:
                console.print(f"  신규 시그널: [yellow]{len(new_signal_dates)}개월[/]")
                sub = progress.add_task("  시그널 생성", total=len(new_signal_dates), status="")
                for date in new_signal_dates:
                    progress.update(sub, status=date)
                    generate_signals_for_date(date)
                    progress.advance(sub)
                progress.remove_task(sub)
            else:
                console.print(f"  [dim]{len(monthly_dates)}개월 시그널 있음 → 스킵[/]")

        run_step("시그널 생성 (Step 6)", _step6)
        progress.advance(overall)

        # ── Step 7: 백테스트 ──
        progress.update(overall, status="[4/6] 백테스트")
        from step7_backtest import run_all_backtests, save_backtest_cache

        def _step7():
            results = run_all_backtests()
            if results:
                save_backtest_cache(results)

        run_step("백테스트 (Step 7)", _step7)
        progress.advance(overall)

        # ── Step 8: 강건성 검증 ──
        progress.update(overall, status="[5/6] 강건성 검증")
        from step8_robustness import (
            test_is_oos_split,
            test_statistical_significance,
            test_rolling_window,
            save_robustness_cache,
        )

        def _step8():
            is_oos = test_is_oos_split()
            stat = test_statistical_significance()
            rolling = test_rolling_window(stat["full_results"])
            save_robustness_cache(is_oos, stat, rolling)

        run_step("강건성 검증 (Step 8)", _step8)
        progress.advance(overall)

        # ── Railway 배포 ──
        progress.update(overall, status="[6/6] Railway 배포")
        run_step("Railway 배포", deploy_to_railway)
        progress.advance(overall)

        progress.update(overall, status="[green]완료![/]")

    # ── 요약 ──
    print_summary_table()

    elapsed = time.time() - t0
    console.print(
        Panel(
            f"[bold green]전체 완료![/]  소요: {elapsed / 60:.1f}분\n"
            f"[dim]streamlit run app.py 로 대시보드를 실행하세요.[/]",
            box=box.ROUNDED,
            style="green",
        )
    )

    notify(f"파이프라인 완료 ({elapsed / 60:.1f}분)", "Step 1~8 + Railway 배포 성공")


def deploy_to_railway():
    """DB + 캐시를 복사하고 Railway에 재배포."""
    import shutil
    import subprocess

    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / "data"
    data_dir.mkdir(exist_ok=True)

    from config.settings import DB_PATH
    dest_db = data_dir / "alpha_radar.db"
    console.print(f"  DB 복사: [dim]{DB_PATH} → {dest_db}[/]")
    shutil.copy2(str(DB_PATH), str(dest_db))

    console.print("  [dim]railway up 실행 중...[/]")
    result = subprocess.run(
        ["railway", "up"],
        cwd=str(project_dir),
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode == 0:
        console.print("  [green]Railway 배포 성공[/]")
    else:
        console.print(f"  [red]Railway 배포 실패:[/] {result.stderr}")


def notify(title, message):
    """macOS 알림 전송"""
    import subprocess
    try:
        subprocess.run([
            "osascript", "-e",
            f'display notification "{message}" with title "Alpha Lab" subtitle "{title}" sound name "Glass"'
        ], timeout=5)
    except Exception:
        pass


if __name__ == "__main__":
    main()
