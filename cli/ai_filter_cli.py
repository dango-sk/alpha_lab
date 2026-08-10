"""AI 종목 필터 CLI (Rich UI).

사용:
    python -m cli.ai_filter_cli --date 2026-05-20
    python -m cli.ai_filter_cli --date 2026-05-20 --top-n 30 --max-rounds 5

직렬 디베이트 (tech → news 순). 토큰 스트리밍 X, 응답 완료 시점에 패널 갱신.
"""
import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table
from rich.text import Text


PHASE_LABEL = {
    "collect_tech": "기술적 지표 계산",
    "collect_news": "뉴스 수집 (Gemini)",
    "agent_r0_tech": "R0 기술 에이전트",
    "agent_r0_news": "R0 뉴스 에이전트",
    "debate": "디베이트 시작",
    "judge": "Judge (최종 결정)",
    "save": "로그 저장",
}


class FilterUI:
    """파이프라인 콜백을 받아 Rich Live 화면을 갱신."""

    PANEL_TAIL_LINES = 18  # 패널에 보여줄 마지막 N줄

    def __init__(self, calc_date: str, max_rounds: int, threshold: float, n_stocks: int):
        self.console = Console()
        self.calc_date = calc_date
        self.max_rounds = max_rounds
        self.threshold = threshold
        self.n_stocks = n_stocks

        self.t0 = time.time()
        self.phase = "준비"
        self.current_agent: str | None = None  # "tech" | "news" | "judge"

        # 각 에이전트 상태: "idle" | "streaming" | "done"
        self.agent_state = {"tech": "idle", "news": "idle", "judge": "idle"}
        self.agent_buffer = {"tech": "", "news": "", "judge": ""}  # 스트리밍 누적
        self.agent_done_text = {"tech": "", "news": "", "judge": ""}  # 완료 요약

        self.agreement_history: list[tuple[int, float]] = []
        self.current_round = 0

    # ── 콜백 ────────────────────────────────────────────
    def on_phase(self, name: str, meta: dict):
        if name == "collect_news_item":
            i = meta.get("i", 0)
            total = meta.get("total", 0)
            self.phase = f"뉴스 수집 ({i}/{total}) {meta.get('name','')}({meta.get('code','')})"
            return
        self.phase = PHASE_LABEL.get(name, name)
        if name == "agent_r0_tech":
            self._start_agent("tech")
        elif name == "agent_r0_news":
            self._start_agent("news")
        elif name == "debate":
            pass  # 라운드 시작은 on_token 첫 도착으로 감지
        elif name == "judge":
            self._start_agent("judge")

    def _start_agent(self, label: str):
        self.current_agent = label
        self.agent_state[label] = "streaming"
        self.agent_buffer[label] = ""

    def on_token(self, agent_label: str, delta: str):
        """토큰 한 조각 도착 (스트리밍)."""
        # 라운드 도중 새 에이전트 시작 감지 (debate_loop는 phase 안 보냄)
        if self.agent_state[agent_label] != "streaming":
            self._start_agent(agent_label)
        self.agent_buffer[agent_label] += delta

    def on_agent_done(self, agent_label: str, full_text: str):
        """에이전트 응답 완료. 요약 모드로 전환."""
        self.agent_state[agent_label] = "done"
        self.agent_done_text[agent_label] = self._summarize_response(full_text)

    def on_round_end(self, round_result: dict):
        r = round_result.get("round", 0)
        rate = round_result.get("agreement_rate", 0.0)
        self.current_round = r
        self.agreement_history.append((r, rate))

    # ── 화면 구성 ───────────────────────────────────────
    def _summarize_response(self, text: str) -> str:
        """응답 텍스트에서 핵심만 추출 (Top10 코드 + 짧은 사유)."""
        import json
        import re

        # JSON 파싱 시도
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return text[:400]
        try:
            data = json.loads(m.group())
        except Exception:
            return text[:400]

        lines = []
        top10 = data.get("top_10") or data.get("final_portfolio") or []
        for i, item in enumerate(top10, 1):
            code = item.get("stock_code", "?")
            name = item.get("stock_name", "?")
            weight = item.get("weight_pct")
            score = item.get("score")
            reason = item.get("reason") or ""
            tag = f"{weight}%" if weight is not None else (f"s={score}" if score is not None else "")
            lines.append(f"  {i:2d}. {name}({code}) {tag} — {reason}")
        if data.get("rebuttal"):
            lines.append(f"\n[반론] {data['rebuttal'][:200]}")
        if data.get("judge_reasoning"):
            lines.append(f"\n[Judge] {data['judge_reasoning'][:300]}")
        return "\n".join(lines) if lines else text[:400]

    def _header(self) -> Panel:
        elapsed = int(time.time() - self.t0)
        mins, secs = divmod(elapsed, 60)
        round_info = f"R{self.current_round}/{self.max_rounds}" if self.current_round else "R0 준비"
        title = (f"[bold]AI Stock Filter[/bold]  "
                 f"{self.calc_date}  ·  입력 {self.n_stocks}종목  ·  "
                 f"{round_info}  ·  ⏱ {mins:02d}:{secs:02d}")
        sub = f"[dim]Phase:[/dim] {self.phase}"
        return Panel(Text.from_markup(f"{title}\n{sub}"), border_style="blue")

    def _agent_panel_body(self, label: str, style: str) -> Text:
        state = self.agent_state[label]
        if state == "idle":
            return Text("(대기)", style="dim", overflow="fold")
        if state == "done":
            return Text(self.agent_done_text[label] or "(빈 응답)",
                        style=style, overflow="fold")
        # streaming
        buf = self.agent_buffer[label]
        if not buf:
            return Text("⏳ 응답 시작 대기...", style="yellow")
        # ```json 코드블록 이후는 가리기 (JSON 정리 단계)
        json_start = buf.find("```json")
        if json_start >= 0:
            visible = buf[:json_start].rstrip()
            suffix = "\n\n[📋 JSON 정리 중...]"
        else:
            visible = buf
            suffix = ""
        # 마지막 N줄만 표시
        lines = visible.splitlines()
        tail = lines[-self.PANEL_TAIL_LINES:]
        return Text("\n".join(tail) + suffix, style="yellow", overflow="fold")

    def _agents_panel(self) -> Panel:
        tech = Panel(self._agent_panel_body("tech", "cyan"),
                     title="🔧 기술 분석가", border_style="cyan")
        news = Panel(self._agent_panel_body("news", "magenta"),
                     title="📰 뉴스 분석가", border_style="magenta")
        layout = Layout()
        layout.split_row(Layout(tech, name="tech"), Layout(news, name="news"))
        return Panel(layout, height=22, border_style="dim")

    def _agreement_panel(self) -> Panel:
        if not self.agreement_history:
            body = Text("(아직 라운드 없음)", style="dim")
        else:
            lines = []
            for r, rate in self.agreement_history:
                bar_len = int(rate * 20)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                marker = " ✓" if rate >= self.threshold else ""
                lines.append(f"R{r}: {rate*100:5.1f}% {bar}{marker}")
            body = Text("\n".join(lines))
        return Panel(body, title=f"📊 합의율 (수렴 ≥ {self.threshold*100:.0f}%)",
                     border_style="yellow")

    def _judge_panel(self) -> Panel:
        return Panel(self._agent_panel_body("judge", "green"),
                     title="⚖️  Judge", border_style="green", height=10)

    def render(self) -> Group:
        return Group(
            self._header(),
            self._agents_panel(),
            self._agreement_panel(),
            self._judge_panel(),
        )

    # ── 종료 후 최종 리포트 ─────────────────────────────
    def print_final_report(self, result: dict):
        self.console.print()
        self.console.rule("[bold green]최종 포트폴리오")

        table = Table(show_lines=True, header_style="bold")
        table.add_column("#", justify="right", style="dim", no_wrap=True)
        table.add_column("종목명", no_wrap=True)
        table.add_column("코드", style="dim", no_wrap=True)
        table.add_column("비중", justify="right", style="cyan", no_wrap=True)
        table.add_column("신뢰", justify="center", no_wrap=True)
        table.add_column("Tech", justify="center", no_wrap=True)
        table.add_column("News", justify="center", no_wrap=True)
        table.add_column("팩터#", justify="right", style="dim", no_wrap=True)
        table.add_column("사유", overflow="fold")

        for i, item in enumerate(result.get("final_portfolio", []), 1):
            table.add_row(
                str(i),
                str(item.get("stock_name", "?")),
                str(item.get("stock_code", "?")),
                f"{item.get('weight_pct', '?')}%",
                str(item.get("confidence", "?")),
                "✓" if item.get("tech_selected_final") else "·",
                "✓" if item.get("news_selected_final") else "·",
                str(item.get("factor_rank", "?")),
                item.get("reason") or "",
            )
        self.console.print(table)

        judge_reason = result.get("judge", {}).get("judge_reasoning")
        if judge_reason:
            self.console.print(Panel(judge_reason, title="Judge 종합 사유",
                                     border_style="green"))

        debate = result.get("debate", {})
        elapsed = int(time.time() - self.t0)
        mins, secs = divmod(elapsed, 60)
        self.console.print(
            f"\n[dim]디베이트 {debate.get('total_rounds')}라운드 · "
            f"converged={debate.get('converged')} · "
            f"final_agreement={debate.get('final_agreement_rate', 0):.2f} · "
            f"⏱ {mins:02d}:{secs:02d}[/dim]"
        )


def _load_stocks_from_cache(name: str, cache_date: str) -> tuple[list[tuple[str, float]], str]:
    """alpha_lab.backtest_cache에서 특정 전략의 특정 날짜 holdings를 로드.

    Returns: (stocks, resolved_date)
        stocks: [(stock_code, factor_score), ...]
        resolved_date: cache_date 이하 가장 가까운 리밸런싱 날짜
    """
    import os
    import psycopg2
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    cur = conn.cursor()
    cur.execute("SELECT holdings_json FROM alpha_lab.backtest_cache WHERE name=%s LIMIT 1", (name,))
    row = cur.fetchone()
    conn.close()
    if not row:
        raise SystemExit(f"backtest_cache에 전략 '{name}' 없음. id로 직접 SELECT 해서 name 확인 권장.")
    holdings = row[0]
    if cache_date in holdings:
        resolved = cache_date
    else:
        candidates = sorted(d for d in holdings.keys() if d <= cache_date)
        if not candidates:
            raise SystemExit(f"'{name}'의 holdings 중 {cache_date} 이하 날짜 없음. "
                             f"가용 날짜: {sorted(holdings.keys())[:5]}...")
        resolved = candidates[-1]
    items = holdings[resolved]
    stocks: list[tuple[str, float]] = []
    for it in items:
        if isinstance(it, list) and len(it) >= 2:
            stocks.append((str(it[0]), float(it[1])))
        elif isinstance(it, dict):
            stocks.append((str(it.get("stock_code", it.get("code"))),
                           float(it.get("factor_score", it.get("score", 0)))))
    return stocks, resolved


def main():
    parser = argparse.ArgumentParser(description="AI 종목 필터 (디베이트 + Rich UI)")
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"),
                        help="기준일 (YYYY-MM-DD)")
    parser.add_argument("--top-n", type=int, default=30, help="입력 종목 수")
    parser.add_argument("--strategy", default=None, help="전략 파일 경로 (없으면 기본 전략)")
    parser.add_argument("--from-cache", default=None,
                        help="alpha_lab.backtest_cache의 전략 name. "
                             "지정하면 --strategy 무시하고 캐시된 holdings 사용. "
                             "예: --from-cache '레짐조합_수정전략_코스피_cap30%%_top30_tx30bp_월간↑_cap30%%_손절율15%%(고점)↓'")
    parser.add_argument("--cache-date", default=None,
                        help="--from-cache와 함께 사용. 이 날짜 이하 가장 가까운 리밸런싱 날짜 선택. "
                             "기본은 --date.")
    parser.add_argument("--max-rounds", type=int, default=5, help="디베이트 최대 라운드")
    parser.add_argument("--threshold", type=float, default=0.8,
                        help="조기 종료 합의율 (0~1)")
    parser.add_argument("--no-ui", action="store_true",
                        help="Rich UI 없이 로그만 (디버깅용)")
    parser.add_argument("--no-news-cache", action="store_true",
                        help="뉴스 캐시 무시하고 Gemini로 새로 수집")
    args = parser.parse_args()

    load_dotenv(Path(__file__).parent.parent / ".env")
    logging.basicConfig(
        level=logging.INFO if args.no_ui else logging.WARNING,
        format="%(asctime)s %(message)s",
    )

    from lib.ai_stock_filter import run_ai_filter_with_debate
    from lib.db import get_conn
    from lib.factor_engine import (
        DEFAULT_STRATEGY_CODE,
        code_to_module,
        load_strategy_module,
        score_stocks_from_strategy,
    )

    conn = get_conn()

    if args.from_cache:
        cache_date = args.cache_date or args.date
        stocks, resolved = _load_stocks_from_cache(args.from_cache, cache_date)
        print(f"[from-cache] '{args.from_cache[:50]}...' @ {resolved} → {len(stocks)}종목")
        stocks = stocks[: args.top_n]
    else:
        if args.strategy:
            strategy = load_strategy_module(args.strategy)
        else:
            strategy = code_to_module(DEFAULT_STRATEGY_CODE)
        stocks = score_stocks_from_strategy(conn, args.date, strategy)
        stocks = stocks[: args.top_n]

    ui = FilterUI(
        calc_date=args.date,
        max_rounds=args.max_rounds,
        threshold=args.threshold,
        n_stocks=len(stocks),
    )

    if args.no_ui:
        result = run_ai_filter_with_debate(
            stocks=stocks,
            calc_date=args.date,
            conn=conn,
            max_rounds=args.max_rounds,
            convergence_threshold=args.threshold,
            use_news_cache=not args.no_news_cache,
        )
        print(result)
        return

    with Live(
        ui.render(),
        console=ui.console,
        refresh_per_second=20,
        screen=True,
        redirect_stdout=True,
        redirect_stderr=True,
    ) as live:
        def refresh_phase(name, meta):
            ui.on_phase(name, meta)
            live.update(ui.render())

        def refresh_token(label, delta):
            ui.on_token(label, delta)
            live.update(ui.render())

        def refresh_done(label, full_text):
            ui.on_agent_done(label, full_text)
            live.update(ui.render())

        def refresh_round(round_result):
            ui.on_round_end(round_result)
            live.update(ui.render())

        result = run_ai_filter_with_debate(
            stocks=stocks,
            calc_date=args.date,
            conn=conn,
            max_rounds=args.max_rounds,
            convergence_threshold=args.threshold,
            on_phase=refresh_phase,
            on_token=refresh_token,
            on_agent_done=refresh_done,
            on_round_end=refresh_round,
            use_news_cache=not args.no_news_cache,
        )

    ui.print_final_report(result)


if __name__ == "__main__":
    main()
