from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mplconfig")

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class NChain:
    """NChain environment with optional action flips."""

    n_states: int = 5
    flip_alpha: float = 0.0
    small_reward: float = 2.0
    large_reward: float = 10.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.flip_alpha <= 1.0:
            raise ValueError("flip_alpha must be between 0 and 1.")
        self.state = 0

    def reset(self) -> int:
        self.state = 0
        return self.state

    def step(self, action: int, rng: np.random.Generator) -> tuple[int, float]:
        if action not in (0, 1):
            raise ValueError("action must be 0(forward) or 1(backward).")

        if rng.random() < self.flip_alpha:
            action = 1 - action

        if action == 1:
            self.state = 0
            return self.state, self.small_reward

        if self.state == self.n_states - 1:
            return self.state, self.large_reward

        self.state += 1
        return self.state, 0.0


def choose_action(q_table: np.ndarray, state: int, epsilon: float, rng: np.random.Generator) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(0, q_table.shape[1]))
    return int(np.argmax(q_table[state]))


def run_q_learning(
    *,
    flip_alpha: float = 0.0,
    episodes: int = 4000,
    max_steps: int = 100,
    learning_rate: float = 0.1,
    discount: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.01,
    epsilon_decay: float = 0.995,
    seed: int = 7,
) -> dict[str, np.ndarray | list[float]]:
    rng = np.random.default_rng(seed)
    env = NChain(flip_alpha=flip_alpha)
    q_table = np.zeros((env.n_states, 2), dtype=float)

    episode_average_rewards: list[float] = []
    q_4_1_history: list[float] = []
    q_table_history: list[np.ndarray] = []
    epsilon = epsilon_start

    for _ in range(episodes):
        state = env.reset()
        total_reward = 0.0

        for _step in range(max_steps):
            action = choose_action(q_table, state, epsilon, rng)
            next_state, reward = env.step(action, rng)
            total_reward += reward

            td_target = reward + discount * np.max(q_table[next_state])
            q_table[state, action] += learning_rate * (td_target - q_table[state, action])
            state = next_state

        episode_average_rewards.append(total_reward / max_steps)
        q_4_1_history.append(float(q_table[4, 1]))
        q_table_history.append(q_table.copy())
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return {
        "episode_average_rewards": episode_average_rewards,
        "q_4_1_history": q_4_1_history,
        "q_table_history": q_table_history,
        "last_q_table": q_table,
    }


def moving_average(values: list[float], window: int = 100) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if window <= 1:
        return arr
    kernel = np.ones(window) / window
    padded = np.pad(arr, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def save_outputs(results: dict[str, np.ndarray | list[float]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    rewards = np.asarray(results["episode_average_rewards"], dtype=float)
    q_4_1 = np.asarray(results["q_4_1_history"], dtype=float)
    last_q_table = np.asarray(results["last_q_table"], dtype=float)
    episodes = np.arange(1, len(rewards) + 1)

    np.savetxt(
        output_dir / "average_reward.csv",
        np.column_stack([episodes, rewards]),
        delimiter=",",
        header="episode,average_reward",
        comments="",
    )
    np.savetxt(
        output_dir / "q_4_1_history.csv",
        np.column_stack([episodes, q_4_1]),
        delimiter=",",
        header="episode,Q[4,1]",
        comments="",
    )
    np.savetxt(
        output_dir / "last_q_table.csv",
        last_q_table,
        delimiter=",",
        header="action_0_forward,action_1_backward",
        comments="",
    )

    q_history = np.asarray(results["q_table_history"], dtype=float)
    np.save(output_dir / "q_table_history.npy", q_history)

    summary = {
        "episodes": int(len(rewards)),
        "final_average_reward": float(rewards[-1]),
        "mean_average_reward_last_100": float(np.mean(rewards[-100:])),
        "last_q_table": last_q_table.tolist(),
        "last_Q[4,1]": float(last_q_table[4, 1]),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plt.figure(figsize=(9, 5))
    plt.plot(episodes, rewards, alpha=0.35, label="episode average reward")
    plt.plot(episodes, moving_average(rewards.tolist(), 100), linewidth=2, label="100-episode moving average")
    plt.xlabel("Episode")
    plt.ylabel("Average reward per step")
    plt.title("NChain Q-Learning Average Reward (flip alpha = 0)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "average_reward_plot.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(episodes, q_4_1, color="tab:orange")
    plt.xlabel("Episode")
    plt.ylabel("Q[4, 1]")
    plt.title("Q[4, 1] Value During Q-Learning")
    plt.tight_layout()
    plt.savefig(output_dir / "q_4_1_plot.png", dpi=160)
    plt.close()


def main() -> None:
    output_dir = Path("nchain_results")
    results = run_q_learning(flip_alpha=0.0, episodes=4000)
    save_outputs(results, output_dir)

    last_q_table = np.asarray(results["last_q_table"], dtype=float)
    print("Last Q-table")
    print("Rows: states 0..4, Columns: action 0(forward), action 1(backward)")
    print(np.round(last_q_table, 4))
    print(f"\nSaved results to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
