"""Plot rollout reward curves for all seeds belonging to an algorithm.

Usage example:
    python plot_reward_curves.py --algo ppo --out ppo_reward.png
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd

# plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams['font.size'] = 24
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['legend.fontsize']  = 24

MAX_STEP = 2_000_000


def summarize_series(series: List[pd.Series]) -> tuple[pd.Series, pd.Series]:
    """Align seed curves and return capped mean/std."""
    combined = pd.concat(series, axis=1).sort_index()
    combined = combined.loc[combined.index <= MAX_STEP]
    mean_curve = combined.mean(axis=1, skipna=True)
    std_curve = combined.std(axis=1, skipna=True, ddof=0).fillna(0.0)
    return mean_curve, std_curve


def find_reward_column(columns: List[str]) -> str:
    for col in columns:
        if "ep_rew_mean" in col.lower():
            return col
    raise ValueError("Could not locate a column containing 'ep_rew_mean'.")


def load_curve(
    csv_path: Path,
    scale_steps: float = 1.0,
    smooth_window: int | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "global_step" not in df.columns:
        raise KeyError(f"Column 'global_step' missing in {csv_path}.")
    reward_col = find_reward_column(df.columns.tolist())
    curve = df[["global_step", reward_col]].copy()
    curve = curve.dropna(subset=["global_step", reward_col])
    if scale_steps != 1.0:
        curve["global_step"] = curve["global_step"] * scale_steps
    if smooth_window and smooth_window > 1:
        curve[reward_col] = (
            curve[reward_col]
            .rolling(window=smooth_window, min_periods=1, center=False)
            .mean()
        )
    curve.sort_values("global_step", inplace=True)
    curve.rename(columns={reward_col: "ep_rew_mean"}, inplace=True)
    return curve


def plot_algorithm_curves(
    algos: List[str],
    data_dir: Path,
    out_path: Path | None = None,
    show_seeds: bool = False,
    smooth_window: int | None = None,
) -> None:
    data_dir = Path(data_dir)
    if not algos:
        raise ValueError("At least one algorithm name must be provided.")

    plt.figure(figsize=(10, 6), dpi=120)
    ax = plt.gca()
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

    def plot_algo(algo_name: str, style_idx: int) -> None:
        csv_paths = sorted(data_dir.glob(f"{algo_name}*_*.csv"))
        if not csv_paths:
            print(f"[skip] No CSV files found for algo '{algo_name}'.")
            return
        scale_steps = 1.0 if algo_name.lower().startswith("ppo") else 0.5
        series = []
        for csv_path in csv_paths:
            curve = load_curve(
                csv_path,
                scale_steps=scale_steps,
                smooth_window=smooth_window,
            )
            series.append(curve.set_index("global_step")["ep_rew_mean"].rename(csv_path.stem))
            if show_seeds:
                ax.plot(
                    curve["global_step"],
                    curve["ep_rew_mean"],
                    alpha=0.15,
                    linewidth=0.6,
                    color=color_cycle[style_idx % len(color_cycle)] if color_cycle else f"C{style_idx}",
                )
        if not series:
            return
        mean_curve, std_curve = summarize_series(series)
        color = color_cycle[style_idx % len(color_cycle)] if color_cycle else f"C{style_idx}"
        ax.plot(
            mean_curve.index,
            mean_curve.values,
            label=algo_name,
            color=color,
            linewidth=2.0,
        )
        ax.fill_between(
            mean_curve.index,
            mean_curve.values - std_curve.values,
            mean_curve.values + std_curve.values,
            color=color,
            alpha=0.15,
        )

    for idx, algo_name in enumerate(algos):
        plot_algo(algo_name, idx % 10)

    title_suffix = (
        f" (MA window={smooth_window})" if smooth_window and smooth_window > 1 else ""
    )
    title_algos = ", ".join(algos)
    ax.set_title(f"Training Rewards")
    ax.set_xlabel("Simulation steps")
    ax.set_ylabel("Episode Reward (mean)")
    ax.set_ylim(bottom=1)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.4, linestyle="--", linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_alpha(0.3)
    plt.tight_layout()

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path)
        print(f"Saved figure to {out_path}")
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--algos",
        nargs="+",
        help="Algorithm prefixes (first one is primary with shading; others overlay).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Directory containing reward CSV files.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to save the plot. If omitted, the plot window is shown.",
    )
    parser.add_argument(
        "--show-seeds",
        action="store_true",
        help="If set, plot each seed curve faintly in addition to the mean/std band.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=None,
        help="Optional moving-average window size applied to ep_rew_mean before aggregation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_algorithm_curves(
        args.algos,
        data_dir=args.data_dir,
        out_path=args.out,
        show_seeds=args.show_seeds,
        smooth_window=args.smooth_window,
    )


if __name__ == "__main__":
    main()
