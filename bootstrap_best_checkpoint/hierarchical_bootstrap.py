"""Hierarchical bootstrap utilities for RL evaluation results.

This module loads per-seed evaluation CSVs (named like
``episode_info_<algo>_<seed>_*.csv``), converts the ``is_success`` column
into binary outcomes, and estimates uncertainty in the mean success rate
using a two-level bootstrap that resamples both seeds and goals.

We treat seeds as random effects: each seed is an IID draw from a
population of training runs. For comparing algorithms, we assume:
  * Training seeds are independent across algorithms (unpaired).
  * The evaluation set (goals) is the same across algorithms, so
    we use paired resampling of evaluation points.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

ALGO_TAGS = ("bc", "h_ppo", "ppo")
FILENAME_RE = re.compile(r"episode_info_(?P<algo>bc|h_ppo|ppo)_(?P<seed>\d+)_.*\.csv$")


def _coerce_success(series: pd.Series) -> np.ndarray:
    """Return a binary numpy array for the ``is_success`` column."""
    if series.empty:
        raise ValueError("Encountered an empty is_success column.")

    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(dtype=int)

    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int).to_numpy()

    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"1", "true", "t", "yes"}).astype(int).to_numpy()


def load_success_by_seed(run_type: str, data_dir: Path) -> Dict[int, np.ndarray]:
    """Load success indicators grouped by seed for the requested run type."""
    run_type = run_type.lower()
    if run_type not in ALGO_TAGS:
        raise ValueError(f"Unsupported run type '{run_type}'. Choose from {ALGO_TAGS}.")

    data_dir = Path(data_dir)
    success_by_seed: Dict[int, np.ndarray] = {}

    for csv_path in sorted(data_dir.glob(f"episode_info_{run_type}_*.csv")):
        match = FILENAME_RE.match(csv_path.name)
        if not match or match.group("algo") != run_type:
            continue

        seed = int(match.group("seed"))
        df = pd.read_csv(csv_path)
        if "is_success" not in df.columns:
            raise KeyError(f"Column 'is_success' missing in {csv_path}.")

        success_array = _coerce_success(df["is_success"])
        success_by_seed[seed] = success_array

    if not success_by_seed:
        raise FileNotFoundError(
            f"No CSV files found for run type '{run_type}' in '{data_dir}'."
        )

    return success_by_seed


def hierarchical_bootstrap(
    success_by_seed: Mapping[int, np.ndarray],
    num_bootstrap: int = 10_000,
    rng: np.random.Generator | int | None = None,
) -> Tuple[float, Tuple[float, float], float, np.ndarray]:
    """Estimate success-rate uncertainty via hierarchical bootstrapping.

    We treat seeds as random effects. Each bootstrap replicate:
      1) Resamples seeds with replacement.
      2) For each selected seed, resamples goals within that seed.

    Returns a tuple of ``(mean_estimate, (ci_low, ci_high), std, distribution)``.
    """
    if not success_by_seed:
        raise ValueError("No success data supplied for bootstrapping.")

    rng = np.random.default_rng(rng)
    seeds = list(success_by_seed.keys())
    bootstrap_means = np.empty(num_bootstrap, dtype=float)

    for b in range(num_bootstrap):
        seed_indices = rng.integers(0, len(seeds), len(seeds))
        total_success = 0.0
        total_goals = 0

        for idx in seed_indices:
            goals = success_by_seed[seeds[idx]]
            if goals.size == 0:
                raise ValueError(f"Seed {seeds[idx]} contains zero goals; cannot resample.")
            sampled_goals = rng.choice(goals, size=goals.size, replace=True)
            total_success += sampled_goals.sum()
            total_goals += sampled_goals.size

        bootstrap_means[b] = total_success / total_goals

    mean_estimate = float(bootstrap_means.mean())
    ci_low, ci_high = np.percentile(bootstrap_means, [2.5, 97.5])
    std_dev = float(bootstrap_means.std(ddof=1))

    return mean_estimate, (float(ci_low), float(ci_high)), std_dev, bootstrap_means


def bootstrap_difference(
    success_a: Mapping[int, np.ndarray],
    success_b: Mapping[int, np.ndarray],
    num_bootstrap: int = 10_000,
    rng: np.random.Generator | int | None = None,
) -> Tuple[float, Tuple[float, float], float, np.ndarray]:
    """Bootstrap the difference in mean success rates between two algorithms.

    Assumptions:
      * All seeds are random / independent within each algorithm.
      * Training seeds are NOT paired across algorithms.
      * The evaluation set (goals) is the same across algorithms, so for
        each bootstrap replicate we resample a shared set of goal indices
        and use these for both algorithms (paired goals).

    Each bootstrap replicate:
      1) Draw a shared multiset of goal indices (paired across algorithms).
      2) Resample seeds with replacement for algorithm A, aggregate success
         over the shared goal indices.
      3) Resample seeds with replacement for algorithm B, aggregate success
         over the same goal indices.
      4) Compute mean_A - mean_B.

    Returns:
        (mean_delta, (ci_low, ci_high), std_delta, bootstrap_deltas)
    """
    if not success_a or not success_b:
        raise ValueError("Both algorithms need non-empty success data for comparison.")

    # Ensure consistent number of evaluation goals across seeds and algorithms.
    lens_a = {arr.size for arr in success_a.values()}
    lens_b = {arr.size for arr in success_b.values()}

    if len(lens_a) != 1:
        raise ValueError(
            f"Algorithm A has seeds with differing numbers of evaluation goals: {lens_a}"
        )
    if len(lens_b) != 1:
        raise ValueError(
            f"Algorithm B has seeds with differing numbers of evaluation goals: {lens_b}"
        )

    n_goals_a = next(iter(lens_a))
    n_goals_b = next(iter(lens_b))

    if n_goals_a != n_goals_b:
        raise ValueError(
            f"Algorithms have different numbers of evaluation goals: "
            f"{n_goals_a} (A) vs {n_goals_b} (B). "
            "Paired goal resampling requires them to be equal."
        )

    n_goals = n_goals_a

    rng = np.random.default_rng(rng)
    seeds_a = list(success_a.keys())
    seeds_b = list(success_b.keys())

    bootstrap_deltas = np.empty(num_bootstrap, dtype=float)

    for b in range(num_bootstrap):
        # Shared goal indices across BOTH algorithms and ALL seeds in this replicate.
        goal_indices = rng.integers(0, n_goals, n_goals)

        # Algorithm A
        seed_indices_a = rng.integers(0, len(seeds_a), len(seeds_a))
        total_a = 0.0
        goals_a = 0

        for idx in seed_indices_a:
            seed = seeds_a[idx]
            goals_a_seed = success_a[seed]
            if goals_a_seed.size == 0:
                raise ValueError(
                    f"Seed {seed} has zero goals for algorithm A; cannot resample."
                )
            sampled_a = goals_a_seed[goal_indices]
            total_a += sampled_a.sum()
            goals_a += sampled_a.size

        # Algorithm B
        seed_indices_b = rng.integers(0, len(seeds_b), len(seeds_b))
        total_b = 0.0
        goals_b = 0

        for idx in seed_indices_b:
            seed = seeds_b[idx]
            goals_b_seed = success_b[seed]
            if goals_b_seed.size == 0:
                raise ValueError(
                    f"Seed {seed} has zero goals for algorithm B; cannot resample."
                )
            sampled_b = goals_b_seed[goal_indices]
            total_b += sampled_b.sum()
            goals_b += sampled_b.size

        mean_a = total_a / goals_a
        mean_b = total_b / goals_b
        bootstrap_deltas[b] = mean_a - mean_b

    mean_delta = float(bootstrap_deltas.mean())
    ci_low, ci_high = np.percentile(bootstrap_deltas, [2.5, 97.5])
    std_delta = float(bootstrap_deltas.std(ddof=1))

    return mean_delta, (float(ci_low), float(ci_high)), std_delta, bootstrap_deltas


def run_bootstrap(
    run_type: str,
    data_dir: Path,
    num_bootstrap: int = 1000000,
    rng_seed: int | None = None,
) -> Dict[str, float | Tuple[float, float]]:
    """Helper to load data and compute summary statistics for one run type."""
    success_by_seed = load_success_by_seed(run_type, data_dir)
    mean_est, (ci_low, ci_high), std_dev, _ = hierarchical_bootstrap(
        success_by_seed, num_bootstrap=num_bootstrap, rng=rng_seed
    )
    return {
        "run_type": run_type,
        "mean": mean_est,
        "ci": (ci_low, ci_high),
        "std": std_dev,
        "seeds": len(success_by_seed),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Directory containing episode_info CSV files.",
    )
    parser.add_argument(
        "--run-types",
        nargs="*",
        default=list(ALGO_TAGS),
        help="Subset of run types to analyze (default: all).",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10_000,
        help="Number of hierarchical bootstrap replicates.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible resampling.",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("ALGO_A", "ALGO_B"),
        help="Optionally bootstrap the success-rate difference between two run types.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summaries = []

    for run_type in args.run_types:
        try:
            summary = run_bootstrap(
                run_type=run_type,
                data_dir=args.data_dir,
                num_bootstrap=args.bootstrap_samples,
                rng_seed=args.seed,
            )
            summaries.append(summary)
        except FileNotFoundError:
            print(f"[skip] No files found for run type '{run_type}'.")
        except Exception as exc:  # pragma: no cover - surfaced to CLI immediately
            print(f"[error] Failed to process '{run_type}': {exc}")

    if not summaries:
        print("No run types produced results.")
        return

    for summary in summaries:
        ci_low, ci_high = summary["ci"]
        print(
            f"{summary['run_type']}: mean={summary['mean']:.4f}, "
            f"std={summary['std']:.4f}, ci95=({ci_low:.4f}, {ci_high:.4f}), "
            f"seeds={summary['seeds']}"
        )

    if args.compare:
        run_a, run_b = [rt.lower() for rt in args.compare]
        try:
            success_a = load_success_by_seed(run_a, args.data_dir)
            success_b = load_success_by_seed(run_b, args.data_dir)
            mean_delta, (ci_low, ci_high), std_delta, _ = bootstrap_difference(
                success_a,
                success_b,
                num_bootstrap=args.bootstrap_samples,
                rng=args.seed,
            )
            print(
                f"delta {run_a} - {run_b}: mean={mean_delta:.4f}, "
                f"std={std_delta:.4f}, ci95=({ci_low:.4f}, {ci_high:.4f}), "
                f"seeds_a={len(success_a)}, seeds_b={len(success_b)}"
            )
        except FileNotFoundError as exc:
            print(f"[skip] {exc}")
        except Exception as exc:  # pragma: no cover
            print(f"[error] Failed to compare '{run_a}' vs '{run_b}': {exc}")


if __name__ == "__main__":
    main()
