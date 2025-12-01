import os
import pandas as pd

def summarize_run(run_name, base_dir="."):
    """
    Summarize evaluation CSVs for a given run_name.
    Computes success rate, average total reward, and component-level stats.
    """
    prefix = f"episode_info_{run_name}_"
    suffix = "_uniform.csv"

    files = [
        f for f in os.listdir(base_dir)
        if f.startswith(prefix) and f.endswith(suffix)
    ]
    files.sort()

    if not files:
        raise FileNotFoundError(f"No CSV files found for run_name '{run_name}' in {base_dir}")

    summaries = []

    for fname in files:
        # Extract checkpoint number
        checkpoint_str = fname[len(prefix):-len(suffix)]
        try:
            checkpoint = int(checkpoint_str)
        except ValueError:
            continue

        path = os.path.join(base_dir, fname)
        df = pd.read_csv(path)

        # --- SUCCESS RATE (handle bool/strings safely) ---
        success_rate = None
        success_std = None
        if "is_success" in df.columns:
            # True/False -> 1/0, "0"/"1" -> 0/1
            success_numeric = pd.to_numeric(df["is_success"], errors="coerce")
            success_rate = float(success_numeric.mean())
            success_std = float(success_numeric.std())

        # --- NUMERIC COLUMNS FOR REWARDS ---
        num_df = df.select_dtypes(include="number")

        # Reward columns (all numeric cols ending with "_reward")
        reward_cols = [c for c in num_df.columns if c.endswith("_reward")]

        # Total reward = sum of all reward components per episode
        if reward_cols:
            total_reward = num_df[reward_cols].sum(axis=1)
        else:
            total_reward = pd.Series([0] * len(df))

        metrics = {
            "run_name": run_name,
            "checkpoint": checkpoint,
            "n_episodes": len(df),
            "success_rate": success_rate,
            "success_rate_std": success_std,
            "avg_total_reward": float(total_reward.mean()),
            "total_reward_std": float(total_reward.std()),
        }

        # Add mean/std for each reward component
        for col in reward_cols:
            metrics[f"{col}_mean"] = float(num_df[col].mean())
            metrics[f"{col}_std"] = float(num_df[col].std())

        summaries.append(metrics)

    summary_df = pd.DataFrame(summaries).sort_values("checkpoint").reset_index(drop=True)
    return summary_df

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

print(summarize_run("bc_3", base_dir="bootstrap_checkpoint_results"))
