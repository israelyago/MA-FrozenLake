from pathlib import Path

from matplotlib import pyplot as plt
import pandas as pd

plt.rcParams.update(
    {
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "text.usetex": False,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
    }
)


def get_experiment_dirs(artifacts_dir: Path):
    """Return only directories inside artifacts/."""
    return [d for d in artifacts_dir.iterdir() if d.is_dir()]


def load_metrics(exp_dir: Path, filename: str) -> pd.DataFrame | None:
    """Load a metrics file (metrics.csv or smooth.csv), return None if missing."""
    file_path = exp_dir / filename
    if not file_path.exists():
        print(f"⚠️ {filename} not found in '{exp_dir.name}', skipping.")
        return None
    return pd.read_csv(file_path)


def format_label(exp_dir: Path) -> str:
    """Human-friendly experiment name."""
    return exp_dir.name.replace("_", " ")


def compute_confidence_intervals(df: pd.DataFrame):
    """Compute mean ± 1.96 * std confidence bands if std exists."""
    if "std_reward" in df.columns:
        df["upper"] = df["mean_reward"] + 1.96 * df["std_reward"]
        df["lower"] = df["mean_reward"] - 1.96 * df["std_reward"]
    return df


def smooth(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Apply rolling-window smoothing to mean + CI."""
    df["smooth_mean_reward"] = df["mean_reward"].rolling(window, min_periods=1).mean()

    if "upper" in df.columns:
        df["smooth_upper"] = df["upper"].rolling(window, min_periods=1).mean()
        df["smooth_lower"] = df["lower"].rolling(window, min_periods=1).mean()

    return df


def setup_plot(title: str):
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Mean Reward")
    plt.grid(alpha=0.3, linestyle="--")


def save_plot(path: Path):
    plt.legend()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📊 Saved plot to '{path}'")


def plot_ci_curve(df: pd.DataFrame, label: str, smooth: bool):
    """Plot mean curve + CI band."""
    if smooth:
        mean_col = "smooth_mean_reward"
        upper_col = "smooth_upper"
        lower_col = "smooth_lower"
    else:
        mean_col = "mean_reward"
        upper_col = "upper"
        lower_col = "lower"

    plt.plot(df["iteration"], df[mean_col], label=label)

    if upper_col in df.columns and lower_col in df.columns:
        plt.fill_between(df["iteration"], df[lower_col], df[upper_col], alpha=0.2)


# ============================================================
# Main Plotting Functions
# ============================================================


def plot(artifacts_dir: Path):
    """Raw plot: no smoothing."""
    setup_plot("Comparison of Experiments (Mean Reward ± 95% CI)")

    for exp_dir in get_experiment_dirs(artifacts_dir):
        df = load_metrics(exp_dir, "metrics.csv")
        if df is None:
            continue

        df = compute_confidence_intervals(df)
        plot_ci_curve(df, label=format_label(exp_dir), smooth=False)

    save_plot(artifacts_dir / "comparison_plot.png")


def plot_smooth(artifacts_dir: Path, rolling_window: int):
    """Full experiment plot with smoothing."""
    setup_plot("Comparison of Experiments (Smoothed Mean Reward ± 95% CI)")

    for exp_dir in get_experiment_dirs(artifacts_dir):
        df = load_metrics(exp_dir, "metrics.csv")
        if df is None:
            continue

        df = compute_confidence_intervals(df)
        df = smooth(df, rolling_window)

        plot_ci_curve(df, label=format_label(exp_dir), smooth=True)

        # Save per-experiment smooth dataframe
        df.to_csv(exp_dir / "smooth.csv", index=False)

    save_plot(artifacts_dir / "comparison_smooth_plot.png")


def plot_last_iters(artifacts_dir: Path, last_iterations: int, rolling_window: int):
    """Plot only the last N iterations from smooth.csv."""
    setup_plot(f"Mean Reward of Last {last_iterations} Iterations (Smoothed ± 95% CI)")

    for exp_dir in get_experiment_dirs(artifacts_dir):
        df = load_metrics(exp_dir, "smooth.csv")
        if df is None:
            continue

        df = df.tail(last_iterations)
        df = compute_confidence_intervals(df)  # applies only if present
        df = smooth(df, rolling_window)

        plot_ci_curve(df, label=format_label(exp_dir), smooth=True)

    save_plot(artifacts_dir / f"comparison_last_{last_iterations}_plot.png")


if __name__ == "__main__":
    artifacts_dir = Path("artifacts")
    rolling_window = 20
    last_iters = 100

    plot(artifacts_dir)
    plot_smooth(artifacts_dir, rolling_window)
    plot_last_iters(artifacts_dir, last_iters, rolling_window)
