import argparse
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Analyze and plot results from a run.")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to the run directory (e.g., runs/my_run_id)")
    parser.add_argument("--fig-root", type=str, default=None, help="Root for figures (defaults to figures/<run_id> if not set)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        sys.exit(1)

    run_id = run_dir.name
    
    # Determine output directory
    # User req: figures/<run_id>/
    if args.fig_root:
        out_dir = Path(args.fig_root) / run_id
    else:
        # Assuming run_dir is like .../runs/<run_id>
        # We want .../figures/<run_id>
        # Check if parent is 'runs'
        if run_dir.parent.name == "runs":
             out_dir = run_dir.parent.parent / "figures" / run_id
        else:
             out_dir = Path("figures") / run_id

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving figures to: {out_dir}")

    # Paths to CSVs
    csv_dir = run_dir / "csv"
    train_csv = csv_dir / "train_episodes.csv"
    eval_csv = csv_dir / "eval_summary.csv"
    frontier_csv = csv_dir / "frontier.csv" # If exists

    # 1. Learning Curve (Train Episodes)
    if train_csv.exists():
        try:
            df_train = pd.read_csv(train_csv)
            # Assuming columns like 'step', 'reward', 'success', 'global_step'
            # Adjust based on likely column names. 
            # If standard SB3/custom logging: often 'ep_rew_mean' in monitor.csv, but here we expect 'train_episodes.csv'
            # Let's assume cols: episode, step, reward, success
            
            # 1. Learning Curve (Success Rate vs Step)
            if "env_step_end" in df_train.columns and "success" in df_train.columns:
                plt.figure(figsize=(10, 6))
                # Smooth the success rate
                df_train["success_smooth"] = df_train["success"].rolling(window=100).mean()
                sns.lineplot(data=df_train, x="env_step_end", y="success_smooth", label="Train Success (MA100)")
                plt.title(f"Learning Curve: {run_id}")
                plt.xlabel("Environment Steps")
                plt.ylabel("Success Rate")
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(out_dir / "learning_curve.png")
                plt.close()
                print("Generated learning_curve.png")
            else:
                print(f"Warning: Expected columns (env_step_end, success) not found in {args.run_dir}/csv/train_episodes.csv")
        except Exception as e:
            print(f"Failed to plot learning curve: {e}")
    else:
        print(f"Not found: {train_csv}")

    # 2. Main Results Bar (Eval Summary)
    if eval_csv.exists():
        try:
            df_eval = pd.read_csv(eval_csv)
            # Columns: step, setting, mean_success, ...
            if 'setting' in df_eval.columns and 'mean_success' in df_eval.columns:
                # Plot the LATEST evaluation for each setting
                last_step = df_eval['step'].max()
                df_last = df_eval[df_eval['step'] == last_step]
                
                plt.figure(figsize=(10, 6))
                sns.barplot(data=df_last, x='setting', y='mean_success')
                plt.title(f'Evaluation Results (Step {last_step})')
                plt.ylim(0, 1.0)
                plt.ylabel('Success Rate')
                plt.savefig(out_dir / "main_results_bar.png")
                plt.close()
                print("Generated main_results_bar.png")
            else:
                 print(f"Warning: Expected columns (setting, mean_success) not found in {eval_csv}")
        except Exception as e:
            print(f"Failed to plot main results: {e}")
    else:
        print(f"Not found: {eval_csv}")

    # 3. Frontier/Heatmap (if applicable)
    if frontier_csv and frontier_csv.exists():
        # TODO: Implement frontier plotting if schema helps
        pass

if __name__ == "__main__":
    main()


























