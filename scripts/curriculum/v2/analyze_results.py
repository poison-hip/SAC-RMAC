import os
import csv
import glob
import argparse
import numpy as np
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser("Analyze Results")
    parser.add_argument("--log-dir", type=str, required=True, help="Root directory containing run folders")
    parser.add_argument("--last-n", type=int, default=3, help="Average over last N eval checkpoints per seed (for stability)")
    return parser.parse_args()

def analyze_eval_summary(log_dir, last_n=1):
    # Find all eval_summary.csv files
    # Structure: log_dir/RunID/eval_summary.csv
    pattern = os.path.join(log_dir, "**", "eval_summary.csv")
    files = glob.glob(pattern, recursive=True)
    
    print(f"Found {len(files)} eval_summary.csv files.")
    
    # Data structure:
    # results[method][setting_id][seed] = [list of (step, sr, ret, len)]
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    # Track stats
    runs_found = 0
    
    for fpath in files:
        # We can extract run_id from path or use the one in CSV
        # Let's rely on CSV content for method/seed/setting
        with open(fpath, "r") as f:
            reader = csv.DictReader(f)
            # Check headers
            if "method" not in reader.fieldnames:
                print(f"[Warn] Skipping {fpath}: missing 'method' column")
                continue
                
            runs_found += 1
            
            # Read all rows
            rows = list(reader)
            if not rows:
                continue
                
            # We want to separate by setting_id
            # Filter rows by setting
            by_setting = defaultdict(list)
            for r in rows:
                by_setting[r["eval_setting_id"]].append(r)
                
            # For each setting, getting entries
            for setting, s_rows in by_setting.items():
                # Sort by env_step
                s_rows.sort(key=lambda x: int(float(x["env_step"]))) # cast to float first just in case
                
                # Take last N
                final_rows = s_rows[-last_n:]
                
                if not final_rows:
                    continue
                    
                # Compute average over last N checkpoints for this seed
                avg_sr = np.mean([float(r["success_rate"]) for r in final_rows])
                avg_ret = np.mean([float(r["return_mean"]) for r in final_rows])
                avg_len = np.mean([float(r["len_mean"]) for r in final_rows])
                
                # Extract meta info from the last row
                method = final_rows[0]["method"]
                seed = final_rows[0]["train_seed"]
                
                results[method][setting][seed] = {
                    "sr": avg_sr,
                    "ret": avg_ret,
                    "len": avg_len,
                    "eval_episodes": sum(int(r["num_eval_episodes"]) for r in final_rows)
                }

    # Now aggregate over seeds
    # Output Table rows
    table_rows = []
    
    for method, settings_dict in results.items():
        for setting, seeds_dict in settings_dict.items():
            srs = [v["sr"] for v in seeds_dict.values()]
            rets = [v["ret"] for v in seeds_dict.values()]
            lens = [v["len"] for v in seeds_dict.values()]
            
            n_seeds = len(srs)
            
            row = {
                "method": method,
                "eval_setting": setting,
                "sr_mean": np.mean(srs),
                "sr_std": np.std(srs),
                "ret_mean": np.mean(rets),
                "ret_std": np.std(rets),
                "len_mean": np.mean(lens),
                "len_std": np.std(lens),
                "n_seeds": n_seeds,
                "total_eval_episodes": sum(v["eval_episodes"] for v in seeds_dict.values())
            }
            table_rows.append(row)
            
    # Sort for readability: Method then Setting
    table_rows.sort(key=lambda x: (x["method"], x["eval_setting"]))
    
    # Write to CSV
    out_path = os.path.join(log_dir, "paper_table_main.csv")
    headers = [
        "method", "eval_setting", 
        "sr_mean", "sr_std", 
        "ret_mean", "ret_std", 
        "len_mean", "len_std", 
        "n_seeds", "total_eval_episodes"
    ]
    
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in table_rows:
            # Format floats for cleaner CSV
            r_fmt = r.copy()
            for k, v in r_fmt.items():
                if isinstance(v, float):
                    r_fmt[k] = f"{v:.4f}"
            writer.writerow(r_fmt)
            
    print(f"\nAnalysis complete. Aggregated {runs_found} runs.")
    print(f"Saved summary table to: {out_path}")
    
    # Print preview
    print("\nPreview (SR avg):")
    print(f"{'Method':<25} | {'Setting':<15} | {'SR':<15} | {'Ref':<15}")
    print("-" * 80)
    for r in table_rows:
        print(f"{r['method']:<25} | {r['eval_setting']:<15} | {r['sr_mean']:.3f} ± {r['sr_std']:.3f} | {r['ret_mean']:.1f}")

if __name__ == "__main__":
    args = parse_args()
    analyze_eval_summary(args.log_dir, args.last_n)


























