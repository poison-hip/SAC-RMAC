"""
Export training curves from Stable-Baselines3 TensorBoard event files.

This is useful for papers/reports where you want PNG/CSV instead of screenshots.

Default behavior:
  - auto-detect the newest TensorBoard event file under ./logs/**/tensorboard/**/
  - export ALL scalar tags to docs/training_curves/<event_stem>/
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _find_event_files(root: str) -> list[Path]:
    p = Path(root)
    if not p.exists():
        return []
    return sorted(p.rglob("events.out.tfevents.*"), key=lambda x: x.stat().st_mtime, reverse=True)


def _moving_average(y: list[float], window: int) -> list[float]:
    if window <= 1 or len(y) == 0:
        return y
    out: list[float] = []
    s = 0.0
    q: list[float] = []
    for v in y:
        q.append(v)
        s += v
        if len(q) > window:
            s -= q.pop(0)
        out.append(s / len(q))
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Export SB3 TensorBoard scalars to PNG/CSV.")
    p.add_argument(
        "--tb-root",
        type=str,
        default="./logs",
        help="Search root for TensorBoard event files (default: ./logs).",
    )
    p.add_argument(
        "--event-file",
        type=str,
        default=None,
        help="Explicit TensorBoard event file path (overrides auto-detect).",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="./docs/training_curves",
        help="Output directory for exported PNG/CSV.",
    )
    p.add_argument(
        "--tags",
        type=str,
        default=None,
        help="Comma-separated list of scalar tags to export (default: export all).",
    )
    p.add_argument(
        "--smooth",
        type=int,
        default=1,
        help="Moving-average smoothing window for plots (1 = no smoothing).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.event_file is not None:
        event_path = Path(os.path.expanduser(args.event_file))
        if not event_path.exists():
            raise FileNotFoundError(f"Event file not found: {event_path}")
    else:
        files = _find_event_files(args.tb_root)
        if not files:
            raise FileNotFoundError(
                f"No TensorBoard event files found under: {args.tb_root}\n"
                "Expected something like logs/<env>/tensorboard/<run>/events.out.tfevents.*"
            )
        event_path = files[0]

    # Import here so tensorboard is only required when using this script.
    from tensorboard.backend.event_processing import event_accumulator  # noqa: WPS433

    ea = event_accumulator.EventAccumulator(str(event_path), size_guidance={"scalars": 0})
    ea.Reload()
    scalar_tags = ea.Tags().get("scalars", [])
    if not scalar_tags:
        raise RuntimeError(f"No scalar tags found in event file: {event_path}")

    if args.tags is not None:
        wanted = {t.strip() for t in args.tags.split(",") if t.strip()}
        scalar_tags = [t for t in scalar_tags if t in wanted]
        if not scalar_tags:
            raise RuntimeError(f"No matching tags found. Available tags: {ea.Tags().get('scalars', [])}")

    out_base = Path(args.out_dir) / event_path.stem
    out_base.mkdir(parents=True, exist_ok=True)

    print(f"Using event file: {event_path}")
    print(f"Exporting to: {out_base}")
    print(f"Tags: {len(scalar_tags)}")

    for tag in sorted(scalar_tags):
        events = ea.Scalars(tag)
        steps = [int(e.step) for e in events]
        values = [float(e.value) for e in events]

        # CSV
        safe_tag = tag.replace("/", "__")
        csv_path = out_base / f"{safe_tag}.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("step,value\n")
            for s, v in zip(steps, values):
                f.write(f"{s},{v}\n")

        # Plot
        y = _moving_average(values, args.smooth)
        plt.figure(figsize=(6, 4))
        plt.plot(steps, y, linewidth=1.8)
        plt.grid(True, alpha=0.3)
        plt.xlabel("step")
        plt.ylabel(tag)
        plt.title(tag)
        plt.tight_layout()
        png_path = out_base / f"{safe_tag}.png"
        plt.savefig(png_path, dpi=200)
        plt.close()

    print("Done.")


if __name__ == "__main__":
    main()


