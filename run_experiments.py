"""Reproduce the experiments of the paper: 5 folds of HaRS-Mamba and of the two ablation variants,
test-set evaluation, statistics and the ablation tables.  Finished folds are skipped, so the
script can be restarted at any time.

    python run_experiments.py                 # everything (about 3 x 17 h on one laptop GPU)
    python run_experiments.py hars_mamba      # only the main model
"""
from __future__ import annotations

import subprocess
import sys

from config import N_FOLDS, run_dir

VARIANTS = {
    "hars_mamba": [],
    "no_synthetic_hair": ["--synthetic-hair", "0"],
    "no_agss": ["--no-artifact"],
}


def run(*args):
    print(">", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], check=True)


def main():
    names = sys.argv[1:] or list(VARIANTS)
    for name in names:
        for k in range(1, N_FOLDS + 1):
            if not (run_dir(name) / f"fold_{k}" / "done.json").exists():
                run("train_fold.py", "--fold", str(k), "--name", name, *VARIANTS[name])
        run("evaluate_folds.py", "--name", name)
    if "hars_mamba" in names:
        run("paper_stats.py", "--name", "hars_mamba")
    if all((run_dir(n) / "ensemble" / "test_per_image.csv").exists() for n in VARIANTS):
        run("compare_variants.py")


if __name__ == "__main__":
    main()
