## Table 5. Ablation (official ISIC 2018 test set, 5 folds)

| Configuration | AGSS | Synthetic hair | Val Dice | Test Dice (fold mean ± SD) | Test IoU (fold mean ± SD) | Ensemble Dice | Ensemble Dice, lesions > 60% | Ensemble Dice, hair-heavy 20% | Images with Dice < 0.5 |
|---|---|---|---|---|---|---|---|---|---|
| HaRS-Mamba | ✓ | ✓ | 0.9040 | 0.8891 ± 0.0043 | 0.8191 ± 0.0067 | 0.8980 | 0.8502 | 0.9154 | 18 |
| w/o synthetic hair | ✓ | — | 0.9044 | 0.8879 ± 0.0035 | 0.8180 ± 0.0047 | 0.8951 | 0.8425 | 0.9134 | 25 |
| w/o AGSS | — | — | 0.9037 | 0.8852 ± 0.0035 | 0.8130 ± 0.0055 | 0.8919 | 0.8338 | 0.9110 | 22 |

## Table 6. Paired comparisons of the ensembles (Wilcoxon signed-rank, Holm-adjusted over 3 tests)

| Comparison | Isolates | Δ Dice, mean / median | Images better / worse | Folds better | p | Δ Dice, hair-heavy (p) |
|---|---|---|---|---|---|---|
| hars_mamba vs no_agss | AGSS with synthetic hair | +0.0062 / +0.0031 | 637 / 353 | 4 / 5 | 3e-17 | +0.0045 (8.3e-09) |
| no_synthetic_hair vs no_agss | AGSS alone | +0.0032 / +0.0035 | 659 / 332 | 4 / 5 | 9.6e-22 | +0.0024 (3.6e-06) |
| hars_mamba vs no_synthetic_hair | synthetic hair | +0.0030 / -0.0004 | 472 / 523 | 3 / 5 | 0.13 | +0.0021 (0.077) |
