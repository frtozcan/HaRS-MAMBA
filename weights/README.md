# HaRS-Mamba weights (ISIC 2018, 5-fold cross-validation)

One checkpoint per fold, selected on that fold's held-out validation images. Load with:

```python
import torch
from hars import build_model
ck = torch.load('weights/hars_mamba_fold1.pt', map_location='cpu')
model = build_model('base')
model.load_state_dict(ck['model'])
model.eval()
```

To evaluate all five (and the ensemble) on the ISIC 2018 test set, copy them to `$HARS_RUNS/hars_mamba/fold_k/best.pt` and run `python evaluate_folds.py --name hars_mamba`.

Input: RGB image resized to 256×256, ImageNet normalisation; output `model(x)['seg']` are logits (threshold sigmoid at 0.5).

| Fold | File | Selected epoch | Val Dice | Test Dice | SHA-256 (first 16) |
|---|---|---|---|---|---|
| 1 | `hars_mamba_fold1.pt` | 116 | 0.9006 | 0.8916 | `b780d651c94b4727` |
| 2 | `hars_mamba_fold2.pt` | 139 | 0.9040 | 0.8926 | `cb3cf9343ef8ff09` |
| 3 | `hars_mamba_fold3.pt` | 181 | 0.9043 | 0.8873 | `48ba7f5a34b6744a` |
| 4 | `hars_mamba_fold4.pt` | 97 | 0.9051 | 0.8823 | `68d26a759d8bbb69` |
| 5 | `hars_mamba_fold5.pt` | 111 | 0.9061 | 0.8915 | `2cfd426182aedb56` |
