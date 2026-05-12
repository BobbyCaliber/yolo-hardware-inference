# Stage-2 ablation: do arch features replace `model_name`?

Each cell shows mean across folds. **R²** is on log-time; **MAPE** is in seconds-space.

## CV by hardware (GroupKFold on cpu+gpu)

| variant | description | n_feat | R² log | MAPE | MAE sec |
|---|---|---:|---:|---:|---:|
| **A** | baseline (with model_name) | 33 | 0.907 | 38.8% | 0.753 |
| **B** | +arch, model_name kept | 79 | 0.906 | 37.4% | 0.679 |
| **C** | −id +arch (no model_name/family) | 77 | 0.926 | 31.7% | 0.745 |
| **D** | C + roofline (log_t_theoretical) | 78 | 0.949 | 27.5% | 0.607 |

## CV by model_family (leave-one-family-out)

| variant | description | n_feat | R² log | MAPE | MAE sec |
|---|---|---:|---:|---:|---:|
| **A** | baseline (with model_name) | 33 | 0.949 | 28.4% | 0.480 |
| **B** | +arch, model_name kept | 79 | 0.964 | 20.8% | 0.423 |
| **C** | −id +arch (no model_name/family) | 77 | 0.959 | 22.4% | 0.466 |
| **D** | C + roofline (log_t_theoretical) | 78 | 0.969 | 19.1% | 0.360 |

## Pass-gate

- A under **hardware** CV: R² = 0.907
- D under **family**   CV: R² = 0.969
- Δ = -6.2 pp

**PASS** (Δ ≤ 5 pp): arch features successfully replace model identity. Proceed to Stage 3 (ModelRunner abstraction).

## Raw results

```json
[
  {
    "r2_log_mean": 0.9073941353290131,
    "r2_log_std": 0.04974229534379922,
    "mape_mean": 0.38769875558772016,
    "mae_sec_mean": 0.7526929560354707,
    "n_features": 33,
    "n_categorical": 4,
    "n_splits": 5,
    "code": "A",
    "label": "baseline (with model_name)",
    "scheme": "hardware"
  },
  {
    "r2_log_mean": 0.9494457467888427,
    "r2_log_std": 0.01987524684483742,
    "mape_mean": 0.2837820490071955,
    "mae_sec_mean": 0.4799730801731891,
    "n_features": 33,
    "n_categorical": 4,
    "n_splits": 6,
    "code": "A",
    "label": "baseline (with model_name)",
    "scheme": "family"
  },
  {
    "r2_log_mean": 0.906023346606009,
    "r2_log_std": 0.06194784000890257,
    "mape_mean": 0.3739275874025023,
    "mae_sec_mean": 0.6791631458732938,
    "n_features": 79,
    "n_categorical": 4,
    "n_splits": 5,
    "code": "B",
    "label": "+arch, model_name kept",
    "scheme": "hardware"
  },
  {
    "r2_log_mean": 0.9636694048007071,
    "r2_log_std": 0.019877596990859454,
    "mape_mean": 0.20797241914767603,
    "mae_sec_mean": 0.42347017977471707,
    "n_features": 79,
    "n_categorical": 4,
    "n_splits": 6,
    "code": "B",
    "label": "+arch, model_name kept",
    "scheme": "family"
  },
  {
    "r2_log_mean": 0.9264971239879,
    "r2_log_std": 0.04503715020720553,
    "mape_mean": 0.3169222081402923,
    "mae_sec_mean": 0.7452228601804805,
    "n_features": 77,
    "n_categorical": 2,
    "n_splits": 5,
    "code": "C",
    "label": "\u2212id +arch (no model_name/family)",
    "scheme": "hardware"
  },
  {
    "r2_log_mean": 0.9594498371929813,
    "r2_log_std": 0.023247269892831945,
    "mape_mean": 0.2237039892829196,
    "mae_sec_mean": 0.46600040176024327,
    "n_features": 77,
    "n_categorical": 2,
    "n_splits": 6,
    "code": "C",
    "label": "\u2212id +arch (no model_name/family)",
    "scheme": "family"
  },
  {
    "r2_log_mean": 0.9486409814408037,
    "r2_log_std": 0.01847977783677279,
    "mape_mean": 0.2754859864118547,
    "mae_sec_mean": 0.6066525030486088,
    "n_features": 78,
    "n_categorical": 2,
    "n_splits": 5,
    "code": "D",
    "label": "C + roofline (log_t_theoretical)",
    "scheme": "hardware"
  },
  {
    "r2_log_mean": 0.9693851983547753,
    "r2_log_std": 0.014268369331114358,
    "mape_mean": 0.19070403083362994,
    "mae_sec_mean": 0.35954548581271245,
    "n_features": 78,
    "n_categorical": 2,
    "n_splits": 6,
    "code": "D",
    "label": "C + roofline (log_t_theoretical)",
    "scheme": "family"
  }
]
```