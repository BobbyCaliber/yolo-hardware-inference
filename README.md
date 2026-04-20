# yolo-hardware-predict

**End-to-end pipeline for predicting YOLO inference time on arbitrary hardware.**
Build and run three Docker containers to probe the host's CPU/GPU and benchmark
inference across a matrix of YOLO variants × image sizes × batch sizes;
merge the per-run CSVs into a training dataset; fit a CatBoost regressor;
and use the trained model to predict inference time for unseen
hardware/workload combinations — all driven from a single `Makefile`.

## Repository layout

```
.
├── src/
│   ├── check_cpu_config/            # CPU descriptor (Docker, ubuntu:22.04)
│   ├── check_gpu_config/            # GPU descriptor (Docker, pycuda on cuda:12.6)
│   ├── check_yolo_predict/          # YOLO inference benchmark (Docker, cuda:12.6)
│   ├── run_benchmark.py             # orchestrates the 3 containers via Docker SDK
│   ├── merge_results.py             # cross-joins tmp/data/*.csv → data_new/
│   ├── train_model.py               # trains CatBoost on data_base + data_new
│   └── predict.py                   # predicts inference time for new hardware
├── specs/
│   ├── amd-cpus.csv                 # AMD cpu specs database
│   ├── intel-cpus.csv               # Intel cpu specs database
│   ├── benchmark-cpus.csv           # All cpu specs database
│   └── gpu_1986-2026.csv            # gpu specs database
├── data_base/                       # baseline prediction weights/db
│   ├── data_base.csv                # 35-column dataset (7200 benchmark rows)
│   └── reg_weights_base/            # baseline CatBoost weights
│       ├── catboost_model.cbm
│       └── metadata.json
├── data_new/                        # produced by `make merge` / `make train`
│   └── reg_weights_new/             # fresh weights after `make train`
├── tmp/                             # temp cache folder/unmerged benchmark data
│   ├── data/                        # per-run raw CSVs
│   └── weights/                     # YOLO .pt cache
├── research/                        # Jupyter notebooks (model selection, EDA)
├── Makefile
├── requirements.txt                 # docker SDK + pandas + sklearn + catboost
└── README.md
```

## Requirements

- **Docker Engine 24+** with the **NVIDIA Container Toolkit** for GPU benchmarks
- **Python 3.10+** on the host (for `run_benchmark.py`, `merge_results.py`, `train_model.py`, `predict.py`)
- **NVIDIA GPU** with CUDA 12.6-compatible drivers for `check_gpu_config` and `check_yolo_predict` (the CPU descriptor runs anywhere)

## Quickstart

The commands below take a cold repo to a trained predictor:

```bash
make help                          # list make targets + current variable values
make install                       # pip install requirements.txt
make run                           # run 3 benchmark containers → tmp/data/*.csv
make merge MERGE_TAG=$(hostname)   # → data_new/merged_<ts>_<host>.csv
make train                         # train CatBoost on data_base + data_new → weights go to data_new/reg_weights_new/
make predict CPU="AMD Ryzen 9 7900X" GPU="RTX 5060" RAM=32 MODEL=yolov8m.pt IMG=640 BATCH=5 USED_GPU=1 # example of making predictions on unseen hardware
```

Expected final output:

```
[INFO] weights: data_new/reg_weights_new/catboost_model.cbm  [data_new (fresh)]
[INFO] CPU matched: AMD Ryzen™ 7 5700X  (source: amd-cpus.csv)
       8c/16t  3400.0→4600.0 MHz  L2=4096.0KB  L3=32.0MB
[INFO] GPU matched: NVIDIA GeForce RTX 3060 12 GB
       28 SMs × 128 = 3584 cores, FP32 12.74 TFLOPS, CC 8.6

predicted inference time: 0.0847 sec  (84.7 ms)  for yolov8m.pt @ 640px × batch 5 on GPU
```

If `make train` has not been run, `make predict` transparently falls back
to the baseline weights shipped in `data_base/reg_weights_base/` — so **you can run predictions with base data** without building and running containers.

## Before training - configure VM memory on Docker Desktop / WSL2 / macOS

Raise the VM memory cap before collecting training data.
`check_yolo_predict.py` reads `psutil.virtual_memory()` from *inside*
the container, not the global value of RAM.

- **Windows + WSL2:** edit `%USERPROFILE%\.wslconfig` on the Windows host:
   ```
   [wsl2]
   memory="full memory size"
   swap=0
   ```
   then run `wsl --shutdown` (Docker will re-create the VM on next start).
 - **macOS / Windows Hyper-V:** Docker Desktop → Settings → Resources →
   drag the Memory slider to the desired value → Apply & Restart.
 - **Native Linux:** no action needed — containers inherit host RAM
   unless you pass `--memory=<N>` explicitly.



## Pipeline stages

### 1 · Collect — benchmark the host

Three containers write CSVs into `tmp/data/`:

| Container | Output | Contents |
|---|---|---|
| `check_cpu_config` | `cpu_config.csv` | CPU model, freq, cores/threads, L1/L2/L3, SMP |
| `check_gpu_config` | `gpu_config.csv` | GPU, compute capability, SMs, CUDA/tensor cores, FP32/FP16/FP64 TFLOPS |
| `check_yolo_predict` | `yolo_predict.csv` | One row per inference run over YOLOv5/6/8/9/10/11 × imgsz × batch × device |

`merge_results.py` cross-joins them into a single file in `data_new/`, matching the exact schema of `data_base/data_base.csv`.

### 2 · Train — fit CatBoost

`train_model.py` concatenates `data_base/data_base.csv` with every CSV in
`data_new/`, applies the
preprocessing pipeline, runs a cross-validation and
fits a final CatBoost regressor on the full dataset.

Model and metadata are
saved to `data_new/reg_weights_new/`.

### 3 · Predict — ask the model

`predict.py` resolves CPU/GPU specs from the `specs/*.csv` lookup tables,
pulls the corresponding YOLO model param count from `data_base.csv`,
applies the same preprocessing, and calls `CatBoostRegressor.predict()`.

Defaults for freshly-trained weights (`data_new/reg_weights_new/`) if
present, otherwise uses repo baseline weights(`data_base/reg_weights_base/`).


## API reference

### `run_benchmark.py` — container orchestration

```python
@dataclass(frozen=True)
class Benchmark:
    name: str                    # container name, e.g. "check_yolo_predict"
    context: Path                # docker build context directory
    image: str                   # image tag
    needs_gpu: bool              # adds DeviceRequest(count=-1, capabilities=[["gpu"]])
    needs_images: bool = False   # mounts --images-dir → /app/data/images (ro)
    needs_weights: bool = False  # mounts --weights-dir → /app/weights (rw)

BENCHMARKS: dict[str, Benchmark]  # registry keyed by "cpu"/"gpu"/"yolo"
```

**CLI flags:** `--data-dir`, `--images-dir`, `--weights-dir`, `--only`,
`--skip-build`, `--pull`, `--log-level`. Handles `KeyboardInterrupt` by
forwarding `SIGINT` into the container so the Python `with open(...)`
blocks can flush CSV buffers before exit.

### `merge_results.py` — schema-aligned CSV join

```python
def merge(data_dir: Path) -> pandas.DataFrame
```

Reads `cpu_config.csv`, `gpu_config.csv`, `yolo_predict.csv` from
`data_dir`, cross-joins via `DataFrame.join(…, how="cross")`, and reorders
columns to match `data_base/data_base.csv`.


**CLI flags:** `--data-dir`, `--out-dir`, `--tag` (suffix for filename).
One invocation → one timestamped file; no accumulation.

### `train_model.py` — regression fit

```python
def load_all_csvs(ref_path: Path, new_dir: Path) -> pandas.DataFrame
def preprocess(raw: pandas.DataFrame) -> pandas.DataFrame
def cv_score(df: pandas.DataFrame) -> dict[str, float] | None
def fit_final(df: pandas.DataFrame) -> tuple[CatBoostRegressor, list[str]]
def save(model: CatBoostRegressor,
         features: list[str],
         report: TrainReport,
         out_dir: Path, ref_csv: Path, new_dir: Path) -> None

MODEL_PARAMS = dict(iterations=600, depth=8,
                    learning_rate=0.05, random_seed=42, verbose=0)
CATEGORICAL = ['cpu_name', 'gpu_name', 'model_name', 'model_family']
```

`preprocess()` steps, in order:
1. Drop: `model_param_dict`, `Processor`, `GPU Name`, `FP64 TENSOR FLOPS`, `BF16 FLOPS`, `TF32 FLOPS`.
2. Coerce `l1_*` / `l2_*` / `l3_cache_size` to numeric (`"Unknown"` → NaN).
3. Parse `Compute Capability` tuple-repr `"(8, 6)"` → `8.6` float.
4. Cast `used_gpu` → int.
5. Derive `pixels`, `work_estimate`, `cpu_throughput`, `compute_throughput`.
6. Log-transform target → `log_time = log(predicting_times)`.
7. Extract `model_family` from `model_name`.

### `predict.py` — inference-time estimation

```python
def resolve_cpu(cpu_name: str) -> dict[str, Any]
def resolve_gpu(gpu_name: str) -> dict[str, Any]
def load_model_param_lookup() -> dict[str, int]
def build_row(cpu: dict, gpu: dict, *,
              used_gpu: int, ram: int,
              model_name: str, img_size: int, batch: int,
              model_params: dict[str, int]) -> pandas.DataFrame
def predict(row_df: pandas.DataFrame,
            weights_path: Path, meta_path: Path) -> float

BASE_WEIGHTS: Path   # data_base/reg_weights_base/catboost_model.cbm
NEW_WEIGHTS:  Path   # data_new/reg_weights_new/catboost_model.cbm

def _default_weights() -> Path  # NEW if exists else BASE
```

**`resolve_cpu(cpu_name)`** — dispatches by vendor token in `cpu_name`:
- `amd`/`ryzen`/`epyc`/`threadripper` → `specs/amd-cpus.csv` (L1+L2+L3 present)
- `intel`/`core`/`xeon`/`pentium`/`celeron` → `specs/intel-cpus.csv` (L2+L3)
- else → `specs/benchmark-cpus.csv` (cache fields → NaN)

Returns a dict with keys matching `data_base.csv` CPU columns.

**`resolve_gpu(gpu_name)`** — always queries `specs/gpu_1986-2026.csv`.
Strips `NVIDIA`/`GeForce`/`AMD`/`Radeon`/`Intel`/`Arc` prefixes as a
fallback. Computes `Cores per SM = Shading Units / SM Count` (matches the
arch-derived value from `check_gpu_config.py` to within rounding).

**`build_row()`** — signature uses keyword-only args (`*`) to prevent
positional confusion. Constructs a single-row DataFrame with exactly the
columns `preprocess()` expects.

**`predict(row_df, weights_path, meta_path) -> float`** — returns
inference time in **seconds** (post-exp). Raises `SystemExit` if weights
or metadata files are missing.

**Caveats:**
- Predictions for hardware dramatically outside the training distribution
  (e.g., H100 when no H100 rows were in training) are unreliable. CatBoost
  does not extrapolate gracefully outside tree splits seen during training.
- Predictions below ~0.02s or above ~100s are likely extrapolation errors
  — the training target ranged 0.015s … 127.5s but long-tail regions have
  sparse coverage.
- The `num_all_params` lookup requires `model_name` to have appeared in
  `data_base.csv` at least once. Use `python src/predict.py --list-models`
  to see the 24 supported YOLO weights.

**CLI flags:** `--cpu` / `--gpu` / `--used-gpu` / `--ram` / `--model` /
`--img-size` / `--batch` / `--weights` / `--meta` / `--list-models`.


## External data sources

- GPU database source: <https://www.kaggle.com/datasets/ellimaaac/gpus-specs-from-1986-to-2026>
- CPU database source: <https://github.com/felixsteinke/cpu-spec-dataset>
