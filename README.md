# yolo-hardware-predict

**Predict CV-model inference time on arbitrary hardware.**

A trained CatBoost regressor that, given any `(cpu, gpu, model, image_size,
batch)` tuple, returns expected inference time in seconds. The shipped
baseline covers **5 hardware platforms × 24 YOLO models**; the model zoo
itself spans **73 models across 21 architecture families** (YOLO v5–11,
RT-DETR, DETR, Segformer, Faster/Mask/Keypoint-RCNN, DeepLabV3, FCN,
LR-ASPP, ViT, DeiT, Swin, EfficientNet, ResNet/ResNeXt, ConvNeXt).
Hardware features for unseen CPUs/GPUs come from shipped spec catalogues
(~7800 CPUs, ~2900 GPUs); model features come from pre-computed
architectural fingerprints (FLOPs, op-type histogram, attention/depthwise
shares, roofline estimate). So you can predict for hardware that's never
been benchmarked.

If you want better-than-baseline accuracy on YOUR machine, run
`make collect` to add real benchmarks for your platform and retrain.

See `scalability_plan.md` for the methodology and
`research/ablation_arch_features.md` for the ablation that validates
arch features replace the per-model identity (R² 0.968 leave-one-family-out).
Stage-4 RT-DETR gate (held-out family MAPE ≤ 40%) was confirmed at 23–27%
across configurations once structural arch features were added.

## Requirements

- **Python 3.10+** for `predict.py`, `streamlit_app.py`, and the helper scripts
- **Docker Engine 24+** with the **NVIDIA Container Toolkit** — only needed
  if you want to bench on your own hardware (the shipped baseline weights
  predict without it)

## Quickstart — predict only

```bash
make install
make ui                                  # browser UI; pick CPU/GPU/model and predict
# or:
make predict CPU="AMD Ryzen 9 7950X" GPU="NVIDIA GeForce RTX 4090" \
    USED_GPU=1 RAM=64 MODEL=yolov8m.pt IMG=640 BATCH=5
```

`predict` works for any CPU/GPU you can spell from the spec catalogues:

```bash
python src/predict.py --list-platforms        # 5 baseline pairs (best accuracy)
python src/predict.py --list-known-cpus       # ~7800 CPUs
python src/predict.py --list-known-gpus       # ~2900 GPUs
python src/predict.py --list-models           # registered model_names
```

If both CPU and GPU are in the baseline (5 platforms), the prediction uses
measured cache sizes etc. directly. Otherwise the regressor falls back on
hardware features synthesised from spec CSVs — works but less accurate.

## Reproducing on a new platform — improve baseline accuracy

**Prerequisites:**

- Make sure Docker Engine 24+ is installed and the daemon is running (`docker info`
  succeeds). On Linux, your user is in the `docker` group; on Windows/macOS,
  Docker Desktop is open.
- The **NVIDIA Container Toolkit** must also be set up if you want GPU
  benchmarks — see install instructions below. Verify with:
  ```bash
  docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
  ```
  (a CUDA image alone is not enough — the toolkit is what exposes the host
  driver and `/dev/nvidia*` to the container).
- A clean Python 3.10+ virtual environment for the host-side scripts
  (orchestrator, merge, train, predict). Avoid installing into the system
  interpreter:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate          # Windows: .venv\Scripts\activate
  make install                       # pip install -r requirements.txt
  ```

#### Installing the NVIDIA Container Toolkit

Pre-requisite: a working NVIDIA driver on the host (`nvidia-smi` runs and
shows your GPU). The toolkit only bridges the existing driver into Docker —
it does not install one.

**Ubuntu / Debian (and WSL2 Ubuntu):**

```bash
# 1. Add NVIDIA's package repo
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 2. Install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# 3. Wire it into the docker daemon and restart
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**Windows + WSL2:** install the **Windows** NVIDIA driver normally (it
exposes the GPU into WSL automatically — do **not** install a Linux
NVIDIA driver inside WSL). Then open your WSL distro and run the
Ubuntu/Debian steps above. Docker Desktop with the WSL2 backend will pick
up the toolkit from the distro.

**macOS:** GPU benchmarks are not supported (no NVIDIA GPU passthrough on
macOS). `make collect` will still work but will only produce CPU rows —
your model coverage on GPU will rely on the shipped baseline.

After install, the verification command above `docker run --gpus all … nvidia-smi` should print your GPU. If it errors with `could not select device driver "" with capabilities: gpu`, the toolkit is not wired
into the docker daemon yet — re-run the `nvidia-ctk runtime configure`
step and restart Docker.

To add real benchmarks for YOUR hardware in **one command**:

```bash
git clone …
make collect                     # build → bench every family → merge → enrich → train
# Default FAMILIES covers all 21 registered families (full zoo, ~1.5–2 h on
# a single platform). Override e.g. FAMILIES=rtdetr,yolov8 to bench a subset.
# After completion: prints "[collect] total elapsed: HH:MM:SS" with the
# wall-clock time for the whole pipeline.
```

`make collect` runs three docker containers (`check_cpu_config`,
`check_gpu_config`, `check_model_predict`) and chains the rest of the
pipeline (merge → enrich → train). Output: a refreshed
`data_new/reg_weights_new/catboost_model.cbm` that the predictor will
prefer over the shipped baseline.

The CPU/GPU containers detect everything they need from the host — no
external bandwidth lookup required:

| Container | Output | Auto-detects |
|---|---|---|
| `check_cpu_config` | `cpu_config.csv` | name, freq, cores, cache, **memory bandwidth** (best-effort) |
| `check_gpu_config` | `gpu_config.csv` | name, SMs, FLOPS, **memory bandwidth** (via pycuda mem-clock × bus-width) |
| `check_model_predict` | `family_<name>_predict.csv` | per-row inference time across configured `(model, img_size, batch, device)` sweep |

If you only want one family or sweep, use `make bench-family`:

```bash
make bench-family FAMILY=rtdetr IMG=320,640,800,1120 BATCH=1,4,8
make merge                                   # join with cpu/gpu config
make train                                   # retrain regressor
```

By default each family registers 6 input sizes × 5 batch sizes × CPU+GPU =
**60 measurements per model**, matching the YOLO baseline coverage.
Classification families sweep 160–512 px, detection/segmentation sweep
320–1120 px (per-spec, see `default_img_sizes` in each runner).

### Adding a new model family

1. Register the model in `src/runners/<family>_runner.py` (one
   `ModelSpec(...)` line — see existing runners for examples). Set
   `arch_ref_img_size` to the model's native size (used only for
   FLOPs/activation profiling — it does not constrain bench sizes).
2. Pre-compute its arch features (one-time, platform-agnostic). The host
   doesn't need timm/transformers installed — `arch-features-docker`
   profiles inside the bench container:
   ```bash
   make arch-features-docker FAMILY=segformer
   ```
3. Bench it on each target platform:
   ```bash
   make bench-family FAMILY=segformer SKIP_BUILD=1
   make merge && make train
   ```

## Pipeline at a glance

```
                ┌──────────────────────────┐
                │   check_cpu_config       │ ─── cpu_config.csv (host CPU + bandwidth)
                ├──────────────────────────┤
   make collect │   check_gpu_config       │ ─── gpu_config.csv (host GPU + bandwidth)
                ├──────────────────────────┤
                │   check_model_predict    │ ─── family_<n>_predict.csv (one row per inference)
                └──────────────────────────┘
                              │
                       merge_results.py
                              │
                              ▼
                    data_new/merged_*.csv   ──┐
                                              │
                    data_base/data_base.csv ──┤   ──> train_model.py ──> data_new/reg_weights_new/
                                              │       (preprocess + enrich_helpers
                    model_arch_features.csv ──┘        to add arch + structural-share +
                                                       roofline features)
```

`predict.py` then loads those weights, looks up host features
(baseline first, spec catalogues second), looks up model arch features,
re-derives the same enrichment, and calls `CatBoostRegressor.predict()`.

## Repository layout

```
.
├── src/
│   ├── check_cpu_config/            # CPU descriptor + memory bandwidth
│   ├── check_gpu_config/            # GPU descriptor + memory bandwidth
│   ├── check_model_predict/         # generic model benchmark (uses runners/)
│   ├── runners/                     # ModelRunner abstractions per framework
│   │   ├── base.py / registry.py
│   │   ├── yolo.py / rt_detr.py
│   │   └── timm_runner.py / torchvision_runner.py / hf_runner.py
│   ├── enrich_helpers.py            # arch features + roofline (shared)
│   ├── hardware_lookup.py           # CPU/GPU exact-name lookup from specs/
│   ├── run_benchmark.py             # docker SDK orchestrator
│   ├── merge_results.py             # cross-join tmp/data/*.csv
│   ├── train_model.py               # CatBoost trainer
│   ├── predict.py                   # CLI predictor
│   └── streamlit_app.py             # browser UI
├── scripts/
│   ├── compute_arch_features.py     # profile any registered model → arch features
│   ├── enrich_dataset.py            # apply enrich_helpers to a CSV
│   └── run_ablation.py              # Stage-2 feature-set ablation
├── specs/                           # CPU / GPU spec catalogues
│   ├── amd-cpus.csv / intel-cpus.csv / benchmark-cpus.csv
│   └── gpu_1986-2026.csv
├── data_base/                       # shipped baseline
│   ├── data_base.csv                # 7200 measurements (5 platforms × 24 YOLO × sweep)
│   ├── model_arch_features.csv      # 73 models × arch features (FLOPs, op-type
│   │                                #   histogram, attention/depthwise shares, …)
│   └── reg_weights_base/            # baseline CatBoost weights
├── data_new/                        # produced by make merge / make train
├── research/
│   └── ablation_arch_features.md    # Stage-2 results
├── Makefile
└── requirements.txt
```

## VM memory note (Docker Desktop / WSL2)

The benchmark containers read `psutil.virtual_memory()` from inside the VM,
not the host. If you're on Docker Desktop or WSL2, raise the VM cap before
running `make collect`:

- **Windows + WSL2:** `%USERPROFILE%\.wslconfig` → `[wsl2]\nmemory="<N>GB"`
  → `wsl --shutdown`.
- **macOS / Hyper-V:** Docker Desktop → Settings → Resources → Memory slider.
- **Native Linux:** containers inherit host RAM; nothing to change.

## External data sources

- GPU specs: <https://www.kaggle.com/datasets/ellimaaac/gpus-specs-from-1986-to-2026>
- CPU specs: <https://github.com/felixsteinke/cpu-spec-dataset>
