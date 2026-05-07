"""Streamlit UI — a thin wrapper around the Makefile.

Launch:
    make ui         # or:  streamlit run src/streamlit_app.py

Every action (build/run/merge/train/predict/clean) is dispatched as
`make <target>` in a subprocess — logs are streamed into the browser.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPECS_DIR = PROJECT_ROOT / "specs"
DATA_BASE = PROJECT_ROOT / "data_base" / "data_base.csv"
ARCH_CSV = PROJECT_ROOT / "data_base" / "model_arch_features.csv"


# ------------------------------------------------------------------ helpers

def run_make(target: str, extra_env: dict[str, str] | None = None,
             extra_args: list[str] | None = None,
             log_height: int = 500) -> int:
    """Run `make <target> [K=V ...]`, stream stdout/stderr into the page.

    Logs are written into a fixed-height scrollable container.
    """
    env = os.environ.copy()
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items() if v is not None and v != ""})

    cmd = ["make", target] + (extra_args or [])
    st.caption("`" + " ".join(cmd) + "`")

    log_box = st.container(height=log_height, border=True)
    placeholder = log_box.empty()
    log_lines: list[str] = []
    proc = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        log_lines.append(_ANSI_RE.sub("", line.rstrip()))
        placeholder.code("\n".join(log_lines[-1000:]), language="bash")
    rc = proc.wait()
    if rc == 0:
        st.success(f"`make {target}` — done")
    else:
        st.error(f"`make {target}` — exit {rc}")
    return rc


@st.cache_data(show_spinner=False)
def load_cpu_names() -> list[str]:
    names: list[str] = []
    for path, col in [
        (SPECS_DIR / "amd-cpus.csv", "Model"),
        (SPECS_DIR / "intel-cpus.csv", "CpuName"),
        (SPECS_DIR / "benchmark-cpus.csv", "CpuName"),
    ]:
        if path.is_file():
            try:
                df = pd.read_csv(path, low_memory=False, usecols=[col])
                names.extend(df[col].dropna().astype(str).tolist())
            except Exception:
                pass
    return sorted(set(names))


@st.cache_data(show_spinner=False)
def load_gpu_names() -> list[str]:
    path = SPECS_DIR / "gpu_1986-2026.csv"
    if not path.is_file():
        return []
    df = pd.read_csv(path, low_memory=False, usecols=["Brand", "Name"])
    full = (df["Brand"].fillna("") + " " + df["Name"].fillna("")).str.strip()
    return sorted(set(full[full.str.len() > 0].tolist()))


@st.cache_data(show_spinner=False)
def load_models() -> list[str]:
    """All registered models — pulled from model_arch_features.csv (every model
    we have arch features for is predictable, regardless of whether it's in
    the measured baseline)."""
    if ARCH_CSV.is_file():
        return sorted(pd.read_csv(ARCH_CSV)["model_name"].dropna().astype(str).unique().tolist())
    if DATA_BASE.is_file():
        return sorted(pd.read_csv(DATA_BASE, usecols=["model_name"])["model_name"]
                      .dropna().astype(str).unique().tolist())
    return []


@st.cache_data(show_spinner=False)
def load_baseline_platforms() -> pd.DataFrame:
    if not DATA_BASE.is_file():
        return pd.DataFrame(columns=["cpu_name", "gpu_name"])
    df = pd.read_csv(DATA_BASE, usecols=["cpu_name", "gpu_name"], low_memory=False)
    return df.drop_duplicates().reset_index(drop=True)


# ------------------------------------------------------------------ layout

st.set_page_config(page_title="YOLO Hardware Predict", page_icon="▶️", layout="wide")
st.title("YOLO Hardware Predict")

with st.sidebar:
    st.header("Global settings")
    only = st.multiselect("ONLY (benchmarks) — pick which benchmarks to run",
                          ["cpu", "gpu", "model"], default=["cpu", "gpu", "model"])
    skip_build = st.checkbox("SKIP_BUILD (reuse existing docker images)", value=False)
    log_level = st.selectbox("LOG_LEVEL", ["DEBUG", "INFO", "WARNING", "ERROR"], index=1)
    merge_tag = st.text_input("MERGE_TAG (suffix for merged CSV)", value="")

    st.divider()
    st.caption(f"Project: `{PROJECT_ROOT}`")
    st.caption(f"Python: `{sys.executable}`")

common_env = {
    "ONLY": ",".join(only) if only else "cpu,gpu,model",
    "SKIP_BUILD": "1" if skip_build else "0",
    "LOG_LEVEL": log_level,
    "MERGE_TAG": merge_tag,
}

tab_pipeline, tab_predict, tab_maintenance, tab_data = st.tabs(
    ["Pipeline", "Predict", "Maintenance", "Data"]
)

# ---------- Pipeline --------------------------------------------------------
with tab_pipeline:
    st.subheader("Benchmark → Merge → Train")
    st.markdown(
        "1. **Build** — build the three docker images\n"
        "2. **Run** — run the benchmarks (use the sidebar 'ONLY' widget to run a subset)\n"
        "3. **Merge** — merge `tmp/data/*.csv` → `data_new/`\n"
        "4. **Train** — train CatBoost on `data_base + data_new/*.csv`"
    )

    c1, c2, c3, c4 = st.columns(4)
    if c1.button("build", use_container_width=True):
        run_make("build")
    if c2.button("run", use_container_width=True):
        run_make("run", extra_env=common_env)
    if c3.button("merge", use_container_width=True):
        run_make("merge", extra_env=common_env)
    if c4.button("train", use_container_width=True):
        run_make("train")

    st.divider()
    st.caption("Or do everything in one shot — runs `make collect` (build → bench → merge → enrich → train).")
    if st.button("collect (end-to-end on this host)", type="primary", use_container_width=True):
        run_make("collect", extra_env=common_env)

# ---------- Predict ---------------------------------------------------------
with tab_predict:
    st.subheader("Predict inference time")
    st.caption(
        "CPU/GPU dropdowns cover the full spec catalogues "
        "(~7800 CPUs, ~2900 GPUs). Hardware features come from "
        "the baseline if the pair is among the 5 already-benchmarked "
        "platforms; otherwise from `specs/*.csv`. Model dropdown lists every "
        "model with cached arch features."
    )
    cpu_names = load_cpu_names()
    gpu_names = load_gpu_names()
    models = load_models()

    with st.form("predict_form"):
        c1, c2 = st.columns(2)
        with c1:
            cpu = st.selectbox("CPU", cpu_names or [""], index=0 if cpu_names else None)
            ram = st.number_input("RAM (GB)", min_value=1, max_value=2048, value=32)
            model = st.selectbox("Model", models or ["yolov8m.pt"],
                                 index=(models.index("yolov8m.pt") if "yolov8m.pt" in models else 0))
        with c2:
            gpu = st.selectbox("GPU", gpu_names or [""], index=0 if gpu_names else None)
            used_gpu = st.checkbox("If Yolo inference on GPU", value=True)
            img_size = st.number_input("Image size (px)", min_value=32, max_value=4096,
                                        value=640, step=32)
            batch = st.number_input("Batch size", min_value=1, max_value=256, value=5)

        submitted = st.form_submit_button("predict", type="primary", use_container_width=True)

    if submitted:
        extra_env = {
            "CPU": cpu,
            "GPU": gpu,
            "RAM": str(ram),
            "MODEL": model,
            "IMG": str(img_size),
            "BATCH": str(batch),
            "USED_GPU": "1" if used_gpu else "0",
        }
        run_make("predict", extra_env=extra_env)

    st.divider()
    if st.button("List models (predict → --list-models)"):
        # direct script invocation — there's no dedicated Makefile target
        placeholder = st.container(height=400, border=True).empty()
        log: list[str] = []
        proc = subprocess.Popen(
            [sys.executable, str(PROJECT_ROOT / "src" / "predict.py"), "--list-models"],
            cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log.append(line.rstrip())
            placeholder.code("\n".join(log), language="bash")
        proc.wait()

# ---------- Maintenance -----------------------------------------------------
with tab_maintenance:
    st.subheader("Cleanup")
    st.warning("`make clean` will remove `tmp/`, `data_new/` and the benchmark docker images.")
    confirm = st.checkbox("I understand this is irreversible")
    if st.button("Clean", disabled=not confirm):
        run_make("clean")

    st.divider()
    st.subheader("make help")
    if st.button("Show all targets"):
        run_make("help")

# ---------- Data preview ----------------------------------------------------
with tab_data:
    st.subheader("Data overview")
    if DATA_BASE.is_file():
        st.markdown(f"**`data_base/data_base.csv`** — {DATA_BASE.stat().st_size / 1024:.0f} KB")
        df_base = pd.read_csv(DATA_BASE)
        st.caption(f"{len(df_base):,} rows × {len(df_base.columns)} cols")
        st.dataframe(df_base.head(200), use_container_width=True, height=300)
    else:
        st.info("`data_base/data_base.csv` not found.")

    data_new = PROJECT_ROOT / "data_new"
    if data_new.is_dir():
        csvs = sorted(data_new.glob("*.csv"))
        if csvs:
            st.markdown("**`data_new/*.csv`** (fresh benchmark runs)")
            for p in csvs:
                st.markdown(f"- `{p.relative_to(PROJECT_ROOT)}` — {p.stat().st_size / 1024:.0f} KB")
