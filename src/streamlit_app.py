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
DNS_JSON = PROJECT_ROOT / "specs" / "dns_pcs.json"

sys.path.insert(0, str(PROJECT_ROOT / "src"))


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

tab_pipeline, tab_predict, tab_recommend, tab_maintenance, tab_data = st.tabs(
    ["Pipeline", "Predict", "Recommend", "Maintenance", "Data"]
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
        from predict import build_row, predict as _predict, _default_weights
        weights_path = _default_weights()
        meta_path = weights_path.parent / "metadata.json"
        try:
            row = build_row(cpu_name=cpu, gpu_name=gpu,
                            used_gpu=1 if used_gpu else 0, ram_gb=int(ram),
                            model_name=model, img_size=int(img_size),
                            batch=int(batch))
            t_pred, t_lo, t_hi = _predict(row, weights_path, meta_path,
                                          with_uncertainty=True)
        except SystemExit as e:
            st.error(str(e))
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("predicted", f"{t_pred*1000:.1f} ms")
            c2.metric("lower (~16%)", f"{t_lo*1000:.1f} ms",
                      delta=f"-{(t_pred - t_lo)*1000:.1f}", delta_color="off")
            c3.metric("upper (~84%)", f"{t_hi*1000:.1f} ms",
                      delta=f"+{(t_hi - t_pred)*1000:.1f}", delta_color="off")
            band = "(no uncertainty: legacy weights)" if t_lo == t_hi else \
                   f"band is `exp(μ ± σ)` in log-space — wider = model is less certain about this exact hardware combo"
            st.caption(band)
            st.caption(f"weights: `{weights_path.relative_to(PROJECT_ROOT)}`")

# ---------- Recommend -------------------------------------------------------
with tab_recommend:
    st.subheader("Cheapest PC for a latency target")
    st.caption(
        "Walks `specs/dns_pcs.json` (1500 pre-built PCs scraped from "
        "dns-shop.ru/custompc/user-pc/) and returns builds whose predicted "
        "latency UPPER bound (`t_hi`, ~84%-ile) fits your target. Conservative "
        "by design — high-uncertainty unseen hardware gets rejected."
    )

    if not DNS_JSON.is_file():
        st.warning(
            f"`{DNS_JSON.relative_to(PROJECT_ROOT)}` not found. "
            f"Run `python scripts/scrape_dns_pcs.py` first."
        )
    else:
        import json as _json
        _dns_meta = _json.loads(DNS_JSON.read_text(encoding="utf-8"))
        st.caption(
            f"catalogue: {_dns_meta['scraped_count']} builds  ·  "
            f"scraped {_dns_meta['scraped_at'][:10]}  ·  "
            f"total listed on DNS: {_dns_meta.get('total_listed', '?')}"
        )

    with st.form("recommend_form"):
        c1, c2 = st.columns(2)
        with c1:
            r_model = st.selectbox(
                "Model", models or ["yolov8m.pt"],
                index=(models.index("yolov8m.pt") if "yolov8m.pt" in models else 0),
                key="r_model",
            )
            r_img = st.number_input("Image size (px)", min_value=32, max_value=4096,
                                    value=640, step=32, key="r_img")
            r_batch = st.number_input("Batch size", min_value=1, max_value=256,
                                      value=1, key="r_batch")
        with c2:
            r_latency_ms = st.number_input(
                "Max latency (ms)", min_value=1.0, max_value=10000.0,
                value=40.0, step=5.0,
                help="t_hi (upper 84% bound) must be ≤ this",
            )
            r_budget = st.number_input(
                "Max budget (₽)", min_value=10_000, max_value=2_000_000,
                value=200_000, step=10_000,
            )
            r_top_k = st.number_input("Top K (cheapest)", min_value=1,
                                       max_value=50, value=10)

        r_submitted = st.form_submit_button("find builds", type="primary",
                                            use_container_width=True,
                                            disabled=not DNS_JSON.is_file())

    if r_submitted and DNS_JSON.is_file():
        from recommend import recommend as _recommend
        with st.spinner("matching hardware + predicting latency …"):
            try:
                result = _recommend(
                    model_name=r_model,
                    img_size=int(r_img),
                    batch=int(r_batch),
                    max_latency_s=r_latency_ms / 1000.0,
                    max_budget_rub=float(r_budget),
                    dns_json=DNS_JSON,
                    top_k=int(r_top_k),
                )
            except SystemExit as e:
                st.error(str(e))
                result = pd.DataFrame()

        if result.empty:
            # Re-run with the latency cap removed to surface the achievable floor.
            probe = _recommend(
                model_name=r_model, img_size=int(r_img), batch=int(r_batch),
                max_latency_s=10.0, max_budget_rub=float(r_budget),
                dns_json=DNS_JSON, top_k=1,
            )
            if probe.empty:
                st.warning(
                    f"No builds within {r_budget:,} ₽ have matchable hardware. "
                    f"Raise the budget."
                )
            else:
                t_hi_min = probe["t_hi"].min() * 1000
                t_pred_min = probe["predicted_t_s"].min() * 1000
                st.warning(
                    f"No builds meet your latency cap of {r_latency_ms:.0f} ms "
                    f"at batch={int(r_batch)}. The fastest matchable build within "
                    f"{r_budget:,} ₽ has  t_pred ≈ {t_pred_min:.0f} ms  /  "
                    f"t_hi ≈ {t_hi_min:.0f} ms. Latency scales roughly linearly "
                    f"with batch — try batch=1 or raise the latency target above "
                    f"{t_hi_min:.0f} ms."
                )
        else:
            display = result.copy()
            display["price"] = display["price_rub"].apply(lambda x: f"{int(x):,} ₽")
            display["t_pred (ms)"] = (display["predicted_t_s"] * 1000).round(1)
            display["t_lo (ms)"] = (display["t_lo"] * 1000).round(1)
            display["t_hi (ms)"] = (display["t_hi"] * 1000).round(1)
            display = display[[
                "price", "t_pred (ms)", "t_lo (ms)", "t_hi (ms)",
                "cpu_matched", "gpu_matched", "ram_gb",
                "dns_cpu_name", "dns_gpu_name", "url",
            ]]
            st.dataframe(
                display,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "url": st.column_config.LinkColumn("link", display_text="open"),
                    "dns_cpu_name": st.column_config.TextColumn(width="medium"),
                    "dns_gpu_name": st.column_config.TextColumn(width="medium"),
                },
            )
            st.caption(
                f"Showing top {len(display)} by price. `cpu_matched` / "
                f"`gpu_matched` are the spec-catalogue names the predictor "
                f"actually used (fuzzy-matched from the DNS retail names)."
            )

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
