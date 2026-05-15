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

from hardware_lookup import list_known_cpus, list_known_gpus  # noqa: E402


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
    return list_known_cpus()


@st.cache_data(show_spinner=False)
def load_gpu_names() -> list[str]:
    return list_known_gpus()


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

st.set_page_config(page_title="CV Hardware Predict", page_icon="▶️", layout="wide")
st.title("CV Hardware Predict")

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

tab_pipeline, tab_predict, tab_recommend, tab_models, tab_maintenance, tab_data = st.tabs(
    ["Pipeline", "Predict", "Recommend", "Models", "Maintenance", "Data"]
)


def model_family_of(model_name: str) -> str | None:
    """Look up family for a model_name from the arch CSV (cache-free, cheap)."""
    if not ARCH_CSV.is_file():
        return None
    df = pd.read_csv(ARCH_CSV, usecols=["model_name", "model_family"])
    hit = df[df["model_name"] == model_name]
    if hit.empty:
        return None
    return str(hit.iloc[0]["model_family"])


CUSTOM_FAMILY_HINT = (
    "Custom model — CatBoost was trained on native families (yolov8, rtdetr, "
    "timm, torchvision-detection, …). For `family=custom` the regressor "
    "extrapolates from arch features only, so accuracy is lower than for native "
    "models. Treat the band as wider than shown."
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
        "the baseline if the pair is among the 8 already-benchmarked "
        "platforms; otherwise from `specs/*.csv` (uncertainty band widens "
        "the further you stray from a benched platform). Model dropdown "
        "lists every model with cached arch features."
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
            used_gpu = st.checkbox("If CV inference on GPU", value=True)
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
            if model_family_of(model) == "custom":
                st.info(CUSTOM_FAMILY_HINT)

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
            if model_family_of(r_model) == "custom":
                st.info(CUSTOM_FAMILY_HINT)

# ---------- Models (custom uploads) -----------------------------------------
with tab_models:
    st.subheader("Add your own model")
    st.caption(
        "Upload a CV model and it appears in the Predict / Recommend dropdowns. "
        "We profile arch features once (FLOPs, params, op-histogram) and the "
        "regressor predicts latency from those — no actual inference is run."
    )
    st.markdown(
        "**Accepted formats**\n"
        "- `.pt` — full pickled `nn.Module` (saved via `torch.save(model, 'name.pt')`). "
        "State_dicts and TorchScript files are rejected.\n"
        "- `.onnx` — exported via `torch.onnx.export(...)`. Converted to torch "
        "internally with `onnx2torch`."
    )
    st.warning(
        "Loading a `.pt` file executes arbitrary pickle code. Only upload models "
        "you trust. ONNX is safer if the source is untrusted."
    )

    from custom_model_import import (  # noqa: E402
        CustomModelError,
        WEIGHTS_DIR,
        list_custom_models,
        register_custom_model,
        remove_custom_model,
    )

    with st.form("upload_model_form", clear_on_submit=False):
        uploaded = st.file_uploader("model file", type=["pt", "onnx"])
        c1, c2, c3 = st.columns([2, 1, 1])
        u_name = c1.text_input(
            "model name",
            value="",
            placeholder="e.g. my_yolo_v1 (defaults to filename)",
            help="Used as the dropdown label. Must be unique; an existing row "
                 "with the same name is replaced.",
        )
        u_family = c2.text_input("family", value="custom")
        u_img = c3.number_input("ref img size (px)", min_value=32, max_value=2048,
                                value=640, step=32)
        u_submit = st.form_submit_button("Profile and add", type="primary",
                                         use_container_width=True)

    if u_submit:
        if uploaded is None:
            st.error("pick a `.pt` or `.onnx` file first")
        else:
            stem = Path(uploaded.name).stem
            target_name = (u_name.strip() or stem)
            ext = Path(uploaded.name).suffix.lower()
            tmp_dir = PROJECT_ROOT / "tmp" / "uploads"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = tmp_dir / f"_upload_{target_name}{ext}"
            tmp_path.write_bytes(uploaded.getbuffer())

            with st.spinner(f"profiling {target_name} @ {u_img}px (one forward pass) …"):
                try:
                    feats = register_custom_model(
                        src_path=tmp_path,
                        name=target_name,
                        family=u_family or "custom",
                        img_size=int(u_img),
                    )
                except CustomModelError as e:
                    st.error(str(e))
                    feats = None
                except Exception as e:
                    st.exception(e)
                    feats = None
                finally:
                    try:
                        tmp_path.unlink()
                    except OSError:
                        pass

            if feats:
                load_models.clear()
                st.success(f"added `{target_name}` — family `{u_family or 'custom'}`")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("FLOPs", f"{feats['flops'] / 1e9:.2f} G")
                m2.metric("params", f"{feats['params'] / 1e6:.2f} M")
                m3.metric("activations", f"{feats['activation_bytes'] / 1e6:.1f} MB")
                m4.metric("# ops", f"{feats['num_ops']}")
                st.caption(
                    f"weights saved to `{(WEIGHTS_DIR / (target_name + ext)).relative_to(PROJECT_ROOT)}` · "
                    f"row appended to `data_base/model_arch_features.csv` · "
                    "switch to the Predict or Recommend tab — it's in the dropdown now."
                )

    st.divider()
    st.subheader("Uploaded models")
    custom_df = list_custom_models()
    if custom_df.empty:
        st.info("no custom models yet — upload one above.")
    else:
        for _, row in custom_df.iterrows():
            name = str(row["model_name"])
            cc1, cc2, cc3, cc4, cc5 = st.columns([3, 2, 2, 2, 1])
            cc1.write(f"**{name}**")
            cc2.caption(f"{row.get('params', 0) / 1e6:.2f} M params")
            cc3.caption(f"{row.get('flops', 0) / 1e9:.2f} GFLOPs")
            cc4.caption(f"ref {int(row.get('img_size_ref', 0))}px · {int(row.get('num_ops', 0))} ops")
            if cc5.button("delete", key=f"del_{name}"):
                if remove_custom_model(name):
                    load_models.clear()
                    st.success(f"removed `{name}`")
                    st.rerun()
                else:
                    st.error(f"`{name}` not found or is not a custom model")


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

    # Current weights / CV metrics summary
    import json as _json2
    new_meta = PROJECT_ROOT / "data_new" / "reg_weights_new" / "metadata.json"
    base_meta = PROJECT_ROOT / "data_base" / "reg_weights_base" / "metadata.json"
    meta_path = new_meta if new_meta.is_file() else (base_meta if base_meta.is_file() else None)
    if meta_path:
        meta = _json2.loads(meta_path.read_text())
        cv = meta.get("cv") or {}
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("rows trained", f"{meta.get('rows_trained', '?'):,}")
        c2.metric("groups (cpu × gpu)", cv.get("n_groups", "?"))
        c3.metric("CV R² log",
                  f"{cv.get('cv_r2_log', 0):.3f}" if cv else "—")
        c4.metric("CV MAE (s)",
                  f"{cv.get('cv_mae_sec', 0):.3f}" if cv else "—")
        st.caption(
            f"weights: `{meta_path.parent.relative_to(PROJECT_ROOT)}` · "
            f"loss: `{meta.get('model_params', {}).get('loss_function', '?')}` · "
            f"trained: {meta.get('trained_at_utc', '?')[:10]}"
        )

    if DATA_BASE.is_file():
        st.markdown(f"**`data_base/data_base.csv`** — {DATA_BASE.stat().st_size / 1024:.0f} KB")
        df_base = pd.read_csv(DATA_BASE)
        st.caption(f"{len(df_base):,} rows × {len(df_base.columns)} cols  ·  "
                   f"{df_base.groupby(['cpu_name','gpu_name']).ngroups} platforms  ·  "
                   f"{df_base['model_name'].nunique()} models")

        with st.expander("Benched platforms", expanded=False):
            plat = (df_base
                    .groupby(['cpu_name', 'gpu_name'])
                    .size().reset_index(name='rows')
                    .sort_values('rows', ascending=False))
            st.dataframe(plat, use_container_width=True, hide_index=True)

        st.dataframe(df_base.head(200), use_container_width=True, height=300)
    else:
        st.info("`data_base/data_base.csv` not found.")

    data_new = PROJECT_ROOT / "data_new"
    if data_new.is_dir():
        csvs = sorted(data_new.glob("*.csv"))
        if csvs:
            st.markdown("**`data_new/*.csv`** (fresh benchmark runs — added to "
                        "training on top of `data_base.csv`)")
            for p in csvs:
                st.markdown(f"- `{p.relative_to(PROJECT_ROOT)}` — {p.stat().st_size / 1024:.0f} KB")
            st.caption(
                "Heads-up: if any of these CSVs has already been folded into "
                "`data_base.csv`, remove it from `data_new/` before retraining — "
                "otherwise the rows get counted twice."
            )
