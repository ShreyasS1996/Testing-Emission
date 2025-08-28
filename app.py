import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

# -----------------------------
# Optional Plotly (graceful fallback)
# -----------------------------
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_OK = True
    PLOTLY_IMPORT_ERROR = None
except Exception as _e:
    px = None
    go = None
    make_subplots = None
    PLOTLY_OK = False
    PLOTLY_IMPORT_ERROR = _e

st.set_page_config(page_title="Engine Test Dashboard", layout="wide")

# =========================
# Helpers & Config
# =========================
EXPECTED_MAP = {
    # identifiers
    "BatchNumber": ["batch", "batch_no", "batchnumber", "batch id", "batch_id", "batch number"],
    "EngineModel": ["engine_model", "model", "engine model", "variant"],
    "EngineNo": ["engine_no", "engine number", "engine_id", "engine", "serial", "serial_no", "serial number"],
    # performance metrics
    "ActualTorque": ["actual_torque", "torque", "measured torque", "act torque", "tq"],
    "ActualPower": ["actual_power", "power", "measured power", "act power", "kw", "bhp"],
    "FuelFlow": ["fuel_flow", "fuel rate", "fuel consumption", "ff", "kg/h", "lph", "l/h"],
    "BSFC": ["bsfc", "g/kwh", "brake specific fuel consumption"],
    "WaterInTemp": ["water_in_temp", "coolant in temp", "jacket water in", "jw in", "water inlet temp"],
    "WaterOutTemp": ["water_out_temp", "coolant out temp", "jacket water out", "jw out", "water outlet temp"],
    "LubeOilTemp": ["lube_oil_temp", "oil temp", "lot", "lube oil temp"],
    "FuelTemp": ["fuel_temp", "ft", "fuel temperature"],
}

PERF_METRICS = [
    ("ActualTorque", "Actual Torque"),
    ("ActualPower", "Actual Power"),
    ("FuelFlow", "Fuel Flow"),
    ("BSFC", "BSFC"),
    ("WaterInTemp", "Water In Temperature"),
    ("WaterOutTemp", "Water Out Temperature"),
    ("LubeOilTemp", "Lube Oil Temperature"),
    ("FuelTemp", "Fuel Temperature"),
]

REQUIRED_KEYS = ["BatchNumber", "EngineModel", "EngineNo"]  # required for filtering

# ---------- Robust file loader ----------
def load_table(uploaded_file) -> pd.DataFrame:
    """
    Load CSV or Excel robustly with good error messages.
    - Supports .csv, .xlsx, .xls
    - Tries delimiter inference and encoding fallbacks for CSV
    - Raises Streamlit-friendly errors when file is empty or malformed
    """
    if uploaded_file is None:
        return pd.DataFrame()

    raw = uploaded_file.read()
    uploaded_file.seek(0)

    if not raw:
        st.error("Uploaded file is empty (0 bytes). Please upload a valid CSV/XLSX file.")
        st.stop()

    name = getattr(uploaded_file, "name", "") or ""
    lower_name = name.lower()

    try:
        if lower_name.endswith(".xlsx") or lower_name.endswith(".xls"):
            return pd.read_excel(uploaded_file, engine="openpyxl")
        else:
            try:
                df = pd.read_csv(uploaded_file, sep=None, engine="python")
                return df
            except pd.errors.EmptyDataError:
                st.error("The CSV appears to have no data rows. Check that the file has a header and data.")
                st.stop()
            except Exception:
                uploaded_file.seek(0)
                try:
                    df = pd.read_csv(uploaded_file, sep=None, engine="python", encoding="latin-1")
                    return df
                except pd.errors.EmptyDataError:
                    st.error("The CSV appears to have no data rows (even after retry).")
                    st.stop()
    finally:
        uploaded_file.seek(0)

def _normalize(s: str) -> str:
    return s.strip().lower().replace("-", "_").replace(" ", "_").replace("/", "_")

def suggest_column_mapping(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    norm_cols = {c: _normalize(c) for c in df.columns}
    mapping = {k: None for k in EXPECTED_MAP.keys()}
    for canon, aliases in EXPECTED_MAP.items():
        for c, nc in norm_cols.items():
            if nc == _normalize(canon):
                mapping[canon] = c
                break
        if mapping[canon] is not None:
            continue
        alias_norms = [_normalize(a) for a in aliases]
        for c, nc in norm_cols.items():
            if nc in alias_norms:
                mapping[canon] = c
                break
    return mapping

def apply_mapping(df: pd.DataFrame, mapping: Dict[str, Optional[str]]) -> pd.DataFrame:
    rename_dict = {v: k for k, v in mapping.items() if v is not None and v in df.columns}
    return df.rename(columns=rename_dict)

def missing_required(mapping: Dict[str, Optional[str]], keys: List[str]) -> List[str]:
    return [k for k in keys if not mapping.get(k)]

def agg_by_engine(df: pd.DataFrame, metric_cols: List[str], agg_fn: str) -> pd.DataFrame:
    present = [m for m in metric_cols if m in df.columns]
    if not present or df.empty:
        return df.iloc[0:0].copy()
    agg_map = {m: agg_fn for m in present}
    base_cols = [c for c in ["EngineNo", "BatchNumber", "EngineModel"] if c in df.columns]
    dfx = df[base_cols + present].copy()
    grouped = dfx.groupby("EngineNo", as_index=False).agg(agg_map)
    meta_cols = {}
    if "BatchNumber" in dfx.columns:
        meta_cols["BatchNumber"] = lambda x: x.dropna().iloc[0] if len(x.dropna()) else np.nan
    if "EngineModel" in dfx.columns:
        meta_cols["EngineModel"] = lambda x: x.dropna().iloc[0] if len(x.dropna()) else np.nan
    if meta_cols:
        meta = dfx.groupby("EngineNo", as_index=False).agg(meta_cols)
        grouped = grouped.merge(meta, on="EngineNo", how="left")
    return grouped

def safe_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def pct_apply(series: pd.Series, pct: float) -> pd.Series:
    factor = 1.0 + pct / 100.0
    return pd.to_numeric(series, errors="coerce") * factor

# --------- Single-metric line (used mainly for EngineNo view summaries) ----------
def plot_engine_lines(df: pd.DataFrame, y_orig: str, y_sim: Optional[str],
                      y_label: str, title: str):
    if df.empty or "EngineNo" not in df.columns or y_orig not in df.columns:
        st.info("Nothing to plot for this selection.")
        return
    data = df.sort_values("EngineNo").copy()
    if PLOTLY_OK:
        if y_sim and y_sim in data.columns:
            melt = data.melt(id_vars=["EngineNo"], value_vars=[y_orig, y_sim],
                             var_name="Series", value_name="Value")
            series_map = {y_orig: "Original", y_sim: "Simulated"}
            melt["Series"] = melt["Series"].map(series_map).fillna(melt["Series"])
            fig = px.line(melt, x="EngineNo", y="Value", color="Series", markers=True,
                          title=title, labels={"EngineNo": "Engine No.", "Value": y_label})
        else:
            fig = px.line(data, x="EngineNo", y=y_orig, markers=True,
                          title=title, labels={"EngineNo": "Engine No.", y_orig: y_label})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Plotly is not installed; showing simplified line chart.")
        if y_sim and y_sim in data.columns:
            wide = data[["EngineNo", y_orig, y_sim]].set_index("EngineNo")
            st.line_chart(wide)
        else:
            simple = data[["EngineNo", y_orig]].set_index("EngineNo")
            st.line_chart(simple)

# --------- Multi-metric overlay with optional secondary axis ----------
def plot_overlay_lines(
    df: pd.DataFrame,
    primary_metrics: List[str],
    secondary_metric: Optional[str],
    label_map: Dict[str, str],
    title: str,
    compare_sim: bool,
):
    if df.empty or "EngineNo" not in df.columns:
        st.info("Nothing to plot for this selection.")
        return

    data = df.sort_values("EngineNo").copy()
    prim = [m for m in primary_metrics if m in data.columns]
    sec = secondary_metric if (secondary_metric and secondary_metric in data.columns) else None

    if not prim and not sec:
        st.info("No selected metrics available in the current view.")
        return

    series = []
    for m in prim:
        series.append((m, "primary"))
        if compare_sim and f"{m}_Sim" in data.columns:
            series.append((f"{m}_Sim", "primary"))
    if sec:
        series.append((sec, "secondary"))
        if compare_sim and f"{sec}_Sim" in data.columns:
            series.append((f"{sec}_Sim", "secondary"))

    if not series:
        st.info("No plottable series found for the selected options.")
        return

    if PLOTLY_OK and go is not None and make_subplots is not None:
        use_secondary = any(ax == "secondary" for _, ax in series)
        fig = make_subplots(specs=[[{"secondary_y": use_secondary}]])
        xvals = data["EngineNo"].astype(str).tolist()

        for s_col, axis in series:
            base = s_col.replace("_Sim", "")
            sim_flag = " (Sim)" if s_col.endswith("_Sim") else ""
            name = f"{label_map.get(base, base)}{sim_flag}"
            fig.add_trace(
                go.Scatter(x=xvals, y=data[s_col], mode="lines+markers", name=name),
                secondary_y=(axis == "secondary"),
            )

        prim_names = [label_map.get(c.replace("_Sim",""), c.replace("_Sim","")) for c, ax in series if ax == "primary"]
        sec_names  = [label_map.get(c.replace("_Sim",""), c.replace("_Sim","")) for c, ax in series if ax == "secondary"]

        fig.update_layout(
            title=title,
            legend_title_text="Series",
            hovermode="x unified",
            margin=dict(l=10, r=10, t=50, b=10),
        )
        fig.update_xaxes(title_text="Engine No.")
        fig.update_yaxes(title_text=", ".join(sorted(set(prim_names))) or "Value", secondary_y=False)
        if use_secondary:
            fig.update_yaxes(title_text=", ".join(sorted(set(sec_names))) or "Value", secondary_y=True)

        st.plotly_chart(fig, use_container_width=True)
    else:
        # Fallback: normalize each series to [0,1] and overlay on a single axis
        st.warning("Plotly is not installed; showing normalized overlay on a single axis.")
        norm_df = pd.DataFrame({"EngineNo": data["EngineNo"].astype(str)})
        col_names = []
        for s_col, axis in series:
            y = pd.to_numeric(data[s_col], errors="coerce")
            y_min, y_max = np.nanmin(y), np.nanmax(y)
            if np.isfinite(y_min) and np.isfinite(y_max) and y_max != y_min:
                y_norm = (y - y_min) / (y_max - y_min)
            else:
                y_norm = y * 0
            base = s_col.replace("_Sim", "")
            sim_flag = " (Sim)" if s_col.endswith("_Sim") else ""
            name = f"{label_map.get(base, base)}{sim_flag}"
            norm_df[name] = y_norm
            col_names.append(name)
        norm_df = norm_df.set_index("EngineNo")
        st.line_chart(norm_df[col_names])
        st.caption("Note: normalized to [0,1] per series due to missing Plotly (no secondary axis).")

# =========================
# Sidebar (global)
# =========================
st.sidebar.title("Controls")
st.sidebar.info("Upload CSV/XLSX, map columns, select a filter, and (optionally) apply simulations.")

if not PLOTLY_OK:
    st.sidebar.error("Plotly is not available. Install `plotly==5.23.0` to enable rich charts.")

agg_fn = st.sidebar.selectbox("Aggregation for batch/model charts", ["mean", "median", "max", "min"], index=0)
sim_enabled = st.sidebar.toggle("Enable Simulations", value=True, help="Apply % changes and compare Original vs Simulated.")

# =========================
# Tabs
# =========================
tab_perf, tab_emis = st.tabs(["🔧 Engine Performance Testing", "🌫️ Emission Testing"])

# =========================
# PERFORMANCE TAB
# =========================
with tab_perf:
    st.subheader("Data Upload")
    perf_file = st.file_uploader("Upload Performance file (CSV/XLSX)", type=["csv", "xlsx", "xls"], key="perf_csv")

    if perf_file is not None:
        perf_df_raw = load_table(perf_file)
        perf_df_raw = perf_df_raw.dropna(axis=1, how="all").dropna(axis=0, how="all")
        if perf_df_raw.empty:
            st.error("The uploaded Performance file has no usable data after removing empty rows/columns.")
            st.stop()

        st.caption(f"Loaded {perf_df_raw.shape[0]} rows × {perf_df_raw.shape[1]} columns.")

        st.markdown("#### Column Mapping")
        suggested = suggest_column_mapping(perf_df_raw)
        mapping: Dict[str, Optional[str]] = {}
        cols = ["— None —"] + list(perf_df_raw.columns)
        for canon in EXPECTED_MAP.keys():
            default_idx = 0
            if suggested.get(canon) in perf_df_raw.columns:
                default_idx = cols.index(suggested[canon]) if suggested[canon] in cols else 0
            mapping[canon] = st.selectbox(f"Map **{canon}**", cols, index=default_idx, key=f"perf_map_{canon}")
            if mapping[canon] == "— None —":
                mapping[canon] = None

        perf_df = apply_mapping(perf_df_raw, mapping)
        req_missing = missing_required(mapping, REQUIRED_KEYS)
        if req_missing:
            st.error(f"Missing required columns: {', '.join(req_missing)}. Map them above to proceed.")
            st.stop()

        perf_df = safe_numeric(perf_df, [c for c, _ in PERF_METRICS])

        st.markdown("### Filters")
        filter_type = st.radio("Choose filter type",
                               ["Batch Number", "Engine Model", "Engine Number"],
                               horizontal=True, key="perf_filter_type")
        filtered_df = perf_df.copy()
        chosen = None

        if filter_type == "Batch Number":
            values = sorted(filtered_df["BatchNumber"].dropna().unique().tolist())
            if not values:
                st.error("No batch values found in the data after mapping.")
                st.stop()
            chosen = st.selectbox("Select batch", values)
            filtered_df = filtered_df[filtered_df["BatchNumber"] == chosen]
            st.success(f"Showing batch: {chosen}")

        elif filter_type == "Engine Model":
            values = sorted(filtered_df["EngineModel"].dropna().unique().tolist())
            if not values:
                st.error("No engine model values found in the data after mapping.")
                st.stop()
            chosen = st.selectbox("Select engine model", values)
            filtered_df = filtered_df[filtered_df["EngineModel"] == chosen]
            st.success(f"Showing engine model: {chosen}")

        else:
            values = sorted(filtered_df["EngineNo"].dropna().unique().tolist())
            if not values:
                st.error("No engine numbers found in the data after mapping.")
                st.stop()
            chosen = st.selectbox("Select engine number", values)
            filtered_df = filtered_df[filtered_df["EngineNo"] == chosen]
            st.success(f"Showing engine no.: {chosen}")

        # --- Simulation controls (Performance)
        sim_cols_perf = {
            "ActualTorque": "Torque % change",
            "ActualPower": "Power % change",
            "BSFC": "BSFC % change",
            "FuelFlow": "Fuel Flow % change",
        }
        sim_vals_perf = {k: 0.0 for k in sim_cols_perf}
        if sim_enabled:
            st.markdown("### 🔁 Simulations (Performance)")
            c1, c2, c3, c4 = st.columns(4)
            with c1: sim_vals_perf["ActualTorque"] = st.slider(sim_cols_perf["ActualTorque"], -50.0, 50.0, 0.0, 0.5)
            with c2: sim_vals_perf["ActualPower"] = st.slider(sim_cols_perf["ActualPower"], -50.0, 50.0, 0.0, 0.5)
            with c3: sim_vals_perf["BSFC"] = st.slider(sim_cols_perf["BSFC"], -50.0, 50.0, 0.0, 0.5)
            with c4: sim_vals_perf["FuelFlow"] = st.slider(sim_cols_perf["FuelFlow"], -50.0, 50.0, 0.0, 0.5)
            apply_to_charts = st.checkbox("Compare Original vs Simulated in charts", value=True, key="perf_cmp")

        sim_df = filtered_df.copy()
        if sim_enabled and not sim_df.empty:
            for metric_key in ["ActualTorque", "ActualPower", "BSFC", "FuelFlow"]:
                if metric_key in sim_df.columns:
                    sim_df[f"{metric_key}_Sim"] = pct_apply(sim_df[metric_key], sim_vals_perf[metric_key])

        st.markdown("### Analytics / Charts")
        metric_cols_present = [m for m, _ in PERF_METRICS if m in filtered_df.columns]

        if filter_type in ["Batch Number", "Engine Model"]:
            eng_agg = agg_by_engine(filtered_df, metric_cols_present, agg_fn)
            eng_agg_sim = agg_by_engine(sim_df, [f"{m}_Sim" for m in metric_cols_present], agg_fn) if sim_enabled else None
            if eng_agg_sim is not None and not eng_agg_sim.empty:
                join_cols = ["EngineNo"] + [c for c in eng_agg_sim.columns if c.endswith("_Sim")]
                eng_agg = eng_agg.merge(eng_agg_sim[join_cols], on="EngineNo", how="left")

            # === Overlay controls (Performance)
            options_perf = [m for m, _ in PERF_METRICS if m in eng_agg.columns]
            label_map_perf = {c: lbl for c, lbl in PERF_METRICS}

            st.markdown("#### 📈 Overlay Chart (select multiple)")
            c1, c2 = st.columns([2, 1])
            with c1:
                prim_sel_perf = st.multiselect(
                    "Primary-axis metrics",
                    options=options_perf,
                    default=[m for m in ["ActualTorque", "ActualPower"] if m in options_perf],
                    help="Choose 1–3 metrics for the primary Y-axis",
                )
            with c2:
                sec_options = ["— None —"] + options_perf
                default_idx = sec_options.index("BSFC") if "BSFC" in options_perf else 0
                sec_sel_perf = st.selectbox(
                    "Secondary-axis metric (optional)",
                    sec_options,
                    index=default_idx if default_idx < len(sec_options) else 0,
                    help="Good choice: BSFC on secondary axis",
                )
            sec_metric_perf = None if sec_sel_perf == "— None —" else sec_sel_perf

            plot_overlay_lines(
                eng_agg,
                primary_metrics=prim_sel_perf,
                secondary_metric=sec_metric_perf,
                label_map=label_map_perf,
                title="EngineNo overlays (Performance)",
                compare_sim=(sim_enabled and apply_to_charts),
            )

            st.download_button(
                "⬇️ Download Aggregated (current view)",
                eng_agg.to_csv(index=False).encode("utf-8"),
                file_name=f"performance_{filter_type.replace(' ','_').lower()}_aggregated.csv",
                mime="text/csv",
            )

        else:
            st.markdown("#### Selected Engine Summary")
            eng_df = filtered_df.copy()
            numeric_cols = eng_df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                summary = eng_df[numeric_cols].agg(["mean", "min", "max"]).T.reset_index()
                summary.columns = ["Metric", "Mean", "Min", "Max"]
                st.dataframe(summary, use_container_width=True)

            label_map = {c: lbl for c, lbl in PERF_METRICS}
            means_orig = (
                eng_df[[c for c in label_map.keys() if c in eng_df.columns]]
                .mean(numeric_only=True).to_frame(name="Original").reset_index().rename(columns={"index": "Metric"})
            )
            if sim_enabled:
                sim_cols = [f"{c}_Sim" for c in label_map.keys() if f"{c}_Sim" in sim_df.columns]
                if sim_cols:
                    means_sim = (
                        sim_df[sim_cols].mean(numeric_only=True).to_frame(name="Simulated")
                        .reset_index().rename(columns={"index": "Metric_Sim"})
                    )
                    means_sim["Metric"] = means_sim["Metric_Sim"].str.replace("_Sim", "", regex=False)
                    merged = means_orig.merge(means_sim[["Metric", "Simulated"]], on="Metric", how="left")
                else:
                    merged = means_orig.copy()
            else:
                merged = means_orig.copy()

            merged["MetricLabel"] = merged["Metric"].map(label_map).fillna(merged["Metric"])

            if PLOTLY_OK:
                melt = merged.melt(
                    id_vars=["Metric", "MetricLabel"],
                    value_vars=[c for c in ["Original", "Simulated"] if c in merged.columns],
                    var_name="Series", value_name="Value"
                )
                fig = px.line(
                    melt.sort_values(["MetricLabel", "Series"]),
                    x="MetricLabel", y="Value", color="Series", markers=True,
                    title=f"Performance Summary — Engine {chosen}",
                    labels={"MetricLabel": "Metric", "Value": "Value"}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Plotly is not installed; showing simplified line chart.")
                simple = merged.set_index("MetricLabel")[["Original"] + (["Simulated"] if "Simulated" in merged.columns else [])]
                st.line_chart(simple)

            export = (sim_df if sim_enabled else eng_df).copy()
            st.download_button(
                "⬇️ Download Rows (selected engine; incl. simulated if enabled)",
                export.to_csv(index=False).encode("utf-8"),
                file_name=f"performance_engine_{chosen}_rows.csv",
                mime="text/csv",
            )

            with st.expander("Show raw records for selected engine"):
                st.dataframe(eng_df, use_container_width=True)
    else:
        st.info("Upload a CSV/XLSX to begin with Engine Performance.")

# =========================
# EMISSION TAB
# =========================
with tab_emis:
    st.subheader("Data Upload")
    emis_file = st.file_uploader("Upload Emission file (CSV/XLSX)", type=["csv", "xlsx", "xls"], key="emis_csv")

    if emis_file is not None:
        emis_df_raw = load_table(emis_file)
        emis_df_raw = emis_df_raw.dropna(axis=1, how="all").dropna(axis=0, how="all")
        if emis_df_raw.empty:
            st.error("The uploaded Emission file has no usable data after removing empty rows/columns.")
            st.stop()

        st.caption(f"Loaded {emis_df_raw.shape[0]} rows × {emis_df_raw.shape[1]} columns.")

        st.markdown("#### Column Mapping")
        suggested_em = suggest_column_mapping(emis_df_raw)
        mapping_em: Dict[str, Optional[str]] = {}
        cols_em = ["— None —"] + list(emis_df_raw.columns)
        for canon in EXPECTED_MAP.keys():
            default_idx = 0
            if suggested_em.get(canon) in emis_df_raw.columns:
                default_idx = cols_em.index(suggested_em[canon]) if suggested_em[canon] in cols_em else 0
            mapping_em[canon] = st.selectbox(f"Map **{canon}**", cols_em, index=default_idx, key=f"emis_map_{canon}")
            if mapping_em[canon] == "— None —":
                mapping_em[canon] = None

        emis_df = apply_mapping(emis_df_raw, mapping_em)
        req_missing = missing_required(mapping_em, REQUIRED_KEYS)
        if req_missing:
            st.error(f"Missing required columns for filtering: {', '.join(req_missing)}. Map them above to proceed.")
            st.stop()

        st.markdown("### Filters")
        filter_type_e = st.radio("Choose filter type",
                                 ["Batch Number", "Engine Model", "Engine Number"],
                                 horizontal=True, key="emis_filter_type")
        emis_filtered = emis_df.copy()
        chosen_e = None

        if filter_type_e == "Batch Number":
            values = sorted(emis_filtered["BatchNumber"].dropna().unique().tolist())
            if not values:
                st.error("No batch values found in the Emission data after mapping.")
                st.stop()
            chosen_e = st.selectbox("Select batch", values, key="emis_batch")
            emis_filtered = emis_filtered[emis_filtered["BatchNumber"] == chosen_e]
            st.success(f"Showing batch: {chosen_e}")

        elif filter_type_e == "Engine Model":
            values = sorted(emis_filtered["EngineModel"].dropna().unique().tolist())
            if not values:
                st.error("No engine model values found in the Emission data after mapping.")
                st.stop()
            chosen_e = st.selectbox("Select engine model", values, key="emis_model")
            emis_filtered = emis_filtered[emis_filtered["EngineModel"] == chosen_e]
            st.success(f"Showing engine model: {chosen_e}")

        else:
            values = sorted(emis_filtered["EngineNo"].dropna().unique().tolist())
            if not values:
                st.error("No engine numbers found in the Emission data after mapping.")
                st.stop()
            chosen_e = st.selectbox("Select engine number", values, key="emis_engine")
            emis_filtered = emis_filtered[emis_filtered["EngineNo"] == chosen_e]
            st.success(f"Showing engine no.: {chosen_e}")

        st.markdown("### Analytics / Charts")
        numeric_cols = [c for c in emis_filtered.columns if pd.api.types.is_numeric_dtype(emis_filtered[c])]
        numeric_cols = [c for c in numeric_cols if c not in ["BatchNumber", "EngineModel", "EngineNo"]]

        if not numeric_cols:
            st.warning("No numeric emission metrics found after filtering. Check column mapping / data.")
        else:
            st.caption("Pick up to 6 emission metrics to visualize vs Engine No.")
            chosen_metrics = st.multiselect(
                "Emission metrics",
                options=numeric_cols,
                default=numeric_cols[: min(6, len(numeric_cols))],
                help="Examples: NOx, CO, HC, PM, Smoke, O2, CO2, Lambda, etc.",
            )
            emis_filtered = safe_numeric(emis_filtered, chosen_metrics)

            # Simulations
            sim_vals_em: Dict[str, float] = {}
            if sim_enabled:
                st.markdown("### 🔁 Simulations (Emissions)")
                uni = st.slider("Apply a uniform % change to selected emission metrics",
                                -90.0, 90.0, 0.0, 0.5, key="emis_uni")
                per_metric = st.checkbox("Use per-metric overrides", value=False, key="emis_overrides")
                if per_metric:
                    for m in chosen_metrics:
                        sim_vals_em[m] = st.slider(f"{m} % change", -90.0, 90.0, uni, 0.5, key=f"sim_{m}")
                else:
                    for m in chosen_metrics:
                        sim_vals_em[m] = uni
                apply_to_charts_e = st.checkbox("Compare Original vs Simulated in charts (emissions)",
                                                value=True, key="emis_cmp")

            sim_em = emis_filtered.copy()
            if sim_enabled:
                for m in chosen_metrics:
                    if m in sim_em.columns:
                        sim_em[f"{m}_Sim"] = pct_apply(sim_em[m], sim_vals_em[m])

            if filter_type_e in ["Batch Number", "Engine Model"]:
                emis_agg = agg_by_engine(emis_filtered, chosen_metrics, agg_fn)
                emis_agg_sim = agg_by_engine(sim_em, [f"{m}_Sim" for m in chosen_metrics], agg_fn) if sim_enabled else None
                if emis_agg_sim is not None and not emis_agg_sim.empty:
                    join_cols = ["EngineNo"] + [c for c in emis_agg_sim.columns if c.endswith("_Sim")]
                    emis_agg = emis_agg.merge(emis_agg_sim[join_cols], on="EngineNo", how="left")

                # === Overlay controls (Emissions)
                options_em = [m for m in chosen_metrics if m in emis_agg.columns]
                label_map_em = {m: m for m in options_em}

                st.markdown("#### 📈 Overlay Chart (select multiple)")
                c1, c2 = st.columns([2, 1])
                with c1:
                    prim_sel_em = st.multiselect(
                        "Primary-axis metrics",
                        options=options_em,
                        default=options_em[: min(2, len(options_em))],
                        help="Choose 1–3 metrics for the primary Y-axis",
                    )
                with c2:
                    sec_options_em = ["— None —"] + options_em
                    sec_sel_em = st.selectbox(
                        "Secondary-axis metric (optional)",
                        sec_options_em,
                        index=0,
                        help="Pick a metric with a different scale, if needed",
                    )
                sec_metric_em = None if sec_sel_em == "— None —" else sec_sel_em

                plot_overlay_lines(
                    emis_agg,
                    primary_metrics=prim_sel_em,
                    secondary_metric=sec_metric_em,
                    label_map=label_map_em,
                    title="EngineNo overlays (Emissions)",
                    compare_sim=(sim_enabled and apply_to_charts_e),
                )

                st.download_button(
                    "⬇️ Download Aggregated (current view)",
                    emis_agg.to_csv(index=False).encode("utf-8"),
                    file_name=f"emissions_{filter_type_e.replace(' ','_').lower()}_aggregated.csv",
                    mime="text/csv",
                )
            else:
                st.markdown("#### Selected Engine — Emission Summary")
                df_eng = emis_filtered.copy()
                if not df_eng.empty and chosen_metrics:
                    summ = df_eng[chosen_metrics].agg(["mean", "min", "max"]).T.reset_index()
                    summ.columns = ["Metric", "Mean", "Min", "Max"]
                    st.dataframe(summ, use_container_width=True)

                    means_o = (
                        df_eng[chosen_metrics].mean(numeric_only=True)
                        .to_frame(name="Original").reset_index().rename(columns={"index": "Metric"})
                    )

                    if sim_enabled:
                        sims = [f"{m}_Sim" for m in chosen_metrics if f"{m}_Sim" in sim_em.columns]
                        if sims:
                            means_s = (
                                sim_em[sims].mean(numeric_only=True)
                                .to_frame(name="Simulated").reset_index().rename(columns={"index": "Metric_Sim"})
                            )
                            means_s["Metric"] = means_s["Metric_Sim"].str.replace("_Sim", "", regex=False)
                            merged = means_o.merge(means_s[["Metric", "Simulated"]], on="Metric", how="left")
                        else:
                            merged = means_o.copy()
                    else:
                        merged = means_o.copy()

                    if PLOTLY_OK:
                        melt = merged.melt(
                            id_vars=["Metric"],
                            value_vars=[c for c in ["Original", "Simulated"] if c in merged.columns],
                            var_name="Series", value_name="Value"
                        )
                        fig = px.line(melt, x="Metric", y="Value", color="Series", markers=True,
                                      title=f"Emission Summary — Engine {chosen_e}")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Plotly is not installed; showing simplified line chart.")
                        simple = merged.set_index("Metric")[["Original"] + (["Simulated"] if "Simulated" in merged.columns else [])]
                        st.line_chart(simple)

                export_e = (sim_em if sim_enabled else df_eng).copy()
                st.download_button(
                    "⬇️ Download Rows (selected engine; incl. simulated if enabled)",
                    export_e.to_csv(index=False).encode("utf-8"),
                    file_name=f"emissions_engine_{chosen_e}_rows.csv",
                    mime="text/csv",
                )

                with st.expander("Show raw records for selected engine"):
                    st.dataframe(df_eng, use_container_width=True)
    else:
        st.info("Upload a CSV/XLSX to begin with Emission Testing.")

# Footer
st.caption(
    "Batch/Model views aggregate multiple records per engine using the selected aggregation. "
    "Simulations apply percentage changes to selected metrics and are shown alongside originals. "
    "Overlay charts support a secondary Y-axis for mixed-scale metrics."
)
