import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from typing import Dict, List, Optional

st.set_page_config(page_title="Engine Test Dashboard", layout="wide")

# =========================
# Helpers & Config
# =========================
EXPECTED_MAP = {
    "BatchNumber": ["batch", "batch_no", "batchnumber", "batch id", "batch_id", "batch number"],
    "EngineModel": ["engine_model", "model", "engine model", "variant"],
    "EngineNo": ["engine_no", "engine number", "engine_id", "engine", "serial", "serial_no", "serial number"],
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
REQUIRED_KEYS = ["BatchNumber", "EngineModel", "EngineNo"]

def _normalize(s: str) -> str:
    return s.strip().lower().replace("-", "_").replace(" ", "_").replace("/", "_")

def suggest_column_mapping(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    norm_cols = {c: _normalize(c) for c in df.columns}
    mapping = {k: None for k in EXPECTED_MAP.keys()}
    for canon, aliases in EXPECTED_MAP.items():
        # canonical first
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
    agg_map = {m: agg_fn for m in metric_cols if m in df.columns}
    base_cols = ["BatchNumber", "EngineModel", "EngineNo"]
    use_cols = [c for c in base_cols if c in df.columns] + list(agg_map.keys())
    dfx = df[use_cols].copy()
    grouped = dfx.groupby("EngineNo", as_index=False).agg(agg_map)
    # attach labels per engine
    meta_cols = {}
    if "BatchNumber" in dfx.columns:
        meta_cols["BatchNumber"] = lambda x: x.dropna().iloc[0] if len(x.dropna()) else np.nan
    if "EngineModel" in dfx.columns:
        meta_cols["EngineModel"] = lambda x: x.dropna().iloc[0] if len(x.dropna()) else np.nan
    if meta_cols:
        meta = dfx.groupby("EngineNo", as_index=False).agg(meta_cols)
        grouped = grouped.merge(meta, on="EngineNo", how="left")
    return grouped

def plot_engine_grouped_bars(df: pd.DataFrame, y_orig: str, y_sim: Optional[str], y_label: str, title: str):
    # if sim is None, plot single series
    data = df.sort_values("EngineNo").copy()
    if y_sim and y_sim in data.columns:
        melt = data.melt(id_vars=["EngineNo"], value_vars=[y_orig, y_sim], var_name="Series", value_name="Value")
        series_map = {y_orig: "Original", y_sim: "Simulated"}
        melt["Series"] = melt["Series"].map(series_map).fillna(melt["Series"])
        fig = px.bar(
            melt, x="EngineNo", y="Value", color="Series", barmode="group",
            title=title, labels={"EngineNo": "Engine No.", "Value": y_label}
        )
    else:
        fig = px.bar(
            data, x="EngineNo", y=y_orig,
            title=title, labels={"EngineNo": "Engine No.", y_orig: y_label}
        )
    st.plotly_chart(fig, use_container_width=True)

def safe_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def pct_apply(series: pd.Series, pct: float) -> pd.Series:
    factor = 1.0 + pct / 100.0
    return series.astype(float) * factor

# =========================
# Sidebar (global)
# =========================
st.sidebar.title("Controls")
st.sidebar.info("Upload CSVs, map columns, select a filter, and (optionally) apply simulations.")
agg_fn = st.sidebar.selectbox("Aggregation for batch/model charts", ["mean", "median", "max", "min"], index=0)

# Simulation master toggle
sim_enabled = st.sidebar.toggle("Enable Simulations", value=True, help="Apply % changes to metrics and compare.")

# =========================
# Tabs
# =========================
tab_perf, tab_emis = st.tabs(["🔧 Engine Performance Testing", "🌫️ Emission Testing"])

# =========================
# PERFORMANCE TAB
# =========================
with tab_perf:
    st.subheader("Data Upload")
    perf_file = st.file_uploader("Upload Performance CSV", type=["csv"], key="perf_csv")
    if perf_file is not None:
        try:
            perf_df_raw = pd.read_csv(perf_file)
        except Exception:
            perf_df_raw = pd.read_csv(perf_file, encoding="latin-1")
        st.caption(f"Loaded {perf_df_raw.shape[0]} rows × {perf_df_raw.shape[1]} columns.")

        st.markdown("#### Column Mapping")
        suggested = suggest_column_mapping(perf_df_raw)
        mapping = {}
        cols = ["— None —"] + list(perf_df_raw.columns)
        for canon in EXPECTED_MAP.keys():
            default_idx = 0
            if suggested.get(canon) in perf_df_raw.columns:
                default_idx = cols.index(suggested[canon]) if suggested[canon] in cols else 0
            mapping[canon] = st.selectbox(
                f"Map **{canon}**", cols, index=default_idx, key=f"perf_map_{canon}"
            )
            if mapping[canon] == "— None —":
                mapping[canon] = None

        perf_df = apply_mapping(perf_df_raw, mapping)
        req_missing = missing_required(mapping, REQUIRED_KEYS)
        if req_missing:
            st.error(f"Missing required columns: {', '.join(req_missing)}. Map them above to proceed.")
            st.stop()

        # Make numeric safe
        num_candidates = [c for c, _ in PERF_METRICS]
        perf_df = safe_numeric(perf_df, num_candidates)

        # --- Filters
        st.markdown("### Filters")
        filter_type = st.radio(
            "Choose filter type", ["Batch Number", "Engine Model", "Engine Number"],
            horizontal=True, key="perf_filter_type",
        )
        filtered_df = perf_df.copy()
        chosen = None
        if filter_type == "Batch Number":
            values = sorted(filtered_df["BatchNumber"].dropna().unique().tolist())
            chosen = st.selectbox("Select batch", values)
            filtered_df = filtered_df[filtered_df["BatchNumber"] == chosen]
            st.success(f"Showing batch: {chosen}")
        elif filter_type == "Engine Model":
            values = sorted(filtered_df["EngineModel"].dropna().unique().tolist())
            chosen = st.selectbox("Select engine model", values)
            filtered_df = filtered_df[filtered_df["EngineModel"] == chosen]
            st.success(f"Showing engine model: {chosen}")
        else:
            values = sorted(filtered_df["EngineNo"].dropna().unique().tolist())
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
            with c1:
                sim_vals_perf["ActualTorque"] = st.slider(sim_cols_perf["ActualTorque"], -50.0, 50.0, 0.0, 0.5)
            with c2:
                sim_vals_perf["ActualPower"]  = st.slider(sim_cols_perf["ActualPower"],  -50.0, 50.0, 0.0, 0.5)
            with c3:
                sim_vals_perf["BSFC"]         = st.slider(sim_cols_perf["BSFC"],         -50.0, 50.0, 0.0, 0.5)
            with c4:
                sim_vals_perf["FuelFlow"]     = st.slider(sim_cols_perf["FuelFlow"],     -50.0, 50.0, 0.0, 0.5)
            apply_to_charts = st.checkbox("Compare Original vs Simulated in charts", value=True)

        # Produce a simulated df (only selected metrics)
        sim_df = filtered_df.copy()
        if sim_enabled and not sim_df.empty:
            for metric_key in ["ActualTorque", "ActualPower", "BSFC", "FuelFlow"]:
                if metric_key in sim_df.columns:
                    sim_col = f"{metric_key}_Sim"
                    sim_df[sim_col] = pct_apply(sim_df[metric_key], sim_vals_perf[metric_key])

        # --- Charts
        st.markdown("### Analytics / Charts")
        metric_cols_present = [m for m, _ in PERF_METRICS if m in filtered_df.columns]

        if filter_type in ["Batch Number", "Engine Model"]:
            eng_agg = agg_by_engine(filtered_df, metric_cols_present, agg_fn)
            eng_agg_sim = agg_by_engine(sim_df, [f"{m}_Sim" for m in metric_cols_present], agg_fn) if sim_enabled else None
            if eng_agg_sim is not None:
                eng_agg = eng_agg.merge(eng_agg_sim[["EngineNo"] + [c for c in eng_agg_sim.columns if c.endswith("_Sim")]],
                                        on="EngineNo", how="left")

            cols = st.columns(2)
            for idx, (m, label) in enumerate(PERF_METRICS):
                if m in eng_agg.columns:
                    with cols[idx % 2]:
                        y_sim = f"{m}_Sim" if (sim_enabled and apply_to_charts and f"{m}_Sim" in eng_agg.columns) else None
                        plot_engine_grouped_bars(eng_agg, m, y_sim, label, f"Engine No. vs {label}")

            # Export aggregated
            st.download_button(
                "⬇️ Download Aggregated (current view)",
                eng_agg.to_csv(index=False).encode("utf-8"),
                file_name=f"performance_{filter_type.replace(' ','_').lower()}_aggregated.csv",
                mime="text/csv",
            )

        else:
            # Engine-specific summary
            st.markdown("#### Selected Engine Summary")
            eng_df = filtered_df.copy()
            numeric_cols = eng_df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                summary = eng_df[numeric_cols].agg(["mean", "min", "max"]).T.reset_index()
                summary.columns = ["Metric", "Mean", "Min", "Max"]
                st.dataframe(summary, use_container_width=True)

            # Build mean values for the 8 metrics (original & simulated)
            label_map = {c: lbl for c, lbl in PERF_METRICS}
            means_orig = (
                eng_df[[c for c in label_map.keys() if c in eng_df.columns]]
                .mean(numeric_only=True)
                .to_frame(name="Original")
                .reset_index().rename(columns={"index": "Metric"})
            )
            if sim_enabled:
                means_sim = (
                    sim_df[[f"{c}_Sim" for c in label_map.keys() if f"{c}_Sim" in sim_df.columns]]
                    .mean(numeric_only=True)
                    .to_frame(name="Simulated")
                    .reset_index().rename(columns={"index": "Metric_Sim"})
                )
                # unify labels
                means_sim["Metric"] = means_sim["Metric_Sim"].str.replace("_Sim", "", regex=False)
                merged = means_orig.merge(means_sim[["Metric","Simulated"]], on="Metric", how="left")
            else:
                merged = means_orig.copy()

            merged["MetricLabel"] = merged["Metric"].map(label_map).fillna(merged["Metric"])
            melt = merged.melt(id_vars=["Metric","MetricLabel"], value_vars=[c for c in ["Original","Simulated"] if c in merged.columns],
                               var_name="Series", value_name="Value")
            fig = px.bar(
                melt.sort_values(["MetricLabel","Series"]), x="MetricLabel", y="Value",
                color="Series", barmode="group",
                title=f"Performance Summary — Engine {chosen}",
                labels={"MetricLabel":"Metric", "Value":"Value"}
            )
            st.plotly_chart(fig, use_container_width=True)

            # Export filtered rows (raw & simulated)
            export = sim_df.copy() if sim_enabled else eng_df.copy()
            st.download_button(
                "⬇️ Download Rows (selected engine; incl. simulated if enabled)",
                export.to_csv(index=False).encode("utf-8"),
                file_name=f"performance_engine_{chosen}_rows.csv",
                mime="text/csv",
            )

            with st.expander("Show raw records for selected engine"):
                st.dataframe(eng_df, use_container_width=True)
    else:
        st.info("Upload a CSV to begin with Engine Performance.")

# =========================
# EMISSION TAB
# =========================
with tab_emis:
    st.subheader("Data Upload")
    emis_file = st.file_uploader("Upload Emission CSV", type=["csv"], key="emis_csv")
    if emis_file is not None:
        try:
            emis_df_raw = pd.read_csv(emis_file)
        except Exception:
            emis_df_raw = pd.read_csv(emis_file, encoding="latin-1")
        st.caption(f"Loaded {emis_df_raw.shape[0]} rows × {emis_df_raw.shape[1]} columns.")

        st.markdown("#### Column Mapping")
        suggested_em = suggest_column_mapping(emis_df_raw)
        mapping_em: Dict[str, Optional[str]] = {}
        cols_em = ["— None —"] + list(emis_df_raw.columns)
        for canon in EXPECTED_MAP.keys():
            default_idx = 0
            if suggested_em.get(canon) in emis_df_raw.columns:
                default_idx = cols_em.index(suggested_em[canon]) if suggested_em[canon] in cols_em else 0
            mapping_em[canon] = st.selectbox(
                f"Map **{canon}**", cols_em, index=default_idx, key=f"emis_map_{canon}"
            )
            if mapping_em[canon] == "— None —":
                mapping_em[canon] = None

        emis_df = apply_mapping(emis_df_raw, mapping_em)
        req_missing = missing_required(mapping_em, REQUIRED_KEYS)
        if req_missing:
            st.error(f"Missing required columns for filtering: {', '.join(req_missing)}. Map them above to proceed.")
            st.stop()

        # numeric detection after mapping
        emis_df = emis_df.copy()

        st.markdown("### Filters")
        filter_type_e = st.radio(
            "Choose filter type", ["Batch Number", "Engine Model", "Engine Number"],
            horizontal=True, key="emis_filter_type",
        )
        emis_filtered = emis_df.copy()
        chosen_e = None
        if filter_type_e == "Batch Number":
            values = sorted(emis_filtered["BatchNumber"].dropna().unique().tolist())
            chosen_e = st.selectbox("Select batch", values, key="emis_batch")
            emis_filtered = emis_filtered[emis_filtered["BatchNumber"] == chosen_e]
            st.success(f"Showing batch: {chosen_e}")
        elif filter_type_e == "Engine Model":
            values = sorted(emis_filtered["EngineModel"].dropna().unique().tolist())
            chosen_e = st.selectbox("Select engine model", values, key="emis_model")
            emis_filtered = emis_filtered[emis_filtered["EngineModel"] == chosen_e]
            st.success(f"Showing engine model: {chosen_e}")
        else:
            values = sorted(emis_filtered["EngineNo"].dropna().unique().tolist())
            chosen_e = st.selectbox("Select engine number", values, key="emis_engine")
            emis_filtered = emis_filtered[emis_filtered["EngineNo"] == chosen_e]
            st.success(f"Showing engine no.: {chosen_e}")

        st.markdown("### Analytics / Charts")
        # choose emission numeric columns
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
            # Ensure numeric
            emis_filtered = safe_numeric(emis_filtered, chosen_metrics)

            # Simulate emissions: single uniform % slider or per-metric?
            sim_vals_em = {}
            if sim_enabled:
                st.markdown("### 🔁 Simulations (Emissions)")
                uni = st.slider("Apply a uniform % change to selected emission metrics", -90.0, 90.0, 0.0, 0.5)
                per_metric = st.checkbox("Use per-metric overrides", value=False)
                if per_metric:
                    for m in chosen_metrics:
                        sim_vals_em[m] = st.slider(f"{m} % change", -90.0, 90.0, uni, 0.5, key=f"sim_{m}")
                else:
                    for m in chosen_metrics:
                        sim_vals_em[m] = uni
                apply_to_charts_e = st.checkbox("Compare Original vs Simulated in charts (emissions)", value=True)

            sim_em = emis_filtered.copy()
            if sim_enabled:
                for m in chosen_metrics:
                    if m in sim_em.columns:
                        sim_em[f"{m}_Sim"] = pct_apply(sim_em[m], sim_vals_em[m])

            if filter_type_e in ["Batch Number", "Engine Model"]:
                emis_agg = agg_by_engine(emis_filtered, chosen_metrics, agg_fn)
                emis_agg_sim = agg_by_engine(sim_em, [f"{m}_Sim" for m in chosen_metrics], agg_fn) if sim_enabled else None
                if emis_agg_sim is not None:
                    emis_agg = emis_agg.merge(
                        emis_agg_sim[["EngineNo"] + [c for c in emis_agg_sim.columns if c.endswith("_Sim")]],
                        on="EngineNo", how="left"
                    )
                cols = st.columns(2)
                for i, m in enumerate(chosen_metrics):
                    with cols[i % 2]:
                        y_sim = f"{m}_Sim" if (sim_enabled and apply_to_charts_e and f"{m}_Sim" in emis_agg.columns) else None
                        plot_engine_grouped_bars(emis_agg, m, y_sim, m, f"Engine No. vs {m}")
                # Export aggregated
                st.download_button(
                    "⬇️ Download Aggregated (current view)",
                    emis_agg.to_csv(index=False).encode("utf-8"),
                    file_name=f"emissions_{filter_type_e.replace(' ','_').lower()}_aggregated.csv",
                    mime="text/csv",
                )
            else:
                # Engine-specific emission summary
                st.markdown("#### Selected Engine — Emission Summary")
                df_eng = emis_filtered.copy()
                if not df_eng.empty and chosen_metrics:
                    summ = df_eng[chosen_metrics].agg(["mean", "min", "max"]).T.reset_index()
                    summ.columns = ["Metric", "Mean", "Min", "Max"]
                    st.dataframe(summ, use_container_width=True)

                    # Means original vs simulated
                    means_o = df_eng[chosen_metrics].mean(numeric_only=True).to_frame(name="Original").reset_index().rename(columns={"index":"Metric"})
                    if sim_enabled:
                        sims = [f"{m}_Sim" for m in chosen_metrics if f"{m}_Sim" in sim_em.columns]
                        if sims:
                            means_s = sim_em[sims].mean(numeric_only=True).to_frame(name="Simulated").reset_index().rename(columns={"index":"Metric_Sim"})
                            means_s["Metric"] = means_s["Metric_Sim"].str.replace("_Sim","",regex=False)
                            merged = means_o.merge(means_s[["Metric","Simulated"]], on="Metric", how="left")
                        else:
                            merged = means_o.copy()
                    else:
                        merged = means_o.copy()

                    melt = merged.melt(id_vars=["Metric"], value_vars=[c for c in ["Original","Simulated"] if c in merged.columns],
                                       var_name="Series", value_name="Value")
                    fig = px.bar(melt, x="Metric", y="Value", color="Series", barmode="group",
                                 title=f"Emission Summary — Engine {chosen_e}")
                    st.plotly_chart(fig, use_container_width=True)

                export_e = sim_em.copy() if sim_enabled else df_eng.copy()
                st.download_button(
                    "⬇️ Download Rows (selected engine; incl. simulated if enabled)",
                    export_e.to_csv(index=False).encode("utf-8"),
                    file_name=f"emissions_engine_{chosen_e}_rows.csv",
                    mime="text/csv",
                )

                with st.expander("Show raw records for selected engine"):
                    st.dataframe(df_eng, use_container_width=True)
    else:
        st.info("Upload a CSV to begin with Emission Testing.")

# Footer
st.caption(
    "Batch/Model views aggregate multiple records per engine using the selected aggregation. "
    "Simulations apply percentage changes to selected metrics and are shown alongside originals."
)
