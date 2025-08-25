import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from typing import Dict, List, Optional

st.set_page_config(page_title="Engine Test Dashboard", layout="wide")


# ---------- Helpers ----------
EXPECTED_MAP = {
    # canonical_name: possible aliases (case-insensitive, stripped)
    "BatchNumber": ["batch", "batch_no", "batchnumber", "batch id", "batch_id", "batch number"],
    "EngineModel": ["engine_model", "model", "engine model", "variant"],
    "EngineNo": ["engine_no", "engine number", "engine_id", "engine", "serial", "serial_no", "serial number"],

    # Performance metrics
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

REQUIRED_KEYS = ["BatchNumber", "EngineModel", "EngineNo"]  # always required to filter


def _normalize(s: str) -> str:
    return s.strip().lower().replace("-", "_").replace(" ", "_").replace("/", "_")


def suggest_column_mapping(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Auto-suggest mapping from EXPECTED_MAP to existing df columns."""
    norm_cols = {c: _normalize(c) for c in df.columns}
    mapping = {k: None for k in EXPECTED_MAP.keys()}
    for canon, aliases in EXPECTED_MAP.items():
        # Try exact match on canonical first
        for c, nc in norm_cols.items():
            if nc == _normalize(canon):
                mapping[canon] = c
                break
        if mapping[canon] is not None:
            continue
        # Try alias match
        alias_norms = [_normalize(a) for a in aliases]
        for c, nc in norm_cols.items():
            if nc in alias_norms:
                mapping[canon] = c
                break
    return mapping


def apply_mapping(df: pd.DataFrame, mapping: Dict[str, Optional[str]]) -> pd.DataFrame:
    """Rename columns to canonical names if present; leave others as-is."""
    rename_dict = {v: k for k, v in mapping.items() if v is not None and v in df.columns}
    return df.rename(columns=rename_dict)


def missing_required(mapping: Dict[str, Optional[str]], keys: List[str]) -> List[str]:
    return [k for k in keys if not mapping.get(k)]


def agg_by_engine(df: pd.DataFrame, metric_cols: List[str], agg_fn: str) -> pd.DataFrame:
    """Aggregate to EngineNo level for the selected metric columns."""
    agg_map = {m: agg_fn for m in metric_cols if m in df.columns}
    base_cols = ["BatchNumber", "EngineModel", "EngineNo"]
    use_cols = [c for c in base_cols if c in df.columns] + list(agg_map.keys())
    dfx = df[use_cols].copy()
    grouped = dfx.groupby("EngineNo", as_index=False).agg(agg_map)
    # bring first non-null batch/model labels per engine for context
    meta = dfx.groupby("EngineNo", as_index=False).agg({
        "BatchNumber": lambda x: x.dropna().iloc[0] if "BatchNumber" in dfx.columns else np.nan,
        "EngineModel": lambda x: x.dropna().iloc[0] if "EngineModel" in dfx.columns else np.nan,
    })
    return grouped.merge(meta, on="EngineNo", how="left")


def plot_engine_bar(df: pd.DataFrame, y_col: str, y_label: str, title: str):
    fig = px.bar(
        df.sort_values("EngineNo"),
        x="EngineNo",
        y=y_col,
        title=title,
        labels={"EngineNo": "Engine No.", y_col: y_label},
    )
    st.plotly_chart(fig, use_container_width=True)


def kpi_triplet(label: str, value, delta=None, help_text: Optional[str] = None):
    col = st.columns(1)[0]
    with col:
        st.metric(label, value, delta=delta, help=help_text)


# ---------- Sidebar ----------
st.sidebar.title("Controls")
st.sidebar.info(
    "Upload CSVs for each section. If your column names differ, use the mapping panel to align them."
)
agg_fn = st.sidebar.selectbox("Aggregation for batch/model charts", ["mean", "median", "max", "min"], index=0)


# ---------- Tabs ----------
tab_perf, tab_emis = st.tabs(["🔧 Engine Performance Testing", "🌫️ Emission Testing"])


# ---------- PERFORMANCE TAB ----------
with tab_perf:
    st.subheader("Data Upload")
    perf_file = st.file_uploader("Upload Performance CSV", type=["csv"], key="perf_csv")
    if perf_file is not None:
        try:
            perf_df_raw = pd.read_csv(perf_file)
        except Exception:
            perf_df_raw = pd.read_csv(perf_file, encoding="latin-1")

        st.caption(f"Loaded {perf_df_raw.shape[0]} rows × {perf_df_raw.shape[1]} columns.")

        # Column mapping
        st.markdown("#### Column Mapping")
        suggested = suggest_column_mapping(perf_df_raw)
        mapping = {}
        cols = ["— None —"] + list(perf_df_raw.columns)
        for canon, _ in EXPECTED_MAP.items():
            default_idx = 0
            if suggested.get(canon) in perf_df_raw.columns:
                default_idx = cols.index(suggested[canon]) if suggested[canon] in cols else 0
            mapping[canon] = st.selectbox(
                f"Map **{canon}**",
                cols,
                index=default_idx,
                key=f"perf_map_{canon}",
            )
            if mapping[canon] == "— None —":
                mapping[canon] = None

        # Apply mapping (renaming)
        perf_df = apply_mapping(perf_df_raw, mapping)

        # Validate required
        req_missing = missing_required(mapping, REQUIRED_KEYS)
        if req_missing:
            st.error(f"Missing required columns: {', '.join(req_missing)}. Map them above to proceed.")
            st.stop()

        # Filter selector
        st.markdown("### Filters")
        filter_type = st.radio(
            "Choose filter type",
            ["Batch Number", "Engine Model", "Engine Number"],
            horizontal=True,
            key="perf_filter_type",
        )

        # Dynamic filter value
        filtered_df = perf_df.copy()
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

        else:  # Engine Number
            values = sorted(filtered_df["EngineNo"].dropna().unique().tolist())
            chosen = st.selectbox("Select engine number", values)
            filtered_df = filtered_df[filtered_df["EngineNo"] == chosen]
            st.success(f"Showing engine no.: {chosen}")

        # Charts
        st.markdown("### Analytics / Charts")
        metric_cols_present = [m for m, _ in PERF_METRICS if m in filtered_df.columns]

        if filter_type in ["Batch Number", "Engine Model"]:
            # Aggregate to EngineNo for "Engine no. vs <metric>"
            eng_agg = agg_by_engine(filtered_df, metric_cols_present, agg_fn)
            # Display charts
            grid = st.container()
            with grid:
                # 2 columns * 4 rows to display 8 charts neatly
                cols = st.columns(2)
                for idx, (m, label) in enumerate(PERF_METRICS):
                    if m in eng_agg.columns:
                        with cols[idx % 2]:
                            plot_engine_bar(eng_agg, m, label, f"Engine No. vs {label}")

        else:  # Engine Number
            # Show performance result of that specific engine
            st.markdown("#### Selected Engine Summary")
            sel_engine = chosen
            eng_df = filtered_df.copy()

            # If multiple rows (e.g., multiple speed/load points), show a quick summary
            numeric_cols = eng_df.select_dtypes(include=[np.number]).columns.tolist()
            summary = eng_df[numeric_cols].agg(["mean", "min", "max"]).T.reset_index()
            summary.columns = ["Metric", "Mean", "Min", "Max"]
            st.dataframe(summary, use_container_width=True)

            # Show the eight key metrics as bars (using mean if multiple rows)
            eng_means = (
                eng_df[[c for c, _ in PERF_METRICS if c in eng_df.columns]]
                .mean(numeric_only=True)
                .to_frame(name="value")
                .reset_index()
                .rename(columns={"index": "Metric"})
            )
            # Prettify labels
            label_map = {c: lbl for c, lbl in PERF_METRICS}
            eng_means["MetricLabel"] = eng_means["Metric"].map(label_map).fillna(eng_means["Metric"])

            fig = px.bar(
                eng_means.sort_values("MetricLabel"),
                x="MetricLabel",
                y="value",
                title=f"Performance Summary — Engine {sel_engine}",
                labels={"MetricLabel": "Metric", "value": "Value"},
            )
            st.plotly_chart(fig, use_container_width=True)

            # Raw records (useful for traceability)
            with st.expander("Show raw records for selected engine"):
                st.dataframe(eng_df, use_container_width=True)

    else:
        st.info("Upload a CSV to begin with Engine Performance.")


# ---------- EMISSION TAB ----------
with tab_emis:
    st.subheader("Data Upload")
    emis_file = st.file_uploader("Upload Emission CSV", type=["csv"], key="emis_csv")
    if emis_file is not None:
        try:
            emis_df_raw = pd.read_csv(emis_file)
        except Exception:
            emis_df_raw = pd.read_csv(emis_file, encoding="latin-1")

        st.caption(f"Loaded {emis_df_raw.shape[0]} rows × {emis_df_raw.shape[1]} columns.")

        # Reuse mapping UI so filters still work (Batch/Model/EngineNo)
        st.markdown("#### Column Mapping")
        suggested_em = suggest_column_mapping(emis_df_raw)
        mapping_em: Dict[str, Optional[str]] = {}
        cols_em = ["— None —"] + list(emis_df_raw.columns)
        # Only map the three required + any columns that match emissions automatically
        base_keys = REQUIRED_KEYS
        # (Optional) If user's emission file includes performance columns, we allow mapping too—but not required.
        for canon in EXPECTED_MAP.keys():
            default_idx = 0
            if suggested_em.get(canon) in emis_df_raw.columns:
                default_idx = cols_em.index(suggested_em[canon]) if suggested_em[canon] in cols_em else 0
            mapping_em[canon] = st.selectbox(
                f"Map **{canon}**",
                cols_em,
                index=default_idx,
                key=f"emis_map_{canon}",
            )
            if mapping_em[canon] == "— None —":
                mapping_em[canon] = None

        emis_df = apply_mapping(emis_df_raw, mapping_em)
        req_missing = missing_required(mapping_em, REQUIRED_KEYS)
        if req_missing:
            st.error(f"Missing required columns for filtering: {', '.join(req_missing)}. Map them above to proceed.")
            st.stop()

        # Generic filters identical to performance
        st.markdown("### Filters")
        filter_type_e = st.radio(
            "Choose filter type",
            ["Batch Number", "Engine Model", "Engine Number"],
            horizontal=True,
            key="emis_filter_type",
        )

        emis_filtered = emis_df.copy()
        if filter_type_e == "Batch Number":
            values = sorted(emis_filtered["BatchNumber"].dropna().unique().tolist())
            chosen = st.selectbox("Select batch", values, key="emis_batch")
            emis_filtered = emis_filtered[emis_filtered["BatchNumber"] == chosen]
            st.success(f"Showing batch: {chosen}")
        elif filter_type_e == "Engine Model":
            values = sorted(emis_filtered["EngineModel"].dropna().unique().tolist())
            chosen = st.selectbox("Select engine model", values, key="emis_model")
            emis_filtered = emis_filtered[emis_filtered["EngineModel"] == chosen]
            st.success(f"Showing engine model: {chosen}")
        else:
            values = sorted(emis_filtered["EngineNo"].dropna().unique().tolist())
            chosen = st.selectbox("Select engine number", values, key="emis_engine")
            emis_filtered = emis_filtered[emis_filtered["EngineNo"] == chosen]
            st.success(f"Showing engine no.: {chosen}")

        st.markdown("### Analytics / Charts")

        # Emission datasets vary a lot. Let the user pick up to 6 numeric emission metrics to plot vs EngineNo.
        numeric_cols = [c for c in emis_filtered.columns if pd.api.types.is_numeric_dtype(emis_filtered[c])]
        # Exclude identifiers
        numeric_cols = [c for c in numeric_cols if c not in ["BatchNumber", "EngineModel", "EngineNo"]]

        if not numeric_cols:
            st.warning("No numeric emission metrics found after filtering. Check your column mapping / data.")
        else:
            st.caption("Pick up to 6 metrics to visualize vs Engine No.")
            chosen_metrics = st.multiselect(
                "Emission metrics",
                options=numeric_cols,
                default=numeric_cols[: min(6, len(numeric_cols))],
                help="Common examples: NOx, CO, HC, PM, Smoke, O2, CO2, Lambda, etc.",
            )

            if filter_type_e in ["Batch Number", "Engine Model"]:
                # Aggregate to engine level
                emis_agg = agg_by_engine(emis_filtered, chosen_metrics, agg_fn)
                cols = st.columns(2)
                for i, m in enumerate(chosen_metrics):
                    with cols[i % 2]:
                        plot_engine_bar(emis_agg, m, m, f"Engine No. vs {m}")
            else:
                # Engine-specific view: show summary and bars
                st.markdown("#### Selected Engine — Emission Summary")
                df_eng = emis_filtered.copy()
                if not df_eng.empty:
                    summ = df_eng[chosen_metrics].agg(["mean", "min", "max"]).T.reset_index()
                    summ.columns = ["Metric", "Mean", "Min", "Max"]
                    st.dataframe(summ, use_container_width=True)

                    means = (
                        df_eng[chosen_metrics]
                        .mean(numeric_only=True)
                        .to_frame(name="value")
                        .reset_index()
                        .rename(columns={"index": "Metric"})
                    )
                    fig = px.bar(
                        means.sort_values("Metric"),
                        x="Metric",
                        y="value",
                        title=f"Emission Summary — Engine {chosen}",
                        labels={"Metric": "Metric", "value": "Value"},
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with st.expander("Show raw records for selected engine"):
                    st.dataframe(df_eng, use_container_width=True)

    else:
        st.info("Upload a CSV to begin with Emission Testing.")


# ---------- Footer ----------
st.caption(
    "Tip: Use the column mapping to align your CSV headers with expected names. "
    "Batch/Model views aggregate multiple records per engine using the selected aggregation."
)
