#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PA Transplant Equity Dashboard + Local Chatbot
Reads merged CSVs exported by your analysis script and builds an interactive app
with KPIs, charts, tables, downloads, and a simple Q&A chatbot that answers
questions from the same data.
"""

from pathlib import Path
import re
import textwrap

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="PA Transplant Equity Dashboard", layout="wide")

# Default data folder (next to this file)
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "merged_data"

# Persist last-used data path across reruns
if "data_root" not in st.session_state:
    st.session_state["data_root"] = str(DEFAULT_DATA_DIR)

# -----------------------------
# Data loading
# -----------------------------
@st.cache_data(show_spinner=False)
def load_csvs(data_dir: Path):
    data = {}
    def _read(name):
        fp = data_dir / f"{name}.csv"
        return pd.read_csv(fp) if fp.exists() else None

    data["race_timeseries"]       = _read("race_timeseries")
    data["center_summary"]        = _read("center_summary")
    data["payment_totals"]        = _read("payment_totals")
    data["payment_center_detail"] = _read("payment_center_detail")

    # Optional extras
    data["top5_centers"]          = _read("top5_centers")
    data["regional_summary"]      = _read("regional_summary")
    data["payment_totals_ranked"] = _read("payment_totals_ranked")
    return data

def pct(n, d):
    return (100 * n / d) if d not in (0, None, np.nan) and d != 0 else np.nan

# -----------------------------
# Sidebar: data path & filters
# -----------------------------
st.sidebar.header("Settings")
data_root = st.sidebar.text_input(
    "Data folder (where CSVs live)",
    value=st.session_state["data_root"],
    help="Points to the folder with CSVs exported by your analysis script."
)
st.session_state["data_root"] = data_root
data_dir = Path(data_root).expanduser().resolve()

if not data_dir.exists():
    st.error(f"Data directory not found: `{data_dir}`")
    st.stop()

data = load_csvs(data_dir)
ts   = data["race_timeseries"]
cent = data["center_summary"]
payt = data["payment_totals"]
payd = data["payment_center_detail"]

if all(v is None for v in [ts, cent, payt, payd]):
    st.error("No CSVs found. Run your analysis script to export merged_data/*.csv first.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.caption(f"Reading from:\n`{data_dir}`")

# -----------------------------
# Header
# -----------------------------
st.title("🫀 PA Transplant Equity Dashboard")
st.caption("Interactive snapshot powered by your merged datasets")

# -----------------------------
# Filters (built from available data)
# -----------------------------
with st.sidebar:
    st.subheader("Filters")

    donor_types = sorted(ts["Donor_Type"].dropna().unique()) if ts is not None and "Donor_Type" in ts.columns else []
    years = sorted(ts["Year"].dropna().unique()) if ts is not None and "Year" in ts.columns else []
    races = sorted(ts["Race_Ethnicity"].dropna().unique()) if ts is not None and "Race_Ethnicity" in ts.columns else []

    regions = sorted(payd["Region"].dropna().unique()) if payd is not None and "Region" in payd.columns else []
    urban_vals = sorted(list({bool(x) for x in (payd["Urban"].dropna().unique() if (payd is not None and "Urban" in payd.columns) else [])}))

    sel_donor = st.multiselect("Donor Type", donor_types, default=[d for d in donor_types if "Deceased" in d] or donor_types[:1])
    yr_min, yr_max = (min(years), max(years)) if years else (2016, 2025)
    sel_years = st.slider("Year range", min_value=int(yr_min), max_value=int(yr_max), value=(int(yr_min), int(yr_max)))

    sel_race = st.multiselect("Race/Ethnicity (for time series)", races,
                              default=["Black_NH"] if "Black_NH" in races else (races[:1] if races else []))

    sel_region = st.multiselect("Region (center table & chart)", regions, default=regions)
    sel_urban = st.multiselect("Urban (True/False)", urban_vals, default=urban_vals)

# -----------------------------
# KPI Cards
# -----------------------------
st.subheader("Key Metrics")
col1, col2, col3, col4 = st.columns(4)

# A) Black share (deceased donor) within selected window
with col1:
    if ts is not None and {"Donor_Type","Race_Ethnicity","Year","Transplant_Count"}.issubset(ts.columns):
        ts_f = ts[(ts["Year"] >= sel_years[0]) & (ts["Year"] <= sel_years[1])]
        if sel_donor:
            ts_f = ts_f[ts_f["Donor_Type"].isin(sel_donor)]
        totals = ts_f.groupby("Year")["Transplant_Count"].sum()
        black_alias = {"Black_NH","Black","Black or African American"}
        black = ts_f[ts_f["Race_Ethnicity"].isin(black_alias)].groupby("Year")["Transplant_Count"].sum()
        share = pct(black.sum(), totals.sum())
        st.metric("Black share (selected years, %)", f"{share:.1f}%" if share==share else "—")
    else:
        st.metric("Black share (selected years, %)", "—")

# B) Medicaid / Private ratio (all years)
with col2:
    if payt is not None and {"Payment_Category","Transplant_Count"}.issubset(payt.columns):
        g = payt.groupby("Payment_Category")["Transplant_Count"].sum()
        med, prv = g.get("Medicaid", 0), g.get("Private_Insurance", 0)
        ratio = (med / prv) if prv else np.nan
        st.metric("Medicaid / Private ratio", f"{ratio:.2f}" if ratio==ratio else "—")
    else:
        st.metric("Medicaid / Private ratio", "—")

# C) Urban/Rural ratio
with col3:
    if cent is not None and {"Urban","Total_Transplants"}.issubset(cent.columns):
        urb = cent[cent["Urban"] == True]["Total_Transplants"].sum()
        rur = cent[cent["Urban"] == False]["Total_Transplants"].sum()
        ur_ratio = (urb / rur) if rur else np.inf
        st.metric("Urban/Rural volume ratio", f"{ur_ratio:.1f}:1" if np.isfinite(ur_ratio) else "∞")
    else:
        st.metric("Urban/Rural volume ratio", "—")

# D) Top center (by volume)
with col4:
    if cent is not None and {"Center_Name","Total_Transplants"}.issubset(cent.columns) and len(cent) > 0:
        top_row = cent.sort_values("Total_Transplants", ascending=False).head(1).iloc[0]
        st.metric("Top center (by volume)", f"{str(top_row['Center_Name'])[:28]}…", delta=int(top_row["Total_Transplants"]))
    else:
        st.metric("Top center (by volume)", "—")

st.markdown("---")

# -----------------------------
# Charts Row 1: Time Series & Payment Distribution
# -----------------------------
c1, c2 = st.columns([2, 1])

with c1:
    st.markdown("### Time Series: Selected Race Share (Deceased Donor)")
    if ts is not None and {"Year","Race_Ethnicity","Donor_Type","Transplant_Count"}.issubset(ts.columns):
        t = ts[(ts["Year"] >= sel_years[0]) & (ts["Year"] <= sel_years[1])]
        if sel_donor:
            t = t[t["Donor_Type"].isin(sel_donor)]

        # totals for percent
        tot = t.groupby("Year")["Transplant_Count"].sum()

        # build share series for each selected race
        lines = []
        for r in (sel_race or []):
            sub = t[t["Race_Ethnicity"] == r].groupby("Year")["Transplant_Count"].sum()
            series = (sub / tot * 100).rename(r)
            lines.append(series)

        if lines:
            dfp = pd.concat(lines, axis=1).sort_index().reset_index().melt(
                id_vars="Year", var_name="Race_Ethnicity", value_name="Percent"
            )
            fig = px.line(dfp, x="Year", y="Percent", color="Race_Ethnicity", markers=True, template="plotly_dark")
            fig.update_layout(yaxis_title="Share of Deceased-Donor Transplants (%)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No races selected or no matching data in the chosen range.")
    else:
        st.info("race_timeseries.csv missing required columns.")

with c2:
    st.markdown("### Payment Distribution (All Years)")
    if payt is not None and {"Payment_Category","Transplant_Count"}.issubset(payt.columns):
        pay_rank = payt.groupby("Payment_Category")["Transplant_Count"].sum().reset_index()
        pay_rank = pay_rank.sort_values("Transplant_Count", ascending=True)
        fig2 = px.bar(pay_rank, x="Transplant_Count", y="Payment_Category",
                      orientation="h", template="plotly_dark")
        fig2.update_layout(xaxis_title="Count", yaxis_title="Payment Category")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("payment_totals.csv missing required columns.")

# -----------------------------
# Charts Row 2: Regional Split & Center Table
# -----------------------------
c3, c4 = st.columns([1, 2])

with c3:
    st.markdown("### Transplants by Region")
    reg = data.get("regional_summary")
    if reg is not None and {"Region","Transplants","Centers"}.issubset(reg.columns):
        reg_f = reg[reg["Region"].isin(sel_region)] if sel_region else reg
        fig3 = px.bar(reg_f.sort_values("Transplants", ascending=True),
                      x="Transplants", y="Region", text="Centers", template="plotly_dark")
        fig3.update_traces(texttemplate="Centers: %{text}")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("regional_summary.csv not found (it’s optional).")

with c4:
    st.markdown("### Centers (Filterable)")
    if cent is not None and len(cent) > 0:
        cent_f = cent.copy()
        if sel_region and "Region" in cent_f.columns:
            cent_f = cent_f[cent_f["Region"].isin(sel_region)]
        if sel_urban and "Urban" in cent_f.columns:
            cent_f = cent_f[cent_f["Urban"].isin(sel_urban)]

        sort_by = st.selectbox("Sort centers by", ["Total_Transplants","Center_Name","Region"])
        cent_f = cent_f.sort_values(sort_by, ascending=False if sort_by=="Total_Transplants" else True)

        st.dataframe(cent_f, use_container_width=True)

        st.download_button(
            "Download centers (CSV)",
            data=cent_f.to_csv(index=False).encode("utf-8"),
            file_name="centers_filtered.csv",
            mime="text/csv"
        )
    else:
        st.info("center_summary.csv missing or empty.")

st.markdown("---")

# =================================================================
# Simple In-App Chatbot (LOCAL; uses the same dataframes as above)
# =================================================================
st.header("💬 Chat")

# initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant",
         "content": "Hi! Ask me about PA transplant stats — e.g., **Black share 2021**, **Medicaid vs Private**, **Urban Rural ratio**, **Top 3 centers**."}
    ]

# helper functions
def _safe_pct(n, d):
    return (100 * n / d) if d and d == d and d != 0 else np.nan

def answer_question(q: str) -> str:
    """Very simple rule-based Q&A over loaded tables."""
    ql = q.lower().strip()

    # 1) Black share by year or range
    m = re.search(r"(black.*share|black.*percent|black.*percentage).*?(\d{4})(?:\D+(\d{4}))?", ql)
    if m and ts is not None and {"Year","Race_Ethnicity","Donor_Type","Transplant_Count"}.issubset(ts.columns):
        y1 = int(m.group(2))
        y2 = int(m.group(3)) if m.group(3) else y1
        t = ts[(ts["Year"]>=y1)&(ts["Year"]<=y2)&(ts["Donor_Type"]=="Deceased Donor")]
        totals = t.groupby("Year")["Transplant_Count"].sum()
        black_alias = {"black_nh","black","black or african american"}
        black = t[t["Race_Ethnicity"].str.lower().isin(black_alias)].groupby("Year")["Transplant_Count"].sum()
        if black.empty or totals.empty:
            return f"I don’t see data for {y1}-{y2}."
        shares = (black/totals*100).dropna()
        if y1==y2:
            val = shares.get(y1, np.nan)
            return f"**Black share {y1}**: {val:.1f}%." if val==val else "No value found."
        else:
            rng = f"{y1}-{y2}"
            w = _safe_pct(black.sum(), totals.sum())
            return f"**Black share {rng} (avg)**: {w:.1f}%."

    # 2) Medicaid vs Private ratio
    if ("medicaid" in ql and "private" in ql) and payt is not None and {"Payment_Category","Transplant_Count"}.issubset(payt.columns):
        g = payt.groupby("Payment_Category")["Transplant_Count"].sum()
        med = g.get("Medicaid",0); prv = g.get("Private_Insurance",0)
        ratio = med/prv if prv else np.nan
        return f"**Medicaid / Private ratio**: {ratio:.2f}  \nMedicaid: {med:,} | Private: {prv:,}"

    # 3) Urban / Rural ratio
    if ("urban" in ql and "rural" in ql) and cent is not None and {"Urban","Total_Transplants"}.issubset(cent.columns):
        urb = cent[cent["Urban"]==True]["Total_Transplants"].sum()
        rur = cent[cent["Urban"]==False]["Total_Transplants"].sum()
        ratio = (urb/rur) if rur else np.inf
        ratio_txt = f"{ratio:.1f}:1" if np.isfinite(ratio) else "∞"
        return f"**Urban / Rural volume ratio**: {ratio_txt}  \nUrban: {urb:,} | Rural: {rur:,}"

    # 4) Top centers (N)
    m = re.search(r"top\s*(\d+)?\s*centers", ql)
    if m and cent is not None and {"Center_Name","Total_Transplants"}.issubset(cent.columns):
        k = int(m.group(1)) if m.group(1) else 5
        topk = cent.sort_values("Total_Transplants", ascending=False).head(min(k, len(cent)))
        lines = [f"{i+1}. {row['Center_Name']} — {int(row['Total_Transplants']):,}"
                 for i, row in topk.reset_index(drop=True).iterrows()]
        return "**Top centers by volume:**\n\n" + "\n".join(lines)

    # 5) Payment breakdown
    if ("payment" in ql or "payer" in ql) and payt is not None and {"Payment_Category","Transplant_Count"}.issubset(payt.columns):
        br = (payt.groupby("Payment_Category")["Transplant_Count"].sum()
              .sort_values(ascending=False))
        lines = [f"- {idx}: {int(v):,}" for idx, v in br.items()]
        return "**Payment totals (all years):**\n\n" + "\n".join(lines)

    # 6) Help / examples
    if ql in {"help","examples","what can you do","what can you answer"}:
        return textwrap.dedent("""
        Try:
        - **Black share 2021**
        - **Black share 2018–2022**
        - **Medicaid vs Private**
        - **Urban Rural ratio**
        - **Top 3 centers**
        - **Payment breakdown**
        """)

    # Fallback
    return "I can answer quick stats from the loaded data. Try **Black share 2021**, **Medicaid vs Private**, **Top 3 centers**, or **Urban Rural ratio**."

# render chat history
for m in st.session_state.chat_history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# chat input
q = st.chat_input("Ask about this dataset…")
if q:
    st.session_state.chat_history.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    a = answer_question(q)
    st.session_state.chat_history.append({"role": "assistant", "content": a})
    with st.chat_message("assistant"):
        st.markdown(a)

# -----------------------------
# Raw Tables (expanders)
# -----------------------------
st.markdown("---")
st.subheader("Data Tables")
with st.expander("Race Time Series"):
    st.dataframe(ts if ts is not None else pd.DataFrame(), use_container_width=True)
with st.expander("Payment Totals"):
    st.dataframe(payt if payt is not None else pd.DataFrame(), use_container_width=True)
with st.expander("Payment Center Detail"):
    st.dataframe(payd if payd is not None else pd.DataFrame(), use_container_width=True)
with st.expander("Center Summary"):
    st.dataframe(cent if cent is not None else pd.DataFrame(), use_container_width=True)

st.caption("Tip: Export CSVs from your analysis pipeline, then point this app at that folder in the sidebar.")