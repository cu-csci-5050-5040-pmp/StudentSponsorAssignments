# app.py
# -------------------------------------------------------------
# A Streamlit app for visualizing and
# improving student→project assignments against top-4 preferences.
# -------------------------------------------------------------
# Features
# - Upload 2 CSVs:
#     preferences.csv -> columns: student, pref1, pref2, pref3, pref4
#     assignments.csv -> columns: student, project
# - Adjustable non-preferred cost (default 10)
# - Live cost table with badges and totals
# - Edit assignments inline (no-code) via st.data_editor
# - Visualizations: cost distribution (bar), Sankey flow
# - Greedy local improve (pairwise swaps)
# - Exact optimize (Hungarian) when #students == #projects
# - Download updated assignments with per-student cost
# -------------------------------------------------------------

import io
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.optimize import linear_sum_assignment
import unicodedata, re

st.set_page_config(
    page_title="Student-Project Assignment Optimizer",
    page_icon="🎓",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container{
        padding-top: 1.1rem !important;
        padding-bottom: 1.0rem !important;
      }

      h1{
        margin-top: 0.2rem !important;
        padding-top: 0rem !important;
        margin-bottom: 0.35rem !important;
      }

      [data-testid="stSidebarContent"]{
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
      }

      [data-testid="stSidebarContent"] h1,
      [data-testid="stSidebarContent"] h2,
      [data-testid="stSidebarContent"] h3{
        margin-top: 0.2rem !important;
        margin-bottom: 0.4rem !important;
      }

      [data-testid="stSidebarContent"] .stFileUploader,
      [data-testid="stSidebarContent"] .stNumberInput,
      [data-testid="stSidebarContent"] .stCheckbox,
      [data-testid="stSidebarContent"] .stButton,
      [data-testid="stSidebarContent"] .stAlert,
      [data-testid="stSidebarContent"] .stCaption{
        margin-top: 0.25rem !important;
        margin-bottom: 0.25rem !important;
      }
      
      [data-testid="stSidebarUserContent"]{
        padding-top: 2.5rem !important;
        padding-bottom: 0.5rem !important;
      }

      [data-testid="stSidebarNav"]{
        padding-top: 0.25rem !important;
      }
      [data-testid="stSidebarNav"] li{
        margin: 0.08rem 0 !important;
      }
      [data-testid="stSidebarNav"] a{
        padding: 0.22rem 0.35rem !important;
        line-height: 1.15 !important;
      }

      [data-testid="stSidebarHeader"]{
        padding-top: 0.25rem !important;
        padding-bottom: 0.25rem !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)



# -------------------- Session State Init --------------------
if "ready" not in st.session_state:
    st.session_state.ready = False
if "missing_files_error" not in st.session_state:
    st.session_state.missing_files_error = False
PRIMARY = "#0ea5e9"

# ------------------------- Utilities -------------------------

def _canon_text(s: str) -> str:
    s = "" if pd.isna(s) else str(s)
    s = unicodedata.normalize("NFKC", s).strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _canon_project(s: str) -> str:
    return _canon_text(s).casefold()

def _canon_student(s: str) -> str:
    s = _canon_text(s)
    if "," in s:
        last, first = [p.strip() for p in s.split(",", 1)]
        s = f"{first} {last}"
    return s.casefold()

def rank_cost(project: str, prefs_canon: list[str], non_pref_cost: int) -> int:
    p = _canon_project(project)
    try:
        return prefs_canon.index(p) + 1
    except ValueError:
        return int(non_pref_cost)

def compute_costs(assign_df: pd.DataFrame, pref_map: dict, non_pref_cost: int) -> pd.DataFrame:
    rows = []
    for _, r in assign_df.iterrows():
        stu_key = _canon_student(r["student"])
        prefs = pref_map.get(stu_key, [])
        c = rank_cost(r["project"], prefs, non_pref_cost)
        rows.append({"student": r["student"], "project": r["project"], "cost": c})
    return pd.DataFrame(rows)

def greedy_improve(assign_df: pd.DataFrame, pref_map: dict, non_pref_cost: int, max_iter: int = 100) -> pd.DataFrame:
    df = assign_df.copy().reset_index(drop=True)
    def total(df_):
        return compute_costs(df_, pref_map, non_pref_cost)["cost"].sum()

    best_total = total(df)
    n = len(df)
    improved = False
    for _ in range(max_iter):
        best_delta = 0
        best_pair: Tuple[int, int] | None = None
        for i in range(n):
            for j in range(i + 1, n):
                swapped = df.copy()
                pi, pj = swapped.at[i, "project"], swapped.at[j, "project"]
                swapped.at[i, "project"], swapped.at[j, "project"] = pj, pi
                new_total = total(swapped)
                delta = best_total - new_total
                if delta > best_delta:
                    best_delta = delta
                    best_pair = (i, j)
        if best_pair is None:
            break
        i, j = best_pair
        df.at[i, "project"], df.at[j, "project"] = df.at[j, "project"], df.at[i, "project"]
        best_total -= best_delta
        improved = True
    return df if improved else assign_df


def exact_optimize(students: List[str], projects: List[str], pref_map: dict, non_pref_cost: int) -> pd.DataFrame:
    n = len(students)
    m = len(projects)
    if n != m:
        raise ValueError("Exact optimize requires the same number of students and projects.")
    C = np.zeros((n, m), dtype=float)
    for i, s in enumerate(students):
        prefs = pref_map.get(s, [])
        for j, p in enumerate(projects):
            C[i, j] = rank_cost(p, prefs, non_pref_cost)
    row_idx, col_idx = linear_sum_assignment(C)
    out = pd.DataFrame({
        "student": [students[i] for i in row_idx],
        "project": [projects[j] for j in col_idx],
    })
    return out

def _short(s, n=24):
    s = str(s)
    return s if len(s) <= n else s[:n-1] + "…"

def sankey_from(assign_df: pd.DataFrame, cost_df: pd.DataFrame, non_pref_cost: int) -> go.Figure:
    students = assign_df["student"].tolist()
    projects = assign_df["project"].unique().tolist()

    students_sorted = sorted(students, key=lambda x: str(x).casefold())
    projects_sorted = sorted(projects, key=lambda x: str(x).casefold())

    nodes = students_sorted + projects_sorted
    idx = {n: i for i, n in enumerate(nodes)}

    sources, targets, values, colors, hover = [], [], [], [], []

    def thickness_weight(cost: int) -> float:
        if cost == 1: return 4.0
        if cost == 2: return 3.0
        if cost == 3: return 2.0
        if cost == 4: return 1.0
        return 0.35

    for _, r in cost_df.iterrows():
        s, p, c = r["student"], r["project"], int(r["cost"])
        sources.append(idx[s])
        targets.append(idx[p])
        values.append(thickness_weight(c))

        if c <= 2:
            colors.append("rgba(16,185,129,0.55)")  # green
        elif c <= 4:
            colors.append("rgba(245,158,11,0.55)")  # amber
        else:
            colors.append("rgba(239,68,68,0.55)")   # red

        pref_label = f"Pref #{c}" if 1 <= c <= 4 else f"Non-preferred (cost {c})"
        hover.append(f"Student: {s}<br>Project: {p}<br>{pref_label}")

    fig = go.Figure(
        data=[go.Sankey(
            arrangement="snap",
            node=dict(
                pad=12,
                thickness=14,
                line=dict(width=0.6, color="#cbd5e1"),
                label=nodes,
                color=["rgba(226,232,240,1)"] * len(nodes),
                hovertemplate="%{label}<extra></extra>",
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=colors,
                customdata=hover,
                hovertemplate="%{customdata}<extra></extra>",
            ),
        )]
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=560,
    )
    return fig


def cost_bar(cost_df: pd.DataFrame, non_pref_cost: int) -> go.Figure:
    """Return a stable cost distribution bar chart (Cost on x, Count on y).

    Handles empty data and forces numeric costs. Shows all cost buckets from 1..non_pref_cost
    (and includes 0 only if it exists in the data).
    """
    if cost_df is None or len(cost_df) == 0 or "cost" not in cost_df.columns:
        fig = go.Figure()
        fig.update_layout(
            xaxis_title="Cost",
            yaxis_title="Count",
            margin=dict(l=10, r=10, t=10, b=10),
            height=360,
        )
        fig.add_annotation(
            text="No cost data yet. Upload both files and click Re-evaluate.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
        return fig

    costs = pd.to_numeric(cost_df["cost"], errors="coerce").dropna()
    if len(costs) == 0:
        fig = go.Figure()
        fig.update_layout(
            xaxis_title="Cost",
            yaxis_title="Count",
            margin=dict(l=10, r=10, t=10, b=10),
            height=360,
        )
        fig.add_annotation(
            text="Cost column could not be parsed as numbers.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
        return fig

    costs = costs.astype(int)

    include_zero = int(costs.min()) == 0
    start = 0 if include_zero else 1
    cats = list(range(start, int(non_pref_cost) + 1))

    dist = costs.value_counts().reindex(cats, fill_value=0).sort_index()

    x = [str(i) for i in dist.index]   
    y = dist.values.tolist()

    fig = go.Figure(data=[go.Bar(x=x, y=y, text=y, textposition="auto")])
    fig.update_layout(
        xaxis_title="Cost",
        yaxis_title="Count",
        margin=dict(l=10, r=10, t=10, b=10),
        height=360,
    )
    return fig


# -------------------------- Sidebar --------------------------
st.sidebar.title("Optimizer Settings")

# File inputs
st.sidebar.markdown("### Input Files")

prefs_file = st.sidebar.file_uploader("Upload preferences.csv", type=["csv"], key="prefs_file")
assign_file = st.sidebar.file_uploader("Upload assignments.csv", type=["csv"], key="assign_file")

if "non_pref_cost_applied" not in st.session_state:
    st.session_state.non_pref_cost_applied = 10
if "non_pref_cost_input" not in st.session_state:
    st.session_state.non_pref_cost_input = st.session_state.non_pref_cost_applied

if "use_example_applied" not in st.session_state:
    st.session_state.use_example_applied = False
if "use_example_input" not in st.session_state:
    st.session_state.use_example_input = st.session_state.use_example_applied

st.sidebar.number_input(
    "Non-preferred cost",
    min_value=1,
    max_value=100,
    value=int(st.session_state.non_pref_cost_input),
    step=1,
    key="non_pref_cost_input",
    help="Applied when a project is not in a student's top-4.",
)

st.sidebar.checkbox("Load small example", key="use_example_input")


if st.sidebar.button("Evaluate", use_container_width=True):
    st.session_state.non_pref_cost_applied = int(st.session_state.non_pref_cost_input)
    st.session_state.use_example_applied = bool(st.session_state.use_example_input)

    ok = bool(st.session_state.use_example_applied) or (prefs_file is not None and assign_file is not None)
    if ok:
        st.session_state.ready = True
        st.session_state.missing_files_error = False
        st.rerun()
    else:
        st.session_state.ready = False
        st.session_state.missing_files_error = True

mismatch_cost = int(st.session_state.non_pref_cost_input) != int(st.session_state.non_pref_cost_applied)
mismatch_example = bool(st.session_state.use_example_input) != bool(st.session_state.use_example_applied)

if mismatch_cost or mismatch_example:
    parts = []
    if mismatch_cost:
        parts.append(
            f"Cost set to {int(st.session_state.non_pref_cost_input)} (not applied yet); currently using {int(st.session_state.non_pref_cost_applied)}."
        )
    if mismatch_example:
        parts.append(
            f"Example mode set to {bool(st.session_state.use_example_input)} (not applied yet); currently using {bool(st.session_state.use_example_applied)}."
        )
    st.sidebar.info(" ".join(parts))
else:
    st.sidebar.caption(
        f"Applied cost: {int(st.session_state.non_pref_cost_applied)} | "
        f"Example mode: {bool(st.session_state.use_example_applied)}"
    )


if st.session_state.missing_files_error:
    st.sidebar.error("Please upload BOTH CSV files (or apply example mode) and click **Evaluate**.")
non_pref_cost = int(st.session_state.non_pref_cost_applied)



# --------------------------- Body ----------------------------
st.title("Student - Project Assignment Optimizer")

# ------------------------- Load Inputs -------------------------
if not st.session_state.ready:
    st.info("Upload both CSVs (or enable example mode) in the sidebar, then click **Evaluate**.")
    st.stop()

if bool(st.session_state.use_example_applied):
    # Small built-in example dataset
    prefs_df = pd.DataFrame([
        {"student": "Alice", "pref1": "GridIt!", "pref2": "Cisco #1", "pref3": "BioCollate", "pref4": "GriffMonster"},
        {"student": "Ben",   "pref1": "Cisco #2", "pref2": "GridIt!",   "pref3": "Gravity Foundation", "pref4": "Credible"},
        {"student": "Cara",  "pref1": "BioCollate", "pref2": "Credible", "pref3": "GridIt!", "pref4": "Cisco #1"},
        {"student": "Dee",   "pref1": "GriffMonster", "pref2": "Cisco #2", "pref3": "GridIt!", "pref4": "BioCollate"},
    ])
    assign_df = pd.DataFrame([
        {"student": "Alice", "project": "Cisco #1"},
        {"student": "Ben",   "project": "GridIt!"},
        {"student": "Cara",  "project": "BioCollate"},
        {"student": "Dee",   "project": "Credible"},
    ])
else:
    # Uploaded files are required in non-example mode
    if prefs_file is None or assign_file is None:
        st.error("Please upload BOTH CSV files in the sidebar, then click **Evaluate**.")
        st.stop()
    prefs_df = pd.read_csv(prefs_file)
    assign_df = pd.read_csv(assign_file)

# Normalize
prefs_df = prefs_df.rename(columns=str.lower)
assign_df = assign_df.rename(columns=str.lower)
required_prefs_cols = {"student", "pref1", "pref2", "pref3", "pref4"}
required_assign_cols = {"student", "project"}
if not required_prefs_cols.issubset(prefs_df.columns):
    st.error(f"preferences.csv must have columns: {sorted(required_prefs_cols)}")
    st.stop()
if not required_assign_cols.issubset(assign_df.columns):
    st.error(f"assignments.csv must have columns: {sorted(required_assign_cols)}")
    st.stop()

# Pref map
# pref_map = {
#     str(r["student"]).strip(): [
#         str(r["pref1"]).strip(), str(r["pref2"]).strip(), str(r["pref3"]).strip(), str(r["pref4"]).strip()
#     ]
#     for _, r in prefs_df.iterrows()
# }

pref_map: dict[str, list[str]] = {}
for _, r in prefs_df.iterrows():
    stu_raw = r["student"]
    prefs_raw = [r["pref1"], r["pref2"], r["pref3"], r["pref4"]]
    prefs_canon = [_canon_project(x) for x in prefs_raw]

    stu_key = _canon_student(stu_raw)
    pref_map[stu_key] = prefs_canon

    s = _canon_text(stu_raw)
    if " " in s:
        parts = s.split()
        first, last = parts[0], parts[-1]
        lf = f"{last}, {first}"
        pref_map[_canon_student(lf)] = prefs_canon


# ----------------------- Preferences Table -----------------------
st.subheader("Preferences (Fixed)")
st.caption("Read-only view of each student's top-4 preferences (from preferences.csv).")

prefs_view = prefs_df[["student", "pref1", "pref2", "pref3", "pref4"]].copy()

st.data_editor(
    prefs_view,
    num_rows="fixed",
    use_container_width=True,
    hide_index=True,
    key="prefs_view_table",
    column_config={
        "student": st.column_config.TextColumn(disabled=True),
        "pref1": st.column_config.TextColumn("Pref 1", disabled=True),
        "pref2": st.column_config.TextColumn("Pref 2", disabled=True),
        "pref3": st.column_config.TextColumn("Pref 3", disabled=True),
        "pref4": st.column_config.TextColumn("Pref 4", disabled=True),
    },
)


# Editable assignments
st.subheader("Assignments (Editable)")
st.caption("Tip: Edit the ‘project’ column directly. Changes update all charts and totals.")
assign_df = st.data_editor(
    assign_df,
    num_rows="fixed",
    use_container_width=True,
    hide_index=True,
    column_config={
        "student": st.column_config.TextColumn(disabled=True),
        "project": st.column_config.TextColumn(help="Type or paste a project name."),
    },
)

cost_df = compute_costs(assign_df, pref_map, non_pref_cost)

unmatched_students = sorted(
    set(_canon_student(s) for s in assign_df["student"]) - set(pref_map.keys())
)
if unmatched_students:
    st.warning(
        f"{len(unmatched_students)} students in assignments have no preferences. "
        "Example: " + "; ".join(next(iter(unmatched_students)) for _ in range(1))
    )


col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Students", value=cost_df.shape[0])
with col2:
    st.metric("Projects (unique)", value=assign_df["project"].nunique())
with col3:
    st.metric("Total cost", value=int(cost_df["cost"].sum()))

with st.expander("Per-student cost table", expanded=True):
    st.dataframe(cost_df, use_container_width=True, hide_index=True)

st.markdown("---")
a1, a2, a3 = st.columns([1,1,2])
with a1:
    if st.button("✨ Greedy improve (pairwise swaps)"):
        assign_df = greedy_improve(assign_df, pref_map, non_pref_cost)
        cost_df = compute_costs(assign_df, pref_map, non_pref_cost)
        st.success("Applied greedy local improvements.")
with a2:
    same_count = cost_df.shape[0] == assign_df["project"].nunique()
    if st.button("📐 Exact optimize (Hungarian)"):
        students = assign_df["student"].tolist()
        projects = sorted(assign_df["project"].unique().tolist())
        try:
            exact_df = exact_optimize(students, projects, pref_map, non_pref_cost)
            assign_df = exact_df
            cost_df = compute_costs(assign_df, pref_map, non_pref_cost)
            st.success("Exact minimum-cost matching applied.")
        except Exception as e:
            st.error(str(e))

# Visuals
st.markdown("---")
vc1, vc2 = st.columns([1,1])
with vc1:
    st.subheader("Cost distribution")
    st.plotly_chart(cost_bar(cost_df, non_pref_cost), use_container_width=True)
with vc2:
    min_cost = st.select_slider("Show links with cost ≥", [1, 2, 3, 4, int(non_pref_cost)], 1)
    filtered_cost_df = cost_df[cost_df["cost"] >= min_cost]
    st.subheader("Student-Project Sankey (link thickness≈cost)")
    st.plotly_chart(sankey_from(assign_df, filtered_cost_df, non_pref_cost), use_container_width=True)

# Download
st.markdown("---")
st.markdown("### Download Assignment Sheet")
final = cost_df.copy()
buf = io.StringIO()
final.to_csv(buf, index=False)
st.download_button(
    "Download assignments_with_cost.csv",
    buf.getvalue(),
    file_name="assignments_with_cost.csv",
    mime="text/csv",
)

# st.caption(
#     "Non-preferred cost controls how strongly the app penalizes non-top-4 matches. For multi-seat projects, load per-project capacities and extend the solver (min-cost flow)."
# )