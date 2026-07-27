#!/usr/bin/env python3
import glob
import hashlib
import os
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import tables as tb

from invisible_cities.cities.components import build_pmap
from invisible_cities.cities.components import calibrate_pmts
from invisible_cities.cities.components import calibrate_sipms
from invisible_cities.cities.components import deconv_pmt
from invisible_cities.cities.components import get_actual_sipm_thr
from invisible_cities.cities.components import select_cutting_algorithm
from invisible_cities.cities.components import zero_suppress_wfs
from invisible_cities.core import system_of_units as units
from invisible_cities.core.configure import read_config_file
from invisible_cities.database import load_db
from invisible_cities.types.symbols import CutAlgo
from invisible_cities.types.symbols import SiPMThreshold


ROOT_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("ICTDIR", str(ROOT_DIR))
DEFAULT_DATA_DIR = Path("/analysis")
CONFIG_FILE = ROOT_DIR / "invisible_cities" / "config" / "irene.conf"

CFG = read_config_file(str(CONFIG_FILE)) if CONFIG_FILE.exists() else {}

print("Irene app startup variables:")
print(f"  ROOT_DIR = {ROOT_DIR}")
print(f"  ICTDIR = {os.environ.get('ICTDIR')}")
print(f"  DEFAULT_DATA_DIR = {DEFAULT_DATA_DIR}")
print(f"  CONFIG_FILE = {CONFIG_FILE}")
print(f"  CFG loaded = {bool(CFG)}")

DEFAULT_RUN_NUMBER = int(CFG.get("run_number", 0))
DEFAULT_DETECTOR_DB = CFG.get("detector_db", "next_100")
DEFAULT_EVENT_RANGE = CFG.get("event_range", (0, 1))
DEFAULT_N_BASELINE = int(CFG.get("n_baseline", 28000))
DEFAULT_N_MAW = int(CFG.get("n_maw", 100))
DEFAULT_THR_MAW = float(CFG.get("thr_maw", 3.0))
DEFAULT_THR_CSUM_S1 = float(CFG.get("thr_csum_s1", 0.5))
DEFAULT_THR_CSUM_S2 = float(CFG.get("thr_csum_s2", 1.0))
DEFAULT_S1_TMIN_US = float(CFG.get("s1_tmin", 99 * units.mus)) / units.mus
DEFAULT_S1_TMAX_US = float(CFG.get("s1_tmax", 101 * units.mus)) / units.mus
DEFAULT_S1_STRIDE = int(CFG.get("s1_stride", 4))
DEFAULT_S1_LMIN = int(CFG.get("s1_lmin", 8))
DEFAULT_S1_LMAX = int(CFG.get("s1_lmax", 20))
DEFAULT_S1_REBIN = int(CFG.get("s1_rebin_stride", 1))
DEFAULT_S2_TMIN_US = float(CFG.get("s2_tmin", 101 * units.mus)) / units.mus
DEFAULT_S2_TMAX_US = float(CFG.get("s2_tmax", 1199 * units.mus)) / units.mus
DEFAULT_S2_STRIDE = int(CFG.get("s2_stride", 40))
DEFAULT_S2_LMIN = int(CFG.get("s2_lmin", 80))
DEFAULT_S2_LMAX = int(CFG.get("s2_lmax", 200000))
DEFAULT_S2_REBIN = int(CFG.get("s2_rebin_stride", 40))
DEFAULT_THR_SIPM = float((CFG.get("cutting_params", {}) or {}).get("thr_sipm", 3.5 * units.pes)) / units.pes
DEFAULT_THR_SIPM_S2 = float((CFG.get("cutting_params", {}) or {}).get("thr_sipm_s2", 10 * units.pes)) / units.pes
DEFAULT_PMT_SAMP_WID_NS = float(CFG.get("pmt_samp_wid", 25 * units.ns)) / units.ns
DEFAULT_SIPM_SAMP_WID_US = float(CFG.get("sipm_samp_wid", 1 * units.mus)) / units.mus
DEFAULT_CUTTING_FUNCTION = CFG.get("cutting_function", CutAlgo.threshold)
DEFAULT_THR_SIPM_TYPE = (CFG.get("cutting_params", {}) or {}).get("thr_sipm_type", SiPMThreshold.common)
AUTHORIZED_PASSWORD_HASH = "10fd760b961e9b2e83d1f870b23f4d47cbc18896457e76afbce09062ed7ec1e4"


def write_parameters_to_file(file_path, params):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    values = dict(params)

    int_keys = {
        "run_number",
        "print_mod",
        "event_range",
        "n_baseline",
        "n_maw",
        "s1_stride",
        "s1_lmin",
        "s1_lmax",
        "s1_rebin_stride",
        "s2_stride",
        "s2_lmin",
        "s2_lmax",
        "s2_rebin_stride",
    }
    float_keys = {
        "thr_maw",
        "thr_csum_s1",
        "thr_csum_s2",
        "s1_tmin",
        "s1_tmax",
        "s2_tmin",
        "s2_tmax",
        "thr_sipm",
        "thr_sipm_s2",
        "pmt_samp_wid",
        "sipm_samp_wid",
    }

    def format_value(key, value):
        if key in int_keys:
            return str(int(value))
        if key in float_keys:
            return repr(float(value))
        return str(value)

    template = f"""files_in = '{values.get('files_in', DEFAULT_FILE if 'DEFAULT_FILE' in globals() else '$ICDIR/database/test_data/electrons_40keV_z25_RWF.h5')}'

# REPLACE /tmp with your output directory
file_out = '{values.get('file_out', '/tmp/irene_pmaps.h5')}'

# compression library
compression = '{values.get('compression', 'ZLIB4')}'

# run number 0 is for MC
run_number = {format_value('run_number', values.get('run_number', 0))}
detector_db = '{values.get('detector_db', 'next_100')}'

# How frequently to print events
print_mod = {format_value('print_mod', values.get('print_mod', 1))}

# max number of events to run
event_range =  {format_value('event_range', values.get('event_range', 1))}

n_baseline =   {format_value('n_baseline', values.get('n_baseline', 28000))} # for a window of 800 mus

# Set MAW for calibrated sum
n_maw   = {format_value('n_maw', values.get('n_maw', 100))}
thr_maw =   {format_value('thr_maw', values.get('thr_maw', 3.0))} * adc

# Set thresholds for calibrated sum
thr_csum_s1 = {format_value('thr_csum_s1', values.get('thr_csum_s1', 0.5))} * pes
thr_csum_s2 = {format_value('thr_csum_s2', values.get('thr_csum_s2', 1.0))} * pes

# Set parameters to search for S1
# Notice that in MC file S1 is in t=100 mus
s1_tmin       = {format_value('s1_tmin', values.get('s1_tmin', 99))} * mus # position of S1 in MC files at 100 mus
s1_tmax       = {format_value('s1_tmax', values.get('s1_tmax', 101))} * mus # change tmin and tmax if S1 not at 100 mus
s1_stride     =   {format_value('s1_stride', values.get('s1_stride', 4))}       # minimum number of 25 ns bins in S1 searches
s1_lmin       =   {format_value('s1_lmin', values.get('s1_lmin', 8))}       # 8 x 25 = 200 ns
s1_lmax       =   {format_value('s1_lmax', values.get('s1_lmax', 20))}       # 20 x 25 = 500 ns
s1_rebin_stride = {format_value('s1_rebin_stride', values.get('s1_rebin_stride', 1))}       # Do not rebin S1 by default

# Set parameters to search for S2
s2_tmin     =    {format_value('s2_tmin', values.get('s2_tmin', 101))} * mus # assumes S1 at 100 mus, change if S1 not at 100 mus
s2_tmax     =   {format_value('s2_tmax', values.get('s2_tmax', 1199))} * mus # end of the window
s2_stride   =     {format_value('s2_stride', values.get('s2_stride', 40))}       #  40 x 25 = 1   mus
s2_lmin     =    {format_value('s2_lmin', values.get('s2_lmin', 80))}       # 100 x 25 = 2.5 mus
s2_lmax     = {format_value('s2_lmax', values.get('s2_lmax', 200000))}       # maximum value of S2 width
s2_rebin_stride = {format_value('s2_rebin_stride', values.get('s2_rebin_stride', 40))}       # Rebin by default, 40 25 ns time bins to make one 1us time bin

# Set S2Si parameters
thr_sipm      = {format_value('thr_sipm', values.get('thr_sipm', 3.5))} * pes
thr_sipm_s2   = {format_value('thr_sipm_s2', values.get('thr_sipm_s2', 10.0))} * pes  # Threshold for the full sipm waveform
thr_sipm_type = {values.get('thr_sipm_type', 'common')}

pmt_samp_wid  = {format_value('pmt_samp_wid', values.get('pmt_samp_wid', 25))} * ns
sipm_samp_wid = {format_value('sipm_samp_wid', values.get('sipm_samp_wid', 1))} * mus

cutting_function = {values.get('cutting_function', 'threshold')}
cutting_params   = dict(  thr_sipm_type = {values.get('thr_sipm_type', 'common')} 
                        , thr_sipm      = thr_sipm
                        , thr_sipm_s2   = thr_sipm_s2
                        , detector_db   = detector_db 
                        , run_number    = run_number)
"""

    path.write_text(template)


def discover_run_numbers(data_root: Path):
    runs = []
    if not data_root.exists():
        return runs

    for path in sorted(data_root.iterdir()):
        if not path.is_dir() or not path.name.isdigit():
            continue
        if any((path / "hdf5" / "data" / f"ldc{ldc}").exists() for ldc in range(1, 8)):
            runs.append(int(path.name))
    return runs


def discover_ldc_files(data_root: Path, run_number: int, ldc: int):
    pattern = str(data_root / str(run_number) / "hdf5" / "data" / f"ldc{ldc}" / "*.h5")
    files = []
    for path in sorted(glob.glob(pattern)):
        try:
            with tb.open_file(path, "r") as h5in:
                if "RD" in h5in.root and "pmtrwf" in h5in.root.RD and "sipmrwf" in h5in.root.RD:
                    files.append(path)
        except Exception:
            continue
    return files


@st.cache_data(show_spinner=False)
def get_dataset_shape(file_path: str):
    with tb.open_file(file_path, "r") as h5in:
        return h5in.root.RD.pmtrwf.shape


@st.cache_data(show_spinner=False)
def load_event(file_path: str, event_idx: int):
    with tb.open_file(file_path, "r") as h5in:
        rd = h5in.root.RD
        pmt_rwf = rd.pmtrwf[event_idx]
        sipm_rwf = rd.sipmrwf[event_idx]
        pmt_blr = rd.pmtblr[event_idx] if "pmtblr" in rd else None
        event_number = int(h5in.root.Run.events[event_idx][0])
    return pmt_rwf, pmt_blr, sipm_rwf, event_number


@st.cache_data(show_spinner=False)
def load_sensor_tables(detector_db: str, run_number: int):
    return load_db.DataPMT(detector_db, run_number), load_db.DataSiPM(detector_db, run_number)


def inject_sidebar_styles():
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] div[data-testid="stNumberInput"] {
            background: #ffffff;
            border: 1.5px solid #9ca3af;
            border-radius: 0.55rem;
            padding: 0.15rem 0.35rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.08);
        }

        section[data-testid="stSidebar"] div[data-testid="stNumberInput"]:focus-within {
            border-color: #0b5fff;
            box-shadow: 0 0 0 3px rgba(11, 95, 255, 0.12);
        }

        section[data-testid="stSidebar"] div[data-testid="stNumberInput"] input {
            background: transparent;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def sidebar_labeled_number_input(label, **kwargs):
    label_col, input_col = st.columns([2, 1])
    label_col.markdown(f"**{label}**")
    with input_col:
        return st.number_input(label, label_visibility="collapsed", **kwargs)


def option_labels(df, kind: str):
    active = df.loc[df.Active.astype(bool)]
    labels = []
    lookup = {}
    for row_idx, row in active.iterrows():
        label = f"{kind} {int(row_idx):03d} | SensorID {int(row.SensorID)} | ElecID {int(row.ChannelID)}"
        labels.append(label)
        lookup[label] = int(row_idx)
    return labels, lookup


def pmt_id_from_index(pmt_df, idx: int):
    row = pmt_df.iloc[int(idx)]
    return int(row.SensorID), int(row.ChannelID)


def sipm_id_from_index(sipm_df, idx: int):
    row = sipm_df.iloc[int(idx)]
    return int(row.SensorID), int(row.ChannelID)


def overlay_plot(t_us, a, b, title, name_a, name_b, y_title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_us, y=a, mode="lines", name=name_a, line=dict(width=1.2)))
    fig.add_trace(go.Scatter(x=t_us, y=b, mode="lines", name=name_b, line=dict(width=1.2)))
    fig.update_layout(
        title=title,
        xaxis_title="Time (us)",
        yaxis_title=y_title,
        height=350,
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
    )
    return fig


def threshold_plot(
    t_us,
    y,
    thr,
    title,
    selected,
    rejected,
    selected_color,
    rejected_color,
    allowed_window=None,
    extra_regions=None,
):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_us, y=y, mode="lines", name="summed waveform", line=dict(width=1.2)))
    fig.add_hline(y=thr, line_dash="dash")

    if allowed_window is not None:
        t0, t1 = allowed_window
        fig.add_vrect(x0=float(t0), x1=float(t1), fillcolor="#a5d8ff", opacity=0.12, line_width=0)

    if extra_regions:
        for t0, t1 in extra_regions:
            fig.add_vrect(x0=float(t0), x1=float(t1), fillcolor="#f4a261", opacity=0.18, line_width=0)

    for seg in selected:
        fig.add_vrect(
            x0=float(t_us[seg[0]]),
            x1=float(t_us[seg[-1]]),
            fillcolor=selected_color,
            opacity=0.35,
            line_width=1,
            line_color=selected_color,
        )

    for seg in rejected:
        fig.add_vrect(
            x0=float(t_us[seg[0]]),
            x1=float(t_us[seg[-1]]),
            fillcolor=rejected_color,
            opacity=0.28,
            line_width=1,
            line_color=rejected_color,
        )

    fig.update_layout(
        title=title,
        xaxis_title="Time (us)",
        yaxis_title="pes",
        height=350,
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
    )
    return fig


def split_with_stride(indices: np.ndarray, stride: int):
    if len(indices) == 0:
        return []
    breaks = np.where(np.diff(indices) > stride)[0] + 1
    return np.split(indices, breaks)


def classify_candidate_segments(indices, stride, t_us, sample_width_ns, tmin_us, tmax_us, lmin, lmax):
    candidates = split_with_stride(np.asarray(indices, dtype=int), stride)
    analyzed, selected, rejected = [], [], []

    for seg in candidates:
        if len(seg) == 0:
            continue

        t0 = float(t_us[seg[0]])
        t1 = float(t_us[seg[-1]] + sample_width_ns * 1e-3)
        width = int(seg[-1] + 1 - seg[0])

        reasons = []
        if t0 < tmin_us:
            reasons.append(f"starts before tmin ({t0:.3f} < {tmin_us:.3f} us)")
        if t1 > tmax_us:
            reasons.append(f"ends after tmax ({t1:.3f} > {tmax_us:.3f} us)")
        if not (lmin <= width <= lmax):
            reasons.append(f"length out of range ({width} not in [{lmin}, {lmax}] bins)")

        info = dict(segment=seg, t0=t0, t1=t1, width_bins=width, passed=len(reasons) == 0, reasons=reasons)
        analyzed.append(info)
        (selected if info["passed"] else rejected).append(seg)

    return analyzed, selected, rejected


def build_stage_a_markdown(label, analyzed, n_in_pmap=None, pmap_error=None):
    selected = sum(c["passed"] for c in analyzed)
    rejected = len(analyzed) - selected
    lines = [
        f"**Summary**  ",
        (
            f"Candidates: **{len(analyzed)}** | "
            f"<span style='color:#1b7f3a;font-weight:700;'>Selected: {selected}</span> | "
            f"<span style='color:#b42318;font-weight:700;'>Rejected: {rejected}</span>"
        ),
        "",
        "**Candidate Details**",
    ]

    if not analyzed:
        lines.append("- <span style='color:#6b7280;'>No candidate regions found above threshold.</span>")
    else:
        for i, c in enumerate(analyzed, 1):
            status = (
                "<span style='color:#1b7f3a;font-weight:700;'>Selected</span>"
                if c["passed"]
                else "<span style='color:#b42318;font-weight:700;'>Rejected</span>"
            )
            lines.append(
                f"- **{label} {i:02d}**: {c['t0']:.3f}-{c['t1']:.3f} us | width {c['width_bins']} bins | {status}"
            )
            if not c["passed"] and c["reasons"]:
                for reason in c["reasons"]:
                    lines.append(f"  - <span style='color:#b42318;'>reason: {reason}</span>")

    lines.append("")
    lines.append("**PMAP**")
    if n_in_pmap is not None:
        lines.append(f"- {label} peaks in PMAP: <span style='color:#1b7f3a;font-weight:700;'>{n_in_pmap}</span>")
    else:
        lines.append(f"- {label} peaks in PMAP: <span style='color:#b42318;'>unavailable</span>")
        if pmap_error:
            lines.append(f"- <span style='color:#b42318;'>PMAP build note: {pmap_error}</span>")

    return "\n".join(lines)


def get_s2_windows_us(pmap_evt, s2_selected, t_us, pmt_samp_wid_ns):
    windows = []

    if pmap_evt is not None and len(pmap_evt.s2s):
        try:
            for s2 in pmap_evt.s2s:
                times = np.asarray(s2.times, dtype=float)
                if len(times) == 0:
                    continue
                t0_us = float(times[0]) * 1e-3
                dt_us = float(np.median(np.diff(times))) * 1e-3 if len(times) > 1 else float(pmt_samp_wid_ns) * 1e-3
                windows.append((t0_us, float(times[-1]) * 1e-3 + dt_us))
            if windows:
                return windows
        except Exception:
            pass

    for seg in s2_selected:
        windows.append((float(t_us[seg[0]]), float(t_us[seg[-1]] + float(pmt_samp_wid_ns) * 1e-3)))
    return windows


def sipm_charge_map_figure(
    sipm_wf_evt,
    sipm_df,
    s2_windows_us,
    sipm_samp_wid_us,
    detector_db,
    run_number,
    sipm_thr,
):
    if not s2_windows_us:
        return None, 0

    active = sipm_df.loc[sipm_df.Active.astype(bool)]
    if active.empty:
        return None, 0

    n_samples = sipm_wf_evt.shape[1]
    t_sipm_us = np.arange(n_samples, dtype=float) * float(sipm_samp_wid_us)
    mask = np.zeros_like(t_sipm_us, dtype=bool)
    for t0, t1 in s2_windows_us:
        mask |= (t_sipm_us >= float(t0)) & (t_sipm_us <= float(t1))
    if not np.any(mask):
        return None, 0

    sipm_cal = calibrate_sipms(detector_db, run_number)
    calibrated_wfs = sipm_cal(sipm_wf_evt)

    q_vals, amp_vals, x_vals, y_vals, labels, sensor_indices = [], [], [], [], [], []
    for row_idx, row in active.iterrows():
        elecid = int(row.ChannelID)
        x_vals.append(float(row.X))
        y_vals.append(float(row.Y))
        labels.append(elecid)
        sensor_indices.append(int(row_idx))
        waveform = np.asarray(sipm_wf_evt[int(row_idx)], dtype=float)
        baseline = np.median(waveform[: max(10, min(50, waveform.size // 5))])
        corrected = np.where(waveform - baseline > 0, waveform - baseline, 0.0)
        q_vals.append(float(np.sum(corrected[mask]) * float(sipm_samp_wid_us)))
        amp_vals.append(float(np.max(np.asarray(calibrated_wfs[int(row_idx)], dtype=float)[mask])))

    q_vals = np.asarray(q_vals, dtype=float)
    amp_vals = np.asarray(amp_vals, dtype=float)
    selected_mask = amp_vals >= float(sipm_thr)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("All mapped SiPMs", "SiPMs passing threshold selection"),
        horizontal_spacing=0.08,
    )
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="markers",
            marker=dict(size=5, color=q_vals, colorscale="Turbo", colorbar=dict(title="Integrated charge"), line=dict(color="black", width=0.4)),
            text=[f"ElecID {eid}<br>Q={qq:.2f}" for eid, qq in zip(labels, q_vals)],
            customdata=np.asarray(sensor_indices, dtype=int),
            hovertemplate="%{text}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=np.asarray(x_vals, dtype=float)[selected_mask],
            y=np.asarray(y_vals, dtype=float)[selected_mask],
            mode="markers",
            marker=dict(size=5, color=q_vals[selected_mask], colorscale="Turbo", showscale=False, line=dict(color="black", width=0.4)),
            text=[f"ElecID {eid}<br>Q={qq:.2f}" for eid, qq in zip(np.asarray(labels, dtype=int)[selected_mask], q_vals[selected_mask])],
            customdata=np.asarray(sensor_indices, dtype=int)[selected_mask],
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        title="SiPM integrated charge in S2 valid window(s)",
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=520,
        font=dict(color="black"),
    )
    fig.update_xaxes(title_text="X", row=1, col=1)
    fig.update_yaxes(title_text="Y", row=1, col=1, scaleanchor="x")
    fig.update_xaxes(title_text="X", row=1, col=2)
    fig.update_yaxes(title_text="Y", row=1, col=2, scaleanchor="x2")
    return fig, int(len(active))


def sipm_waveform_figure(
    sipm_wf_evt,
    sensor_idx,
    elecid,
    thr_sipm_s2,
    sipm_samp_wid_us,
    detector_db,
    run_number,
    pmap_windows_us=None,
):
    raw = np.asarray(sipm_wf_evt[sensor_idx], dtype=float)
    n_samples = raw.size
    t_us = np.arange(n_samples, dtype=float) * float(sipm_samp_wid_us)

    sipm_cal = calibrate_sipms(detector_db, run_number)
    calibrated = sipm_cal(sipm_wf_evt)[sensor_idx]
    sipm_thr = get_actual_sipm_thr(SiPMThreshold.common, float(thr_sipm_s2), detector_db, run_number)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    fig.add_trace(go.Scatter(x=t_us, y=raw, mode="lines", name="raw ADC", line=dict(width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t_us, y=calibrated, mode="lines", name="calibrated (pes)", line=dict(width=1.2)), row=2, col=1)
    fig.add_hline(y=float(sipm_thr), line_dash="dash", line_color="#d94a4a", row=2, col=1)
    if pmap_windows_us:
        for t0, t1 in pmap_windows_us:
            fig.add_vrect(x0=float(t0), x1=float(t1), fillcolor="#1f77b4", opacity=0.16, line_width=0, row=2, col=1)
    fig.update_layout(
        title=f"Selected SiPM waveform: ElecID {elecid}",
        height=620,
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
        legend=dict(orientation="h"),
    )
    fig.update_xaxes(title_text="Time (us)", row=2, col=1)
    fig.update_yaxes(title_text="ADC", row=1, col=1)
    fig.update_yaxes(title_text="pes", row=2, col=1)
    return fig


def pmt_summary_figure(t_us, pmt_raw, pmt_cwf, pmt_ccwf, pmt_ccwf_maw, pmt_blr, channel_idx):
    traces = [
        ("raw ADC", pmt_raw[channel_idx]),
        ("deconvolved", pmt_cwf[channel_idx]),
        ("calibrated", pmt_ccwf[channel_idx]),
        ("calibrated MAW", pmt_ccwf_maw[channel_idx]),
    ]
    if pmt_blr is not None:
        traces.insert(1, ("file BLR", pmt_blr[channel_idx]))

    fig = go.Figure()
    for name, values in traces:
        fig.add_trace(go.Scatter(x=t_us, y=values, mode="lines", name=name, line=dict(width=1.2)))
    fig.update_layout(
        title=f"PMT channel {channel_idx}",
        xaxis_title="Time (us)",
        yaxis_title="value",
        height=360,
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
    )
    return fig


def main():
    st.set_page_config(page_title="Irene Interactive Pipeline", layout="wide")
    inject_sidebar_styles()
    st.title("Irene Interactive Pipeline")
    st.caption("Inspect IC Irene waveforms, candidate windows, PMAPs, and SiPM response live.")

    available_runs = discover_run_numbers(DEFAULT_DATA_DIR)
    if not available_runs:
        st.error(f"No run directories found under {DEFAULT_DATA_DIR}")
        st.stop()

    with st.sidebar:
        st.header("Input")
        if st.button("Refresh"):
            st.cache_data.clear()
            st.rerun()

        default_run = DEFAULT_RUN_NUMBER if DEFAULT_RUN_NUMBER in available_runs else available_runs[0]
        run_number = st.selectbox("Run number", options=available_runs, index=available_runs.index(default_run))
        ldc_options = [ldc for ldc in range(1, 8) if (DEFAULT_DATA_DIR / str(run_number) / "hdf5" / "data" / f"ldc{ldc}").exists()]
        if not ldc_options:
            st.error(f"No ldc1-7 folders found for run {run_number}")
            st.stop()

        ldc = st.selectbox("LDC", options=ldc_options, index=0)
        ldc_files = discover_ldc_files(DEFAULT_DATA_DIR, int(run_number), int(ldc))
        if not ldc_files:
            st.error(f"No HDF5 waveform files found for run {run_number} in ldc{ldc}")
            st.stop()

        file_path = st.selectbox("Waveform file", options=ldc_files, index=0)
        n_events, n_pmts, n_samples = get_dataset_shape(file_path)

        detector_db = st.selectbox("Detector DB", ["next100", "flex100"], index=["next100", "flex100"].index(DEFAULT_DETECTOR_DB) if DEFAULT_DETECTOR_DB in ["next100", "flex100"] else 0)

        event_idx_requested = st.number_input(
            "Event index",
            min_value=0,
            max_value=int(n_events) - 1,
            value=min(int(DEFAULT_EVENT_RANGE[0]) if isinstance(DEFAULT_EVENT_RANGE, (list, tuple)) else 0, int(n_events) - 1),
            step=1,
        )
        event_idx = int(event_idx_requested)

        pmt_df, sipm_df = load_sensor_tables(str(detector_db), int(run_number))
        pmt_labels, pmt_lookup = option_labels(pmt_df, "PMT")
        sipm_labels, sipm_lookup = option_labels(sipm_df, "SiPM")

        selected_pmt_label = st.selectbox("Detailed PMT channel", options=pmt_labels, index=0 if pmt_labels else 0)
        selected_sipm_label = st.selectbox("Selected SiPM channel", options=sipm_labels, index=0 if sipm_labels else 0)

        st.markdown('<hr style="margin: 0.2rem 0;">', unsafe_allow_html=True)
        st.header("Parameters")
        st.markdown('<hr style="margin: 0.2rem 0;">', unsafe_allow_html=True)
        n_baseline = sidebar_labeled_number_input("n_baseline", min_value=1, max_value=max(int(n_samples), 1_000_000), value=DEFAULT_N_BASELINE, step=100)
        n_maw = sidebar_labeled_number_input("n_maw", min_value=1, max_value=100000, value=DEFAULT_N_MAW, step=1)
        thr_maw = sidebar_labeled_number_input("thr_maw", min_value=0.0, max_value=1e6, value=DEFAULT_THR_MAW, step=0.1)
        thr_csum_s1 = sidebar_labeled_number_input("thr_csum_s1 (pes)", min_value=0.0, max_value=1e6, value=DEFAULT_THR_CSUM_S1, step=0.1)
        thr_csum_s2 = sidebar_labeled_number_input("thr_csum_s2 (pes)", min_value=0.0, max_value=1e6, value=DEFAULT_THR_CSUM_S2, step=0.1)

        st.markdown('<hr style="margin: 0.2rem 0;">', unsafe_allow_html=True)
        st.subheader("S1 selection")
        st.markdown('<hr style="margin: 0.2rem 0;">', unsafe_allow_html=True)
        s1_tmin_us = sidebar_labeled_number_input("s1_tmin (us)", min_value=0.0, max_value=1e6, value=DEFAULT_S1_TMIN_US, step=1.0)
        s1_tmax_us = sidebar_labeled_number_input("s1_tmax (us)", min_value=0.0, max_value=1e6, value=DEFAULT_S1_TMAX_US, step=1.0)
        s1_stride = sidebar_labeled_number_input("s1_stride", min_value=1, max_value=10000, value=DEFAULT_S1_STRIDE, step=1)
        s1_lmin = sidebar_labeled_number_input("s1_lmin", min_value=1, max_value=1000000, value=DEFAULT_S1_LMIN, step=1)
        s1_lmax = sidebar_labeled_number_input("s1_lmax", min_value=1, max_value=1000000, value=DEFAULT_S1_LMAX, step=1)
        s1_rebin_stride = sidebar_labeled_number_input("s1_rebin_stride", min_value=1, max_value=100000, value=DEFAULT_S1_REBIN, step=1)

        st.markdown('<hr style="margin: 0.2rem 0;">', unsafe_allow_html=True)
        st.subheader("S2 selection")
        st.markdown('<hr style="margin: 0.2rem 0;">', unsafe_allow_html=True)
        s2_tmin_us = sidebar_labeled_number_input("s2_tmin (us)", min_value=0.0, max_value=1e6, value=DEFAULT_S2_TMIN_US, step=1.0)
        s2_tmax_us = sidebar_labeled_number_input("s2_tmax (us)", min_value=0.0, max_value=1e6, value=DEFAULT_S2_TMAX_US, step=1.0)
        s2_stride = sidebar_labeled_number_input("s2_stride", min_value=1, max_value=10000, value=DEFAULT_S2_STRIDE, step=1)
        s2_lmin = sidebar_labeled_number_input("s2_lmin", min_value=1, max_value=1000000, value=DEFAULT_S2_LMIN, step=1)
        s2_lmax = sidebar_labeled_number_input("s2_lmax", min_value=1, max_value=1000000, value=DEFAULT_S2_LMAX, step=1)
        s2_rebin_stride = sidebar_labeled_number_input("s2_rebin_stride", min_value=1, max_value=100000, value=DEFAULT_S2_REBIN, step=1)

        st.markdown('<hr style="margin: 0.2rem 0;">', unsafe_allow_html=True)
        st.subheader("SiPM selection")
        st.markdown('<hr style="margin: 0.2rem 0;">', unsafe_allow_html=True)
        thr_sipm_type = st.selectbox("thr_sipm_type", [SiPMThreshold.common, SiPMThreshold.individual], index=0 if DEFAULT_THR_SIPM_TYPE == SiPMThreshold.common else 1)
        thr_sipm = sidebar_labeled_number_input("thr_sipm (pes)", min_value=0.0, max_value=1e6, value=DEFAULT_THR_SIPM, step=0.1)
        thr_sipm_s2 = sidebar_labeled_number_input("thr_sipm_s2 (pes)", min_value=0.0, max_value=1e6, value=DEFAULT_THR_SIPM_S2, step=0.1)
        cut_algo = st.selectbox("cutting_function", [CutAlgo.threshold, CutAlgo.pyrrha, CutAlgo.no_cut], index=[CutAlgo.threshold, CutAlgo.pyrrha, CutAlgo.no_cut].index(DEFAULT_CUTTING_FUNCTION) if DEFAULT_CUTTING_FUNCTION in [CutAlgo.threshold, CutAlgo.pyrrha, CutAlgo.no_cut] else 0)

        st.markdown('<hr style="margin: 0.2rem 0;">', unsafe_allow_html=True)
        st.subheader("PMAP sampling")
        st.markdown('<hr style="margin: 0.2rem 0;">', unsafe_allow_html=True)
        pmt_samp_wid_ns = sidebar_labeled_number_input("pmt_samp_wid (ns)", min_value=1.0, max_value=1000.0, value=DEFAULT_PMT_SAMP_WID_NS, step=1.0)
        sipm_samp_wid_us = sidebar_labeled_number_input("sipm_samp_wid (us)", min_value=0.1, max_value=1000.0, value=DEFAULT_SIPM_SAMP_WID_US, step=0.1)

        st.markdown('<hr style="margin: 0.2rem 0;">', unsafe_allow_html=True)
        password = st.text_input("Password", type="password", placeholder="Enter password to save")
        if st.button("Save current values to irene.conf", use_container_width=True):
            if not password:
                st.error("Password required to save.")
            elif hashlib.sha256(password.encode()).hexdigest() != AUTHORIZED_PASSWORD_HASH:
                st.error("Incorrect password.")
            else:
                config_updates = {
                    "files_in": str(file_path),
                    "file_out": str(CFG.get("file_out", "/tmp/irene_pmaps.h5")),
                    "compression": str(CFG.get("compression", "ZLIB4")),
                    "run_number": int(run_number),
                    "detector_db": str(detector_db),
                    "print_mod": int(CFG.get("print_mod", 1)),
                    "event_range": int(CFG.get("event_range", 1)),
                    "n_baseline": int(n_baseline),
                    "n_maw": int(n_maw),
                    "thr_maw": float(thr_maw),
                    "thr_csum_s1": float(thr_csum_s1),
                    "thr_csum_s2": float(thr_csum_s2),
                    "s1_tmin": float(s1_tmin_us),
                    "s1_tmax": float(s1_tmax_us),
                    "s1_stride": int(s1_stride),
                    "s1_lmin": int(s1_lmin),
                    "s1_lmax": int(s1_lmax),
                    "s1_rebin_stride": int(s1_rebin_stride),
                    "s2_tmin": float(s2_tmin_us),
                    "s2_tmax": float(s2_tmax_us),
                    "s2_stride": int(s2_stride),
                    "s2_lmin": int(s2_lmin),
                    "s2_lmax": int(s2_lmax),
                    "s2_rebin_stride": int(s2_rebin_stride),
                    "thr_sipm": float(thr_sipm),
                    "thr_sipm_s2": float(thr_sipm_s2),
                    "thr_sipm_type": thr_sipm_type.name,
                    "pmt_samp_wid": float(pmt_samp_wid_ns),
                    "sipm_samp_wid": float(sipm_samp_wid_us),
                    "cutting_function": cut_algo.name,
                }
                try:
                    write_parameters_to_file(CONFIG_FILE, config_updates)
                    st.success(f"Saved {len(config_updates)} values to {CONFIG_FILE.name}")
                except Exception as exc:
                    st.error(f"Failed to save configuration: {exc}")

    if not pmt_labels:
        st.error("No PMT channels found in the selected file.")
        st.stop()
    if not sipm_labels:
        st.error("No SiPM channels found in the selected file.")
        st.stop()

    selected_pmt_idx = pmt_lookup[selected_pmt_label]
    selected_sipm_idx = sipm_lookup[selected_sipm_label]

    st.info(
        f"Run {int(run_number)} | LDC {int(ldc)} | {Path(file_path).name} | Event {event_idx}/{n_events - 1} | "
        f"PMTs {n_pmts} | Samples {n_samples}"
    )

    try:
        pmt_rwf, pmt_blr, sipm_rwf, event_number = load_event(file_path, event_idx)
        t_us = np.arange(n_samples, dtype=float) * float(pmt_samp_wid_ns) * 1e-3

        deconv = deconv_pmt(str(detector_db), int(run_number), int(n_baseline))
        pmt_cwf = deconv(pmt_rwf)
        pmt_cal = calibrate_pmts(str(detector_db), int(run_number), int(n_maw), float(thr_maw))
        ccwfs, ccwfs_maw, cwf_sum, cwf_sum_maw = pmt_cal(pmt_cwf)
        suppress = zero_suppress_wfs(float(thr_csum_s1), float(thr_csum_s2))
        s1_indices, s2_indices = suppress(cwf_sum, cwf_sum_maw)

        cutting_params = dict(
            thr_sipm_type=thr_sipm_type,
            thr_sipm=float(thr_sipm),
            thr_sipm_s2=float(thr_sipm_s2),
            detector_db=str(detector_db),
            run_number=int(run_number),
        )
        sipm_selection_algo = select_cutting_algorithm(cut_algo, **cutting_params)

        pmap_error = None
        pmap_evt = None
        try:
            pmap_builder = build_pmap(
                str(detector_db),
                int(run_number),
                float(pmt_samp_wid_ns) * units.ns,
                float(sipm_samp_wid_us) * units.mus,
                int(s1_lmax),
                int(s1_lmin),
                int(s1_rebin_stride),
                int(s1_stride),
                float(s1_tmax_us) * units.mus,
                float(s1_tmin_us) * units.mus,
                int(s2_lmax),
                int(s2_lmin),
                int(s2_rebin_stride),
                int(s2_stride),
                float(s2_tmax_us) * units.mus,
                float(s2_tmin_us) * units.mus,
                sipm_selection_algo,
            )
            pmap_evt = pmap_builder(ccwfs, s1_indices, s2_indices, sipm_rwf)
        except Exception as exc:
            pmap_error = str(exc)

        s1_analyzed, s1_selected, s1_rejected = classify_candidate_segments(
            s1_indices,
            int(s1_stride),
            t_us,
            float(pmt_samp_wid_ns),
            float(s1_tmin_us),
            float(s1_tmax_us),
            int(s1_lmin),
            int(s1_lmax),
        )
        s2_analyzed, s2_selected, s2_rejected = classify_candidate_segments(
            s2_indices,
            int(s2_stride),
            t_us,
            float(pmt_samp_wid_ns),
            float(s2_tmin_us),
            float(s2_tmax_us),
            int(s2_lmin),
            int(s2_lmax),
        )

        if pmap_evt is not None:
            s1_md = build_stage_a_markdown("S1", s1_analyzed, n_in_pmap=len(pmap_evt.s1s))
            s2_md = build_stage_a_markdown("S2", s2_analyzed, n_in_pmap=len(pmap_evt.s2s))
        else:
            s1_md = build_stage_a_markdown("S1", s1_analyzed, pmap_error=pmap_error)
            s2_md = build_stage_a_markdown("S2", s2_analyzed, pmap_error=pmap_error)

        s2_windows_us = get_s2_windows_us(pmap_evt, s2_selected, t_us, float(pmt_samp_wid_ns))
        sipm_thr = get_actual_sipm_thr(SiPMThreshold.common, float(thr_sipm_s2), str(detector_db), int(run_number))
        sipm_map_fig, sipm_mapped = sipm_charge_map_figure(
            sipm_rwf,
            sipm_df,
            s2_windows_us,
            float(sipm_samp_wid_us),
            str(detector_db),
            int(run_number),
            float(sipm_thr),
        )

    except Exception as exc:
        st.error("Pipeline execution failed with current settings.")
        st.exception(exc)
        st.stop()

    st.subheader(f"Event number: {event_number}")

    pmt_sensor_id, pmt_elecid = pmt_id_from_index(pmt_df, selected_pmt_idx)
    sipm_sensor_id, sipm_elecid = sipm_id_from_index(sipm_df, selected_sipm_idx)
    st.caption(
        f"Selected PMT SensorID {pmt_sensor_id} / ElecID {pmt_elecid} | Selected SiPM SensorID {sipm_sensor_id} / ElecID {sipm_elecid}"
    )

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            overlay_plot(
                t_us,
                pmt_rwf[selected_pmt_idx],
                pmt_cwf[selected_pmt_idx],
                f"PMT {selected_pmt_idx}: raw vs deconvolved",
                "raw ADC",
                "deconvolved",
                "ADC",
            ),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(
            overlay_plot(
                t_us,
                pmt_cwf[selected_pmt_idx],
                ccwfs[selected_pmt_idx],
                f"PMT {selected_pmt_idx}: deconvolved vs calibrated",
                "deconvolved",
                "calibrated (pes)",
                "value",
            ),
            use_container_width=True,
        )

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(
            threshold_plot(
                t_us,
                cwf_sum_maw,
                float(thr_csum_s1),
                "PMT sum with S1 selected/rejected windows",
                s1_selected,
                s1_rejected,
                "#2faa60",
                "#d94a4a",
                allowed_window=(float(s1_tmin_us), float(s1_tmax_us)),
            ),
            use_container_width=True,
        )
    with col4:
        st.plotly_chart(
            threshold_plot(
                t_us,
                cwf_sum,
                float(thr_csum_s2),
                "PMT sum with S2 selected/rejected windows",
                s2_selected,
                s2_rejected,
                "#1e8e5a",
                "#c73e3e",
                allowed_window=(float(s2_tmin_us), float(s2_tmax_us)),
                extra_regions=get_s2_windows_us(pmap_evt, s1_selected, t_us, float(pmt_samp_wid_ns)),
            ),
            use_container_width=True,
        )

    with st.expander("Stage A candidate diagnostic", expanded=True):
        left, right = st.columns(2)
        with left:
            st.markdown("### S1 diagnostics")
            st.markdown(s1_md, unsafe_allow_html=True)
        with right:
            st.markdown("### S2 diagnostics")
            st.markdown(s2_md, unsafe_allow_html=True)

    st.subheader("SiPM charge map")
    if sipm_map_fig is None:
        st.info("No S2 window available for SiPM integration with current settings.")
    else:
        st.caption(f"Mapped active SiPMs: {sipm_mapped}")
        sipm_map_event = st.plotly_chart(
            sipm_map_fig,
            use_container_width=True,
            key="sipm_charge_map",
            on_select="rerun",
            selection_mode="points",
        )

        selected_sipm_idx_from_plot = None
        selection_state = sipm_map_event.selection if sipm_map_event is not None else None
        if selection_state:
            points = selection_state.get("points", []) if isinstance(selection_state, dict) else getattr(selection_state, "points", [])
            if points:
                point = points[0]
                selected_sipm_idx_from_plot = point.get("customdata") if isinstance(point, dict) else getattr(point, "customdata", None)
                if isinstance(selected_sipm_idx_from_plot, (list, tuple, np.ndarray)):
                    selected_sipm_idx_from_plot = int(selected_sipm_idx_from_plot[0])
                elif selected_sipm_idx_from_plot is not None:
                    selected_sipm_idx_from_plot = int(selected_sipm_idx_from_plot)

        if selected_sipm_idx_from_plot is not None:
            selected_sipm_idx = int(selected_sipm_idx_from_plot)
        if 0 <= int(selected_sipm_idx) < len(sipm_df):
            st.plotly_chart(
                sipm_waveform_figure(
                    sipm_rwf,
                    int(selected_sipm_idx),
                    sipm_elecid,
                    float(thr_sipm_s2),
                    float(sipm_samp_wid_us),
                    str(detector_db),
                    int(run_number),
                    pmap_windows_us=s2_windows_us,
                ),
                use_container_width=True,
            )
        else:
            st.warning("Selected SiPM index is outside the available sensor range.")

    with st.expander("Current configuration"):
        st.json(
            {
                "file": file_path,
                "run_number": int(run_number),
                "event_idx": int(event_idx),
                "n_events_file": int(n_events),
                "detector_db": detector_db,
                "pmts": int(n_pmts),
                "samples": int(n_samples),
                "selected_pmt_idx": int(selected_pmt_idx),
                "selected_sipm_idx": int(selected_sipm_idx),
                "n_baseline": int(n_baseline),
                "n_maw": int(n_maw),
                "thr_maw": float(thr_maw),
                "thr_csum_s1": float(thr_csum_s1),
                "thr_csum_s2": float(thr_csum_s2),
                "s1_tmin_us": float(s1_tmin_us),
                "s1_tmax_us": float(s1_tmax_us),
                "s1_stride": int(s1_stride),
                "s1_lmin": int(s1_lmin),
                "s1_lmax": int(s1_lmax),
                "s1_rebin_stride": int(s1_rebin_stride),
                "s2_tmin_us": float(s2_tmin_us),
                "s2_tmax_us": float(s2_tmax_us),
                "s2_stride": int(s2_stride),
                "s2_lmin": int(s2_lmin),
                "s2_lmax": int(s2_lmax),
                "s2_rebin_stride": int(s2_rebin_stride),
                "thr_sipm_type": str(thr_sipm_type),
                "thr_sipm": float(thr_sipm),
                "thr_sipm_s2": float(thr_sipm_s2),
                "cutting_function": str(cut_algo),
                "pmt_samp_wid_ns": float(pmt_samp_wid_ns),
                "sipm_samp_wid_us": float(sipm_samp_wid_us),
            }
        )


if __name__ == "__main__":
    main()
