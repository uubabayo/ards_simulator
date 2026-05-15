# app.py
"""
ARDS Ventilator Simulation Dashboard
Full-Stack Clinical Simulation for ARDS Supportive Management
Author: Senior Biomedical & Software Engineer
"""

import streamlit as st
import numpy as np
import pandas as pd
import math

# ------------------------------- PAGE CONFIG -------------------------------
st.set_page_config(
    page_title="ARDS Ventilator Simulator",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------- CONSTANTS -------------------------------
PATMOSPHERE = 760      # mmHg
PH2O = 47              # mmHg (water vapor pressure at 37°C)
RESPIRATORY_QUOTIENT = 0.8
BASELINE_PaCO2 = 40    # mmHg
REFERENCE_VT = 420     # mL, used for PaCO2 dynamic estimation
MIXED_VENOUS_PO2 = 40  # mmHg, assumed for shunt calculation
RESPIRATORY_RATE = 15  # breaths per minute

# ------------------------------- CLINICAL PROFILES -------------------------------
PROFILES = {
    "Mild ARDS": {
        "baseline_compliance": 40,    # mL/cmH2O (reference)
        "target_shunt": 0.15,         # Qs/Qt fraction
        "min_peep": 8,                # below this PEEP, derecruitment penalty applies
    },
    "Moderate ARDS": {
        "baseline_compliance": 25,
        "target_shunt": 0.25,
        "min_peep": 10,
    },
    "Severe ARDS": {
        "baseline_compliance": 15,
        "target_shunt": 0.40,
        "min_peep": 12,
    },
}

# ------------------------------- HELPER FUNCTIONS -------------------------------
def alveolar_gas_equation(fio2: float, paco2: float) -> float:
    """
    PAO2 = (FiO2 * (Patm - PH2O)) - (PaCO2 / R)
    """
    return (fio2 * (PATMOSPHERE - PH2O)) - (paco2 / RESPIRATORY_QUOTIENT)

def dynamic_paco2(vt: float, ref_vt: float = REFERENCE_VT, baseline_paco2: float = BASELINE_PaCO2) -> float:
    """
    Simulates PaCO2 change with tidal volume.
    Assuming constant CO2 production and dead space, PaCO2 ∝ 1/VT.
    """
    if vt <= 0:
        return float('inf')
    return baseline_paco2 * (ref_vt / vt)

def apply_derecruitment_penalty(profile: dict, peep: float):
    """
    If PEEP is below the profile-specific minimum, apply a derecruitment penalty:
    - reduce compliance by a penalty factor
    - increase shunt fraction
    Returns (compliance_factor, effective_shunt)
    """
    min_peep = profile["min_peep"]
    baseline_shunt = profile["target_shunt"]

    if peep < min_peep:
        deficit = min_peep - peep
        penalty = min(0.5, deficit * 0.05)   # 5% loss per cmH2O below threshold, capped at 50%
        compliance_factor = 1.0 - penalty
        shunt_increase = penalty * 0.5        # shunt increases proportionally to penalty
        effective_shunt = min(0.7, baseline_shunt + shunt_increase)  # cap at 70% shunt
        return compliance_factor, effective_shunt
    else:
        return 1.0, baseline_shunt

def arterial_oxygenation(pao2: float, shunt_fraction: float, pvo2: float = MIXED_VENOUS_PO2) -> float:
    """
    Simple shunt model: PaO2 = PAO2*(1-Qs/Qt) + PvO2*(Qs/Qt)
    """
    return pao2 * (1 - shunt_fraction) + pvo2 * shunt_fraction

def pf_ratio(pao2: float, fio2: float) -> float:
    """Return PaO2/FiO2 ratio. Handle FiO2=0 gracefully."""
    if fio2 == 0:
        return 0
    return pao2 / fio2

def generate_pressure_waveform(peep, pplat, rr=RESPIRATORY_RATE, duration=12, sampling_rate=100):
    """
    P(t) = PEEP + (Pplat - PEEP) * sin^2(π * RR * t)
    RR in breaths per minute, t in seconds.
    Returns a DataFrame with columns 'time' and 'airway_pressure'.
    """
    t = np.linspace(0, duration, int(duration * sampling_rate), endpoint=False)
    rr_hz = rr / 60.0  # convert to Hz
    # sin^2(π * RR * t) produces smooth waveform between 0 and 1
    pressure = peep + (pplat - peep) * (np.sin(np.pi * rr_hz * t) ** 2)
    df = pd.DataFrame({
        "Time (s)": t,
        "Airway Pressure (cmH₂O)": pressure
    })
    return df

# ------------------------------- STREAMLIT UI -------------------------------
st.title("🫁 ARDS Ventilator Management Simulation")
st.markdown("Interactive dashboard integrating alveolar gas equation, lung mechanics, and clinical risk markers.")

# SIDEBAR INPUTS
with st.sidebar:
    st.header("⚙️ Ventilator Settings")
    profile_name = st.selectbox(
        "Patient Profile",
        list(PROFILES.keys()),
        index=1  # default Moderate
    )
    fio2 = st.slider("FiO₂", 0.21, 1.00, 0.40, 0.01)
    peep = st.slider("PEEP (cmH₂O)", 5, 24, 10, 1)
    vt_ml = st.slider("Tidal Volume (mL)", 300, 600, 420, 10)
    pplat = st.slider("Plateau Pressure (cmH₂O)", 15, 45, 25, 1)

# FETCH PROFILE DATA
profile = PROFILES[profile_name]
baseline_compliance = profile["baseline_compliance"]
target_shunt = profile["target_shunt"]

# ------------------------------- CORE COMPUTATIONS -------------------------------
# 1. Dynamic PaCO2 based on tidal volume
paco2 = dynamic_paco2(vt_ml, REFERENCE_VT, BASELINE_PaCO2)

# 2. Alveolar oxygen tension (PAO2)
pao2_alveolar = alveolar_gas_equation(fio2, paco2)

# 3. Lung mechanics (raw values)
delta_p = pplat - peep  # Driving pressure ΔP
if delta_p != 0:
    cstat_raw = vt_ml / delta_p
else:
    cstat_raw = float('inf')

# 4. Derecruitment penalty on compliance and shunt
compliance_factor, effective_shunt = apply_derecruitment_penalty(profile, peep)
cstat_effective = cstat_raw * compliance_factor if cstat_raw != float('inf') else float('inf')

# 5. Arterial oxygenation incorporating shunt
pao2_arterial = arterial_oxygenation(pao2_alveolar, effective_shunt)
pf = pf_ratio(pao2_arterial, fio2)

# 6. Generate dynamic waveform
waveform_df = generate_pressure_waveform(peep, pplat, RESPIRATORY_RATE)

# ------------------------------- DISPLAY METRICS -------------------------------
st.subheader("📊 Real-Time Physiological Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="PAO₂ (Alveolar)",
        value=f"{pao2_alveolar:.1f} mmHg",
        delta=None,
        delta_color="off"
    )

with col2:
    # Driving Pressure ΔP
    delta_color = "inverse" if delta_p > 15 else "normal"
    st.metric(
        label="Driving Pressure (ΔP)",
        value=f"{delta_p:.1f} cmH₂O",
        delta=f"{delta_p - 15:.1f} cmH₂O above limit" if delta_p > 15 else None,
        delta_color=delta_color
    )
    if delta_p > 15:
        st.warning("⚠️ Driving Pressure > 15 cmH₂O – consider lung protective strategy.")

with col3:
    # Static Compliance (effective)
    if cstat_effective != float('inf'):
        cstat_display = f"{cstat_effective:.1f} mL/cmH₂O"
    else:
        cstat_display = "∞"
    st.metric(
        label="Static Compliance (Cstat)",
        value=cstat_display,
        delta=f"Reduced by {(1-compliance_factor)*100:.0f}%" if compliance_factor < 1 else "No derecruitment penalty",
        delta_color="off"
    )
    if compliance_factor < 1.0:
        st.caption("PEEP below profile threshold – compliance penalised for alveolar derecruitment.")

with col4:
    st.metric(
        label="PaO₂ (Arterial)",
        value=f"{pao2_arterial:.1f} mmHg",
        delta=None
    )
    st.metric(
        label="P/F Ratio",
        value=f"{pf:.0f} mmHg",
        delta="Severe ARDS" if pf < 100 else "Moderate" if pf < 200 else "Mild" if pf < 300 else "Normal",
        delta_color="off"
    )

# Safety alerts
st.subheader("🚨 Clinical Alerts")
alert_container = st.container()
with alert_container:
    if pplat > 30:
        st.error("🔥 BAROTRAUMA RISK: Plateau Pressure > 30 cmH₂O! Immediate action required.")
    if compliance_factor < 0.8:
        st.warning("📉 Significant compliance loss due to low PEEP — alveolar derecruitment likely.")
    if pao2_arterial < 60:
        st.error("🩸 Severe hypoxemia: Arterial PaO₂ < 60 mmHg.")
    if pf < 100:
        st.error("🔴 P/F ratio < 100 — severe ARDS, refractory hypoxemia.")

# ------------------------------- PRESSURE WAVEFORM -------------------------------
st.subheader("🌬️ Airway Pressure Waveform (3 Respiratory Cycles)")
st.caption(f"PEEP = {peep} cmH₂O, Pplat = {pplat} cmH₂O, RR = {RESPIRATORY_RATE} /min")

st.line_chart(
    waveform_df,
    x="Time (s)",
    y="Airway Pressure (cmH₂O)",
    use_container_width=True
)

# ------------------------------- FOOTER -------------------------------
st.markdown("---")
st.markdown("*Developed for educational and simulation purposes. Not for clinical use.*")