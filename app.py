import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Page Config
st.set_page_config(page_title="Geotechnical Lab Verifier (IS 2720)", layout="wide")

# App Header with Author Branding
st.title("🏗️ Geotechnical Lab Verifier (IS 2720)")
st.caption("Developed by: **Vishal Raj** | Research Scholar, IIT Hyderabad, doing research under Prof. B.Munwar Basha🎓")
st.write("Triaxial Compression & Direct Shear Test Verification Tool")

st.markdown("---")

# Sidebar - Test Controls & Developer Info
st.sidebar.title("👨‍🔬 Developer Info")
st.sidebar.info("""
**Vishal Raj**  
*Research Scholar @ IIT Hyderabad, doing research under Prof. B.Munwar Basha*  
[LinkedIn Profile](https://www.linkedin.com/in/vishalraj-2002) 🔗
""")

st.sidebar.header("Test Inputs & Setup")

test_type = st.sidebar.selectbox("Select Test:", ["Triaxial Compression Test", "Direct Shear Test"])
unit_str = st.sidebar.radio("Select Unit:", ["kg/cm²", "kPa"])

conv = 98.0665 if unit_str == "kg/cm²" else 1.0

st.sidebar.subheader("Lab Test Data Sets")

# Default Data
if test_type == "Triaxial Compression Test":
    default_df = pd.DataFrame({
        "Set": [1, 2, 3],
        "σ3 (Cell)": [0.5, 1.0, 1.5],
        "σ1 (Major)": [1.7, 3.4, 5.1]
    })
else:
    default_df = pd.DataFrame({
        "Set": [1, 2, 3],
        "σ (Normal)": [0.5, 1.0, 1.5],
        "τ (Shear)": [0.32, 0.50, 0.68]
    })

edited_df = st.sidebar.data_editor(default_df, num_rows="dynamic")

manual_c = st.sidebar.number_input(f"Manual c' Reading ({unit_str}):", value=0.00, step=0.01)
manual_phi = st.sidebar.number_input("Manual φ' (Degrees):", value=33.0, step=0.1)

# Analysis Button
if st.sidebar.button("🚀 ANALYZE & VERIFY", use_container_width=True):
    col1, col2 = st.columns([1.5, 1])

    data = edited_df.to_numpy()
    
    if test_type == "Triaxial Compression Test":
        s3 = data[:, 1] * conv
        s1 = data[:, 2] * conv

        centers = (s1 + s3) / 2.0
        radii = (s1 - s3) / 2.0

        # p-q Best-fit Linear Regression
        p = centers
        q = radii
        P = np.column_stack([p, np.ones_like(p)])
        m_pq, d_pq = np.linalg.lstsq(P, q, rcond=None)[0]

        sin_phi = np.clip(m_pq, 0.01, 0.99)
        phi_rad = np.arcsin(sin_phi)
        phi_calc = np.degrees(phi_rad)
        c_calc_kpa = max(0.0, d_pq / np.cos(phi_rad))
        c_calc = c_calc_kpa / conv if unit_str == "kg/cm²" else c_calc_kpa

        # Plotting
        fig, ax = plt.subplots(figsize=(7, 5))
        theta = np.linspace(0, np.pi, 200)
        max_x = max(s1) * 1.15

        for i in range(len(s3)):
            x_circle = centers[i] + radii[i] * np.cos(theta)
            y_circle = radii[i] * np.sin(theta)
            ax.fill_between(x_circle, y_circle, color='red', alpha=0.25)
            ax.plot(x_circle, y_circle, 'r-', linewidth=1.3)
            ax.plot(centers[i], 0, 'ro', markersize=5)

        x_env = np.linspace(0, max_x, 200)
        y_env = c_calc_kpa + x_env * np.tan(phi_rad)
        ax.plot(x_env, y_env, 'g-', linewidth=2.5, label='Failure Envelope')

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(0, max_x)
        ax.set_ylim(0, max(radii) * 1.3)
        ax.set_xlabel('Normal Stress σ (kPa)')
        ax.set_ylabel('Shear Stress τ (kPa)')
        ax.set_title('Triaxial Mohr Circles & Failure Envelope')
        ax.grid(True, linestyle='--', alpha=0.6)

    else:
        sig = data[:, 1] * conv
        tau = data[:, 2] * conv

        P = np.column_stack([sig, np.ones_like(sig)])
        m, c_calc_kpa = np.linalg.lstsq(P, tau, rcond=None)[0]

        phi_rad = np.arctan(m)
        phi_calc = np.degrees(phi_rad)
        c_calc = c_calc_kpa / conv if unit_str == "kg/cm²" else c_calc_kpa

        # Plotting
        fig, ax = plt.subplots(figsize=(7, 5))
        max_x = max(sig) * 1.2
        ax.plot(sig, tau, 'rs', markersize=8)

        x_line = np.linspace(0, max_x, 100)
        y_line = c_calc_kpa + x_line * np.tan(phi_rad)
        ax.plot(x_line, y_line, 'b-', linewidth=2.5)

        ax.set_xlim(0, max_x)
        ax.set_ylim(0, max(tau) * 1.3)
        ax.set_xlabel('Normal Stress σ (kPa)')
        ax.set_ylabel('Shear Stress τ (kPa)')
        ax.set_title('Direct Shear Envelope')
        ax.grid(True, linestyle='--', alpha=0.6)

    with col1:
        st.pyplot(fig)

    # Verification Checks
    err_c = abs(c_calc - manual_c) / (c_calc + 1e-5) * 100
    err_phi = abs(phi_calc - manual_phi) / phi_calc * 100

    verdict = "✅ ACCEPTABLE / ACCURATE (< 5% Error)" if (err_c <= 5.0 and err_phi <= 5.0) else "❌ RE-CHECK GRAPH (> 5% Error)"

    with col2:
        st.subheader("Analysis Results")
        st.metric("Calculated Cohesion (c')", f"{c_calc:.3f} {unit_str}")
        st.metric("Calculated Friction Angle (φ')", f"{phi_calc:.2f}°")
        st.divider()
        st.write(f"**c' Error:** {err_c:.2f}%")
        st.write(f"**φ' Error:** {err_phi:.2f}%")
        
        if "ACCEPTABLE" in verdict:
            st.success(f"**VERDICT:** {verdict}")
        else:
            st.error(f"**VERDICT:** {verdict}")

# Footer Section
st.markdown("---")
st.markdown("Designed & Developed with ❤️ by **Vishal Raj** (Research Scholar @ IIT Hyderabad)")
