import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Page Config
st.set_page_config(
    page_title="Geotechnical Lab Verifier (IS 2720)",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Premium Dark Theme UI)
st.markdown("""
    <style>
    .main {
        background-color: #0f172a;
    }
    .title-text {
        font-size: 2.4rem !important;
        font-weight: 800 !important;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #94a3b8;
        font-weight: 500;
        margin-bottom: 20px;
    }
    .custom-card {
        background: rgba(30, 41, 59, 0.7);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(4px);
        margin-bottom: 20px;
    }
    .badge {
        background-color: #0284c7;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 10px;
    }
    .footer-text {
        text-align: center;
        color: #64748b;
        font-size: 0.9rem;
        margin-top: 30px;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# App Header
st.markdown('<div class="badge">IS 2720 (Part 11 & 12) Standard</div>', unsafe_allow_html=True)
st.markdown('<h1 class="title-text">🏗️ Geotechnical Lab Verifier</h1>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Soil Shear Strength Analysis & Verification Tool</div>', unsafe_allow_html=True)

# Developer Branding Banner
st.markdown("""
<div class="custom-card" style="border-left: 5px solid #38bdf8; padding: 12px 20px;">
    <b>Developed by:</b> <span style="color: #38bdf8; font-weight: bold;">Vishal Raj</span> | 
    <i>Research Scholar under Prof. B Munawar Basha @ Indian Institute of Technology Hyderabad (IITH)</i> 🎓
</div>
""", unsafe_allow_html=True)

# Sidebar - Test Controls
with st.sidebar:
    st.title("⚙️ Control Panel")
    st.markdown("---")
    
    st.subheader("1️⃣ Select Test Type")
    test_type = st.selectbox("", ["Triaxial Compression Test", "Direct Shear Test"])
    
    st.subheader("2️⃣ Select Measurement Unit")
    unit_str = st.radio("", ["kg/cm²", "kPa"], horizontal=True)

    st.markdown("---")
    st.subheader("3️⃣ Test Data Entry")

    # Fixed syntax error in default_df (Added [1, 2, 3] to "Set")
    if test_type == "Triaxial Compression Test":
        default_df = pd.DataFrame({
            "Set": [1, 2, 3],
            "σ3 (Cell)": [0.5, 1.0, 1.5] if unit_str == "kg/cm²" else [50.0, 100.0, 150.0],
            "σ1 (Major)": [1.7, 3.4, 5.1] if unit_str == "kg/cm²" else [170.0, 340.0, 510.0]
        })
    else:
        default_df = pd.DataFrame({
            "Set": [1, 2, 3],
            "σ (Normal)": [0.5, 1.0, 1.5] if unit_str == "kg/cm²" else [50.0, 100.0, 150.0],
            "τ (Shear)": [0.32, 0.50, 0.68] if unit_str == "kg/cm²" else [32.0, 50.0, 68.0]
        })

    edited_df = st.data_editor(default_df, num_rows="dynamic", use_container_width=True)

    st.markdown("---")
    st.subheader("4️⃣ Manual Observations")
    manual_c = st.number_input(f"Manual c' Reading ({unit_str}):", value=0.00, step=0.01)
    manual_phi = st.number_input("Manual φ' (Degrees):", value=33.0, step=0.1)

    st.markdown("---")
    analyze_btn = st.button("🚀 ANALYZE & VERIFY", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown("""
    👨‍🔬 **Developer Profile**  
    **Vishal Raj**  
    [LinkedIn Profile](https://www.linkedin.com/in/vishalraj-2002) 🔗
    """)

# Main Workspace Tabs
tab1, tab2 = st.tabs(["📊 Interactive Analysis", "📖 Benchmark Verification Details"])

with tab1:
    if analyze_btn:
        col1, col2 = st.columns([1.6, 1])

        data = edited_df.to_numpy()
        
        if test_type == "Triaxial Compression Test":
            s3 = data[:, 1]
            s1 = data[:, 2]

            centers = (s1 + s3) / 2.0
            radii = (s1 - s3) / 2.0

            # Linear regression: R = Xc * sin(phi) + c * cos(phi)
            X_mat = np.column_stack([centers, np.ones_like(centers)])
            lstsq_res = np.linalg.lstsq(X_mat, radii, rcond=None)[0]
            
            slope_m = lstsq_res[0]       # sin(phi)
            intercept_d = lstsq_res[1]   # c * cos(phi)

            sin_phi = np.clip(slope_m, 0.01, 0.99)
            phi_rad = np.arcsin(sin_phi)
            phi_calc = np.degrees(phi_rad)
            c_calc = max(0.0, intercept_d / np.cos(phi_rad))

            # Matplotlib Dark Styling
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(7, 5))
            fig.patch.set_facecolor('#1e293b')
            ax.set_facecolor('#0f172a')

            theta = np.linspace(0, np.pi, 200)
            max_x = max(s1) * 1.15

            for i in range(len(s3)):
                x_circle = centers[i] + radii[i] * np.cos(theta)
                y_circle = radii[i] * np.sin(theta)
                ax.fill_between(x_circle, y_circle, color='#f43f5e', alpha=0.2)
                ax.plot(x_circle, y_circle, color='#f43f5e', linewidth=1.8)
                ax.plot(centers[i], 0, 'o', color='#fb7185', markersize=6)

            x_env = np.linspace(0, max_x, 200)
            y_env = c_calc + x_env * np.tan(phi_rad)
            ax.plot(x_env, y_env, color='#38bdf8', linewidth=2.5, label='Failure Envelope')

            ax.set_aspect('equal', adjustable='box')
            ax.set_xlim(0, max_x)
            ax.set_ylim(0, max(radii) * 1.35)
            ax.set_xlabel(f'Normal Stress σ ({unit_str})', color='#e2e8f0', fontweight='bold')
            ax.set_ylabel(f'Shear Stress τ ({unit_str})', color='#e2e8f0', fontweight='bold')
            ax.set_title('Mohr Circles & Failure Envelope', color='#38bdf8', fontsize=12, fontweight='bold', pad=12)
            ax.grid(True, linestyle='--', alpha=0.3, color='#475569')

        else:
            sig = data[:, 1]
            tau = data[:, 2]

            P = np.column_stack([sig, np.ones_like(sig)])
            lstsq_res = np.linalg.lstsq(P, tau, rcond=None)[0]
            m = lstsq_res[0]
            c_calc = max(0.0, lstsq_res[1])

            phi_rad = np.arctan(m)
            phi_calc = np.degrees(phi_rad)

            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(7, 5))
            fig.patch.set_facecolor('#1e293b')
            ax.set_facecolor('#0f172a')

            max_x = max(sig) * 1.2
            ax.plot(sig, tau, 's', color='#f43f5e', markersize=8)

            x_line = np.linspace(0, max_x, 100)
            y_line = c_calc + x_line * np.tan(phi_rad)
            ax.plot(x_line, y_line, color='#38bdf8', linewidth=2.5)

            ax.set_xlim(0, max_x)
            ax.set_ylim(0, max(tau) * 1.35)
            ax.set_xlabel(f'Normal Stress σ ({unit_str})', color='#e2e8f0', fontweight='bold')
            ax.set_ylabel(f'Shear Stress τ ({unit_str})', color='#e2e8f0', fontweight='bold')
            ax.set_title('Direct Shear Envelope', color='#38bdf8', fontsize=12, fontweight='bold', pad=12)
            ax.grid(True, linestyle='--', alpha=0.3, color='#475569')

        with col1:
            st.pyplot(fig)

        # Verification Checks
        err_c = abs(c_calc - manual_c) / (manual_c + 1e-5) * 100
        err_phi = abs(phi_calc - manual_phi) / (manual_phi + 1e-5) * 100

        is_accurate = (err_c <= 5.0 and err_phi <= 5.0)

        with col2:
            st.markdown('### 📌 Analysis Results')
            st.metric(label=f"Calculated Cohesion (c')", value=f"{c_calc:.3f} {unit_str}")
            st.metric(label=f"Calculated Friction Angle (φ')", value=f"{phi_calc:.2f}°")
            
            st.markdown("---")
            st.write(f"**c' Deviation:** `{err_c:.2f}%`")
            st.write(f"**φ' Deviation:** `{err_phi:.2f}%`")
            
            if is_accurate:
                st.success("✅ **VERDICT: ACCEPTABLE**\n\nResults match manual observations within 5% tolerance.")
            else:
                st.error("❌ **VERDICT: RE-CHECK GRAPH**\n\nError exceeds 5% threshold compared to manual entries.")
    else:
        st.info("👈 Enter test parameters in the sidebar and click **'🚀 ANALYZE & VERIFY'** to display plots and verification analysis.")

with tab2:
    st.markdown("""
    ### 📚 Benchmark Validation
    This application utilizes linear regression in $p-q$ stress space and transformed failure envelopes validated against standard benchmark problems from:
    * **Braja M. Das** - *Principles of Geotechnical Engineering*
    * **Dr. B. C. Punmia** - *Soil Mechanics and Foundations*
    * **Gopal Ranjan & A. S. R. Rao** - *Basic and Applied Soil Mechanics*
    
    ### 🎯 Target Use Cases
    * **Undergraduate & Postgraduate Labs**: Real-time cross-checking of graphical calculations.
    * **Geotechnical Consultants**: Automated calculation verification for foundation & slope stability designs.
    """)

# Footer
st.markdown('<div class="footer-text">Built with Streamlit & Python • Designed by <b>Vishal Raj</b> (IIT Hyderabad)</div>', unsafe_allow_html=True)
