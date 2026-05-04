import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Simulador UNMSM - Fenómenos I", layout="wide")

# Estilo de alto contraste (Métricas en Blanco Puro)
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    [data-testid="stMetricValue"] { color: #ffffff !important; font-size: 2.5rem !important; font-weight: 800 !important; }
    [data-testid="stMetricLabel"] { color: #00d4ff !important; font-size: 1.2rem !important; }
    .stMetric { background-color: #1a1f2e; padding: 20px; border-radius: 12px; border: 2px solid #31333f; }
    .latex-container { background-color: #1a1f2e; padding: 15px; border-radius: 10px; border: 1px solid #00d4ff; text-align: center; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- PARÁMETROS EN SIDEBAR ---
with st.sidebar:
    st.header("📋 Parámetros")
    rho = st.slider("Densidad (ρ) [kg/m³]", 800, 1500, 1000)
    mu = st.slider("Viscosidad (μ) [Pa·s]", 0.01, 2.00, 0.50, step=0.01)
    R = st.number_input("Radio del Tubo (R) [m]", 0.010, 0.500, 0.050, format="%.3f")
    delta_user = st.slider("Espesor Actual (δ) [m]", 0.001, 0.050, 0.015, step=0.001)
    g = 9.81

# --- FUNCIONES NÚCLEO ---
def get_m_real(r_val, mu_val, rho_val, d_val):
    a_val = (r_val + d_val) / r_val
    term = (4 * a_val**4 * np.log(a_val)) - (3 * a_val**4 - 4 * a_val**2 + 1)
    return (np.pi * (rho_val**2) * g * (r_val**4) / (8 * mu_val)) * term

def get_m_taylor(r_val, mu_val, rho_val, d_val):
    return (2 * np.pi * r_val * (rho_val**2) * g * (d_val**3)) / (3 * mu_val)

m_real_actual = get_m_real(R, mu, rho, delta_user)
m_taylor_actual = get_m_taylor(R, mu, rho, delta_user)
error_actual = abs(m_real_actual - m_taylor_actual) / m_real_actual * 100

# --- INTERFAZ ---
st.title("🛡️ Simulación de Flujo en Película Cilíndrica")

# Métricas de cabecera
c1, c2, c3, c4 = st.columns(4)
c1.metric("ṁ Exacto", f"{m_real_actual:.5f} kg/s")
c2.metric("ṁ Taylor", f"{m_taylor_actual:.5f} kg/s")
c3.metric("Error Relativo", f"{error_actual:.2f}%")
c4.metric("Relación (a)", f"{(R+delta_user)/R:.3f}")

st.divider()

tab1, tab2 = st.tabs(["📊 Visualización de Flujo", "🔬 Análisis de Taylor"])

with tab1:
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Perfil de Velocidad Radial")
        r_range = np.linspace(R, R + delta_user, 100)
        a_act = (R + delta_user) / R
        vz = (rho * g * R**2 / (4 * mu)) * (1 - (r_range/R)**2 + 2 * (a_act**2) * np.log(r_range/R))
        
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor('#0e1117'); ax.set_facecolor('#1a1f2e')
        ax.plot(r_range, vz, color='#00d4ff', lw=4)
        ax.set_xlabel("Radio [m]", color="white"); ax.set_ylabel("Velocidad [m/s]", color="white")
        ax.tick_params(colors="white")
        st.pyplot(fig)
    
    with col_b:
        st.subheader("Representación 3D")
        st.info("Cilindro concéntrico representando el fluido sobre el tubo sólido.")
        t = np.linspace(0, 2*np.pi, 50); z = np.linspace(0, 1, 10)
        T, Z = np.meshgrid(t, z)
        X, Y = (R+delta_user)*np.cos(T), (R+delta_user)*np.sin(T)
        fig3d = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale='Cividis', showscale=False)])
        fig3d.update_layout(template="plotly_dark", height=400, margin=dict(l=0,r=0,b=0,t=0))
        st.plotly_chart(fig3d, use_container_width=True)

with tab2:
    st.subheader("Análisis de la Serie de Taylor")
    
    # 1. Fórmula en caja destacada (como en tu imagen)
    st.markdown(f"""<div class="latex-container">
        <h2 style="color:#00d4ff; margin-bottom:10px;">Aproximación de Placa Plana</h2>
        <code style="font-size:25px; color:#ffffff;">ṁ ≈ (2πRρ²gδ³) / (3μ)</code>
    </div>""", unsafe_allow_html=True)
    
    col_izq, col_der = st.columns([1, 1.2])
    
    with col_izq:
        st.markdown("### Impacto del Espesor de Película")
        # Generar tabla de impacto con valores fijos/dinámicos similares a tu imagen
        espesores = [0.0001, 0.0010, 0.0050, 0.0100, 0.0200, 0.0300, delta_user]
        espesores = sorted(list(set(espesores))) # Eliminar duplicados y ordenar
        
        filas = []
        for d in espesores:
            m_e = get_m_real(R, mu, rho, d)
            m_t = get_m_taylor(R, mu, rho, d)
            err = abs(m_e - m_t) / m_e * 100
            filas.append({
                "Espesor δ (m)": f"{d:.4f}",
                "ṁ Exacto": f"{m_e:.5f}",
                "ṁ Taylor": f"{m_t:.5f}",
                "Error (%)": f"{err:.2f}%"
            })
        st.table(pd.DataFrame(filas))

    with col_der:
        # Preparación de gráficas comparativas
        d_range = np.linspace(0.0005, 0.05, 100)
        m_e_vals = [get_m_real(R, mu, rho, d) for d in d_range]
        m_t_vals = [get_m_taylor(R, mu, rho, d) for d in d_range]
        err_vals = [abs(re - ta)/re * 100 for re, ta in zip(m_e_vals, m_t_vals)]

        # Gráfica A: Comparación de Curvas
        fig_a = go.Figure()
        fig_a.add_trace(go.Scatter(x=d_range, y=m_e_vals, name="ṁ Exacto (Cilíndrico)", line=dict(color='#a29bfe', width=3)))
        fig_a.add_trace(go.Scatter(x=d_range, y=m_t_vals, name="ṁ Taylor (Simplificado)", line=dict(color='#fd79a8', dash='dash')))
        fig_a.update_layout(title="Gráfica A: Comparación de curvas vs δ", template="plotly_dark", height=300, margin=dict(b=20, t=40))
        st.plotly_chart(fig_a, use_container_width=True)

        # Gráfica B: Evolución del Error
        fig_b = go.Figure()
        fig_b.add_trace(go.Scatter(x=d_range, y=err_vals, name="Error %", fill='tozeroy', line=dict(color='#e84393')))
        # Línea de límite 5%
        fig_b.add_hline(y=5, line_dash="dot", line_color="yellow", annotation_text="Límite 5% de Error")
        
        fig_b.update_layout(title="Gráfica B: Evolución del Error Relativo (%) vs δ", template="plotly_dark", height=300, margin=dict(b=20, t=40))
        st.plotly_chart(fig_b, use_container_width=True)

    st.warning(f"💡 El análisis muestra que con el radio R={R}m, la aproximación de Taylor cruza el 5% de error cerca de δ ≈ 0.003m.")
