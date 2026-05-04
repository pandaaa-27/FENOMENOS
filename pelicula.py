import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Simulador UNMSM - Fenómenos I", layout="wide")

# Estilo visual de alto contraste para métricas y visibilidad
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    /* Estilo para que las métricas sean blancas y muy legibles */
    [data-testid="stMetricValue"] { color: #ffffff !important; font-size: 2.2rem !important; font-weight: 700 !important; }
    [data-testid="stMetricLabel"] { color: #00d4ff !important; font-size: 1.1rem !important; }
    .stMetric { background-color: #1a1f2e; padding: 20px; border-radius: 12px; border: 2px solid #31333f; }
    </style>
    """, unsafe_allow_html=True)

# --- PARÁMETROS DE CONTROL (SIDEBAR) ---
with st.sidebar:
    st.header("📋 Parámetros")
    rho = st.slider("Densidad (ρ) [kg/m³]", 800, 1500, 1000)
    mu = st.slider("Viscosidad (μ) [Pa·s]", 0.01, 2.00, 0.50, step=0.01)
    R = st.number_input("Radio del Tubo (R) [m]", 0.010, 0.500, 0.050, format="%.3f")
    delta = st.slider("Espesor (δ) [m]", 0.001, 0.050, 0.015, step=0.001)
    g = 9.81
    st.divider()
    st.success("Configuración cargada")

# --- FUNCIONES DE CÁLCULO ---
aR = R + delta
a = aR / R

def calcular_vz(r_arr, rho_v, mu_v, R_v, a_v):
    condicion = r_arr >= R_v
    vz = np.zeros_like(r_arr)
    r_filt = r_arr[condicion]
    vz[condicion] = (rho_v * g * R_v**2 / (4 * mu_v)) * (1 - (r_filt/R_v)**2 + 2 * (a_v**2) * np.log(r_filt/R_v))
    return vz

def calcular_m_real(rho_v, mu_v, R_v, d_v):
    a_v = (R_v + d_v) / R_v
    term_a = (4 * a_v**4 * np.log(a_v)) - (3 * a_v**4 - 4 * a_v**2 + 1)
    return (np.pi * (rho_v**2) * g * (R_v**4) / (8 * mu_v)) * term_a

def calcular_m_taylor(rho_v, mu_v, R_v, d_v):
    return (2 * np.pi * R_v * (rho_v**2) * g * (d_v**3)) / (3 * mu_v)

# Cálculos actuales
m_real_act = calcular_m_real(rho, mu, R, delta)
m_taylor_act = calcular_m_taylor(rho, mu, R, delta)
error_actual = abs(m_real_act - m_taylor_act) / m_real_act * 100
r_plot = np.linspace(0.0001, aR * 1.1, 500)
vz_plot = calcular_vz(r_plot, rho, mu, R, a)

# --- INTERFAZ PRINCIPAL ---
st.title("🛡️ Simulador de Flujo: Ejercicio 2B.6")

# Métricas con colores de alto contraste
c_m1, c_m2, c_m3, c_m4 = st.columns(4)
c_m1.metric("Flujo Másico Real", f"{m_real_act:.5f} kg/s")
c_m2.metric("Velocidad Máxima", f"{np.max(vz_plot):.4f} m/s")
c_m3.metric("Relación Radios (a)", f"{a:.3f}")
c_m4.metric("Error de Taylor", f"{error_actual:.2f}%")

st.divider()

# --- SISTEMA DE PESTAÑAS (TABS) ---
tab1, tab2, tab3 = st.tabs(["📊 Visualización de Flujo", "🔬 Análisis de Taylor", "📝 Sustento Teórico"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Perfil de Velocidad")
        fig_r, ax_r = plt.subplots(figsize=(8, 6))
        fig_r.patch.set_facecolor('#0e1117')
        ax_r.set_facecolor('#1a1f2e')
        ax_r.axvspan(0, R, color='#333333', label='Tubo Sólido')
        ax_radial_line, = ax_r.plot(r_plot, vz_plot, color='#00d4ff', lw=3, label='$v_z(r)$')
        ax_r.set_xlabel("Radio [m]", color='white')
        ax_r.set_ylabel("Velocidad [m/s]", color='white')
        ax_r.tick_params(colors='white')
        ax_r.legend()
        st.pyplot(fig_r)

    with col2:
        st.subheader("Simulación 3D")
        t_3d = np.linspace(0, 2*np.pi, 50); z_3d = np.linspace(0, 1, 20)
        T_3, Z_3 = np.meshgrid(t_3d, z_3d)
        X_f, Y_f = aR * np.cos(T_3), aR * np.sin(T_3)
        X_t, Y_t = R * np.cos(T_3), R * np.sin(T_3)
        fig_3 = go.Figure(data=[
            go.Surface(x=X_f, y=Y_f, z=Z_3, colorscale='Ice', opacity=0.5, showscale=False),
            go.Surface(x=X_t, y=Y_t, z=Z_3, colorscale='Greys', opacity=1, showscale=False)
        ])
        fig_3.update_layout(template="plotly_dark", margin=dict(l=0,r=0,b=0,t=0), height=400)
        st.plotly_chart(fig_3, use_container_width=True)

with tab2:
    st.subheader("Comparativa: Geometría Cilíndrica vs. Placa Plana")
    
    # Tabla Comparativa de Valores
    st.markdown("### 📋 Tabla de Resultados")
    data = {
        "Parámetro": ["Flujo Másico (ṁ)", "Espesor de Película (δ)", "Radio Interno (R)", "Área de Flujo"],
        "Modelo Real (Cilíndrico)": [f"{m_real_act:.6f} kg/s", f"{delta:.4f} m", f"{R:.4f} m", f"{np.pi*(aR**2 - R**2):.6f} m²"],
        "Modelo Taylor (Placa)": [f"{m_taylor_act:.6f} kg/s", f"{delta:.4f} m", "∞ (Plano)", f"{2*np.pi*R*delta:.6f} m²"],
        "Diferencia (%)": ["-", "-", "-", f"{abs(1 - (2*R*delta)/(aR**2-R**2))*100:.2f}%"]
    }
    data["Diferencia (%)"][0] = f"{error_actual:.2f}%"
    st.table(pd.DataFrame(data))

    # Gráficas A y B
    d_vals = np.linspace(0.001, R * 1.5, 100)
    m_r_vals = [calcular_m_real(rho, mu, R, dv) for dv in d_vals]
    m_t_vals = [calcular_m_taylor(rho, mu, R, dv) for dv in d_vals]
    err_vals = [abs(rv - tv)/rv * 100 for rv, tv in zip(m_r_vals, m_t_vals)]

    g_a, g_b = st.columns(2)
    with g_a:
        st.markdown("**Gráfica A: Error Relativo (%)**")
        fig_a = go.Figure(go.Scatter(x=d_vals, y=err_vals, fill='tozeroy', line=dict(color='#ff4b4b')))
        fig_a.add_trace(go.Scatter(x=[delta], y=[error_actual], mode='markers', marker=dict(size=15, color='white')))
        fig_a.update_layout(template="plotly_dark", height=350, xaxis_title="Espesor δ", yaxis_title="Error %")
        st.plotly_chart(fig_a, use_container_width=True)

    with g_b:
        st.markdown("**Gráfica B: Convergencia de Modelos**")
        fig_b = go.Figure()
        fig_b.add_trace(go.Scatter(x=d_vals, y=m_r_vals, name="Real", line=dict(color='#00d4ff', width=3)))
        fig_b.add_trace(go.Scatter(x=d_vals, y=m_t_vals, name="Taylor", line=dict(color='yellow', dash='dot')))
        fig_b.update_layout(template="plotly_dark", height=350, xaxis_title="Espesor δ", yaxis_title="ṁ [kg/s]")
        st.plotly_chart(fig_b, use_container_width=True)

with tab3:
    st.header("Sustento Matemático")
    st.markdown("""
    Para el flujo de una película líquida sobre la superficie exterior de un cilindro, la ecuación de continuidad y de movimiento en estado estacionario (Bird 2B.6) nos otorga el perfil:
    """)
    st.latex(r"v_z(r) = \frac{\rho g R^2}{4\mu} \left[ 1 - \left( \frac{r}{R} \right)^2 + 2a^2 \ln \left( \frac{r}{R} \right) \right]")
    st.markdown("Cuando el espesor $\delta$ es pequeño respecto a $R$, la expansión en serie de Taylor simplifica la integración a:")
    st.latex(r"\dot{m} \approx \frac{2\pi R \rho^2 g \delta^3}{3\mu}")
    st.info("Este simulador permite evaluar visual y numéricamente la pérdida de precisión de esta simplificación a medida que la curvatura se vuelve importante.")
