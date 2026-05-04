import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Simulador UNMSM - Fenómenos I", layout="wide")

# Estilo visual oscuro y profesional
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1a1f2e; padding: 15px; border-radius: 10px; border: 1px solid #31333f; }
    div.stButton > button:first-child { background-color: #00d4ff; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- PARÁMETROS DE CONTROL (SIDEBAR) ---
with st.sidebar:
    st.header("📋 Parámetros del Sistema")
    rho = st.slider("Densidad (ρ) [kg/m³]", 800, 1500, 1000)
    mu = st.slider("Viscosidad (μ) [Pa·s]", 0.01, 2.00, 0.50, step=0.01)
    R = st.number_input("Radio del Tubo (R) [m]", 0.010, 0.500, 0.050, format="%.3f")
    delta = st.slider("Espesor de Película (δ) [m]", 0.001, 0.050, 0.015, step=0.001)
    g = 9.81
    st.divider()
    st.info("Este simulador resuelve el transporte de cantidad de movimiento para una película cilíndrica en descenso.")

# --- FUNCIONES DE CÁLCULO ---
aR = R + delta
a = aR / R

def calcular_vz(r_arr, rho_v, mu_v, R_v, a_v):
    condicion = r_arr >= R_v
    vz = np.zeros_like(r_arr)
    r_filt = r_arr[condicion]
    # Ecuación exacta del perfil de velocidad (Bird 2B.6)
    vz[condicion] = (rho_v * g * R_v**2 / (4 * mu_v)) * (1 - (r_filt/R_v)**2 + 2 * (a_v**2) * np.log(r_filt/R_v))
    return vz

def calcular_m_real(rho_v, mu_v, R_v, a_v):
    # Ecuación exacta del flujo másico (Bird 2B.6)
    term_a = (4 * a_v**4 * np.log(a_v)) - (3 * a_v**4 - 4 * a_v**2 + 1)
    return (np.pi * (rho_v**2) * g * (R_v**4) / (8 * mu_v)) * term_a

def calcular_m_taylor(rho_v, mu_v, R_v, d_v):
    # Aproximación de Taylor (Placa plana)
    return (2 * np.pi * R_v * (rho_v**2) * g * (d_v**3)) / (3 * mu_v)

# Cálculos de ejecución
r_plot = np.linspace(0.0001, aR * 1.1, 500)
vz_plot = calcular_vz(r_plot, rho, mu, R, a)
vz_max = np.max(vz_plot)
m_real_act = calcular_m_real(rho, mu, R, a)
m_taylor_act = calcular_m_taylor(rho, mu, R, delta)
error_actual = abs(m_real_act - m_taylor_act) / m_real_act * 100

# --- INTERFAZ PRINCIPAL ---
st.title("🛡️ Fenómenos de Transporte I: Ejercicio 2B.6")
st.markdown("### Simulación de Flujo en Película Cilíndrica Descendente")

# Métricas destacadas
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("Flujo Másico Real", f"{m_real_act:.5f} kg/s")
col_m2.metric("Velocidad Máxima", f"{vz_max:.4f} m/s")
col_m3.metric("Relación de Radios (a)", f"{a:.3f}")
col_m4.metric("Error de Taylor", f"{error_actual:.2f}%")

st.divider()

# --- BLOQUE 1: PERFIL Y 3D ---
c1, c2 = st.columns(2)

with c1:
    st.subheader("📈 Perfil de Velocidad Radial")
    fig_radial, ax_radial = plt.subplots(figsize=(8, 6))
    fig_radial.patch.set_facecolor('#0e1117')
    ax_radial.set_facecolor('#1a1f2e')
    
    ax_radial.axvspan(0, R, color='#444444', label='Tubo (v=0)')
    ax_radial.plot(r_plot, vz_plot, color='#00d4ff', lw=3, label='Perfil $v_z(r)$')
    ax_radial.fill_between(r_plot, 0, vz_plot, where=(r_plot >= R), color='#00d4ff', alpha=0.3)
    
    ax_radial.set_xlabel("Radio (r) [m]", color='white')
    ax_radial.set_ylabel("Velocidad [m/s]", color='white')
    ax_radial.tick_params(colors='white')
    ax_radial.grid(alpha=0.2)
    ax_radial.legend()
    st.pyplot(fig_radial)

with c2:
    st.subheader("🌐 Visualización 3D")
    # Generación de cilindros 3D
    t = np.linspace(0, 2*np.pi, 50)
    z_val = np.linspace(0, 1, 20)
    T, Z = np.meshgrid(t, z_val)
    
    # Superficie película
    X_f, Y_f = aR * np.cos(T), aR * np.sin(T)
    # Superficie tubo
    X_t, Y_t = R * np.cos(T), R * np.sin(T)
    
    fig_3d = go.Figure()
    fig_3d.add_trace(go.Surface(x=X_f, y=Y_f, z=Z, colorscale='Blues', opacity=0.5, showscale=False))
    fig_3d.add_trace(go.Surface(x=X_t, y=Y_t, z=Z, colorscale='Greys', opacity=1, showscale=False))
    
    fig_3d.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'), 
                         margin=dict(l=0, r=0, b=0, t=0), height=400, template="plotly_dark")
    st.plotly_chart(fig_3d, use_container_width=True)

# --- BLOQUE 2: GRÁFICAS A Y B (ANÁLISIS DE ERROR) ---
st.divider()
st.subheader("🔬 Análisis de Error (Aproximación de Taylor)")

# Generar rango de espesores para las gráficas de comparación
d_range = np.linspace(0.0005, R * 1.2, 100)
m_real_range = [calcular_m_real(rho, mu, R, (R + di)/R) for di in d_range]
m_taylor_range = [calcular_m_taylor(rho, mu, R, di) for di in d_range]
error_range = [abs(re - ta)/re * 100 for re, ta in zip(m_real_range, m_taylor_range)]

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("**Gráfica A: Crecimiento del Error Relativo**")
    fig_a = go.Figure()
    fig_a.add_trace(go.Scatter(x=d_range, y=error_range, fill='tozeroy', name='Error %', line=dict(color='#ff4b4b')))
    fig_a.add_trace(go.Scatter(x=[delta], y=[error_actual], mode='markers+text', 
                               text=["Punto Actual"], textposition="top center",
                               marker=dict(color='white', size=12, symbol='x')))
    fig_a.update_layout(template="plotly_dark", xaxis_title="Espesor δ [m]", yaxis_title="Error (%)", height=400)
    st.plotly_chart(fig_a, use_container_width=True)

with col_b:
    st.markdown("**Gráfica B: Comparación Real vs Taylor**")
    fig_b = go.Figure()
    fig_b.add_trace(go.Scatter(x=d_range, y=m_real_range, name="Real (Cilíndrica)", line=dict(color='#00ff00', width=3)))
    fig_b.add_trace(go.Scatter(x=d_range, y=m_taylor_range, name="Taylor (Placa Plana)", line=dict(color='orange', dash='dot')))
    fig_b.update_layout(template="plotly_dark", xaxis_title="Espesor δ [m]", yaxis_title="Flujo Másico [kg/s]", height=400)
    st.plotly_chart(fig_b, use_container_width=True)

# --- SUSTENTO MATEMÁTICO ---
st.divider()
with st.expander("📝 Ver Sustento Matemático y Ecuaciones"):
    st.markdown("""
    Al aplicar un balance de cantidad de movimiento en coordenadas cilíndricas para un fluido Newtoniano en estado estacionario:
    """)
    st.latex(r"v_z(r) = \frac{\rho g R^2}{4\mu} \left[ 1 - \left( \frac{r}{R} \right)^2 + 2a^2 \ln \left( \frac{r}{R} \right) \right]")
    st.markdown("El flujo másico total se obtiene integrando el perfil de velocidad en el área anular:")
    st.latex(r"\dot{m} = \int_{0}^{2\pi} \int_{R}^{R+\delta} \rho v_z(r) r \, dr \, d\theta")
    st.markdown("Resultando en la ecuación exacta utilizada en este simulador:")
    st.latex(r"\dot{m} = \frac{\pi \rho^2 g R^4}{8\mu} \left[ 4a^4 \ln a - (3a^4 - 4a^2 + 1) \right]")
