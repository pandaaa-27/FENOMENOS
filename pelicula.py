import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Simulador UNMSM - Fenómenos I", layout="wide")

# Estilo de alto contraste y diseño de contenedores
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    [data-testid="stMetricValue"] { color: #ffffff !important; }
    .stMetric { background-color: #1a1f2e; padding: 15px; border-radius: 10px; border: 1px solid #31333f; }
    .formula-box { 
        background-color: #000000; 
        padding: 25px; 
        border-radius: 15px; 
        border: 2px solid #00d4ff; 
        text-align: center; 
        margin-top: 20px;
    }
    .formula-text { font-family: 'Courier New', monospace; font-size: 28px; font-weight: bold; color: #ffff00; }
    .section-title { color: #ff00ff; font-weight: bold; font-size: 1.2rem; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- PARÁMETROS EN SIDEBAR ---
with st.sidebar:
    st.header("Parámetros de Entrada")
    rho = st.slider("Densidad ρ [kg/m³]", 800, 1500, 1000)
    mu = st.slider("Viscosidad μ [Pa·s]", 0.01, 1.00, 0.10, step=0.01)
    R = st.number_input("Radio R [m]", 0.010, 0.500, 0.050, format="%.3f")
    delta = st.slider("Espesor δ [m]", 0.001, 0.050, 0.015, step=0.001)
    g = 9.81
    if st.button("🔄 RESETEAR VALORES"):
        st.rerun()

# --- FUNCIONES MATEMÁTICAS ---
aR = R + delta
a = aR / R

def get_vz(r):
    # vz = (rho*g*R^2 / 4*mu) * [1 - (r/R)^2 + 2*a^2*ln(r/R)]
    return (rho * g * R**2 / (4 * mu)) * (1 - (r/R)**2 + 2 * (a**2) * np.log(r/R))

def get_m_real():
    term = (4 * a**4 * np.log(a)) - (3 * a**4 - 4 * a**2 + 1)
    return (np.pi * (rho**2) * g * (R**4) / (8 * mu)) * term

def get_m_taylor():
    return (2 * np.pi * R * (rho**2) * g * (delta**3)) / (3 * mu)

m_real = get_m_real()
m_taylor = get_m_taylor()

# --- INTERFAZ PRINCIPAL ---
st.title("Simulación de Flujo en Película Cilíndrica Descendente")

col_izq, col_der = st.columns(2)

with col_izq:
    st.markdown('<p class="section-title">REPRESENTACIÓN FÍSICA</p>', unsafe_allow_html=True)
    st.write("Representación Física 3D")
    
    # Gráfica 3D con Vectores
    fig3d = go.Figure()
    # Tubo interno
    z_t = np.linspace(0, 10, 20); theta = np.linspace(0, 2*np.pi, 30)
    Z_t, T_t = np.meshgrid(z_t, theta)
    X_t, Y_t = R * np.cos(T_t), R * np.sin(T_t)
    fig3d.add_trace(go.Surface(x=X_t, y=Y_t, z=Z_t, colorscale='Greys', opacity=0.8, showscale=False))
    
    # Película (transparente rosa)
    X_f, Y_f = aR * np.cos(T_t), aR * np.sin(T_t)
    fig3d.add_trace(go.Surface(x=X_f, y=Y_f, z=Z_t, colorscale='RdPu', opacity=0.3, showscale=False))
    
    # Vectores de flujo (flechas verdes)
    z_arrows = np.linspace(2, 8, 3); r_arrows = np.linspace(R, aR, 2)
    for za in z_arrows:
        for ra in r_arrows:
            for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
                fig3d.add_trace(go.Cone(x=[ra*np.cos(angle)], y=[ra*np.sin(angle)], z=[za], 
                                        u=[0], v=[0], w=[-2], colorscale='Greens', showscale=False))

    fig3d.update_layout(scene=dict(xaxis_title='[m]', yaxis_title='[m]', zaxis_title='Altura Z [m]'),
                        margin=dict(l=0,r=0,b=0,t=0), template="plotly_dark", height=450)
    st.plotly_chart(fig3d, use_container_width=True)

with col_der:
    st.markdown('<p class="section-title">PERFIL DE VELOCIDAD</p>', unsafe_allow_html=True)
    st.write("Perfil Matemático de Velocidad")
    
    # Gráfica Radial con área sombreada
    r_vals = np.linspace(R, aR, 100)
    vz_vals = get_vz(r_vals)
    
    fig_rad = go.Figure()
    fig_rad.add_trace(go.Scatter(x=r_vals, y=vz_vals, fill='tozeroy', name='Vz', 
                                 line=dict(color='#00d4ff', width=4), fillcolor='rgba(0, 212, 255, 0.2)'))
    # Puntos de interés
    fig_rad.add_trace(go.Scatter(x=[R], y=[0], mode='markers', marker=dict(color='red', size=10), name='Pared'))
    fig_rad.add_trace(go.Scatter(x=[aR], y=[np.max(vz_vals)], mode='markers', marker=dict(color='springgreen', size=10), name='Vmax'))
    
    fig_rad.update_layout(template="plotly_dark", xaxis_title="Radio r [m]", yaxis_title="Velocidad Vz [m/s]",
                          height=450, margin=dict(l=0,r=0,b=40,t=0))
    st.plotly_chart(fig_rad, use_container_width=True)

# --- SECCIÓN DE ECUACIONES ---
st.divider()
st.markdown("## Ecuaciones del Modelo")
ec1, ec2 = st.columns(2)

with ec1:
    st.markdown("**Perfil de Velocidad:**")
    st.latex(r"v_z(r) = \frac{\rho g R^2}{4\mu} \left[ 1 - \left( \frac{r}{R} \right)^2 + 2a^2 \ln \left( \frac{r}{R} \right) \right]")
    st.markdown("**Flujo Másico Real:**")
    st.latex(r"\dot{m} = \frac{\pi \rho^2 g R^4}{8\mu} \left[ 4a^4 \ln a - (3a^4 - 4a^2 + 1) \right]")

with ec2:
    st.markdown("**Análisis de Taylor (Pestaña dedicada):**")
    st.markdown(f"""<div class="formula-box"><div class="formula-text">ṁ ≈ (2πRρ²gδ³) / (3μ)</div></div>""", unsafe_allow_html=True)
    st.info(f"Diferencia actual: {abs(m_real - m_taylor)/m_real*100:.2f}%")

# --- TABLA DE IMPACTO (TAYLOR) ---
st.subheader("Impacto del Espesor de Película")
espesores = [0.0001, 0.0010, 0.0050, 0.0100, 0.0200, 0.0300]
data = []
for e in espesores:
    a_e = (R + e) / R
    m_e = (np.pi * (rho**2) * g * (R**4) / (8 * mu)) * ((4 * a_e**4 * np.log(a_e)) - (3 * a_e**4 - 4 * a_e**2 + 1))
    m_t = (2 * np.pi * R * (rho**2) * g * (e**3)) / (3 * mu)
    data.append({"Espesor δ (m)": e, "ṁ exacto": f"{m_e:.5f}", "ṁ Taylor": f"{m_t:.5f}", "Error (%)": f"{abs(m_e-m_t)/m_e*100:.2f}"})

st.table(pd.DataFrame(data))
