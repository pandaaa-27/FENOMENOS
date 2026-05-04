import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Simulador UNMSM - Ejercicio 2B6", layout="wide")

# Estilo CSS de alto contraste (Métricas en Blanco Puro)
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    [data-testid="stMetricValue"] { color: #ffffff !important; font-size: 2.5rem !important; font-weight: 800 !important; }
    [data-testid="stMetricLabel"] { color: #00d4ff !important; font-size: 1.1rem !important; }
    .stMetric { background-color: #1a1f2e; padding: 15px; border-radius: 10px; border: 1px solid #31333f; }
    .formula-box { 
        background-color: #000000; padding: 20px; border-radius: 15px; 
        border: 2px solid #00d4ff; text-align: center; margin-bottom: 20px;
    }
    .formula-text { font-family: 'Courier New', monospace; font-size: 28px; font-weight: bold; color: #ffff00; }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("Parámetros de Entrada")
    rho = st.slider("Densidad ρ [kg/m³]", 800, 1500, 1000)
    mu = st.slider("Viscosidad μ [Pa·s]", 0.01, 1.00, 0.10, step=0.01)
    R = st.number_input("Radio del Tubo R [m]", 0.010, 0.200, 0.050, format="%.3f")
    delta_user = st.slider("Espesor Película δ [m]", 0.001, 0.040, 0.015, step=0.001)
    g = 9.81

# --- LÓGICA MATEMÁTICA ---
def get_vz(r_val, r_v, mu_v, rho_v, d_v):
    a_v = (r_v + d_v) / r_v
    return (rho_v * g * r_v**2 / (4 * mu_v)) * (1 - (r_val/r_v)**2 + 2 * (a_v**2) * np.log(r_val/r_v))

def get_m_real(r_v, mu_v, rho_v, d_v):
    a_v = (r_v + d_v) / r_v
    term = (4 * a_v**4 * np.log(a_v)) - (3 * a_v**4 - 4 * a_v**2 + 1)
    return (np.pi * (rho_v**2) * g * (r_v**4) / (8 * mu_v)) * term

def get_m_taylor(r_v, mu_v, rho_v, d_v):
    return (2 * np.pi * r_v * (rho_v**2) * g * (d_v**3)) / (3 * mu_v)

# Cálculos inmediatos para la cabecera
m_r_act = get_m_real(R, mu, rho, delta_user)
m_t_act = get_m_taylor(R, mu, rho, delta_user)
# La velocidad máxima ocurre en la interfase líquido-gas (r = R + delta)
vz_max_act = get_vz(R + delta_user, R, mu, rho, delta_user)
err_act = abs(m_r_act - m_t_act) / m_r_act * 100

# --- INTERFAZ ---
st.title("Simulación de Flujo en Película Cilíndrica Descendente")

# MÉTRICAS SUPERIORES (Incluyendo Velocidad)
m1, m2, m3, m4 = st.columns(4)
m1.metric("ṁ Exacto", f"{m_r_act:.5f} kg/s")
m2.metric("Vz Máxima", f"{vz_max_act:.4f} m/s")
m3.metric("ṁ Taylor", f"{m_t_act:.5f} kg/s")
m4.metric("Error Relativo", f"{err_act:.2f}%")

st.divider()

# --- PESTAÑAS ---
tab1, tab2 = st.tabs(["📊 Visualización Física", "🔬 Análisis de Taylor"])

with tab1:
    col_izq, col_der = st.columns(2)
    with col_izq:
        st.subheader("Representación Física 3D")
        fig3d = go.Figure()
        z_c = np.linspace(0, 10, 20); th = np.linspace(0, 2*np.pi, 50)
        Z_c, T_c = np.meshgrid(z_c, th)
        
        # 1. Tubo Central (Gris Oscuro Sólido)
        fig3d.add_trace(go.Surface(x=R*np.cos(T_c), y=R*np.sin(T_c), z=Z_c, 
                                   colorscale=[[0, '#1a1a1a'], [1, '#404040']], opacity=1, showscale=False))
        
        # 2. Película de Líquido (Cian Neón translúcido para ver el espesor)
        fig3d.add_trace(go.Surface(x=(R+delta_user)*np.cos(T_c), y=(R+delta_user)*np.sin(T_c), z=Z_c, 
                                   colorscale=[[0, '#00d4ff'], [1, '#00d4ff']], opacity=0.35, showscale=False))
        
        # Flechas de flujo (pequeñas hacia abajo)
        for zp in [2, 5, 8]:
            for ang in np.linspace(0, 2*np.pi, 8, endpoint=False):
                fig3d.add_trace(go.Cone(x=[(R+delta_user/2)*np.cos(ang)], y=[(R+delta_user/2)*np.sin(ang)], z=[zp], 
                                        u=[0], v=[0], w=[-0.5], colorscale=[[0, '#00ff00'], [1, '#00ff00']], 
                                        sizemode="absolute", sizeref=0.1, showscale=False))
        
        fig3d.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                            template="plotly_dark", height=500, margin=dict(l=0,r=0,b=0,t=0))
        st.plotly_chart(fig3d, use_container_width=True)

    with col_der:
        st.subheader("Perfil Matemático de Velocidad")
        r_vals = np.linspace(R, R+delta_user, 100)
        vz_vals = get_vz(r_vals, R, mu, rho, delta_user)
        
        fig_rad = go.Figure(go.Scatter(x=r_vals, y=vz_vals, fill='tozeroy', line=dict(color='#00d4ff', width=4), fillcolor='rgba(0, 212, 255, 0.2)'))
        fig_rad.update_layout(template="plotly_dark", xaxis_title="Radio r [m]", yaxis_title="Velocidad Vz [m/s]", height=450)
        st.plotly_chart(fig_rad, use_container_width=True)
        
    st.divider()
    st.markdown("### Ecuaciones del Modelo (Bird 2B.6)")
    c_eq1, c_eq2 = st.columns(2)
    with c_eq1:
        st.markdown("**Perfil de Velocidad $v_z(r)$:**")
        st.latex(r"v_z(r) = \frac{\rho g R^2}{4\mu} \left[ 1 - \left( \frac{r}{R} \right)^2 + 2a^2 \ln \left( \frac{r}{R} \right) \right]")
    with c_eq2:
        st.markdown("**Flujo Másico Total $\dot{m}$:**")
        st.latex(r"\dot{m} = \frac{\pi \rho^2 g R^4}{8\mu} \left[ 4a^4 \ln a - (3a^4 - 4a^2 + 1) \right]")

with tab2:
    st.subheader("Estudio de la Aproximación de Taylor")
    st.markdown(f'<div class="formula-box"><div style="color:#00d4ff; margin-bottom:5px;">Aproximación de Placa Plana:</div><div class="formula-text">ṁ ≈ (2πRρ²gδ³) / (3μ)</div></div>', unsafe_allow_html=True)
    
    c_tab, c_graf = st.columns([1, 1.2])
    
    with c_tab:
        st.markdown("**Impacto del Espesor de Película**")
        espesores = sorted(list(set([0.0001, 0.0010, 0.0050, 0.0100, 0.0200, 0.0300, delta_user])))
        filas = [{"Espesor δ (m)": d, "ṁ exacto": f"{get_m_real(R,mu,rho,d):.5f}", "ṁ Taylor": f"{get_m_taylor(R,mu,rho,d):.5f}", "Error (%)": f"{abs(get_m_real(R,mu,rho,d)-get_m_taylor(R,mu,rho,d))/get_m_real(R,mu,rho,d)*100:.2f}"} for d in espesores]
        st.table(pd.DataFrame(filas))

    with c_graf:
        d_range = np.linspace(0.0001, 0.05, 100)
        m_e = [get_m_real(R,mu,rho,d) for d in d_range]
        m_t = [get_m_taylor(R,mu,rho,d) for d in d_range]
        err = [abs(re-ta)/re*100 for re, ta in zip(m_e, m_t)]
        
        # Gráfica A
        fig_a = go.Figure()
        fig_a.add_trace(go.Scatter(x=d_range, y=m_e, name="m_exacto", line=dict(color='#a29bfe', width=3)))
        fig_a.add_trace(go.Scatter(x=d_range, y=m_t, name="m_taylor", line=dict(color='#fd79a8', dash='dash')))
        fig_a.update_layout(title="Comparación de Curvas", template="plotly_dark", height=300)
        st.plotly_chart(fig_a, use_container_width=True)
        
        # Gráfica B
        fig_b = go.Figure()
        fig_b.add_trace(go.Scatter(x=d_range, y=err, name="Error %", fill='tozeroy', line=dict(color='#e84393')))
        fig_b.add_hline(y=5, line_dash="dot", line_color="yellow", annotation_text="Límite 5% Error")
        fig_b.update_layout(title="Evolución del Error Relativo", template="plotly_dark", height=300)
        st.plotly_chart(fig_b, use_container_width=True)
