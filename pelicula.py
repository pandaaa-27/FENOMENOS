import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Simulador UNMSM - Fenómenos I", layout="wide")

# --- PARÁMETROS DE CONTROL (SIDEBAR) ---
with st.sidebar:
    st.header("📋 Parámetros del Sistema")
    rho = st.slider("Densidad (ρ) [kg/m³]", 800, 1500, 1000)
    mu = st.slider("Viscosidad (μ) [Pa·s]", 0.01, 1.00, 0.10, step=0.01)
    R = st.number_input("Radio del Tubo (R) [m]", 0.010, 0.500, 0.050, format="%.3f")
    delta = st.slider("Espesor de Película (δ) [m]", 0.001, 0.050, 0.030, step=0.001)
    g = 9.81

# --- CÁLCULOS MATEMÁTICOS ---
aR = R + delta
a = aR / R

# Perfil de velocidad: r desde 0 hasta aR
r_total = np.linspace(0.0001, aR, 300) # Evitamos 0 exacto por el logaritmo

def calcular_vz(r_arr):
    # vz = 0 para r < R (dentro del tubo)
    # vz = fórmula para R <= r <= aR
    condicion = r_arr >= R
    vz = np.zeros_like(r_arr)
    
    # Aplicamos la fórmula solo donde r >= R
    r_filt = r_arr[condicion]
    vz[condicion] = (rho * g * R**2 / (4 * mu)) * (1 - (r_filt/R)**2 + 2 * (a**2) * np.log(r_filt/R))
    return vz

vz_total = calcular_vz(r_total)
vz_max = np.max(vz_total)

# Flujo másico (Corregido según la fórmula del Bird 2B.6)
# m_punto = (pi * rho^2 * g * R^4 / 8*mu) * [4*a^4*ln(a) - (3*a^4 - 4*a^2 + 1)]
term_a = (4 * a**4 * np.log(a)) - (3 * a**4 - 4 * a**2 + 1)
m_punto = (np.pi * (rho**2) * g * (R**4) / (8 * mu)) * term_a

# --- 1. RESULTADOS PRINCIPALES ---
st.title("🛡️ Solución de Flujo en Película Cilíndrica - Ejercicio 2B6")

col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.metric("Flujo Másico (ṁ)", f"{m_punto:.4f} kg/s")
with col_m2:
    st.metric("Velocidad Máxima", f"{vz_max:.4f} m/s")
with col_m3:
    st.metric("Relación de Radios (a)", f"{a:.3f}")

st.divider()

# --- 2. VISUALIZACIÓN ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("🖼️ Representación Física (Descenso)")
    fig1, ax1 = plt.subplots(figsize=(6, 7))
    
    ax1.axvspan(0, R, color='#444444', label='Tubo Sólido')
    ax1.axvspan(R, aR, color='#add8e6', alpha=0.5, label='Líquido Descendente')
    
    # Vectores de velocidad
    z_arrows = np.linspace(2, 8, 8)
    r_arrows = np.linspace(R, aR, 5)
    for z in z_arrows:
        for ra in r_arrows:
            v_loc = (rho * g * R**2 / (4 * mu)) * (1 - (ra/R)**2 + 2 * (a**2) * np.log(ra/R))
            v_len = (v_loc / vz_max) * 0.02 if vz_max > 0 else 0
            ax1.arrow(ra, z, 0, -v_len, head_width=0.002, head_length=0.1, fc='blue', ec='blue')

    ax1.set_xlim(0, aR + 0.02)
    ax1.set_ylim(0, 10)
    ax1.set_xlabel("Radio (r) [m]")
    ax1.set_ylabel("Altura (Z)")
    ax1.legend(loc='upper right')
    st.pyplot(fig1)

with col2:
    st.subheader("📈 Perfil Matemático con Tubo")
    fig2, ax2 = plt.subplots(figsize=(6, 7))
    
    ax2.axvspan(0, R, color='#444444', alpha=0.8, label='Tubo (v=0)')
    ax2.plot(r_total, vz_total, color='blue', lw=3, label='Perfil $v_z(r)$')
    ax2.fill_between(r_total, 0, vz_total, where=(r_total >= R), color='blue', alpha=0.1)
    
    ax2.scatter([R], [0], color='red', zorder=5, label='Pared (No-slip)')
    ax2.scatter([aR], [vz_max], color='green', zorder=5, label='$v_{max}$ (Interfase)')
    
    ax2.set_xlabel("Radio (r) [m]")
    ax2.set_ylabel("Velocidad [m/s]")
    ax2.set_xlim(0, aR + 0.01)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend()
    st.pyplot(fig2)

# --- 3. SUSTENTO MATEMÁTICO ---
st.divider()
st.subheader("📝 Sustento Matemático")
c_eq1, c_eq2 = st.columns(2)

with c_eq1:
    st.markdown("**Distribución de Velocidad:**")
    st.latex(r"v_z(r) = \frac{\rho g R^2}{4\mu} \left[ 1 - \left( \frac{r}{R} \right)^2 + 2a^2 \ln \left( \frac{r}{R} \right) \right]")

with c_eq2:
    st.markdown("**Flujo Másico Total:**")
    st.latex(r"\dot{m} = \frac{\pi \rho^2 g R^4}{8\mu} \left[ 4a^4 \ln a - (3a^4 - 4a^2 + 1) \right]")

st.info(f"Nota: Con R={R} y δ={delta}, el valor de a es {a:.3f}. El flujo másico ahora se calcula de forma consistente con las ecuaciones mostradas.")
