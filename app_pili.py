import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import os
import json

# ============================================
# FICHA TÉCNICA EXPLOSIVOS
# ============================================
# ============================================
# FICHA TÉCNICA EXPLOSIVOS (generado desde el CSV explosivos_final (1).csv)
# Nota: la app NO lee el CSV en tiempo de ejecución; los datos están embebidos aquí.
# ============================================

# ============================================
# FÓRMULAS TEÓRICAS ENAEX
# ============================================
# Basado en Manual de Tronadura ENAEX
# Incluye cálculos de parámetros óptimos según teoría

def calcular_Kb_enaex(ucs):
    """
    Constante de burden según dureza de roca (Manual ENAEX).
    
    Parámetros:
    - ucs: Resistencia a compresión uniaxial (MPa)
    
    Retorna:
    - Kb: Constante para cálculo de burden óptimo
    """
    if pd.isna(ucs):
        return 30
    if ucs < 50:
        return 35  # Roca blanda
    elif ucs < 100:
        return 30  # Roca media
    elif ucs < 150:
        return 28  # Roca dura
    else:
        return 25  # Roca muy dura


def calcular_Ks_enaex(ucs):
    """
    Constante S/B (Espaciamiento/Burden) según dureza de roca.
    
    Parámetros:
    - ucs: Resistencia a compresión uniaxial (MPa)
    
    Retorna:
    - Ks: Ratio óptimo S/B
    """
    if pd.isna(ucs):
        return 1.25
    if ucs < 50:
        return 1.40  # Roca blanda - mayor espaciamiento
    elif ucs < 100:
        return 1.30  # Roca media
    elif ucs < 150:
        return 1.20  # Roca dura
    else:
        return 1.15  # Roca muy dura - menor espaciamiento


def calcular_Th_enaex(ucs):
    """
    Constante de timing entre pozos según Konya (Manual ENAEX Tabla 7.1).
    
    Parámetros:
    - ucs: Resistencia a compresión uniaxial (MPa)
    
    Retorna:
    - Th: Constante de timing (ms/m)
    """
    if pd.isna(ucs):
        return 4.5
    if ucs < 50:
        return 6.5  # Arena, margas
    elif ucs < 80:
        return 5.5  # Calizas, esquistos
    elif ucs < 120:
        return 4.5  # Calizas compactas, granitos
    else:
        return 3.5  # Gneis compactos


def calcular_fc_optimo_enaex(ucs):
    """
    Factor de carga óptimo según UCS (kg/m³).
    
    Parámetros:
    - ucs: Resistencia a compresión uniaxial (MPa)
    
    Retorna:
    - FC óptimo (kg/m³)
    """
    if pd.isna(ucs):
        return 0.5
    if ucs < 50:
        return 0.35  # Roca blanda
    elif ucs < 100:
        return 0.50  # Roca media
    elif ucs < 150:
        return 0.75  # Roca dura
    else:
        return 1.00  # Roca muy dura


def recomendar_explosivo_enaex(ucs):
    """
    Recomienda tipo de explosivo según UCS (Manual ENAEX).
    
    Parámetros:
    - ucs: Resistencia a compresión uniaxial (MPa)
    
    Retorna:
    - Explosivo recomendado (string)
    """
    if pd.isna(ucs):
        ucs = 100
    
    if ucs < 50:
        return 'ANFO / Blendex 920-930'
    elif ucs < 80:
        return 'Blendex 940-950 / Vertex ALR'
    elif ucs < 120:
        return 'Emultex BN / Energex 50'
    elif ucs < 180:
        return 'Energex 70 / Pirex S'
    else:
        return 'Pirex S Plus / Energex 70 Plus'


# =============================================================================
# MODELO PILIX v3.6 - FUNCIONES ROBUSTAS DE FRAGMENTACIÓN
# =============================================================================

def calcular_indice_lilly(ucs, rqd=70):
    """
    Calcula el Índice de Lilly (Blastability Index) para estimar Factor de Roca A.
    
    Basado en: Lilly, P.A. (1986) - "An Empirical Method of Assessing Rock Mass Blastability"
    
    Fórmulas:
    ---------
    SGI = 25×SG - 50
    HP = 0.05×UCS
    BI = 0.5 × (RMD + JPS + JPO + SGI + HP)
    A = 0.12 × BI
    
    Parámetros:
    -----------
    - ucs: Resistencia a compresión uniaxial (MPa)
    - rqd: Rock Quality Designation (%)
    
    Retorna:
    --------
    - Diccionario con todos los parámetros de Lilly y el Factor A
    """
    if pd.isna(ucs):
        ucs = 100
    if pd.isna(rqd):
        rqd = 70
    
    # RMD (Rock Mass Description): basado en RQD
    if rqd > 75:
        RMD = 50  # Masivo
    elif rqd >= 50:
        RMD = 20  # Bloques
    else:
        RMD = 10  # Quebradizo
    
    # JPS (Joint Plane Spacing): basado en RQD
    if rqd > 80:
        JPS = 50  # Amplio
    elif rqd >= 50:
        JPS = 20  # Intermedio
    else:
        JPS = 10  # Cercano
    
    # JPO (Joint Plane Orientation): valor promedio
    JPO = 25  # Orientación intermedia
    
    # SG (Densidad): estimación basada en UCS
    SG = min(2.9, max(2.4, 2.5 + (ucs / 400)))
    
    # Tensile (σt): relación típica σt ≈ UCS/10
    tensile = ucs / 10
    
    # Cálculos Lilly
    SGI = 25 * SG - 50
    HP = 0.05 * ucs
    BI = 0.5 * (RMD + JPS + JPO + SGI + HP)
    A = 0.12 * BI
    
    # Limitar A a rango típico
    A = max(3.0, min(13.0, A))
    
    return {
        'RMD': RMD,
        'JPS': JPS,
        'JPO': JPO,
        'SG': round(SG, 2),
        'tensile': round(tensile, 1),
        'SGI': round(SGI, 1),
        'HP': round(HP, 1),
        'BI': round(BI, 1),
        'A': round(A, 2)
    }


def calcular_n_cunningham(B, S, d_pulg, Lc, H, W=0.5):
    """
    Calcula el índice de uniformidad n usando la fórmula de Cunningham corregida.
    
    IMPORTANTE: El diámetro debe estar en MILÍMETROS en la fórmula.
    
    Fórmula:
    --------
    n = (2.2 - 14×B/D_mm) × √[(1+S/B)/2] × (1-W/B) × (Lc/H)
    
    Parámetros:
    -----------
    - B: Burden (m)
    - S: Espaciamiento (m)
    - d_pulg: Diámetro de perforación (pulgadas)
    - Lc: Longitud de carga (m)
    - H: Altura de banco (m)
    - W: Desviación estándar de perforación (m), típico 0.5m
    
    Retorna:
    --------
    - Índice de uniformidad n
    """
    # Convertir diámetro a milímetros (CRÍTICO!)
    d_mm = d_pulg * 25.4
    
    # Calcular cada término
    term1 = 2.2 - 14 * (B / d_mm)
    term2 = np.sqrt((1 + S / B) / 2)
    term3 = 1 - W / B
    term4 = Lc / H
    
    n = term1 * term2 * term3 * term4
    
    # Limitar a rango válido
    n = max(0.5, min(2.5, n))
    
    return round(n, 3)


def estimar_Kx_Kn(ucs, rqd=70):
    """
    Estima factores de calibración Kx y Kn desde UCS y RQD.
    
    Basado en: Mundaca et al. (2015) - Calibración Kuz-Ram UGT-4 Chuquicamata
    
    Fórmulas empíricas:
    -------------------
    Kx ≈ 0.25 + 0.75 × (UCS/200) × (RQD/100)
    Kn ≈ 0.95 + 0.10 × (RQD/100)
    
    Interpretación:
    ---------------
    - Kx < 1: Fragmentación real más fina que predicción teórica
    - Kx > 1: Fragmentación real más gruesa que predicción teórica
    - Kn < 1: Distribución menos uniforme
    - Kn > 1: Distribución más uniforme
    
    Referencia UGT-4 Chuquicamata: Kx=0.31, Kn=1.04 (calibrado con Split)
    
    Parámetros:
    -----------
    - ucs: Resistencia a compresión uniaxial (MPa)
    - rqd: Rock Quality Designation (%)
    
    Retorna:
    --------
    - Tupla (Kx, Kn)
    """
    if pd.isna(ucs):
        ucs = 100
    if pd.isna(rqd):
        rqd = 70
    
    Kx = max(0.2, min(1.5, 0.25 + 0.75 * (ucs / 200) * (rqd / 100)))
    Kn = max(0.85, min(1.15, 0.95 + 0.10 * (rqd / 100)))
    
    return round(Kx, 3), round(Kn, 3)


def calcular_zonas_dano(d_pulg, densidad_exp, vod, ucs, tensile=None):
    """
    Calcula zonas de daño usando modelo Holmberg-Persson.
    
    Fórmulas:
    ---------
    Pd = ρe × VOD² / 4  (Presión de detonación, GPa)
    rc = r0 × (Pd/σc)^0.6  (Radio de trituración)
    rf = r0 × (Pd/σt)^0.45  (Radio de fractura)
    rd = rf × 2.0  (Radio de daño)
    
    Parámetros:
    -----------
    - d_pulg: Diámetro de perforación (pulgadas)
    - densidad_exp: Densidad del explosivo (g/cc)
    - vod: Velocidad de detonación (m/s)
    - ucs: Resistencia a compresión uniaxial (MPa)
    - tensile: Resistencia a tracción (MPa), si None se estima como UCS/10
    
    Retorna:
    --------
    - Diccionario con Pd, r0, rc, rf, rd
    """
    d_m = d_pulg * 0.0254  # metros
    r0 = d_m / 2
    
    # Presión de detonación (GPa)
    rho_e = densidad_exp * 1000  # kg/m³
    Pd = (rho_e * (vod ** 2)) / 4 / 1e9
    
    # Resistencias en GPa
    sigma_c = ucs / 1000
    sigma_t = (tensile if tensile else ucs / 10) / 1000
    
    # Radios de daño
    rc = r0 * ((Pd / sigma_c) ** 0.6)
    rc = max(rc, r0 * 3)
    
    rf = r0 * ((Pd / sigma_t) ** 0.45)
    rf = max(rf, rc * 1.5)
    
    rd = rf * 2.0
    
    return {
        'Pd': round(Pd, 2),
        'r0': round(r0, 3),
        'rc': round(rc, 2),
        'rf': round(rf, 2),
        'rd': round(rd, 2)
    }


def calcular_X50_kuzram_calibrado(A, Kx, V0, Q, RWS=100):
    """
    Calcula X50 usando modelo Kuz-Ram con factor de calibración Kx.
    
    Fórmula completa:
    -----------------
    X50 = Kx × A × (V0/Q)^0.8 × Q^(1/6) × (1.15/Er)^(19/30)
    
    Donde:
    - Kx: Factor de calibración (Mundaca)
    - A: Factor de roca (Lilly)
    - V0: Volumen de roca por pozo (m³)
    - Q: Masa de explosivo por pozo (kg)
    - Er: RWS/100 (ANFO=1.0)
    
    Parámetros:
    -----------
    - A: Factor de roca (3-13)
    - Kx: Factor de calibración (0.2-1.5)
    - V0: Volumen de roca por pozo (m³)
    - Q: Masa de explosivo por pozo (kg)
    - RWS: Relative Weight Strength (ANFO=100)
    
    Retorna:
    --------
    - X50 en centímetros
    """
    Er = RWS / 100
    
    X50_cm = Kx * A * ((V0 / Q) ** 0.8) * (Q ** (1/6)) * ((1.15 / Er) ** (19/30))
    
    # Limitar a rango físicamente razonable
    X50_cm = max(3, min(60, X50_cm))
    
    return round(X50_cm, 2)


def calcular_percentiles_rosin_rammler(X50, n):
    """
    Calcula percentiles usando distribución Rosin-Rammler.
    
    Fórmulas:
    ---------
    Xc = X50 / (ln2)^(1/n)
    P(x) = 1 - exp[-(x/Xc)^n]
    Px = Xc × [-ln(1-P/100)]^(1/n)
    
    Parámetros:
    -----------
    - X50: Tamaño medio de fragmentación (cm)
    - n: Índice de uniformidad
    
    Retorna:
    --------
    - Diccionario con Xc, P10, P25, P50, P80, P95, P100 (en pulgadas)
    """
    # Xc en cm
    Xc = X50 / (0.693 ** (1/n))
    
    # Función para calcular percentil
    def percentil(p_frac):
        return Xc * ((-np.log(1 - p_frac)) ** (1/n))
    
    # Calcular percentiles y convertir a pulgadas
    cm_to_in = 1 / 2.54
    
    return {
        'Xc': round(Xc * cm_to_in, 2),
        'P10': round(percentil(0.10) * cm_to_in, 2),
        'P25': round(percentil(0.25) * cm_to_in, 2),
        'P50': round(X50 * cm_to_in, 2),
        'P80': round(percentil(0.80) * cm_to_in, 2),
        'P95': round(percentil(0.95) * cm_to_in, 2),
        'P100': round(percentil(0.99) * cm_to_in, 2)
    }


def calcular_fragmentacion_pilix(B, S, T, H, J, d_pulg, ucs, rqd=70, 
                                  densidad_exp=1.2, vod=4500, RWS=100,
                                  Kx=None, Kn=None, usar_lilly=True,
                                  doble_taco=False, Ti=2.0, Ti_pos=8.0, eta_alpha=0.12):
    """
    Modelo completo de fragmentación PILIX v3.6.
    
    Integra:
    - Índice de Lilly para Factor A
    - Kuz-Ram con calibración Kx
    - Cunningham para índice n
    - Calibración Kn
    - Modelo doble taco con 3 APDs
    - Rosin-Rammler para percentiles
    
    Parámetros:
    -----------
    - B, S, T, H, J: Geometría (m)
    - d_pulg: Diámetro (pulgadas)
    - ucs: Resistencia UCS (MPa)
    - rqd: Rock Quality Designation (%)
    - densidad_exp: Densidad explosivo (g/cc)
    - vod: Velocidad detonación (m/s)
    - RWS: Relative Weight Strength (ANFO=100)
    - Kx, Kn: Factores de calibración (si None, se estiman)
    - usar_lilly: Si True, calcula A con Lilly; si False, usa A=19.9
    - doble_taco: Activar modelo doble taco
    - Ti: Altura taco intermedio (m)
    - Ti_pos: Posición taco intermedio desde superficie (m)
    - eta_alpha: Factor alpha para distribución (0.10-0.15)
    
    Retorna:
    --------
    - Diccionario completo con todos los resultados
    """
    # 1. Calcular parámetros de Lilly
    lilly = calcular_indice_lilly(ucs, rqd)
    A = lilly['A'] if usar_lilly else 19.9
    
    # 2. Estimar Kx/Kn si no se proporcionan
    if Kx is None or Kn is None:
        Kx_est, Kn_est = estimar_Kx_Kn(ucs, rqd)
        Kx = Kx if Kx is not None else Kx_est
        Kn = Kn if Kn is not None else Kn_est
    
    # 3. Calcular geometría y carga
    L = H + J
    d_m = d_pulg * 0.0254
    area_pozo = np.pi * (d_m / 2) ** 2
    
    if doble_taco:
        # Modelo doble taco
        Lc1 = max(0.5, Ti_pos - T)  # Carga superior
        Lc2 = max(0.5, L - T - Lc1 - Ti - J)  # Carga inferior
        Lc_total = Lc1 + Lc2
        Q = Lc_total * area_pozo * densidad_exp * 1000
        
        # Factor de distribución η
        balance = 1 - abs(Lc1 - Lc2) / (Lc1 + Lc2)
        eta_dist = 1 + eta_alpha * balance
    else:
        Lc_total = L - T - J
        Q = Lc_total * area_pozo * densidad_exp * 1000
        eta_dist = 1.0
        Lc1 = Lc_total
        Lc2 = 0
    
    # 4. Calcular volumen
    V0 = B * S * H
    
    # 5. Calcular n de Cunningham
    n_calc = calcular_n_cunningham(B, S, d_pulg, Lc_total, H)
    n_adj = n_calc * Kn
    
    # 6. Aplicar mejora por doble taco
    if doble_taco:
        beta = 0.5
        n_eff = n_adj * (1 + beta * (eta_dist - 1))
    else:
        n_eff = n_adj
    
    n_eff = max(0.5, min(2.5, n_eff))
    
    # 7. Calcular X50 con Kuz-Ram calibrado
    X50_cm = calcular_X50_kuzram_calibrado(A, Kx, V0, Q, RWS)
    
    # 8. Aplicar factor de distribución si doble taco
    if doble_taco:
        X50_cm = X50_cm / eta_dist
    
    # 9. Calcular percentiles
    percentiles = calcular_percentiles_rosin_rammler(X50_cm, n_eff)
    
    # 10. Reducir P100 si doble taco
    if doble_taco:
        gamma = 0.20
        T_total = T + Ti
        reduccion = 1 - gamma * (Ti / T_total)
        percentiles['P100'] = round(percentiles['P100'] * reduccion, 2)
    
    # 11. Calcular zonas de daño
    zonas = calcular_zonas_dano(d_pulg, densidad_exp, vod, ucs, lilly['tensile'])
    
    # 12. Calcular métricas adicionales
    metros_por_ha = (10000 / (B * S)) * H
    FC = Q / V0  # Factor de carga
    
    return {
        # Parámetros Lilly
        'lilly': lilly,
        'A_usado': A,
        'usar_lilly': usar_lilly,
        
        # Calibración
        'Kx': Kx,
        'Kn': Kn,
        
        # Índice n
        'n_calculado': n_calc,
        'n_ajustado': round(n_adj, 3),
        'n_efectivo': round(n_eff, 3),
        
        # Carga
        'Q': round(Q, 1),
        'Lc_total': round(Lc_total, 2),
        
        # Doble taco
        'doble_taco': doble_taco,
        'eta_dist': round(eta_dist, 3) if doble_taco else 1.0,
        'Lc1': round(Lc1, 2) if doble_taco else Lc_total,
        'Lc2': round(Lc2, 2) if doble_taco else 0,
        
        # Fragmentación
        'X50_cm': X50_cm,
        'X50_pulg': round(X50_cm / 2.54, 2),
        **percentiles,
        
        # Zonas de daño
        'zonas': zonas,
        
        # Métricas
        'metros_por_ha': round(metros_por_ha, 0),
        'FC': round(FC, 3),
        'V0': round(V0, 1)
    }


def calcular_parametros_teoricos_enaex(ucs, densidad_explosivo=1.2, vod=4500, diametro_pulg=6.75):
    """
    Calcula todos los parámetros óptimos de malla según teoría ENAEX.
    
    Fórmulas ENAEX (Manual de Tronadura):
    =====================================
    
    1. BURDEN ÓPTIMO (Ash modificado):
       B = Kb × De × (ρe/ρr)^0.33 × (VOD/4000)^0.5
       Donde:
       - Kb = 25-40 (constante según tipo de roca)
       - De = diámetro de perforación (pulg)
       - ρe = densidad explosivo (g/cc)
       - ρr = densidad roca (g/cc) ≈ 2.65
       - VOD = velocidad detonación (m/s)
    
    2. ESPACIAMIENTO ÓPTIMO:
       S = Ks × B
       Donde Ks = 1.15 (roca dura) a 1.40 (roca blanda)
    
    3. TACO ÓPTIMO:
       T = 0.85 × B (rango 0.7 a 1.0 × B)
    
    4. TIMING POZOS (Konya):
       tp = Th × S (Th = 3.5-6.5 ms/m según roca)
    
    5. TIMING FILAS:
       tf = 11.5 × B (para fragmentación óptima)
    
    Parámetros:
    - ucs: Resistencia a compresión uniaxial (MPa)
    - densidad_explosivo: Densidad del explosivo (g/cc)
    - vod: Velocidad de detonación (m/s)
    - diametro_pulg: Diámetro de perforación (pulgadas)
    
    Retorna:
    - Diccionario con parámetros óptimos
    """
    rho_roca = 2.65  # Densidad típica de roca (g/cc)
    
    # Constantes según UCS
    Kb = calcular_Kb_enaex(ucs)
    Ks = calcular_Ks_enaex(ucs)
    Th = calcular_Th_enaex(ucs)
    fc_optimo = calcular_fc_optimo_enaex(ucs)
    
    # Burden óptimo (Ash modificado) - convertir a metros
    burden_optimo = (
        Kb * diametro_pulg * 
        ((densidad_explosivo / rho_roca) ** 0.33) * 
        ((vod / 4000) ** 0.5)
    ) / 39.37  # pulgadas a metros
    
    # Espaciamiento óptimo
    espaciamiento_optimo = Ks * burden_optimo
    
    # Área de malla
    area_malla = burden_optimo * espaciamiento_optimo
    
    # Taco óptimo
    taco_optimo = 0.85 * burden_optimo
    taco_minimo = 0.70 * burden_optimo
    
    # Timing óptimo
    timing_pozos_optimo = Th * espaciamiento_optimo
    timing_filas_optimo = 11.5 * burden_optimo  # ms/m para fragmentación óptima
    
    # Pasadura (subdrilling)
    pasadura_optima = 0.3 * burden_optimo
    
    # Explosivo recomendado
    explosivo_recomendado = recomendar_explosivo_enaex(ucs)
    
    return {
        'burden_optimo': round(burden_optimo, 2),
        'espaciamiento_optimo': round(espaciamiento_optimo, 2),
        'area_malla': round(area_malla, 2),
        'ratio_SB': round(Ks, 2),
        'taco_optimo': round(taco_optimo, 2),
        'taco_minimo': round(taco_minimo, 2),
        'timing_pozos_optimo': round(timing_pozos_optimo, 2),
        'timing_filas_optimo': round(timing_filas_optimo, 2),
        'fc_optimo': round(fc_optimo, 2),
        'pasadura_optima': round(pasadura_optima, 2),
        'explosivo_recomendado': explosivo_recomendado,
        'Kb': Kb,
        'Ks': Ks,
        'Th': Th
    }


def calcular_malla_minimos_metros(ucs, densidad_explosivo=1.2, vod=4500, diametro_pulg=6.75,
                                   p80_objetivo=8.0, p100_objetivo=15.0, 
                                   factor_expansion=1.15):
    """
    Calcula parámetros de malla optimizados para minimizar metros de perforación
    manteniendo control sobre fragmentación.
    
    Esta función expande la malla teórica ENAEX dentro de límites seguros,
    priorizando la reducción de metros perforados por consideraciones de
    rendimiento de perforadoras.
    
    Parámetros:
    - ucs: Resistencia a compresión uniaxial (MPa)
    - densidad_explosivo: Densidad del explosivo (g/cc)
    - vod: Velocidad de detonación (m/s)
    - diametro_pulg: Diámetro de perforación (pulgadas)
    - p80_objetivo: P80 objetivo máximo (pulgadas)
    - p100_objetivo: P100 objetivo máximo (pulgadas)
    - factor_expansion: Factor de expansión de malla (1.0-1.25)
    
    Retorna:
    - Diccionario con parámetros optimizados para mínimos metros
    """
    # Obtener parámetros teóricos base
    params_teoricos = calcular_parametros_teoricos_enaex(
        ucs, densidad_explosivo, vod, diametro_pulg
    )
    
    # Expandir malla según factor (con límites de seguridad)
    # Factor limitado según dureza de roca
    if pd.isna(ucs) or ucs >= 120:
        # Roca dura: menor expansión permitida
        factor_expansion = min(factor_expansion, 1.10)
    elif ucs >= 80:
        factor_expansion = min(factor_expansion, 1.15)
    else:
        # Roca blanda: puede expandir más
        factor_expansion = min(factor_expansion, 1.20)
    
    # Expandir burden y espaciamiento
    burden_expandido = params_teoricos['burden_optimo'] * factor_expansion
    espaciamiento_expandido = params_teoricos['espaciamiento_optimo'] * factor_expansion
    area_malla_expandida = burden_expandido * espaciamiento_expandido
    
    # Compensar con mayor FC para mantener energía
    fc_compensado = params_teoricos['fc_optimo'] * (factor_expansion ** 1.5)
    
    # Ajustar timing para compensar malla más grande
    timing_pozos_ajustado = params_teoricos['timing_pozos_optimo'] * 0.9  # Más rápido
    timing_filas_ajustado = params_teoricos['timing_filas_optimo'] * 0.95
    
    # Taco ajustado (mantener ratio con nuevo burden)
    taco_ajustado = 0.85 * burden_expandido
    
    # Calcular reducción de metros por m²
    reduccion_pozos = 1 - (params_teoricos['area_malla'] / area_malla_expandida)
    
    return {
        'burden_minmetros': round(burden_expandido, 2),
        'espaciamiento_minmetros': round(espaciamiento_expandido, 2),
        'area_malla_expandida': round(area_malla_expandida, 2),
        'ratio_SB': params_teoricos['ratio_SB'],
        'taco_ajustado': round(taco_ajustado, 2),
        'timing_pozos_ajustado': round(timing_pozos_ajustado, 2),
        'timing_filas_ajustado': round(timing_filas_ajustado, 2),
        'fc_compensado': round(fc_compensado, 2),
        'factor_expansion_usado': round(factor_expansion, 2),
        'reduccion_pozos_pct': round(reduccion_pozos * 100, 2),
        'explosivo_recomendado': params_teoricos['explosivo_recomendado'],
        'p80_objetivo': p80_objetivo,
        'p100_objetivo': p100_objetivo,
        'advertencias': []
    }


def calcular_malla_optimizada_multiobjetivo(ucs, densidad_explosivo=1.2, vod=4500, diametro_pulg=6.75,
                                             p80_objetivo=8.0, p100_objetivo=15.0,
                                             peso_metros=0.4, peso_p80=0.35, peso_p100=0.25,
                                             altura_banco=15.0, rqd=70, 
                                             Kx_input=None, Kn_input=None,
                                             usar_lilly=False, RWS=100):
    """
    ================================================================================
    FÓRMULA DE OPTIMIZACIÓN MULTI-OBJETIVO - MODELO PILIX v3.6
    ================================================================================
    
    INTEGRA:
    - Índice de Lilly para Factor A (opcional)
    - Kuz-Ram con calibración Kx/Kn (Mundaca)
    - Cunningham para índice n
    - Factor de timing (Konya)
    - Zonas de daño (Holmberg-Persson)
    
    MODELO DE FRAGMENTACIÓN KUZ-RAM CALIBRADO:
    ===========================================
    
    1. Metros perforados:
       Metros/ha = (10,000 / (B × S)) × H
    
    2. Modelo Kuz-Ram con Kx:
       X50 = Kx × A × (V0/Q)^0.8 × Q^(1/6) × (1.15/Er)^(19/30)   [cm]
       
       Donde:
       - A = Factor de roca (Lilly: 3-13, o calibrado: 19.9)
       - Kx = Factor de calibración (Mundaca)
       - Er = RWS/100
    
    3. Índice n (Cunningham):
       n = (2.2 - 14×B/D_mm) × √[(1+S/B)/2] × (1-W/B) × (Lc/H)
       n_adj = n × Kn
    
    4. Percentiles (Rosin-Rammler):
       Xc = X50 / (ln2)^(1/n)
       P80 = Xc × [-ln(0.20)]^(1/n)
       P100 = Xc × [-ln(0.01)]^(1/n)
    
    5. Función objetivo PILIX:
       min f = w1×(Metros/1500) + w2×(P80/4.5) + w3×(P100/12) + w4×(1-n/1.5)
    
    Parámetros:
    -----------
    - ucs: Resistencia a compresión uniaxial (MPa)
    - densidad_explosivo: Densidad del explosivo (g/cc)
    - vod: Velocidad de detonación (m/s)
    - diametro_pulg: Diámetro de perforación (pulgadas)
    - p80_objetivo: P80 objetivo máximo (pulgadas)
    - p100_objetivo: P100 objetivo máximo (pulgadas)
    - peso_metros, peso_p80, peso_p100: Pesos de optimización
    - altura_banco: Altura del banco (m)
    - rqd: Rock Quality Designation (%)
    - Kx_input, Kn_input: Factores de calibración (si None, se estiman)
    - usar_lilly: Si True, calcula A con Lilly; si False, usa A=19.9
    - RWS: Relative Weight Strength del explosivo (ANFO=100)
    
    Retorna:
    --------
    - Diccionario con parámetros optimizados y métricas
    """
    
    # Obtener parámetros teóricos base
    params_base = calcular_parametros_teoricos_enaex(ucs, densidad_explosivo, vod, diametro_pulg)
    
    B_base = params_base['burden_optimo']
    S_base = params_base['espaciamiento_optimo']
    FC_base = params_base['fc_optimo']
    tp_base = params_base['timing_pozos_optimo']
    tf_base = params_base['timing_filas_optimo']
    Th = params_base['Th']
    
    # =========================================================================
    # MODELO PILIX v3.6: Calcular Factor A y Kx/Kn
    # =========================================================================
    
    # Calcular Índice de Lilly
    lilly_params = calcular_indice_lilly(ucs, rqd)
    A_lilly = lilly_params['A']
    
    # Usar A de Lilly o calibrado
    if usar_lilly:
        A_CALIBRADO = A_lilly
    else:
        A_CALIBRADO = 19.9  # Valor calibrado Los Pelambres
    
    # Estimar o usar Kx/Kn proporcionados
    if Kx_input is None or Kn_input is None:
        Kx_est, Kn_est = estimar_Kx_Kn(ucs, rqd)
        Kx = Kx_input if Kx_input is not None else Kx_est
        Kn = Kn_input if Kn_input is not None else Kn_est
    else:
        Kx = Kx_input
        Kn = Kn_input
    
    # Altura de banco fija para cálculos (parámetro de entrada)
    H = altura_banco
    
    def calcular_factor_taco(B, taco):
        """
        Factor de taco según tipo de malla (calibrado Los Pelambres).
        
        Malla ABIERTA (B≥11m): f_taco = 1.0 - 0.30×(T-5.0)      → más taco = mejor
        Malla CERRADA (B<8m):  f_taco = 1.0 + 0.30×(T-5.25)²   → MÍNIMO en T=5.25m
        Malla INTERMEDIA:      interpolación lineal
        """
        if B >= 11.0:
            # Malla abierta: más taco = mejor fragmentación
            f_taco = 1.0 - 0.30 * (taco - 5.0)
        elif B < 8.0:
            # Malla cerrada: óptimo en T=5.25m
            f_taco = 1.0 + 0.30 * ((taco - 5.25) ** 2)
        else:
            # Malla intermedia: interpolación
            ratio = (B - 8.0) / (11.0 - 8.0)
            f_abierta = 1.0 - 0.30 * (taco - 5.0)
            f_cerrada = 1.0 + 0.30 * ((taco - 5.25) ** 2)
            f_taco = f_cerrada + ratio * (f_abierta - f_cerrada)
        
        return max(0.5, min(2.0, f_taco))  # Limitar rango
    
    def calcular_carga_pozo(taco):
        """
        Carga por pozo calibrada.
        Q = 68.7 × (H - T)    [kg]
        """
        columna = H - taco
        if columna <= 0:
            return 0.0
        return 68.7 * columna
    
    def calcular_X50_calibrado(B, S, taco, return_n=False):
        """
        Modelo Kuz-Ram PILIX v3.6 con calibración Kx/Kn.
        
        X50 = Kx × A × (V0/Q)^0.8 × Q^(1/6) × (1.15/Er)^(19/30)   [cm]
        n = (2.2 - 14×B/D_mm) × √[(1+S/B)/2] × (1-W/B) × (Lc/H) × Kn
        """
        Q = calcular_carga_pozo(taco)
        
        if Q <= 0:
            if return_n:
                return 30.0, 1.2
            return 30.0
        
        # Volumen por pozo (m³)
        V0 = B * S * H
        
        # RWS relativo (Er)
        Er = RWS / 100.0
        
        # Fórmula Kuz-Ram con Kx (PILIX v3.6)
        X50 = Kx * A_CALIBRADO * ((V0 / Q) ** 0.8) * (Q ** (1/6)) * ((1.15 / Er) ** (19/30))
        
        # Calcular n de Cunningham
        Lc = H - taco  # Longitud de carga
        d_mm = diametro_pulg * 25.4
        
        # Fórmula Cunningham
        term1 = 2.2 - 14 * (B / d_mm)
        term2 = np.sqrt((1 + S / B) / 2)
        term3 = 1 - 0.5 / B  # W = 0.5m típico
        term4 = Lc / H
        
        n_calc = term1 * term2 * term3 * term4 * Kn
        n_calc = max(0.5, min(2.5, n_calc))
        
        X50 = max(5.0, min(60.0, X50))  # Limitar a rango realista (cm)
        
        if return_n:
            return X50, n_calc
        return X50
    
    def calcular_timing_optimo(B, S):
        """
        Calcula timing óptimo según Konya.
        
        Timing pozos = Th × S  (Th según UCS)
        Timing filas = 11.5 × B
        
        Donde Th (ms/m):
        - UCS < 50: Th = 6.5 (roca blanda)
        - UCS 50-80: Th = 5.5
        - UCS 80-120: Th = 4.5
        - UCS > 120: Th = 3.5 (roca dura)
        """
        tp_opt = Th * S
        tf_opt = 11.5 * B
        return tp_opt, tf_opt
    
    def calcular_factor_timing(tp, tf, tp_opt, tf_opt):
        """
        Factor de corrección por timing.
        
        f_timing = 1.0 + 0.08×|tp/tp_opt - 1| + 0.12×|tf/tf_opt - 1|
        
        - Timing óptimo: f_timing = 1.0
        - Desviaciones penalizan la fragmentación
        """
        if tp_opt <= 0 or tf_opt <= 0:
            return 1.0
        
        # Penalización por desviación del timing óptimo
        pen_pozos = 0.08 * abs(tp / tp_opt - 1.0)
        pen_filas = 0.12 * abs(tf / tf_opt - 1.0)
        
        f_timing = 1.0 + pen_pozos + pen_filas
        
        return max(0.85, min(1.3, f_timing))  # Limitar rango
    
    def estimar_fragmentacion(B, S, taco, tp=None, tf=None):
        """
        Estima P80 y P100 usando modelo PILIX v3.6 con Rosin-Rammler.
        
        Modelo Rosin-Rammler:
        - Xc = X50 / (ln2)^(1/n)
        - P80 = Xc × [-ln(0.20)]^(1/n)
        - P100 = Xc × [-ln(0.01)]^(1/n)
        
        Donde:
        - n = índice de uniformidad (Cunningham con Kn)
        - f_taco = factor de taco según tipo de malla
        - f_timing = factor de corrección por timing
        """
        # Calcular X50 y n (cm)
        X50_cm, n = calcular_X50_calibrado(B, S, taco, return_n=True)
        
        # Factor de taco según tipo de malla
        f_taco = calcular_factor_taco(B, taco)
        
        # Timing óptimo
        tp_opt, tf_opt = calcular_timing_optimo(B, S)
        
        # Si no se proporcionan tiempos, usar óptimos
        if tp is None:
            tp = tp_opt
        if tf is None:
            tf = tf_opt
        
        # Factor de timing
        f_timing = calcular_factor_timing(tp, tf, tp_opt, tf_opt)
        
        # Xc de Rosin-Rammler
        Xc = X50_cm / (0.693 ** (1/n))
        
        # P80 y P100 con Rosin-Rammler (cm) + factores de corrección
        P80_cm = Xc * ((-np.log(0.20)) ** (1/n)) * f_taco * f_timing
        P100_cm = Xc * ((-np.log(0.01)) ** (1/n)) * f_taco * f_timing
        
        # Convertir a pulgadas
        P80 = P80_cm / 2.54
        P100 = P100_cm / 2.54
        
        # Carga por pozo
        Q = calcular_carga_pozo(taco)
        
        # Limitar a rangos realistas
        P80 = max(2.0, min(15.0, P80))
        P100 = max(5.0, min(35.0, P100))
        
        return X50_cm, P80, P100, f_taco, f_timing, Q, tp_opt, tf_opt, n
    
    def funcion_objetivo(B, S, taco, tp=None, tf=None):
        """
        Función objetivo PILIX v3.6:
        min f = w1×(Metros/1500) + w2×(P80/4.5) + w3×(P100/12) + w4×(1-n/1.5)
        
        Incluye:
        - Penalización por timing subóptimo
        - Penalización por baja uniformidad (n bajo)
        """
        area = B * S
        
        # Metros perforados por hectárea
        metros_por_ha = (10000 / area) * H
        
        # Fragmentación con modelo PILIX (incluye n de Cunningham)
        X50, P80_est, P100_est, f_taco, f_timing, Q, tp_opt, tf_opt, n = estimar_fragmentacion(B, S, taco, tp, tf)
        
        # Función objetivo normalizada (pesos PILIX v3.6)
        # w1=0.30, w2=0.20, w3=0.35, w4=0.15
        f_metros = metros_por_ha / 1500.0
        f_P80 = P80_est / 4.5  # Normalizar a 4.5" (límite operativo)
        f_P100 = P100_est / 12.0
        
        # Penalización por baja uniformidad (n ideal ≈ 1.5)
        f_uniformidad = max(0, 1 - n / 1.5)
        
        # Usar pesos calibrados
        J = 0.30 * f_metros + 0.20 * f_P80 + 0.35 * f_P100 + 0.15 * f_uniformidad
        
        return J, X50, P80_est, P100_est, f_taco, f_timing, Q, metros_por_ha, area, tp_opt, tf_opt, n
    
    def simular_taco_intermedio(B, S):
        """
        Simula diferentes valores de taco para encontrar el óptimo.
        Incluye timing óptimo y n de Cunningham en la simulación.
        Retorna tabla de simulación y taco óptimo.
        """
        resultados = []
        mejor_taco = 5.25  # Default para malla cerrada
        mejor_J = float('inf')
        
        for taco in np.arange(4.0, 7.0, 0.25):
            # Usar timing óptimo para la simulación
            J, X50, P80, P100, f_taco, f_timing, Q, metros, area, tp_opt, tf_opt, n = funcion_objetivo(B, S, taco)
            
            cumple = P80 <= p80_objetivo and P100 <= p100_objetivo
            
            resultados.append({
                'taco': round(taco, 2),
                'f_taco': round(f_taco, 2),
                'f_timing': round(f_timing, 2),
                'n': round(n, 2),
                'Q_kg': round(Q, 0),
                'X50_cm': round(X50, 2),
                'P80': round(P80, 2),
                'P100': round(P100, 2),
                'tp_opt': round(tp_opt, 1),
                'tf_opt': round(tf_opt, 1),
                'J': round(J, 4),
                'cumple': cumple
            })
            
            if cumple and J < mejor_J:
                mejor_J = J
                mejor_taco = taco
        
        return resultados, mejor_taco
    
    # Buscar configuración óptima
    mejor_J = float('inf')
    mejor_resultado = None
    
    # Limitar factor de expansión según dureza de roca
    if pd.isna(ucs) or ucs >= 120:
        max_factor = 1.10
    elif ucs >= 80:
        max_factor = 1.15
    else:
        max_factor = 1.20
    
    # Grid search sobre factor de expansión y taco
    for factor in np.arange(1.00, max_factor + 0.01, 0.01):
        B = B_base * factor
        S = S_base * factor
        
        # Simular tacos para esta malla
        sim_tacos, taco_opt = simular_taco_intermedio(B, S)
        
        # Calcular con taco óptimo (usar timing óptimo)
        J, X50, P80_est, P100_est, f_taco, f_timing, Q, metros, area, tp_opt, tf_opt, n = funcion_objetivo(B, S, taco_opt)
        
        # Verificar restricciones
        if P80_est <= p80_objetivo and P100_est <= p100_objetivo:
            if J < mejor_J:
                mejor_J = J
                
                # Determinar si necesita taco intermedio
                tipo_malla = "CERRADA" if B < 8.0 else ("ABIERTA" if B >= 11.0 else "INTERMEDIA")
                necesita_taco_int = (B < 8.0 and taco_opt >= 5.0 and taco_opt <= 5.5)
                
                mejor_resultado = {
                    'factor_optimo': round(factor, 2),
                    'burden_optimo': round(B, 2),
                    'espaciamiento_optimo': round(S, 2),
                    'area_malla': round(area, 2),
                    'ratio_SB': round(S/B, 2),
                    'taco_optimo': round(taco_opt, 2),
                    'taco_tradicional': round(0.85 * B, 2),
                    'tipo_malla': tipo_malla,
                    'necesita_taco_intermedio': necesita_taco_int,
                    'f_taco': round(f_taco, 2),
                    'f_timing': round(f_timing, 2),
                    'n_uniformidad': round(n, 2),
                    'carga_pozo_kg': round(Q, 2),
                    'timing_pozos': round(tp_opt, 2),
                    'timing_filas': round(tf_opt, 2),
                    'timing_pozos_formula': f"Th × S = {Th} × {round(S, 2)}",
                    'timing_filas_formula': f"11.5 × B = 11.5 × {round(B, 2)}",
                    'X50_estimado': round(X50, 2),
                    'P80_estimado': round(P80_est, 2),
                    'P100_estimado': round(P100_est, 2),
                    'metros_por_ha': round(metros, 2),
                    'reduccion_metros_pct': round((1 - (B_base * S_base) / area) * 100, 2),
                    'funcion_objetivo': round(J, 4),
                    'A_calibrado': A_CALIBRADO,
                    'Kx': Kx,
                    'Kn': Kn,
                    'lilly_params': lilly_params,
                    'Th_usado': Th,
                    'simulacion_tacos': sim_tacos,
                    'cumple_P80': P80_est <= p80_objetivo,
                    'cumple_P100': P100_est <= p100_objetivo,
                    'fc_ajustado': round(Q / (B * S * H), 2),
                    'explosivo_recomendado': params_base['explosivo_recomendado']
                }
    
    # Si no se encontró solución factible, usar la base
    if mejor_resultado is None:
        taco_default = 5.25 if B_base < 8.0 else 0.85 * B_base
        J, X50, P80_base, P100_base, f_taco, f_timing, Q, metros, area, tp_opt, tf_opt, n = funcion_objetivo(B_base, S_base, taco_default)
        
        mejor_resultado = {
            'factor_optimo': 1.00,
            'burden_optimo': B_base,
            'espaciamiento_optimo': S_base,
            'area_malla': round(B_base * S_base, 2),
            'ratio_SB': round(S_base/B_base, 2),
            'taco_optimo': round(taco_default, 2),
            'taco_tradicional': round(0.85 * B_base, 2),
            'tipo_malla': "CERRADA" if B_base < 8.0 else "ABIERTA",
            'necesita_taco_intermedio': B_base < 8.0,
            'f_taco': round(f_taco, 2),
            'f_timing': round(f_timing, 2),
            'n_uniformidad': round(n, 2),
            'carga_pozo_kg': round(Q, 2),
            'timing_pozos': round(tp_opt, 2),
            'timing_filas': round(tf_opt, 2),
            'timing_pozos_formula': f"Th × S = {Th} × {S_base}",
            'timing_filas_formula': f"11.5 × B = 11.5 × {B_base}",
            'X50_estimado': round(X50, 2),
            'P80_estimado': round(P80_base, 2),
            'P100_estimado': round(P100_base, 2),
            'metros_por_ha': round(metros, 2),
            'reduccion_metros_pct': 0.00,
            'funcion_objetivo': round(J, 4),
            'A_calibrado': A_CALIBRADO,
            'Kx': Kx,
            'Kn': Kn,
            'lilly_params': lilly_params,
            'Th_usado': Th,
            'simulacion_tacos': [],
            'cumple_P80': P80_base <= p80_objetivo,
            'cumple_P100': P100_base <= p100_objetivo,
            'fc_ajustado': round(Q / (B_base * S_base * H), 2),
            'explosivo_recomendado': params_base['explosivo_recomendado'],
            'advertencia': 'No se encontró solución que cumpla objetivos. Se usa malla teórica.'
        }
    
    return mejor_resultado

EXPLOSIVOS_CSV = {'Blendex_920': {'Densidad_g_cc': 0.9,
                 'Diam_Min_pulg': 4.0,
                 'Duracion_dias': 60,
                 'Energia_KJ_Kg': 3590,
                 'Metodo_Carga': 'Vaciable',
                 'Pot_Rel_ANFO_Peso': 0.95,
                 'Pot_Rel_ANFO_Vol': 1.1,
                 'Presion_Det_Kbar': 35.0,
                 'Proporcion_Matriz_ANFO': '80/20',
                 'Resistencia_Agua': 'Nula',
                 'VoD_Minima_m_s': 3670,
                 'VoD_Tipica_m_s': 3900,
                 'Vol_Gases_l_Kg': 1068},
 'Blendex_930': {'Densidad_g_cc': 1.0,
                 'Diam_Min_pulg': 4.0,
                 'Duracion_dias': 60,
                 'Energia_KJ_Kg': 3473,
                 'Metodo_Carga': 'Vaciable',
                 'Pot_Rel_ANFO_Peso': 0.93,
                 'Pot_Rel_ANFO_Vol': 1.19,
                 'Presion_Det_Kbar': 40.0,
                 'Proporcion_Matriz_ANFO': '70/30',
                 'Resistencia_Agua': 'Nula',
                 'VoD_Minima_m_s': 3760,
                 'VoD_Tipica_m_s': 3920,
                 'Vol_Gases_l_Kg': 1076},
 'Blendex_940': {'Densidad_g_cc': 1.2,
                 'Diam_Min_pulg': 5.0,
                 'Duracion_dias': 60,
                 'Energia_KJ_Kg': 3360,
                 'Metodo_Carga': 'Vaciable',
                 'Pot_Rel_ANFO_Peso': 0.91,
                 'Pot_Rel_ANFO_Vol': 1.39,
                 'Presion_Det_Kbar': 52.0,
                 'Proporcion_Matriz_ANFO': '60/40',
                 'Resistencia_Agua': 'Baja',
                 'VoD_Minima_m_s': 3760,
                 'VoD_Tipica_m_s': 3950,
                 'Vol_Gases_l_Kg': 1085},
 'Blendex_945': {'Densidad_g_cc': 1.32,
                 'Diam_Min_pulg': 5.0,
                 'Duracion_dias': 60,
                 'Energia_KJ_Kg': 3305,
                 'Metodo_Carga': 'Vaciable',
                 'Pot_Rel_ANFO_Peso': 0.89,
                 'Pot_Rel_ANFO_Vol': 1.49,
                 'Presion_Det_Kbar': 62.0,
                 'Proporcion_Matriz_ANFO': '55/45',
                 'Resistencia_Agua': 'Baja',
                 'VoD_Minima_m_s': 3800,
                 'VoD_Tipica_m_s': 4200,
                 'Vol_Gases_l_Kg': 1089},
 'Blendex_950': {'Densidad_g_cc': 1.32,
                 'Diam_Min_pulg': 5.0,
                 'Duracion_dias': 60,
                 'Energia_KJ_Kg': 3247,
                 'Metodo_Carga': 'Vaciable',
                 'Pot_Rel_ANFO_Peso': 0.88,
                 'Pot_Rel_ANFO_Vol': 1.47,
                 'Presion_Det_Kbar': 59.0,
                 'Proporcion_Matriz_ANFO': '50/50',
                 'Resistencia_Agua': 'Buena',
                 'VoD_Minima_m_s': 3695,
                 'VoD_Tipica_m_s': 4150,
                 'Vol_Gases_l_Kg': 1094},
 'Emultex_BN_1600': {'Densidad_g_cc': 1.32,
                     'Diam_Min_pulg': 5.5,
                     'Duracion_dias': 60,
                     'Energia_KJ_Kg': 3017,
                     'Metodo_Carga': 'Bombeable',
                     'Pot_Rel_ANFO_Peso': 0.84,
                     'Pot_Rel_ANFO_Vol': 1.41,
                     'Presion_Det_Kbar': 50.0,
                     'Proporcion_Matriz_ANFO': '70/30',
                     'Resistencia_Agua': 'Buena',
                     'VoD_Minima_m_s': 3200,
                     'VoD_Tipica_m_s': 4680,
                     'Vol_Gases_l_Kg': 1111},
 'Energex_50': {'Densidad_g_cc': 1.34,
                'Diam_Min_pulg': 5.0,
                'Duracion_dias': 45,
                'Energia_KJ_Kg': 3252,
                'Metodo_Carga': 'In_Situ',
                'Pot_Rel_ANFO_Peso': 0.88,
                'Pot_Rel_ANFO_Vol': 1.53,
                'Presion_Det_Kbar': 48.0,
                'Proporcion_Matriz_ANFO': None,
                'Resistencia_Agua': 'Baja',
                'VoD_Minima_m_s': 3200,
                'VoD_Tipica_m_s': 3800,
                'Vol_Gases_l_Kg': 1079},
 'Energex_50_Plus': {'Densidad_g_cc': 1.3,
                     'Diam_Min_pulg': 5.0,
                     'Duracion_dias': 45,
                     'Energia_KJ_Kg': 3361,
                     'Metodo_Carga': 'In_Situ',
                     'Pot_Rel_ANFO_Peso': 0.9,
                     'Pot_Rel_ANFO_Vol': 1.52,
                     'Presion_Det_Kbar': 78.0,
                     'Proporcion_Matriz_ANFO': None,
                     'Resistencia_Agua': 'Baja',
                     'VoD_Minima_m_s': 4300,
                     'VoD_Tipica_m_s': 4900,
                     'Vol_Gases_l_Kg': 1058},
 'Energex_70': {'Densidad_g_cc': 1.29,
                'Diam_Min_pulg': 5.0,
                'Duracion_dias': 45,
                'Energia_KJ_Kg': 3232,
                'Metodo_Carga': 'In_Situ',
                'Pot_Rel_ANFO_Peso': 0.88,
                'Pot_Rel_ANFO_Vol': 1.47,
                'Presion_Det_Kbar': 91.0,
                'Proporcion_Matriz_ANFO': None,
                'Resistencia_Agua': 'Buena',
                'VoD_Minima_m_s': 4700,
                'VoD_Tipica_m_s': 5300,
                'Vol_Gases_l_Kg': 1069},
 'Energex_70_Plus': {'Densidad_g_cc': 1.28,
                     'Diam_Min_pulg': 5.0,
                     'Duracion_dias': 45,
                     'Energia_KJ_Kg': 3299,
                     'Metodo_Carga': 'In_Situ',
                     'Pot_Rel_ANFO_Peso': 0.89,
                     'Pot_Rel_ANFO_Vol': 1.48,
                     'Presion_Det_Kbar': 108.0,
                     'Proporcion_Matriz_ANFO': None,
                     'Resistencia_Agua': 'Buena',
                     'VoD_Minima_m_s': 5200,
                     'VoD_Tipica_m_s': 5800,
                     'Vol_Gases_l_Kg': 1057},
 'Hidrex_LD500': {'Densidad_g_cc': 0.5,
                  'Diam_Min_pulg': 6.0,
                  'Duracion_dias': 60,
                  'Energia_KJ_Kg': 3027,
                  'Metodo_Carga': 'Granel',
                  'Pot_Rel_ANFO_Peso': 0.82,
                  'Pot_Rel_ANFO_Vol': 0.53,
                  'Presion_Det_Kbar': 7.81,
                  'Proporcion_Matriz_ANFO': None,
                  'Resistencia_Agua': 'Buena',
                  'VoD_Minima_m_s': 1900,
                  'VoD_Tipica_m_s': 2500,
                  'Vol_Gases_l_Kg': 1006},
 'Hidrex_LD600': {'Densidad_g_cc': 0.6,
                  'Diam_Min_pulg': 6.0,
                  'Duracion_dias': 60,
                  'Energia_KJ_Kg': 2974,
                  'Metodo_Carga': 'Granel',
                  'Pot_Rel_ANFO_Peso': 0.81,
                  'Pot_Rel_ANFO_Vol': 0.78,
                  'Presion_Det_Kbar': 12.0,
                  'Proporcion_Matriz_ANFO': None,
                  'Resistencia_Agua': 'Buena',
                  'VoD_Minima_m_s': 2500,
                  'VoD_Tipica_m_s': 2800,
                  'Vol_Gases_l_Kg': 1035},
 'Hidrex_LD700': {'Densidad_g_cc': 0.7,
                  'Diam_Min_pulg': 6.0,
                  'Duracion_dias': 60,
                  'Energia_KJ_Kg': 3027,
                  'Metodo_Carga': 'Granel',
                  'Pot_Rel_ANFO_Peso': 0.82,
                  'Pot_Rel_ANFO_Vol': 0.74,
                  'Presion_Det_Kbar': 14.72,
                  'Proporcion_Matriz_ANFO': None,
                  'Resistencia_Agua': 'Buena',
                  'VoD_Minima_m_s': 2700,
                  'VoD_Tipica_m_s': 2900,
                  'Vol_Gases_l_Kg': 1006},
 'Hidrex_LD800': {'Densidad_g_cc': 0.8,
                  'Diam_Min_pulg': 6.0,
                  'Duracion_dias': 60,
                  'Energia_KJ_Kg': 2974,
                  'Metodo_Carga': 'Granel',
                  'Pot_Rel_ANFO_Peso': 0.81,
                  'Pot_Rel_ANFO_Vol': 1.04,
                  'Presion_Det_Kbar': 22.0,
                  'Proporcion_Matriz_ANFO': None,
                  'Resistencia_Agua': 'Buena',
                  'VoD_Minima_m_s': 2900,
                  'VoD_Tipica_m_s': 3300,
                  'Vol_Gases_l_Kg': 1035},
 'Hidrex_LD900': {'Densidad_g_cc': 0.9,
                  'Diam_Min_pulg': 6.0,
                  'Duracion_dias': 60,
                  'Energia_KJ_Kg': 3027,
                  'Metodo_Carga': 'Granel',
                  'Pot_Rel_ANFO_Peso': 0.82,
                  'Pot_Rel_ANFO_Vol': 0.95,
                  'Presion_Det_Kbar': 26.01,
                  'Proporcion_Matriz_ANFO': None,
                  'Resistencia_Agua': 'Buena',
                  'VoD_Minima_m_s': 3100,
                  'VoD_Tipica_m_s': 3400,
                  'Vol_Gases_l_Kg': 1006},
 'Pirex_P-20': {'Densidad_g_cc': 0.88,
                'Diam_Min_pulg': 5.0,
                'Duracion_dias': 60,
                'Energia_KJ_Kg': 3599,
                'Metodo_Carga': 'Vaciable/Bombeable',
                'Pot_Rel_ANFO_Peso': 0.95,
                'Pot_Rel_ANFO_Vol': 1.14,
                'Presion_Det_Kbar': 31.0,
                'Proporcion_Matriz_ANFO': '20/80',
                'Resistencia_Agua': 'Nula',
                'VoD_Minima_m_s': 3000,
                'VoD_Tipica_m_s': 3756,
                'Vol_Gases_l_Kg': 1067},
 'Pirex_P-30': {'Densidad_g_cc': 1.0,
                'Diam_Min_pulg': 5.0,
                'Duracion_dias': 60,
                'Energia_KJ_Kg': 3486,
                'Metodo_Carga': 'Vaciable/Bombeable',
                'Pot_Rel_ANFO_Peso': 0.93,
                'Pot_Rel_ANFO_Vol': 1.3,
                'Presion_Det_Kbar': 39.0,
                'Proporcion_Matriz_ANFO': '30/70',
                'Resistencia_Agua': 'Nula',
                'VoD_Minima_m_s': 3000,
                'VoD_Tipica_m_s': 3949,
                'Vol_Gases_l_Kg': 1076},
 'Pirex_P-40': {'Densidad_g_cc': 1.2,
                'Diam_Min_pulg': 5.0,
                'Duracion_dias': 60,
                'Energia_KJ_Kg': 3377,
                'Metodo_Carga': 'Vaciable/Bombeable',
                'Pot_Rel_ANFO_Peso': 0.91,
                'Pot_Rel_ANFO_Vol': 1.56,
                'Presion_Det_Kbar': 48.0,
                'Proporcion_Matriz_ANFO': '40/60',
                'Resistencia_Agua': 'Nula',
                'VoD_Minima_m_s': 3000,
                'VoD_Tipica_m_s': 4018,
                'Vol_Gases_l_Kg': 1084},
 'Pirex_P-45': {'Densidad_g_cc': 1.3,
                'Diam_Min_pulg': 5.0,
                'Duracion_dias': 60,
                'Energia_KJ_Kg': 3318,
                'Metodo_Carga': 'Vaciable/Bombeable',
                'Pot_Rel_ANFO_Peso': 0.9,
                'Pot_Rel_ANFO_Vol': 1.69,
                'Presion_Det_Kbar': 52.0,
                'Proporcion_Matriz_ANFO': '45/55',
                'Resistencia_Agua': 'Nula',
                'VoD_Minima_m_s': 3000,
                'VoD_Tipica_m_s': 4006,
                'Vol_Gases_l_Kg': 1089},
 'Pirex_P-50': {'Densidad_g_cc': 1.3,
                'Diam_Min_pulg': 5.0,
                'Duracion_dias': 60,
                'Energia_KJ_Kg': 3264,
                'Metodo_Carga': 'Vaciable/Bombeable',
                'Pot_Rel_ANFO_Peso': 0.89,
                'Pot_Rel_ANFO_Vol': 1.69,
                'Presion_Det_Kbar': 55.0,
                'Proporcion_Matriz_ANFO': '50/50',
                'Resistencia_Agua': 'Nula',
                'VoD_Minima_m_s': 3000,
                'VoD_Tipica_m_s': 4101,
                'Vol_Gases_l_Kg': 1094},
 'Pirex_P-65': {'Densidad_g_cc': 1.32,
                'Diam_Min_pulg': 5.0,
                'Duracion_dias': 60,
                'Energia_KJ_Kg': 3028,
                'Metodo_Carga': 'Vaciable/Bombeable',
                'Pot_Rel_ANFO_Peso': 0.82,
                'Pot_Rel_ANFO_Vol': 1.4,
                'Presion_Det_Kbar': 33.8,
                'Proporcion_Matriz_ANFO': '65/35',
                'Resistencia_Agua': 'Buena',
                'VoD_Minima_m_s': 3000,
                'VoD_Tipica_m_s': 3200,
                'Vol_Gases_l_Kg': 1070},
 'Pirex_P-70': {'Densidad_g_cc': 1.32,
                'Diam_Min_pulg': 5.0,
                'Duracion_dias': 60,
                'Energia_KJ_Kg': 3207,
                'Metodo_Carga': 'Vaciable/Bombeable',
                'Pot_Rel_ANFO_Peso': 0.82,
                'Pot_Rel_ANFO_Vol': 1.71,
                'Presion_Det_Kbar': 33.0,
                'Proporcion_Matriz_ANFO': '70/30',
                'Resistencia_Agua': 'Buena',
                'VoD_Minima_m_s': 3000,
                'VoD_Tipica_m_s': 3161,
                'Vol_Gases_l_Kg': 951},
 'Pirex_S': {'Densidad_g_cc': 1.29,
             'Diam_Min_pulg': 5.0,
             'Duracion_dias': 60,
             'Energia_KJ_Kg': 2865,
             'Metodo_Carga': 'Vaciable/Bombeable',
             'Pot_Rel_ANFO_Peso': 0.81,
             'Pot_Rel_ANFO_Vol': 1.68,
             'Presion_Det_Kbar': 65.0,
             'Proporcion_Matriz_ANFO': None,
             'Resistencia_Agua': 'Buena',
             'VoD_Minima_m_s': 4000,
             'VoD_Tipica_m_s': 4500,
             'Vol_Gases_l_Kg': 1071},
 'Pirex_S_HP': {'Densidad_g_cc': 1.25,
                'Diam_Min_pulg': 5.0,
                'Duracion_dias': 60,
                'Energia_KJ_Kg': 2836,
                'Metodo_Carga': 'Vaciable/Bombeable',
                'Pot_Rel_ANFO_Peso': 0.81,
                'Pot_Rel_ANFO_Vol': 1.62,
                'Presion_Det_Kbar': 113.0,
                'Proporcion_Matriz_ANFO': None,
                'Resistencia_Agua': 'Buena',
                'VoD_Minima_m_s': 5000,
                'VoD_Tipica_m_s': 6000,
                'Vol_Gases_l_Kg': 1073},
 'Pirex_S_PLUS': {'Densidad_g_cc': 1.27,
                  'Diam_Min_pulg': 5.0,
                  'Duracion_dias': 60,
                  'Energia_KJ_Kg': 2864,
                  'Metodo_Carga': 'Vaciable/Bombeable',
                  'Pot_Rel_ANFO_Peso': 0.81,
                  'Pot_Rel_ANFO_Vol': 1.65,
                  'Presion_Det_Kbar': 93.0,
                  'Proporcion_Matriz_ANFO': None,
                  'Resistencia_Agua': 'Buena',
                  'VoD_Minima_m_s': 4800,
                  'VoD_Tipica_m_s': 5400,
                  'Vol_Gases_l_Kg': 1072},
 'Vertex_ALR_920': {'Densidad_g_cc': 0.86,
                    'Diam_Min_pulg': 5.5,
                    'Duracion_dias': 60,
                    'Energia_KJ_Kg': 3559,
                    'Metodo_Carga': 'Vaciable/Bombeable',
                    'Pot_Rel_ANFO_Peso': 0.95,
                    'Pot_Rel_ANFO_Vol': 1.12,
                    'Presion_Det_Kbar': 24.0,
                    'Proporcion_Matriz_ANFO': None,
                    'Resistencia_Agua': 'Mala',
                    'VoD_Minima_m_s': 3200,
                    'VoD_Tipica_m_s': 3333,
                    'Vol_Gases_l_Kg': 1072},
 'Vertex_ALR_930': {'Densidad_g_cc': 1.0,
                    'Diam_Min_pulg': 5.5,
                    'Duracion_dias': 60,
                    'Energia_KJ_Kg': 3423,
                    'Metodo_Carga': 'Vaciable/Bombeable',
                    'Pot_Rel_ANFO_Peso': 0.92,
                    'Pot_Rel_ANFO_Vol': 1.3,
                    'Presion_Det_Kbar': 29.0,
                    'Proporcion_Matriz_ANFO': None,
                    'Resistencia_Agua': 'Mala',
                    'VoD_Minima_m_s': 3200,
                    'VoD_Tipica_m_s': 3390,
                    'Vol_Gases_l_Kg': 1083},
 'Vertex_ALR_940': {'Densidad_g_cc': 1.23,
                    'Diam_Min_pulg': 5.5,
                    'Duracion_dias': 60,
                    'Energia_KJ_Kg': 3297,
                    'Metodo_Carga': 'Vaciable/Bombeable',
                    'Pot_Rel_ANFO_Peso': 0.89,
                    'Pot_Rel_ANFO_Vol': 1.6,
                    'Presion_Det_Kbar': 52.0,
                    'Proporcion_Matriz_ANFO': None,
                    'Resistencia_Agua': 'Mala',
                    'VoD_Minima_m_s': 3200,
                    'VoD_Tipica_m_s': 4103,
                    'Vol_Gases_l_Kg': 1094},
 'Vertex_ALR_945': {'Densidad_g_cc': 1.3,
                    'Diam_Min_pulg': 5.5,
                    'Duracion_dias': 60,
                    'Energia_KJ_Kg': 3226,
                    'Metodo_Carga': 'Vaciable/Bombeable',
                    'Pot_Rel_ANFO_Peso': 0.88,
                    'Pot_Rel_ANFO_Vol': 1.69,
                    'Presion_Det_Kbar': 38.0,
                    'Proporcion_Matriz_ANFO': None,
                    'Resistencia_Agua': 'Mala',
                    'VoD_Minima_m_s': 3200,
                    'VoD_Tipica_m_s': 3408,
                    'Vol_Gases_l_Kg': 1100},
 'Vertex_ALR_950': {'Densidad_g_cc': 1.32,
                    'Diam_Min_pulg': 5.5,
                    'Duracion_dias': 60,
                    'Energia_KJ_Kg': 3163,
                    'Metodo_Carga': 'Vaciable/Bombeable',
                    'Pot_Rel_ANFO_Peso': 0.87,
                    'Pot_Rel_ANFO_Vol': 1.71,
                    'Presion_Det_Kbar': 37.0,
                    'Proporcion_Matriz_ANFO': None,
                    'Resistencia_Agua': 'Mala',
                    'VoD_Minima_m_s': 3100,
                    'VoD_Tipica_m_s': 3363,
                    'Vol_Gases_l_Kg': 1105},
 'Vertex_ALR_970': {'Densidad_g_cc': 1.34,
                    'Diam_Min_pulg': 5.5,
                    'Duracion_dias': 60,
                    'Energia_KJ_Kg': 2920,
                    'Metodo_Carga': 'Vaciable/Bombeable',
                    'Pot_Rel_ANFO_Peso': 0.82,
                    'Pot_Rel_ANFO_Vol': 1.74,
                    'Presion_Det_Kbar': 36.0,
                    'Proporcion_Matriz_ANFO': None,
                    'Resistencia_Agua': 'Buena',
                    'VoD_Minima_m_s': 3000,
                    'VoD_Tipica_m_s': 3293,
                    'Vol_Gases_l_Kg': 1124},
 'Vertex_S': {'Densidad_g_cc': 1.31,
              'Diam_Min_pulg': 4.0,
              'Duracion_dias': 60,
              'Energia_KJ_Kg': 3032,
              'Metodo_Carga': 'In_Situ',
              'Pot_Rel_ANFO_Peso': 0.83,
              'Pot_Rel_ANFO_Vol': 1.42,
              'Presion_Det_Kbar': 45.0,
              'Proporcion_Matriz_ANFO': None,
              'Resistencia_Agua': 'Buena',
              'VoD_Minima_m_s': 3300,
              'VoD_Tipica_m_s': 4250,
              'Vol_Gases_l_Kg': 1075},
 'Vertex_S_Plus': {'Densidad_g_cc': 1.29,
                   'Diam_Min_pulg': 4.0,
                   'Duracion_dias': 60,
                   'Energia_KJ_Kg': 2994,
                   'Metodo_Carga': 'In_Situ',
                   'Pot_Rel_ANFO_Peso': 0.82,
                   'Pot_Rel_ANFO_Vol': 1.38,
                   'Presion_Det_Kbar': 87.0,
                   'Proporcion_Matriz_ANFO': None,
                   'Resistencia_Agua': 'Buena',
                   'VoD_Minima_m_s': 4700,
                   'VoD_Tipica_m_s': 5400,
                   'Vol_Gases_l_Kg': 1075}}

# Diccionario resumido para la ficha (mantiene el formato esperado por la UI)
EXPLOSIVOS_PROPIEDADES = {'Blendex_920': {'RWS': 95.0, 'Resistencia_Agua': 'Nula', 'VOD': 3900, 'densidad': 0.9, 'energia': 3.59},
 'Blendex_930': {'RWS': 93.0, 'Resistencia_Agua': 'Nula', 'VOD': 3920, 'densidad': 1.0, 'energia': 3.473},
 'Blendex_940': {'RWS': 91.0, 'Resistencia_Agua': 'Baja', 'VOD': 3950, 'densidad': 1.2, 'energia': 3.36},
 'Blendex_945': {'RWS': 89.0, 'Resistencia_Agua': 'Baja', 'VOD': 4200, 'densidad': 1.32, 'energia': 3.305},
 'Blendex_950': {'RWS': 88.0, 'Resistencia_Agua': 'Buena', 'VOD': 4150, 'densidad': 1.32, 'energia': 3.247},
 'Emultex_BN_1600': {'RWS': 84.0, 'Resistencia_Agua': 'Buena', 'VOD': 4680, 'densidad': 1.32, 'energia': 3.017},
 'Energex_50': {'RWS': 88.0, 'Resistencia_Agua': 'Baja', 'VOD': 3800, 'densidad': 1.34, 'energia': 3.252},
 'Energex_50_Plus': {'RWS': 90.0, 'Resistencia_Agua': 'Baja', 'VOD': 4900, 'densidad': 1.3, 'energia': 3.361},
 'Energex_70': {'RWS': 88.0, 'Resistencia_Agua': 'Buena', 'VOD': 5300, 'densidad': 1.29, 'energia': 3.232},
 'Energex_70_Plus': {'RWS': 89.0, 'Resistencia_Agua': 'Buena', 'VOD': 5800, 'densidad': 1.28, 'energia': 3.299},
 'Hidrex_LD500': {'RWS': 82.0, 'Resistencia_Agua': 'Buena', 'VOD': 2500, 'densidad': 0.5, 'energia': 3.027},
 'Hidrex_LD600': {'RWS': 81.0, 'Resistencia_Agua': 'Buena', 'VOD': 2800, 'densidad': 0.6, 'energia': 2.974},
 'Hidrex_LD700': {'RWS': 82.0, 'Resistencia_Agua': 'Buena', 'VOD': 2900, 'densidad': 0.7, 'energia': 3.027},
 'Hidrex_LD800': {'RWS': 81.0, 'Resistencia_Agua': 'Buena', 'VOD': 3300, 'densidad': 0.8, 'energia': 2.974},
 'Hidrex_LD900': {'RWS': 82.0, 'Resistencia_Agua': 'Buena', 'VOD': 3400, 'densidad': 0.9, 'energia': 3.027},
 'Pirex_P-20': {'RWS': 95.0, 'Resistencia_Agua': 'Nula', 'VOD': 3756, 'densidad': 0.88, 'energia': 3.599},
 'Pirex_P-30': {'RWS': 93.0, 'Resistencia_Agua': 'Nula', 'VOD': 3949, 'densidad': 1.0, 'energia': 3.486},
 'Pirex_P-40': {'RWS': 91.0, 'Resistencia_Agua': 'Nula', 'VOD': 4018, 'densidad': 1.2, 'energia': 3.377},
 'Pirex_P-45': {'RWS': 90.0, 'Resistencia_Agua': 'Nula', 'VOD': 4006, 'densidad': 1.3, 'energia': 3.318},
 'Pirex_P-50': {'RWS': 89.0, 'Resistencia_Agua': 'Nula', 'VOD': 4101, 'densidad': 1.3, 'energia': 3.264},
 'Pirex_P-65': {'RWS': 82.0, 'Resistencia_Agua': 'Buena', 'VOD': 3200, 'densidad': 1.32, 'energia': 3.028},
 'Pirex_P-70': {'RWS': 82.0, 'Resistencia_Agua': 'Buena', 'VOD': 3161, 'densidad': 1.32, 'energia': 3.207},
 'Pirex_S': {'RWS': 81.0, 'Resistencia_Agua': 'Buena', 'VOD': 4500, 'densidad': 1.29, 'energia': 2.865},
 'Pirex_S_HP': {'RWS': 81.0, 'Resistencia_Agua': 'Buena', 'VOD': 6000, 'densidad': 1.25, 'energia': 2.836},
 'Pirex_S_PLUS': {'RWS': 81.0, 'Resistencia_Agua': 'Buena', 'VOD': 5400, 'densidad': 1.27, 'energia': 2.864},
 'Vertex_ALR_920': {'RWS': 95.0, 'Resistencia_Agua': 'Mala', 'VOD': 3333, 'densidad': 0.86, 'energia': 3.559},
 'Vertex_ALR_930': {'RWS': 92.0, 'Resistencia_Agua': 'Mala', 'VOD': 3390, 'densidad': 1.0, 'energia': 3.423},
 'Vertex_ALR_940': {'RWS': 89.0, 'Resistencia_Agua': 'Mala', 'VOD': 4103, 'densidad': 1.23, 'energia': 3.297},
 'Vertex_ALR_945': {'RWS': 88.0, 'Resistencia_Agua': 'Mala', 'VOD': 3408, 'densidad': 1.3, 'energia': 3.226},
 'Vertex_ALR_950': {'RWS': 87.0, 'Resistencia_Agua': 'Mala', 'VOD': 3363, 'densidad': 1.32, 'energia': 3.163},
 'Vertex_ALR_970': {'RWS': 82.0, 'Resistencia_Agua': 'Buena', 'VOD': 3293, 'densidad': 1.34, 'energia': 2.92},
 'Vertex_S': {'RWS': 83.0, 'Resistencia_Agua': 'Buena', 'VOD': 4250, 'densidad': 1.31, 'energia': 3.032},
 'Vertex_S_Plus': {'RWS': 82.0, 'Resistencia_Agua': 'Buena', 'VOD': 5400, 'densidad': 1.29, 'energia': 2.994}}

# ============================================
# CONFIGURACIÓN GENERAL STREAMLIT
# ============================================
st.set_page_config(
    page_title="Heatmaps Tronadura",
    layout="wide"
)

st.title("Visualizador de Tronaduras & Explosivos")

# ============================================
# INICIALIZAR SESSION STATE PARA SIMULADOR PILIX
# ============================================
if 'pilix_params' not in st.session_state:
    st.session_state.pilix_params = {
        'UCS': 100,
        'RQD': 70,
        'B': 7.5,
        'S': 8.6,
        'T': 5.5,
        'H': 16.5,
        'J': 1.5,
        'd': 10.625,
        'n': 1.2,
        'Kx': 0.5,
        'Kn': 1.02,
        'explosivo': 'Energex 70',
        'doble_taco': False,
        'Ti': 2.0,
        'Ti_pos': 8.0,
        'nombre_config': 'Configuración Manual'
    }

if 'cargar_config' not in st.session_state:
    st.session_state.cargar_config = None

# ============================================
# TABS PRINCIPALES
# ============================================
tab_analisis, tab_simulador = st.tabs([
    "📊 Análisis y Recomendaciones", 
    "🔬 Simulador PILIX v3.6"
])

with tab_analisis:
    st.markdown(
        """
    Esta app permite:
    1. Ver las **mejores 3 configuraciones** observadas según una métrica (Top 3).
    2. Explorar el desempeño de **P100TRON, P80TRON, P50TRON, P20TRON** en distintos **heatmaps**.
    3. Consultar la **ficha técnica** y comparar propiedades de **explosivos**.
    4. **Simular configuraciones** en el Simulador PILIX v3.6 (pestaña separada).
    """
    )

# ============================================
# 1) CARGA DE DATOS
# ============================================

st.sidebar.header("1. Datos")

uploaded_file = st.sidebar.file_uploader(
    "Sube un archivo CSV (opcional). Si no subes nada, intentaré leer `0_Preparación.csv` del directorio actual.",
    type=["csv"]
)

df_raw = None

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
else:
    default_path = "0_Preparación.csv"
    if os.path.exists(default_path):
        df_raw = pd.read_csv(default_path)
        st.sidebar.success(f"Se cargó automáticamente `{default_path}`.")
    else:
        st.error(
            "No se detectó archivo subido ni pude encontrar `0_Preparación.csv`. "
            "Por favor sube un CSV en la barra lateral."
        )

if df_raw is None:
    st.stop()

# --------------------------------------------
# Normalización básica de algunas columnas
# --------------------------------------------
for col in ["M", "Fase_cat", "Banco", "LITO", "MINZON", "CAT", "ALTERACION", "Tipo_de_tronadura", "Tipo_Explosivo"]:
    if col in df_raw.columns:
        df_raw[col] = (
            df_raw[col]
            .astype(str)
            .str.strip()
            .str.upper()
            .replace({"NAN": np.nan})
        )

st.write(f"**Datos cargados:** {df_raw.shape[0]} filas × {df_raw.shape[1]} columnas")

# ============================================
# 2) COLUMNAS ESPERADAS / AJUSTES
# ============================================

col_m = "M" if "M" in df_raw.columns else None
col_ugt = "UGT_tron" if "UGT_tron" in df_raw.columns else None
col_fase = "Fase_cat" if "Fase_cat" in df_raw.columns else None
col_banco = "Banco" if "Banco" in df_raw.columns else None
col_lito = "LITO" if "LITO" in df_raw.columns else None
col_zonmin = "MINZON" if "MINZON" in df_raw.columns else None
col_cat = "CAT" if "CAT" in df_raw.columns else None
col_alt = "ALTERACION" if "ALTERACION" in df_raw.columns else None
col_ucs = "UCS_MPA" if "UCS_MPA" in df_raw.columns else None
col_fecha = "Fecha_de_tronadura" if "Fecha_de_tronadura" in df_raw.columns else None
col_tipo_tron = "Tipo_de_tronadura" if "Tipo_de_tronadura" in df_raw.columns else None
col_tipo_explo = "Tipo_Explosivo" if "Tipo_Explosivo" in df_raw.columns else None

# NUEVAS COLUMNAS (modas)
col_lito_moda = "LITO_moda" if "LITO_moda" in df_raw.columns else None
col_minzon_moda = "MINZON_moda" if "MINZON_moda" in df_raw.columns else None
col_ugt_moda = "UGT_moda" if "UGT_moda" in df_raw.columns else None

# COLUMNAS PARA EL GRAFICO EXPLOSIVOS
col_malla = "BxS" if "BxS" in df_raw.columns else None
col_diametro = "Diametro" if "Diametro" in df_raw.columns else None
col_fc = "FC" if "FC" in df_raw.columns else None
col_burden = "Burden" if "Burden" in df_raw.columns else None
col_espaciamiento = "Espaciamiento" if "Espaciamiento" in df_raw.columns else None

# Métricas granulométricas
metricas_posibles = [
    m for m in ["P100TRON", "P80TRON", "P50TRON", "P20TRON"]
    if m in df_raw.columns
]
if not metricas_posibles:
    st.error(
        "No encontré columnas de métricas `P100TRON`, `P80TRON`, `P50TRON`, `P20TRON` ni `P100` "
        "en el dataset. Revisa los nombres de columnas."
    )
    st.stop()

# Variables numéricas y categóricas
vars_numericas_base = [
    "fc1", "AreaMalla", "taco_gravilla", "taco_intermedio", "aire",
    "tpozos_ms", "tfilas_ms"
]
vars_categoricas_base = [
    "BxS", "TacoIntermedio", "Tipo_Explosivo", "Banco", "BxS_gravilla",
    "BxS_gravilla_intermedio"
]

vars_numericas = [v for v in vars_numericas_base if v in df_raw.columns]
vars_categoricas = [v for v in vars_categoricas_base if v in df_raw.columns]

# ============================================
# 3) FILTROS EN LA SIDEBAR
# ============================================

st.sidebar.header("2. Filtros")

df_base = df_raw.copy()
tipo_explo_sel = []  # inicialización para evitar NameError

# --- Filtro por Tipo_de_tronadura ---
if col_tipo_tron is not None:
    tipos_tron = sorted(df_base[col_tipo_tron].dropna().unique())
    tipo_tron_sel = st.sidebar.multiselect(
        "Tipo de tronadura",
        tipos_tron,
        default=tipos_tron
    )
    if tipo_tron_sel:
        df_base = df_base[df_base[col_tipo_tron].isin(tipo_tron_sel)]
    st.sidebar.write(f"Filas después de filtro Tipo_de_tronadura: {len(df_base)}")

# --- Filtro M ---
if col_m is not None:
    opciones_m = sorted(df_base[col_m].dropna().unique())
    m_sel = st.sidebar.multiselect(
        "M (por ejemplo M4, M3, etc.)",
        opciones_m,
        default=opciones_m
    )
    if m_sel:
        df_base = df_base[df_base[col_m].isin(m_sel)]
    st.sidebar.write(f"Filas después de filtro M: {len(df_base)}")

# --- Filtro por Fase ---
if col_fase is not None:
    fases = sorted(df_base[col_fase].dropna().unique())
    fase_sel = st.sidebar.multiselect("Fase", fases, default=fases)
    if fase_sel:
        df_base = df_base[df_base[col_fase].isin(fase_sel)]

# --- Filtro por Banco (RANGO NUMÉRICO) ---
if col_banco is not None:
    serie_banco_base = pd.to_numeric(df_base[col_banco], errors="coerce")
    if serie_banco_base.notna().any():
        banco_min = float(serie_banco_base.min())
        banco_max = float(serie_banco_base.max())
        banco_rango = st.sidebar.slider(
            "Rango Banco",
            min_value=float(np.floor(banco_min)),
            max_value=float(np.ceil(banco_max)),
            value=(float(np.floor(banco_min)), float(np.ceil(banco_max))),
            step=1.0
        )
        mascara_banco = serie_banco_base.between(banco_rango[0], banco_rango[1])
        df_base = df_base[mascara_banco]
        st.sidebar.write(f"Filas después de filtro Banco: {len(df_base)}")
    else:
        st.sidebar.info("No hay valores numéricos válidos en Banco para aplicar filtro de rango.")


# --- Filtro por UGT_tron (CATEGÓRICO) ---
if col_ugt_moda is not None:
    ugts = sorted(df_base[col_ugt_moda].dropna().unique())

    ugt_sel = st.sidebar.multiselect(
        "UGT_tron",
        ugts,
        default=ugts
    )

    if ugt_sel:
        df_base = df_base[df_base[col_ugt_moda].isin(ugt_sel)]

    st.sidebar.write(f"Filas después de filtro UGT_tron: {len(df_base)}")



# --- Filtro por LITO_moda ---
if col_lito_moda is not None:
    litos_moda = sorted(df_base[col_lito_moda].dropna().unique())
    lito_moda_sel = st.sidebar.multiselect(
        "LITO (moda)",
        litos_moda,
        default=litos_moda
    )
    if lito_moda_sel:
        df_base = df_base[df_base[col_lito_moda].isin(lito_moda_sel)]
    st.sidebar.write(f"Filas después de filtro LITO_moda: {len(df_base)}")

# --- Filtro por MINZON_moda ---
if col_minzon_moda is not None:
    minzon_moda_vals = sorted(df_base[col_minzon_moda].dropna().unique())
    minzon_moda_sel = st.sidebar.multiselect(
        "MINZON (moda)",
        minzon_moda_vals,
        default=minzon_moda_vals
    )
    if minzon_moda_sel:
        df_base = df_base[df_base[col_minzon_moda].isin(minzon_moda_sel)]
    st.sidebar.write(f"Filas después de filtro MINZON_moda: {len(df_base)}")

# --- Filtro por Tipo de explosivo ---
if col_tipo_explo is not None:

    # ⬇️ achicar tamaño de letra del multiselect
    st.sidebar.markdown("""
        <style>
        div[data-baseweb="select"] span {
            font-size: 12px !important;
        }
        div[data-baseweb="select"] div {
            font-size: 12px !important;
        }
        div[data-baseweb="popover"] div {
            font-size: 12px !important;
        }
        </style>
    """, unsafe_allow_html=True)

    tipos_explo = sorted(df_base[col_tipo_explo].dropna().unique())
    tipo_explo_sel = st.sidebar.multiselect(
        "Tipo de explosivo",
        tipos_explo,
        default=tipos_explo
    )

if col_tipo_explo is not None and tipo_explo_sel:
    df_base = df_base[df_base[col_tipo_explo].isin(tipo_explo_sel)]
    st.sidebar.write(f"Filas después de filtro Tipo_Explosivo: {len(df_base)}")


# Rango de UCS_MPA
if col_ucs is not None:
    ucs_min = float(df_base[col_ucs].min())
    ucs_max = float(df_base[col_ucs].max())
    ucs_rango = st.sidebar.slider(
        "Rango UCS_MPA",
        min_value=float(np.floor(ucs_min)),
        max_value=float(np.ceil(ucs_max)),
        value=(float(np.floor(ucs_min)), float(np.ceil(ucs_max))),
        step=1.0
    )
    df_base = df_base[df_base[col_ucs].between(ucs_rango[0], ucs_rango[1])]

# --- Filtro por rango de FC ---
if col_fc is not None:
    serie_fc = pd.to_numeric(df_base["FC"], errors="coerce")
    if serie_fc.notna().any():
        fc_min = float(serie_fc.min())
        fc_max = float(serie_fc.max())
        fc_rango = st.sidebar.slider(
            "Rango FC",
            min_value=float(np.floor(fc_min)),
            max_value=float(np.ceil(fc_max)),
            value=(float(np.floor(fc_min)), float(np.ceil(fc_max))),
            step=1.0
        )
        mascara_fc = serie_fc.between(fc_rango[0], fc_rango[1])
        df_base = df_base[mascara_fc]
        st.sidebar.write(f"Filas después de filtro FC: {len(df_base)}")
    else:
        st.sidebar.info("No hay valores numéricos válidos en FC para aplicar filtro de rango.")

st.write(f"**Filas después de aplicar filtros:** {df_base.shape[0]}")

if df_base.empty:
    st.warning("No hay filas después de aplicar los filtros. Ajusta los filtros en la barra lateral.")
    st.stop()

# ============================================
# 3.1) CONTROLES DE ALTURA DEL GRÁFICO (para heatmaps)
# ============================================

st.sidebar.header("3. Opciones de gráfico (Heatmaps)")

altura_min = st.sidebar.slider(
    "Altura mínima del heatmap (pulgadas)",
    min_value=1.0,
    max_value=20.0,
    value=3.0,
    step=0.5
)

altura_por_fila = st.sidebar.slider(
    "Altura por categoría (pulgadas)",
    min_value=0.1,
    max_value=2.0,
    value=0.3,
    step=0.1
)

# ============================================
# 4) ASEGURAR NUMÉRICAS Y ORDEN DE BxS_cat
# ============================================

cols_numericas_extra = ["Burden", "Espaciamiento"]
if col_banco is not None:
    cols_numericas_extra.append(col_banco)

cols_a_numerico = list(set(vars_numericas + metricas_posibles + [col_ucs] + cols_numericas_extra))
cols_a_numerico = [c for c in cols_a_numerico if c is not None and c in df_base.columns]

df_base[cols_a_numerico] = df_base[cols_a_numerico].apply(
    lambda c: pd.to_numeric(c, errors="coerce")
)

orden_bxs_global = None
if "BxS_cat" in df_base.columns and "Burden" in df_base.columns and "Espaciamiento" in df_base.columns:
    df_base["BxS_valor"] = df_base["Burden"] * df_base["Espaciamiento"]
    orden_bxs_global = (
        df_base.dropna(subset=["BxS_cat", "BxS_valor"])
        .groupby("BxS_cat")["BxS_valor"]
        .mean()
        .sort_values()
        .index
        .tolist()
    )

# ============================================
# 5) BINS PARA UCS Y OTRAS VARIABLES
# ============================================

if col_ucs is None:
    st.error("No existe la columna UCS_MPA en el dataset. No se pueden generar los heatmaps UCS vs variable.")
    st.stop()

bins_ucs = [0, 60, 80, 100, 120, 140, 1000]
labels_ucs = ["0-60", "60-80", "80-100", "100-120", "120-140", "140+"]

bins_FC = [180, 300, 400, 500, 600, np.inf]
labels_FC = ["180-300", "300-400", "400-500", "500-600", "600+"]

bins_tpozos = [0, 2, 4, 6, 10, 20, 100, np.inf]
labels_tpozos = ["0-2", "2-4", "4-6", "6-10", "10-20", "20-100", "100+"]

bins_tfilas = [0, 50, 70, 90, 110, 150, np.inf]
labels_tfilas = ["0-50", "50-70", "70-90", "90-110", "110-150", "150+"]

# ============================================
# 6) CONFIG DE COLOR POR MÉTRICA (FIJA 10–35)
# ============================================

metric_config = {
    metrica: {
        "cmap": "RdYlGn_r",  # verde bajo, rojo alto
        "vmin": 10.0,
        "vmax": 35.0,
        "center": (10.0 + 35.0) / 2.0
    }
    for metrica in metricas_posibles
}

# ============================================
# 7) FUNCIÓN GENERAL DE HEATMAP (UNA VARIABLE)
# ============================================

def generar_heatmap_una_variable(
    df_subset: pd.DataFrame,
    nombre_subset: str,
    var_y: str,
    metrica: str,
    altura_min: float,
    altura_por_fila: float
):
    if metrica not in df_subset.columns:
        st.error(f"La métrica '{metrica}' no existe en el DataFrame.")
        return

    if var_y not in df_subset.columns:
        st.error(f"La variable '{var_y}' no existe en el DataFrame.")
        return

    if "UCS_bin" not in df_subset.columns:
        st.error("No existe la columna 'UCS_bin' en el DataFrame.")
        return

    color_cfg = metric_config.get(metrica, None)
    if color_cfg is None:
        cmap = "RdYlGn_r"
        vmin = 10.0
        vmax = 35.0
        center = (10.0 + 35.0) / 2.0
    else:
        cmap = color_cfg["cmap"]
        vmin = color_cfg["vmin"]
        vmax = color_cfg["vmax"]
        center = color_cfg["center"]

    df_tmp = df_subset[["UCS_bin", var_y, metrica]].dropna()
    if df_tmp.empty:
        st.warning(f"[{nombre_subset}] No hay datos disponibles para {var_y} y {metrica} después de filtros.")
        return

    # Detectar si la variable es numérica o categórica
    es_numerica = var_y in vars_numericas

    if es_numerica:
        if var_y == "FC":
            df_tmp["var_bin"] = pd.cut(
                df_tmp[var_y],
                bins=bins_FC,
                labels=labels_FC,
                include_lowest=True,
                right=False
            )
        elif var_y == "tpozos_ms":
            df_tmp["var_bin"] = pd.cut(
                df_tmp[var_y],
                bins=bins_tpozos,
                labels=labels_tpozos,
                include_lowest=True
            )
        elif var_y == "tfilas_ms":
            df_tmp["var_bin"] = pd.cut(
                df_tmp[var_y],
                bins=bins_tfilas,
                labels=labels_tfilas,
                include_lowest=True
            )
        elif df_tmp[var_y].nunique() < 3:
            df_tmp["var_bin"] = df_tmp[var_y].astype(str)
        else:
            df_tmp["var_bin"] = pd.qcut(df_tmp[var_y], q=5, duplicates="drop")
    else:
        if var_y == "BxS_cat" and orden_bxs_global is not None:
            df_tmp["var_bin"] = pd.Categorical(
                df_tmp[var_y],
                categories=orden_bxs_global,
                ordered=True
            )
        else:
            df_tmp["var_bin"] = df_tmp[var_y].astype(str)

    agg_df = (
        df_tmp.groupby(["var_bin", "UCS_bin"])[metrica]
              .agg(["mean", "count"])
              .reset_index()
    )
    if agg_df.empty:
        st.warning(f"[{nombre_subset}] No fue posible agrupar datos para {var_y}.")
        return

    heat_pivot_mean = agg_df.pivot(index="var_bin", columns="UCS_bin", values="mean")
    heat_pivot_n    = agg_df.pivot(index="var_bin", columns="UCS_bin", values="count")

    if (not es_numerica) and (var_y == "BxS_cat") and (orden_bxs_global is not None):
        ordered_index = [c for c in orden_bxs_global if c in heat_pivot_mean.index]
        if ordered_index:
            heat_pivot_mean = heat_pivot_mean.reindex(index=ordered_index)
            heat_pivot_n    = heat_pivot_n.reindex(index=ordered_index)

    new_index = []
    for idx in heat_pivot_mean.index:
        if isinstance(idx, pd.Interval):
            left = int(idx.left)
            right = int(idx.right)
            new_index.append(f"{left}–{right}")
        else:
            new_index.append(str(idx))
    heat_pivot_mean.index = new_index
    heat_pivot_n.index = new_index

    annot = heat_pivot_mean.copy().astype(str)
    for r in heat_pivot_mean.index:
        for c in heat_pivot_mean.columns:
            val = heat_pivot_mean.loc[r, c]
            n   = heat_pivot_n.loc[r, c]
            if pd.isna(val):
                annot.loc[r, c] = ""
            else:
                annot.loc[r, c] = f"{val:.2f}\n(n={int(n)})"

    n_rows = max(1, len(heat_pivot_mean.index))
    n_cols = max(1, len(heat_pivot_mean.columns))

    fig_height = max(altura_min, altura_por_fila * n_rows)
    fig_width = max(8.0, 1.2 * n_cols)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.heatmap(
        heat_pivot_mean,
        annot=annot,
        fmt="",
        cmap=cmap,
        center=center,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"shrink": 0.8},
        ax=ax,
        annot_kws={"size": 9}  # Tamaño de texto en anotaciones
    )

    ax.set_title(f"{metrica} medio — UCS vs {var_y} — {nombre_subset}", fontsize=12, pad=20)
    ax.set_xlabel("UCS (bins)", fontsize=11)
    ax.set_ylabel(var_y, fontsize=11)

    ax.tick_params(axis="x", labelrotation=0, labelsize=10)
    ax.tick_params(axis="y", labelrotation=0, labelsize=10)

    fig.subplots_adjust(left=0.25, right=0.95, top=0.9, bottom=0.15)

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ============================================
# 8) PREPARAR UCS_bin EN DF FILTRADO
# ============================================

df_filtrado = df_base.copy()
df_filtrado["UCS_bin"] = pd.cut(
    df_filtrado[col_ucs],
    bins=bins_ucs,
    labels=labels_ucs
)

# ============================================
# SECCIÓN 1: TOP 3 CONFIGURACIONES
# ============================================

st.markdown("---")
st.header("Sección 1: Top 3 configuraciones que minimizan una métrica")

metricas_optimizacion = [m for m in ["P100TRON", "P80TRON", "P50TRON", "P20TRON"] if m in metricas_posibles]
if not metricas_optimizacion:
    metricas_optimizacion = metricas_posibles

opt_metrica_sel = st.selectbox(
    "Métrica a optimizar (para buscar las mejores configuraciones):",
    metricas_optimizacion,
    help="Por ejemplo, selecciona P100TRON o P80TRON."
)

serie_m = pd.to_numeric(df_filtrado[opt_metrica_sel], errors="coerce")

if serie_m.notna().sum() == 0:
    st.info(f"No hay valores numéricos válidos para la métrica **{opt_metrica_sel}** en el subconjunto filtrado.")
else:
    m_min = float(serie_m.min())
    m_max = float(serie_m.max())

    rango_sel = st.slider(
        f"Rango de búsqueda para {opt_metrica_sel}",
        min_value=float(np.floor(m_min)),
        max_value=float(np.ceil(m_max)),
        value=(float(np.floor(m_min)), float(np.ceil(m_max))),
        step=1.0
    )

    # ---- Rango de UCS local para la búsqueda ----
    serie_ucs = pd.to_numeric(df_filtrado[col_ucs], errors="coerce")
    serie_ucs_valida = serie_ucs[serie_ucs.notna()]

    if len(serie_ucs_valida) > 0:
        ucs_min_local = float(serie_ucs_valida.min())
        ucs_max_local = float(serie_ucs_valida.max())

        ucs_rango_local = st.slider(
            "Rango de UCS_MPA para la búsqueda",
            min_value=float(np.floor(ucs_min_local)),
            max_value=float(np.ceil(ucs_max_local)),
            value=(float(np.floor(ucs_min_local)), float(np.ceil(ucs_max_local))),
            step=1.0
        )

        mascara_ucs_local = serie_ucs.between(ucs_rango_local[0], ucs_rango_local[1])
    else:
        ucs_rango_local = None
        mascara_ucs_local = pd.Series(False, index=df_filtrado.index)

    # ---- Rango de Banco local para la búsqueda ----
    if col_banco is not None and col_banco in df_filtrado.columns:
        serie_banco_local = pd.to_numeric(df_filtrado[col_banco], errors="coerce")
        serie_banco_valida = serie_banco_local[serie_banco_local.notna()]

        if len(serie_banco_valida) > 0:
            banco_min_local = float(serie_banco_valida.min())
            banco_max_local = float(serie_banco_valida.max())

            banco_rango_local = st.slider(
                "Rango de Banco para la búsqueda",
                min_value=float(np.floor(banco_min_local)),
                max_value=float(np.ceil(banco_max_local)),
                value=(float(np.floor(banco_min_local)), float(np.ceil(banco_max_local))),
                step=1.0
            )

            mascara_banco_local = serie_banco_local.between(banco_rango_local[0], banco_rango_local[1])
        else:
            banco_rango_local = None
            mascara_banco_local = pd.Series(False, index=df_filtrado.index)
    else:
        banco_rango_local = None
        mascara_banco_local = pd.Series(True, index=df_filtrado.index)

    mascara_rango_metrica = serie_m.between(rango_sel[0], rango_sel[1])
    df_rango = df_filtrado[mascara_rango_metrica & mascara_ucs_local & mascara_banco_local]

    if df_rango.empty:
        msg = (
            f"No se encontraron tronaduras con **{opt_metrica_sel}** dentro del rango "
            f"({rango_sel[0]:.2f} – {rango_sel[1]:.2f})"
        )
        if ucs_rango_local is not None:
            msg += (
                f" y con **UCS_MPA** en el rango "
                f"({ucs_rango_local[0]:.2f} – {ucs_rango_local[1]:.2f})"
            )
        if banco_rango_local is not None:
            msg += (
                f" y con **Banco** en el rango "
                f"({banco_rango_local[0]:.2f} – {banco_rango_local[1]:.2f})"
            )
        msg += "."
        st.warning(msg)
    else:
        df_top = (
            df_rango
            .sort_values(by=opt_metrica_sel, ascending=True)
            .head(3)
        )

        cols_conf = [
            col_fecha,
            col_fase,
            col_banco,
            "UGT_tron",
            col_tipo_tron,
            col_tipo_explo,
            col_ucs,
            "UCS_bin",
            col_lito_moda,
            col_minzon_moda,
        ] + metricas_posibles + [
            "tpozos_ms",
            "tfilas_ms",
            "BxS",
            "taco_gravilla",
            "taco_intermedio",
            "FC",
            "fc1",
            "fc2",
            col_lito,
            col_zonmin,
            col_alt,
            col_cat
        ]

        cols_conf = [c for c in cols_conf if c is not None and c in df_top.columns]

        cols_conf_unicas = []
        vistos = set()
        for c in cols_conf:
            if c not in vistos:
                cols_conf_unicas.append(c)
                vistos.add(c)

        texto_rango = (
            f"dentro del rango de **{opt_metrica_sel}**: {rango_sel[0]:.2f} – {rango_sel[1]:.2f}"
        )
        if ucs_rango_local is not None:
            texto_rango += (
                f", con **UCS_MPA** en el rango: {ucs_rango_local[0]:.2f} – {ucs_rango_local[1]:.2f}"
            )
        if banco_rango_local is not None:
            texto_rango += (
                f", y **Banco** en el rango: {banco_rango_local[0]:.2f} – {banco_rango_local[1]:.2f}"
            )

        st.write(
            f"Se muestran las **{len(df_top)} configuraciones observadas** con menor **{opt_metrica_sel}** "
            f"{texto_rango}."
        )

        st.dataframe(df_top[cols_conf_unicas], use_container_width=True)

# ============================================
# SECCIÓN 1.1: RECOMENDACIONES DE MALLA (SIMPLIFICADA)
# ============================================

# Estilo CSS para cambiar colores a celeste claro
st.markdown("""
<style>
    /* Slider - track y thumb en celeste claro */
    .stSlider > div > div > div > div {
        background-color: #e1f5fe !important;
    }
    .stSlider > div > div > div > div > div {
        background-color: #4fc3f7 !important;
    }
    /* Slider - valor seleccionado */
    .stSlider [data-baseweb="slider"] div[role="slider"] {
        background-color: #29b6f6 !important;
        border-color: #29b6f6 !important;
    }
    /* Botones en celeste claro */
    .stButton > button {
        background-color: #e1f5fe !important;
        color: #0277bd !important;
        border: 2px solid #81d4fa !important;
        border-radius: 8px !important;
    }
    .stButton > button:hover {
        background-color: #b3e5fc !important;
        border: 2px solid #4fc3f7 !important;
    }
    /* Selectbox y multiselect */
    .stSelectbox > div > div {
        border-color: #81d4fa !important;
    }
    .stMultiSelect > div > div {
        border-color: #81d4fa !important;
    }
    /* Number input */
    .stNumberInput > div > div > input {
        border-color: #81d4fa !important;
    }
    /* Expander headers en celeste */
    .streamlit-expanderHeader {
        background-color: #e1f5fe !important;
        border-radius: 8px !important;
    }
    /* Range slider */
    div[data-testid="stSlider"] > div > div > div {
        background-color: #e1f5fe !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("---")
st.header("Sección 1.1: Calculadora de Malla Óptima")

# ============================================
# PANEL DE PARÁMETROS DE ENTRADA
# ============================================
st.markdown("### 📋 Parámetros de Entrada")

# Fila 1: UCS y Tipo de Roca
col_ucs1, col_ucs2, col_ucs3 = st.columns([2, 1, 1])

with col_ucs1:
    ucs_input = st.slider(
        "🎯 UCS objetivo (MPa)",
        min_value=30,
        max_value=200,
        value=100,
        step=5,
        help="Resistencia a compresión uniaxial de la roca"
    )

with col_ucs2:
    # Clasificación automática de roca (semáforo)
    if ucs_input < 50:
        tipo_roca = "🟢 Blanda"
        color_roca = "green"
    elif ucs_input < 100:
        tipo_roca = "🟡 Media"
        color_roca = "orange"
    elif ucs_input < 150:
        tipo_roca = "🟠 Dura"
        color_roca = "darkorange"
    else:
        tipo_roca = "🔴 Muy Dura"
        color_roca = "red"
    
    st.metric("Tipo de Roca", tipo_roca)

with col_ucs3:
    altura_banco_est = st.number_input("Altura banco (m)", 10.0, 20.0, 15.0, 1.0)

# Fila 2: Perforación y Explosivo
st.markdown("**⚙️ Perforación y Explosivo:**")
col_perf1, col_perf2, col_perf3 = st.columns(3)

with col_perf1:
    diametro_input = st.selectbox(
        "Diámetro perforación (pulg)",
        options=[5.0, 5.5, 6.0, 6.5, 6.75, 7.0, 7.875, 9.0, 10.625, 12.25],
        index=4
    )

with col_perf2:
    explosivo_input = st.selectbox(
        "Tipo de explosivo",
        options=list(EXPLOSIVOS_PROPIEDADES.keys()),
        index=0
    )

with col_perf3:
    # Mostrar propiedades del explosivo seleccionado
    prop_explosivo = EXPLOSIVOS_PROPIEDADES.get(explosivo_input, {})
    densidad_exp = prop_explosivo.get('densidad', 1.2)
    vod_exp = prop_explosivo.get('VOD', 4500)
    st.markdown(f"**Propiedades:**")
    st.caption(f"Densidad: {densidad_exp} g/cc | VOD: {vod_exp} m/s")

# Fila 3: Objetivos de fragmentación
st.markdown("**🎯 Objetivos de Fragmentación:**")
col_obj1, col_obj2 = st.columns(2)

with col_obj1:
    p80_objetivo = st.number_input("P80 objetivo máx (pulg)", 3.0, 15.0, 8.0, 0.5)

with col_obj2:
    p100_objetivo = st.number_input("P100 objetivo máx (pulg)", 8.0, 30.0, 15.0, 1.0)

# Fila 4: Prioridades de optimización
st.markdown("**⚖️ Prioridades de Optimización:**")
col_p1, col_p2, col_p3 = st.columns(3)
with col_p1:
    peso_metros = st.slider("Reducir metros", 0.0, 1.0, 0.40, 0.05, help="Mayor peso = prioriza menos metros perforados")
with col_p2:
    peso_p80 = st.slider("Reducir P80", 0.0, 1.0, 0.35, 0.05, help="Mayor peso = prioriza fragmentación fina")
with col_p3:
    peso_p100 = st.slider("Reducir P100", 0.0, 1.0, 0.25, 0.05, help="Mayor peso = prioriza menos sobretamaño")

# ============================================
# BOTÓN DE CÁLCULO CON SESSION STATE
# ============================================
st.markdown("---")

# Inicializar session state para los resultados
if 'resultados_calculados' not in st.session_state:
    st.session_state.resultados_calculados = False
    st.session_state.params_teoricos = None
    st.session_state.params_multiobj = None
    st.session_state.params_minmetros = None

col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    calcular_btn = st.button("🔄 CALCULAR MALLA ÓPTIMA", use_container_width=True, type="primary")

# Ejecutar cálculos cuando se presiona el botón
if calcular_btn:
    st.session_state.resultados_calculados = True
    st.session_state.ultimo_ucs = ucs_input
    st.session_state.ultimo_diametro = diametro_input
    st.session_state.ultimo_explosivo = explosivo_input

# Normalizar pesos
suma_pesos = peso_metros + peso_p80 + peso_p100
peso_metros_norm = peso_metros / suma_pesos if suma_pesos > 0 else 1/3
peso_p80_norm = peso_p80 / suma_pesos if suma_pesos > 0 else 1/3
peso_p100_norm = peso_p100 / suma_pesos if suma_pesos > 0 else 1/3

# ===== CÁLCULOS (siempre ejecutar para tener datos) =====
params_teoricos = calcular_parametros_teoricos_enaex(
    ucs=ucs_input,
    densidad_explosivo=densidad_exp,
    vod=vod_exp,
    diametro_pulg=diametro_input
)

# Factor de expansión automático según UCS
if ucs_input >= 120:
    factor_expansion_auto = 1.10
elif ucs_input >= 80:
    factor_expansion_auto = 1.15
else:
    factor_expansion_auto = 1.20

params_minmetros = calcular_malla_minimos_metros(
    ucs=ucs_input,
    densidad_explosivo=densidad_exp,
    vod=vod_exp,
    diametro_pulg=diametro_input,
    p80_objetivo=p80_objetivo,
    p100_objetivo=p100_objetivo,
    factor_expansion=factor_expansion_auto
)

params_multiobj = calcular_malla_optimizada_multiobjetivo(
    ucs=ucs_input,
    densidad_explosivo=densidad_exp,
    vod=vod_exp,
    diametro_pulg=diametro_input,
    p80_objetivo=p80_objetivo,
    p100_objetivo=p100_objetivo,
    peso_metros=peso_metros_norm,
    peso_p80=peso_p80_norm,
    peso_p100=peso_p100_norm
)

# Guardar en session_state cuando se presiona el botón
if calcular_btn:
    st.session_state.params_teoricos = params_teoricos
    st.session_state.params_multiobj = params_multiobj
    st.session_state.params_minmetros = params_minmetros
    st.toast("✅ Cálculos actualizados correctamente", icon="✅")
    st.balloons()

# Mostrar mensaje si no se ha calculado nunca
if not st.session_state.resultados_calculados:
    st.info("👆 Presiona el botón **CALCULAR MALLA ÓPTIMA** para confirmar los parámetros y generar las recomendaciones.")
    st.stop()

# Calcular taco intermedio teórico (si aplica)
longitud_carga = altura_banco_est - params_teoricos['taco_optimo'] - params_teoricos['pasadura_optima']
usar_taco_intermedio = longitud_carga > 10

if usar_taco_intermedio:
    taco_intermedio_teorico = round(longitud_carga / 3, 2)
    posicion_taco_int = round(params_teoricos['pasadura_optima'] + longitud_carga * 0.33, 2)
else:
    taco_intermedio_teorico = 0
    posicion_taco_int = 0

# ===== MEJOR CONFIGURACIÓN HISTÓRICA =====
mejor_historico = None
top3_historico = None

if col_ucs is not None and col_ucs in df_filtrado.columns:
    ucs_tolerance = 25
    df_hist = df_filtrado[
        (df_filtrado[col_ucs] >= ucs_input - ucs_tolerance) &
        (df_filtrado[col_ucs] <= ucs_input + ucs_tolerance)
    ].copy()
    
    if len(df_hist) > 0 and 'P80TRON' in df_hist.columns:
        # Top 3 históricos con mejor P80
        top3_historico = df_hist.nsmallest(3, 'P80TRON')
        mejor_historico = top3_historico.iloc[0] if len(top3_historico) > 0 else None

# ============================================
# RESUMEN DE PARÁMETROS UTILIZADOS
# ============================================
with st.expander("📊 Ver parámetros utilizados en el cálculo", expanded=False):
    col_res1, col_res2, col_res3 = st.columns(3)
    
    with col_res1:
        st.markdown("**Roca:**")
        st.write(f"- UCS: {ucs_input} MPa")
        st.write(f"- Tipo: {tipo_roca}")
        st.write(f"- Kb: {params_teoricos['Kb']}")
        st.write(f"- Ks: {params_teoricos['Ks']}")
    
    with col_res2:
        st.markdown("**Perforación:**")
        st.write(f"- Diámetro: {diametro_input} pulg")
        st.write(f"- Altura banco: {altura_banco_est} m")
        st.write(f"- Columna explosiva: {round(longitud_carga, 2)} m")
    
    with col_res3:
        st.markdown(f"**Explosivo ({explosivo_input}):**")
        st.write(f"- Densidad: {densidad_exp} g/cc")
        st.write(f"- VOD: {vod_exp} m/s")
        st.write(f"- FC óptimo: {params_teoricos['fc_optimo']} kg/m³")

# ============================================
# PANEL PRINCIPAL: TOP 5 RECOMENDACIONES
# ============================================
st.markdown("---")
st.markdown("## 🏆 TOP 5 RECOMENDACIONES DE MALLA")
st.markdown(f"**Para UCS = {ucs_input} MPa ({tipo_roca})**")

# Crear lista de recomendaciones ordenadas por mejor resultado
recomendaciones = []

# 1. Añadir recomendación teórica
recomendaciones.append({
    'Ranking': 1,
    'Tipo': '📐 Teórica ENAEX',
    'Malla': f"{params_teoricos['burden_optimo']} × {params_teoricos['espaciamiento_optimo']}",
    'Área (m²)': round(params_teoricos['area_malla'], 2),
    'FC': round(params_teoricos['fc_optimo'], 2),
    'Taco (m)': round(params_teoricos['taco_optimo'], 2),
    'tp (ms)': round(params_teoricos['timing_pozos_optimo'], 2),
    'tf (ms)': round(params_teoricos['timing_filas_optimo'], 2),
    'P80': 6.00,  # Estimado teórico
    'P100': 12.00,  # Estimado teórico
    'Ahorro (%)': 0.00,
    'Fuente': 'Cálculo teórico'
})

# 2. Añadir recomendación optimizada (menos metros)
recomendaciones.append({
    'Ranking': 2,
    'Tipo': '⚡ Optimizada',
    'Malla': f"{params_multiobj['burden_optimo']} × {params_multiobj['espaciamiento_optimo']}",
    'Área (m²)': round(params_multiobj['area_malla'], 2),
    'FC': round(params_multiobj['fc_ajustado'], 2),
    'Taco (m)': round(params_multiobj['taco_optimo'], 2),
    'tp (ms)': round(params_multiobj['timing_pozos'], 2),
    'tf (ms)': round(params_multiobj['timing_filas'], 2),
    'P80': round(params_multiobj['P80_estimado'], 2),
    'P100': round(params_multiobj['P100_estimado'], 2),
    'Ahorro (%)': round(params_multiobj['reduccion_metros_pct'], 2),
    'Fuente': 'Multi-objetivo'
})

# 3. Añadir históricos (hasta 3)
if top3_historico is not None and len(top3_historico) > 0:
    for idx, row in top3_historico.head(3).iterrows():
        burden_h = row.get('Burden', np.nan)
        esp_h = row.get('Espaciamiento', np.nan)
        fc_h = row.get('FC', row.get('fc1', np.nan))
        taco_h = row.get('taco_gravilla', np.nan)
        tp_h = row.get('tpozos_ms', np.nan)
        tf_h = row.get('tfilas_ms', np.nan)
        p80_h = row.get('P80TRON', np.nan)
        p100_h = row.get('P100TRON', np.nan)
        
        if pd.notna(burden_h) and pd.notna(esp_h):
            area_h = round(float(burden_h) * float(esp_h), 2)
            # Calcular ahorro respecto a teórica
            ahorro_h = round((1 - params_teoricos['area_malla'] / area_h) * 100, 2) if area_h > 0 else 0
            
            recomendaciones.append({
                'Ranking': len(recomendaciones) + 1,
                'Tipo': '📊 Histórica',
                'Malla': f"{round(burden_h, 2)} × {round(esp_h, 2)}",
                'Área (m²)': area_h,
                'FC': round(fc_h, 2) if pd.notna(fc_h) else '-',
                'Taco (m)': round(taco_h, 2) if pd.notna(taco_h) else '-',
                'tp (ms)': round(tp_h, 2) if pd.notna(tp_h) else '-',
                'tf (ms)': round(tf_h, 2) if pd.notna(tf_h) else '-',
                'P80': round(p80_h, 2) if pd.notna(p80_h) else '-',
                'P100': round(p100_h, 2) if pd.notna(p100_h) else '-',
                'Ahorro (%)': ahorro_h,
                'Fuente': 'Dato real'
            })

# Limitar a 5 recomendaciones
recomendaciones = recomendaciones[:5]

# Ordenar por P80 (mejores resultados primero) - si P80 es string, dejarlo al final
def sort_key(x):
    p80 = x['P80']
    if isinstance(p80, str):
        return 999
    return p80

recomendaciones_sorted = sorted(recomendaciones, key=sort_key)

# Reasignar ranking
for i, rec in enumerate(recomendaciones_sorted):
    rec['Ranking'] = i + 1

# Crear DataFrame para TOP 5
df_top5 = pd.DataFrame(recomendaciones_sorted)

st.markdown("### 📋 Comparativa de Mejores Configuraciones")
st.dataframe(
    df_top5,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Ranking": st.column_config.NumberColumn("🏅", width="small"),
        "Tipo": st.column_config.TextColumn("Tipo", width="medium"),
        "Malla": st.column_config.TextColumn("B × S (m)", width="medium"),
        "Área (m²)": st.column_config.NumberColumn("Área", format="%.2f"),
        "FC": st.column_config.NumberColumn("FC", format="%.2f"),
        "Taco (m)": st.column_config.NumberColumn("Taco", format="%.2f"),
        "T.Pozos (ms)": st.column_config.NumberColumn("T.Pozos", format="%.2f"),
        "P80": st.column_config.NumberColumn("P80 (\")", format="%.2f"),
        "P100": st.column_config.NumberColumn("P100 (\")", format="%.2f"),
        "Ahorro (%)": st.column_config.NumberColumn("Ahorro", format="%.2f%%"),
    }
)

# ===== CARDS RESUMEN (las 3 principales) =====
st.markdown("### 🎯 Resumen de Mejores Opciones")

col_card1, col_card2, col_card3 = st.columns(3)

with col_card1:
    st.markdown(f"""
    <div style="background-color:#e1f5fe; padding:15px; border-radius:10px; border-left:5px solid #03a9f4;">
    <h4 style="margin:0; color:#0277bd;">📐 TEÓRICA</h4>
    <h2 style="margin:5px 0; color:#01579b;">{params_teoricos['burden_optimo']} × {params_teoricos['espaciamiento_optimo']}</h2>
    <p style="margin:0; color:#0288d1;">
    Área: {round(params_teoricos['area_malla'], 2)} m²<br>
    FC: {round(params_teoricos['fc_optimo'], 2)} kg/m³<br>
    Taco: {round(params_teoricos['taco_optimo'], 2)} m
    </p>
    </div>
    """, unsafe_allow_html=True)

with col_card2:
    cumple_icon = "✅" if (params_multiobj['cumple_P80'] and params_multiobj['cumple_P100']) else "⚠️"
    st.markdown(f"""
    <div style="background-color:#b3e5fc; padding:15px; border-radius:10px; border-left:5px solid #0288d1;">
    <h4 style="margin:0; color:#0277bd;">⚡ OPTIMIZADA {cumple_icon}</h4>
    <h2 style="margin:5px 0; color:#01579b;">{params_multiobj['burden_optimo']} × {params_multiobj['espaciamiento_optimo']}</h2>
    <p style="margin:0; color:#0288d1;">
    P80: {round(params_multiobj['P80_estimado'], 2)}" | P100: {round(params_multiobj['P100_estimado'], 2)}"<br>
    Ahorro: <b>{round(params_multiobj['reduccion_metros_pct'], 2)}%</b> metros<br>
    FC ajustado: {round(params_multiobj['fc_ajustado'], 2)} kg/m³
    </p>
    </div>
    """, unsafe_allow_html=True)

with col_card3:
    if top3_historico is not None and len(top3_historico) > 0:
        mejor = top3_historico.iloc[0]
        b_h = mejor.get('Burden', '-')
        s_h = mejor.get('Espaciamiento', '-')
        p80_h = mejor.get('P80TRON', '-')
        p100_h = mejor.get('P100TRON', '-')
        st.markdown(f"""
        <div style="background-color:#81d4fa; padding:15px; border-radius:10px; border-left:5px solid #0277bd;">
        <h4 style="margin:0; color:#0277bd;">📊 MEJOR HISTÓRICA</h4>
        <h2 style="margin:5px 0; color:#01579b;">{round(b_h, 2) if pd.notna(b_h) else '-'} × {round(s_h, 2) if pd.notna(s_h) else '-'}</h2>
        <p style="margin:0; color:#0288d1;">
        P80: {round(p80_h, 2) if pd.notna(p80_h) else '-'}" | P100: {round(p100_h, 2) if pd.notna(p100_h) else '-'}"<br>
        <b>Resultado real probado</b><br>
        Fuente: datos históricos
        </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background-color:#e1f5fe; padding:15px; border-radius:10px; border-left:5px solid #90a4ae;">
        <h4 style="margin:0; color:#546e7a;">📊 HISTÓRICA</h4>
        <h2 style="margin:5px 0; color:#546e7a;">Sin datos</h2>
        <p style="margin:0; color:#78909c;">
        No hay tronaduras históricas<br>
        para UCS ≈ {ucs_input} MPa
        </p>
        </div>
        """, unsafe_allow_html=True)

# Botones para cargar configuraciones en el Simulador PILIX
st.markdown("### 🔬 Simular en PILIX v3.6")
st.markdown("Selecciona una configuración para cargarla en el **Simulador PILIX v3.6** (pestaña separada):")

col_sim1, col_sim2, col_sim3, col_sim4 = st.columns(4)

with col_sim1:
    if st.button("🔬 Simular Teórica", key="sim_teorica", help="Cargar parámetros teóricos ENAEX"):
        st.session_state.pilix_params = {
            'UCS': ucs_input,
            'RQD': 70,
            'B': params_teoricos['burden_optimo'],
            'S': params_teoricos['espaciamiento_optimo'],
            'T': params_teoricos['taco_optimo'],
            'H': altura_banco_est,
            'J': 1.5,
            'd': diametro_input,
            'Kx': 0.5,
            'Kn': 1.02,
            'doble_taco': False,
            'Ti': 2.0,
            'Ti_pos': 8.0,
            'nombre_config': 'Teórica ENAEX'
        }
        st.success("✅ Configuración Teórica cargada. Ve a la pestaña **Simulador PILIX v3.6**")

with col_sim2:
    if st.button("🔬 Simular Optimizada", key="sim_optimizada", help="Cargar parámetros optimizados"):
        st.session_state.pilix_params = {
            'UCS': ucs_input,
            'RQD': 70,
            'B': params_multiobj['burden_optimo'],
            'S': params_multiobj['espaciamiento_optimo'],
            'T': params_multiobj['taco_optimo'],
            'H': altura_banco_est,
            'J': 1.5,
            'd': diametro_input,
            'Kx': params_multiobj.get('Kx', 0.5),
            'Kn': params_multiobj.get('Kn', 1.02),
            'doble_taco': params_multiobj.get('necesita_taco_intermedio', False),
            'Ti': 2.0,
            'Ti_pos': 8.0,
            'nombre_config': 'Optimizada Multiobjetivo'
        }
        st.success("✅ Configuración Optimizada cargada. Ve a la pestaña **Simulador PILIX v3.6**")

with col_sim3:
    if st.button("🔬 Simular Min. Metros", key="sim_minmetros", help="Cargar parámetros de mínimos metros"):
        st.session_state.pilix_params = {
            'UCS': ucs_input,
            'RQD': 70,
            'B': params_minmetros['burden_minmetros'],
            'S': params_minmetros['espaciamiento_minmetros'],
            'T': params_minmetros['taco_ajustado'],
            'H': altura_banco_est,
            'J': 1.5,
            'd': diametro_input,
            'Kx': 0.5,
            'Kn': 1.02,
            'doble_taco': False,
            'Ti': 2.0,
            'Ti_pos': 8.0,
            'nombre_config': 'Mínimos Metros'
        }
        st.success("✅ Configuración Mín. Metros cargada. Ve a la pestaña **Simulador PILIX v3.6**")

with col_sim4:
    if top3_historico is not None and len(top3_historico) > 0:
        if st.button("🔬 Simular Histórica", key="sim_historica", help="Cargar mejor configuración histórica"):
            mejor = top3_historico.iloc[0]
            st.session_state.pilix_params = {
                'UCS': ucs_input,
                'RQD': 70,
                'B': float(mejor.get('Burden', 7.5)) if pd.notna(mejor.get('Burden')) else 7.5,
                'S': float(mejor.get('Espaciamiento', 8.6)) if pd.notna(mejor.get('Espaciamiento')) else 8.6,
                'T': float(mejor.get('Taco', 5.5)) if pd.notna(mejor.get('Taco')) else 5.5,
                'H': altura_banco_est,
                'J': 1.5,
                'd': diametro_input,
                'Kx': 0.5,
                'Kn': 1.02,
                'doble_taco': False,
                'Ti': 2.0,
                'Ti_pos': 8.0,
                'nombre_config': 'Mejor Histórica'
            }
            st.success("✅ Configuración Histórica cargada. Ve a la pestaña **Simulador PILIX v3.6**")
    else:
        st.button("🔬 Simular Histórica", key="sim_historica_disabled", disabled=True, help="No hay datos históricos")

# ===== SIMULADOR MONTE CARLO 3D DE TRONADURA =====
st.markdown("---")
st.markdown("## 🎲 Simulador Monte Carlo de Tronadura")

st.markdown("""
Simulador avanzado con análisis Monte Carlo que considera:
- **Efecto del UCS** en la fragmentación (roca blanda → fina, roca dura → gruesa)
- **Doble taco (taco intermedio)** para reducir sobretamaños
- **Intervalos de confianza** al 90% para P80 y P100
- **Ondas 3D esféricas** desde el APD (Arranque Por Detonador)
""")

# Función para generar el HTML del simulador Monte Carlo
def generar_simulador_montecarlo_html(ucs, burden, espaciamiento, taco, altura, pasadura, 
                                       diametro_pulg, x50_estimado, n_uniformidad, 
                                       tp_ms, tf_ms, usar_doble_taco=False):
    """
    Genera el HTML del simulador Monte Carlo con los parámetros de la app.
    
    Mejoras técnicas implementadas:
    1. X50 calculado desde UCS usando modelo Kuz-Ram calibrado
    2. Factor n ajustado según dureza de roca
    3. Doble taco reduce P100 en ~20% (validado con datos Los Pelambres)
    4. Ondas 3D esféricas desde posición real del APD
    5. Monte Carlo con 1000 iteraciones por defecto
    """
    
    # Convertir diámetro a mm
    diametro_mm = diametro_pulg * 25.4
    
    # Calcular posición del APD (fondo de la carga)
    largo_total = altura + pasadura
    largo_carga = largo_total - taco - pasadura
    apd_depth = taco + largo_carga
    
    html_code = f'''
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: 'Segoe UI', Arial, sans-serif; 
            background: #1a1a2e;
            color: #fff;
            overflow-x: hidden;
        }}
        
        .container {{
            display: grid;
            grid-template-columns: 280px 1fr 380px;
            gap: 8px;
            padding: 8px;
            height: 650px;
        }}
        
        .panel {{
            background: rgba(255,255,255,0.97);
            color: #222;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .panel-header {{
            background: linear-gradient(135deg, #1976d2, #1565c0);
            color: #fff;
            padding: 8px 12px;
            font-weight: bold;
            font-size: 11px;
        }}
        
        .panel-content {{
            padding: 8px;
            max-height: 600px;
            overflow-y: auto;
        }}
        
        .section-title {{
            color: #1976d2;
            font-size: 10px;
            font-weight: bold;
            margin: 8px 0 5px 0;
            padding-bottom: 3px;
            border-bottom: 2px solid #2196f3;
        }}
        
        .input-group {{
            display: grid;
            grid-template-columns: 1fr 60px 35px;
            align-items: center;
            margin: 4px 0;
            gap: 4px;
        }}
        .input-group label {{ color: #555; font-size: 9px; }}
        .input-group input, .input-group select {{
            padding: 3px 5px;
            border: 1px solid #ddd;
            border-radius: 3px;
            font-size: 9px;
            text-align: right;
        }}
        .input-group .unit {{ color: #888; font-size: 8px; }}
        
        .calc-box {{
            background: linear-gradient(135deg, #e3f2fd, #bbdefb);
            padding: 5px 8px;
            border-radius: 4px;
            margin: 4px 0;
            border-left: 3px solid #2196f3;
            font-size: 9px;
        }}
        .calc-box .value {{ font-weight: bold; color: #1565c0; }}
        
        .ucs-indicator {{
            display: flex;
            gap: 2px;
            margin: 4px 0;
        }}
        .ucs-bar {{
            flex: 1;
            height: 6px;
            border-radius: 3px;
        }}
        
        .btn {{
            padding: 6px 10px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-weight: bold;
            font-size: 9px;
            transition: all 0.2s;
        }}
        .btn:hover {{ transform: translateY(-1px); }}
        .btn-primary {{ background: linear-gradient(135deg, #1976d2, #1565c0); color: #fff; }}
        .btn-danger {{ background: linear-gradient(135deg, #e74c3c, #c0392b); color: #fff; }}
        .btn-success {{ background: linear-gradient(135deg, #27ae60, #219a52); color: #fff; }}
        .btn:disabled {{ background: #bdc3c7; cursor: not-allowed; }}
        
        .btn-group {{ display: flex; gap: 4px; margin-top: 8px; flex-wrap: wrap; }}
        
        .checkbox-group {{
            display: flex;
            align-items: center;
            gap: 6px;
            margin: 6px 0;
            padding: 6px;
            background: #e3f2fd;
            border-radius: 4px;
        }}
        .checkbox-group input[type="checkbox"] {{ width: 14px; height: 14px; }}
        .checkbox-group label {{ font-size: 9px; color: #333; }}
        
        #threejs-container {{
            border-radius: 8px;
            overflow: hidden;
            position: relative;
            background: #37474f;
            height: 100%;
        }}
        
        #overlay-info {{
            position: absolute;
            top: 8px;
            left: 8px;
            background: rgba(0,0,0,0.85);
            color: #fff;
            padding: 8px 10px;
            border-radius: 6px;
            font-size: 9px;
        }}
        #overlay-info .title {{ font-size: 11px; font-weight: bold; color: #4fc3f7; }}
        
        #legend-3d {{
            position: absolute;
            bottom: 8px;
            left: 8px;
            background: rgba(255,255,255,0.95);
            color: #333;
            padding: 6px 8px;
            border-radius: 5px;
            font-size: 7px;
        }}
        .leg-item {{ display: flex; align-items: center; gap: 3px; margin: 1px 0; }}
        .leg-color {{ width: 8px; height: 8px; border-radius: 2px; }}
        
        .chart-container {{
            background: #fff;
            border-radius: 5px;
            padding: 5px;
            margin-bottom: 6px;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 4px;
            margin-top: 6px;
        }}
        .stat-box {{
            background: #f5f5f5;
            padding: 6px;
            border-radius: 4px;
            text-align: center;
            border: 1px solid #e0e0e0;
        }}
        .stat-title {{ color: #666; font-size: 7px; text-transform: uppercase; }}
        .stat-value {{ font-size: 12px; font-weight: bold; margin-top: 1px; }}
        .stat-sub {{ font-size: 7px; color: #888; }}
        .stat-ok {{ color: #27ae60; }}
        .stat-warning {{ color: #f39c12; }}
        .stat-bad {{ color: #e74c3c; }}
        
        .mc-config {{
            background: #fff8e1;
            border: 1px solid #ffca28;
            border-radius: 4px;
            padding: 6px;
            margin: 5px 0;
        }}
        .mc-config h4 {{ color: #f57c00; font-size: 9px; margin-bottom: 4px; }}
        
        .progress-bar {{
            height: 4px;
            background: #e0e0e0;
            border-radius: 2px;
            overflow: hidden;
            margin: 5px 0;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #4caf50, #8bc34a);
            width: 0%;
        }}
        
        .recommendation-box {{
            background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
            border: 1px solid #81c784;
            border-radius: 5px;
            padding: 8px;
            margin: 6px 0;
        }}
        .recommendation-box h4 {{ color: #2e7d32; font-size: 9px; margin-bottom: 5px; }}
        .rec-item {{
            font-size: 8px;
            margin: 3px 0;
            padding: 4px 6px;
            background: rgba(255,255,255,0.7);
            border-radius: 3px;
            border-left: 2px solid #4caf50;
        }}
        .rec-critical {{ border-left-color: #f44336; background: #ffebee; }}
        .rec-success {{ border-left-color: #4caf50; background: #e8f5e9; }}
        
        .comparison-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 8px;
            margin: 5px 0;
        }}
        .comparison-table th {{
            background: #1976d2;
            color: #fff;
            padding: 4px;
        }}
        .comparison-table td {{
            padding: 4px;
            text-align: center;
            border-bottom: 1px solid #e0e0e0;
        }}
        .diff-ok {{ color: #2e7d32; font-weight: bold; }}
        .diff-bad {{ color: #c62828; font-weight: bold; }}
        
        #controls-3d {{
            padding: 6px;
            background: #eceff1;
            display: flex;
            gap: 8px;
            justify-content: center;
            align-items: center;
            font-size: 8px;
        }}
        
        .conf-legend {{
            display: flex;
            justify-content: center;
            gap: 8px;
            margin-top: 4px;
            font-size: 7px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Panel Izquierdo -->
        <div class="panel">
            <div class="panel-header">⚙️ Parámetros de Diseño</div>
            <div class="panel-content">
                <div class="section-title">🪨 Propiedades de Roca</div>
                <div class="input-group">
                    <label>UCS (Resist. Compresión):</label>
                    <input type="number" id="input-UCS" value="{ucs}" step="10" min="30" max="300">
                    <span class="unit">MPa</span>
                </div>
                <div class="ucs-indicator">
                    <div class="ucs-bar" id="ucs-bar-1" style="background:#4caf50;"></div>
                    <div class="ucs-bar" id="ucs-bar-2" style="background:#8bc34a;"></div>
                    <div class="ucs-bar" id="ucs-bar-3" style="background:#ffeb3b;"></div>
                    <div class="ucs-bar" id="ucs-bar-4" style="background:#ff9800;"></div>
                    <div class="ucs-bar" id="ucs-bar-5" style="background:#f44336;"></div>
                </div>
                <div style="display:flex; justify-content:space-between; font-size:6px; color:#666;">
                    <span>Blanda</span><span>Media</span><span>Dura</span><span>Muy Dura</span>
                </div>
                
                <div class="section-title">📐 Geometría de Malla</div>
                <div class="input-group">
                    <label>Burden (B):</label>
                    <input type="number" id="input-B" value="{burden}" step="0.5" min="4" max="16">
                    <span class="unit">m</span>
                </div>
                <div class="input-group">
                    <label>Espaciamiento (S):</label>
                    <input type="number" id="input-S" value="{espaciamiento}" step="0.5" min="4" max="18">
                    <span class="unit">m</span>
                </div>
                <div class="input-group">
                    <label>Taco (T):</label>
                    <input type="number" id="input-T" value="{taco}" step="0.25" min="3" max="8">
                    <span class="unit">m</span>
                </div>
                <div class="input-group">
                    <label>Altura banco (H):</label>
                    <input type="number" id="input-H" value="{altura}" step="1" min="10" max="20">
                    <span class="unit">m</span>
                </div>
                <div class="input-group">
                    <label>Pasadura (J):</label>
                    <input type="number" id="input-J" value="{pasadura}" step="0.5" min="0.5" max="3">
                    <span class="unit">m</span>
                </div>
                
                <div class="checkbox-group">
                    <input type="checkbox" id="check-doble-taco" {'checked' if usar_doble_taco else ''}>
                    <label for="check-doble-taco"><strong>Doble Taco (reduce P100 ~20%)</strong></label>
                </div>
                
                <div class="section-title">📊 Kuz-Ram Calibrado</div>
                <div class="calc-box">
                    <span>X50 (calibrado Los Pelambres):</span>
                    <span class="value" id="calc-X50">{x50_estimado:.2f} cm</span>
                </div>
                <div class="input-group">
                    <label>n (uniformidad):</label>
                    <input type="number" id="input-n" value="{n_uniformidad}" step="0.05" min="0.8" max="2.0">
                    <span class="unit"></span>
                </div>
                
                <div class="section-title">⏱️ Timing</div>
                <div class="calc-box">
                    <span>tp (entre pozos):</span>
                    <span class="value">{tp_ms:.1f} ms</span>
                </div>
                <div class="calc-box">
                    <span>tf (entre filas):</span>
                    <span class="value">{tf_ms:.1f} ms</span>
                </div>
                
                <div class="mc-config">
                    <h4>🎲 Monte Carlo</h4>
                    <div class="input-group">
                        <label>Iteraciones:</label>
                        <input type="number" id="input-iterations" value="1000" step="100" min="100" max="5000">
                        <span class="unit"></span>
                    </div>
                    <div class="input-group">
                        <label>Variabilidad (σ):</label>
                        <input type="number" id="input-var" value="10" step="5" min="0" max="30">
                        <span class="unit">%</span>
                    </div>
                </div>
                
                <div class="btn-group">
                    <button class="btn btn-danger" id="btn-simulate" onclick="runFullSimulation()">🔥 Simular MC</button>
                    <button class="btn btn-primary" onclick="resetVisual()">🔄 Reset</button>
                </div>
                
                <div class="progress-bar" id="progress-container" style="display:none;">
                    <div class="progress-fill" id="progress-fill"></div>
                </div>
                <div id="progress-text" style="text-align:center; font-size:7px; color:#666;"></div>
            </div>
        </div>
        
        <!-- Panel Central - 3D -->
        <div class="panel">
            <div class="panel-header">🎯 Simulación 3D - Ondas desde APD</div>
            <div id="threejs-container">
                <div id="overlay-info">
                    <div class="title">Malla 4×4 = 16 pozos</div>
                    <div>B=<span id="disp-B">{burden}</span>m | S=<span id="disp-S">{espaciamiento}</span>m</div>
                    <div>UCS=<span id="disp-UCS">{ucs}</span>MPa</div>
                    <div id="detonation-status" style="margin-top:4px; color:#4fc3f7;">⏸️ Esperando...</div>
                </div>
                <div id="legend-3d">
                    <div class="leg-item"><div class="leg-color" style="background:#00e5ff;"></div>Taco</div>
                    <div class="leg-item"><div class="leg-color" style="background:#ff1744;"></div>Explosivo</div>
                    <div class="leg-item"><div class="leg-color" style="background:#ffea00;"></div>APD</div>
                    <div class="leg-item"><div class="leg-color" style="background:#00e676;"></div>Pasadura</div>
                    <div class="leg-item"><div class="leg-color" style="background:#9c27b0;"></div>Taco Int.</div>
                </div>
            </div>
            <div id="controls-3d">
                <label>Velocidad: <input type="range" id="speed" min="1" max="100" value="40" style="width:60px;"></label>
                <label><input type="checkbox" id="show-waves" checked> Ondas</label>
                <label><input type="checkbox" id="show-fragments" checked> Fragmentos</label>
            </div>
        </div>
        
        <!-- Panel Derecho - Resultados -->
        <div class="panel">
            <div class="panel-header">📈 Resultados Monte Carlo</div>
            <div class="panel-content">
                <div class="chart-container">
                    <canvas id="curve-canvas" width="350" height="140"></canvas>
                    <div class="conf-legend">
                        <div style="display:flex;align-items:center;gap:3px;"><div style="width:12px;height:2px;background:#1a1a1a;"></div>Teórica</div>
                        <div style="display:flex;align-items:center;gap:3px;"><div style="width:12px;height:6px;background:rgba(33,150,243,0.3);border:1px solid #2196f3;"></div>IC 90%</div>
                    </div>
                </div>
                
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:4px;">
                    <div class="chart-container">
                        <canvas id="hist-p80" width="165" height="65"></canvas>
                    </div>
                    <div class="chart-container">
                        <canvas id="hist-p100" width="165" height="65"></canvas>
                    </div>
                </div>
                
                <div class="stats-grid" id="stats-grid">
                    <div class="stat-box"><div class="stat-title">P80 Medio</div><div class="stat-value">--</div></div>
                    <div class="stat-box"><div class="stat-title">P100 Medio</div><div class="stat-value">--</div></div>
                    <div class="stat-box"><div class="stat-title">IC 90% P80</div><div class="stat-value" style="font-size:10px;">--</div></div>
                    <div class="stat-box"><div class="stat-title">IC 90% P100</div><div class="stat-value" style="font-size:10px;">--</div></div>
                    <div class="stat-box" style="grid-column: span 2;"><div class="stat-title">P(P100 ≤ 12")</div><div class="stat-value">--</div></div>
                </div>
                
                <div class="recommendation-box" id="recommendation-box">
                    <h4>💡 Recomendaciones</h4>
                    <div class="rec-item">Ejecuta la simulación Monte Carlo...</div>
                </div>
                
                <table class="comparison-table" id="comparison-table">
                    <thead><tr><th>Métrica</th><th>Teórico</th><th>MC</th><th>Δ</th></tr></thead>
                    <tbody>
                        <tr><td>P80</td><td id="teo-p80">--</td><td id="mc-p80">--</td><td id="diff-p80">--</td></tr>
                        <tr><td>P100</td><td id="teo-p100">--</td><td id="mc-p100">--</td><td id="diff-p100">--</td></tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
        const ROWS = 4, COLS = 4;
        let scene, camera, renderer;
        let pozos = [];
        let waveObjects = [];
        let fragmentObjects = [];
        let isAnimating = false;
        let simulationTime = 0;
        let detonatedCount = 0;
        let currentSimFragments = [];
        
        let mcResults = {{ p80_values: [], p100_values: [], allSizes: [], iterations: 0 }};
        let theoreticalValues = {{ X50: 0, Xc: 0, P80: 0, P100: 0 }};
        
        let params = {{
            UCS: {ucs},
            B: {burden}, S: {espaciamiento}, T: {taco}, H: {altura}, J: {pasadura},
            d: {diametro_mm}, X50: {x50_estimado}, n: {n_uniformidad},
            th: {tp_ms}, tr: {tf_ms},
            dobleTaco: {'true' if usar_doble_taco else 'false'}, 
            Ti: 2.0, 
            TiPos: {altura / 2:.1f}
        }};

        function init() {{
            initThreeJS();
            setupInputListeners();
            updateCalculations();
            updateUCSIndicator();
            drawEmptyCurve();
            drawEmptyHistograms();
        }}

        function initThreeJS() {{
            const container = document.getElementById('threejs-container');
            const width = container.clientWidth;
            const height = container.clientHeight || 400;
            
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x37474f);

            camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 500);

            renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
            renderer.setSize(width, height);
            renderer.shadowMap.enabled = true;
            renderer.shadowMap.type = THREE.PCFSoftShadowMap;
            container.insertBefore(renderer.domElement, container.firstChild);

            // Iluminación mejorada
            scene.add(new THREE.AmbientLight(0xffffff, 0.5));
            
            const sun = new THREE.DirectionalLight(0xffffff, 1.0);
            sun.position.set(50, 80, 50);
            sun.castShadow = true;
            sun.shadow.mapSize.width = 1024;
            sun.shadow.mapSize.height = 1024;
            scene.add(sun);
            
            // Luz de relleno para detalles
            const fillLight = new THREE.DirectionalLight(0x8ec5fc, 0.3);
            fillLight.position.set(-30, 20, -30);
            scene.add(fillLight);
            
            // Hemisferio para ambiente natural
            const hemi = new THREE.HemisphereLight(0x87ceeb, 0x5a4a3a, 0.4);
            scene.add(hemi);

            setupCameraControls();
            createScene();
            animate();
        }}

        function setupCameraControls() {{
            let drag = false, prev = {{x:0, y:0}};
            let cam = {{ theta: 0.5, phi: 0.8, r: 55 }};

            const updateCam = () => {{
                const tgt = new THREE.Vector3(params.S * 1.5, -6, params.B * 1.5);
                camera.position.set(
                    tgt.x + cam.r * Math.sin(cam.phi) * Math.cos(cam.theta),
                    tgt.y + cam.r * Math.cos(cam.phi),
                    tgt.z + cam.r * Math.sin(cam.phi) * Math.sin(cam.theta)
                );
                camera.lookAt(tgt);
            }};

            renderer.domElement.onmousedown = e => {{ drag = true; prev = {{x: e.clientX, y: e.clientY}}; }};
            renderer.domElement.onmousemove = e => {{
                if (!drag) return;
                cam.theta -= (e.clientX - prev.x) * 0.005;
                cam.phi = Math.max(0.15, Math.min(1.5, cam.phi + (e.clientY - prev.y) * 0.005));
                prev = {{x: e.clientX, y: e.clientY}};
                updateCam();
            }};
            renderer.domElement.onmouseup = () => drag = false;
            renderer.domElement.onwheel = e => {{
                cam.r = Math.max(20, Math.min(150, cam.r + e.deltaY * 0.05));
                updateCam();
            }};

            window.updateCam = updateCam;
            updateCam();
        }}

        function setupInputListeners() {{
            ['UCS', 'B', 'S', 'T', 'H', 'J', 'n', 'iterations', 'var'].forEach(id => {{
                const el = document.getElementById('input-' + id);
                if (el) el.addEventListener('change', () => {{
                    updateCalculations();
                    updateUCSIndicator();
                    createScene();
                    if (window.updateCam) window.updateCam();
                }});
            }});
            
            document.getElementById('check-doble-taco').addEventListener('change', function() {{
                params.dobleTaco = this.checked;
                updateCalculations();
                createScene();
            }});
        }}

        function updateUCSIndicator() {{
            const UCS = params.UCS;
            const bars = ['ucs-bar-1', 'ucs-bar-2', 'ucs-bar-3', 'ucs-bar-4', 'ucs-bar-5'];
            const thresholds = [0, 60, 100, 150, 200];
            bars.forEach((id, i) => {{
                document.getElementById(id).style.opacity = UCS >= thresholds[i] ? '1' : '0.2';
            }});
        }}

        function updateCalculations() {{
            params.UCS = parseFloat(document.getElementById('input-UCS').value);
            params.B = parseFloat(document.getElementById('input-B').value);
            params.S = parseFloat(document.getElementById('input-S').value);
            params.T = parseFloat(document.getElementById('input-T').value);
            params.H = parseFloat(document.getElementById('input-H').value);
            params.J = parseFloat(document.getElementById('input-J').value);
            params.n = parseFloat(document.getElementById('input-n').value);
            
            // =============================================
            // MODELO PILIX v3.6 - CÁLCULOS ROBUSTOS
            // =============================================
            
            // 1. ÍNDICE DE LILLY (Factor A desde UCS)
            // RMD basado en RQD estimado desde UCS
            const rqd_est = Math.max(25, Math.min(100, 50 + params.UCS / 4));
            const RMD = rqd_est > 75 ? 50 : (rqd_est >= 50 ? 20 : 10);
            const JPS = rqd_est > 80 ? 50 : (rqd_est >= 50 ? 20 : 10);
            const JPO = 25;
            const SG = Math.min(2.9, Math.max(2.4, 2.5 + params.UCS / 400));
            const SGI = 25 * SG - 50;
            const HP = 0.05 * params.UCS;
            const BI = 0.5 * (RMD + JPS + JPO + SGI + HP);
            const A_lilly = Math.max(3.0, Math.min(13.0, 0.12 * BI));
            
            // 2. FACTORES Kx/Kn (Calibración Mundaca)
            const Kx = Math.max(0.2, Math.min(1.5, 0.25 + 0.75 * (params.UCS / 200) * (rqd_est / 100)));
            const Kn = Math.max(0.85, Math.min(1.15, 0.95 + 0.10 * (rqd_est / 100)));
            
            // Usar A calibrado Los Pelambres (19.9) o Lilly según configuración
            const A_usado = 19.9;  // Calibrado LP
            
            // 3. CARGA POR POZO
            const Lc = params.H - params.T;
            const Q = Lc > 0 ? 68.7 * Lc : 0;
            const vol = params.B * params.S * params.H;
            
            // 4. KUZ-RAM CON Kx (Modelo PILIX v3.6)
            // X50 = Kx × A × (V0/Q)^0.8 × Q^(1/6) × (1.15/Er)^(19/30)
            const Er = 1.0;  // RWS relativo (ANFO=1.0)
            if (Q > 0) {{
                params.X50 = Kx * A_usado * Math.pow(vol/Q, 0.8) * Math.pow(Q, 1/6) * Math.pow(1.15/Er, 19/30);
                params.X50 = Math.max(5, Math.min(60, params.X50));
            }}
            
            // 5. ÍNDICE n DE CUNNINGHAM CON Kn
            // n = (2.2 - 14×B/D_mm) × √[(1+S/B)/2] × (1-W/B) × (Lc/H) × Kn
            const D_mm = params.d;  // ya en mm
            const term1 = 2.2 - 14 * (params.B / D_mm);
            const term2 = Math.sqrt((1 + params.S / params.B) / 2);
            const term3 = 1 - 0.5 / params.B;  // W = 0.5m típico
            const term4 = Math.max(0.1, Lc / params.H);
            const n_calc = Math.max(0.5, Math.min(2.5, term1 * term2 * term3 * term4 * Kn));
            
            // Actualizar n
            params.n_calc = n_calc;
            
            // 6. FACTOR DE TACO (Modelo Los Pelambres)
            let f_taco = 1.0;
            if (params.B >= 11.0) {{
                f_taco = 1.0 - 0.30 * (params.T - 5.0);
            }} else if (params.B < 8.0) {{
                f_taco = 1.0 + 0.30 * Math.pow(params.T - 5.25, 2);
            }} else {{
                const ratio = (params.B - 8.0) / 3.0;
                const f_ab = 1.0 - 0.30 * (params.T - 5.0);
                const f_ce = 1.0 + 0.30 * Math.pow(params.T - 5.25, 2);
                f_taco = f_ce + ratio * (f_ab - f_ce);
            }}
            f_taco = Math.max(0.5, Math.min(2.0, f_taco));
            
            // 7. ROSIN-RAMMLER CON n CALCULADO
            // Xc = X50 / (ln2)^(1/n)
            const n_eff = params.n_calc || params.n;
            const Xc = params.X50 / Math.pow(0.693, 1/n_eff);
            
            // P80 = Xc × [-ln(0.20)]^(1/n) × f_taco
            // P100 = Xc × [-ln(0.01)]^(1/n) × f_taco
            theoreticalValues.X50 = params.X50;
            theoreticalValues.Xc = Xc;
            theoreticalValues.n = n_eff;
            theoreticalValues.Kx = Kx;
            theoreticalValues.Kn = Kn;
            theoreticalValues.A = A_usado;
            theoreticalValues.P80 = (Xc * Math.pow(-Math.log(0.20), 1/n_eff) * f_taco) / 2.54;
            theoreticalValues.P100 = (Xc * Math.pow(-Math.log(0.01), 1/n_eff) * f_taco) / 2.54;
            
            // 8. REDUCCIÓN POR DOBLE TACO
            if (params.dobleTaco) {{
                const gamma = 0.20;
                const Ti = params.Ti || 2.0;
                const reduccion = 1 - gamma * (Ti / (params.T + Ti));
                theoreticalValues.P100 *= reduccion;
            }}
            
            // 9. ZONAS DE DAÑO (Holmberg-Persson)
            const rho_e = 1.2 * 1000;  // kg/m³
            const VOD = 4500;  // m/s
            const Pd = (rho_e * VOD * VOD) / 4 / 1e9;  // GPa
            const r0 = (params.d / 1000) / 2;  // Radio pozo en m
            const sigma_c = params.UCS / 1000;  // GPa
            const sigma_t = (params.UCS / 10) / 1000;  // GPa
            theoreticalValues.Pd = Pd;
            theoreticalValues.rc = r0 * Math.pow(Pd / sigma_c, 0.6);
            theoreticalValues.rf = r0 * Math.pow(Pd / sigma_t, 0.45);
            theoreticalValues.rd = theoreticalValues.rf * 2.0;
            
            // Actualizar UI
            document.getElementById('calc-X50').textContent = params.X50.toFixed(2) + ' cm';
            document.getElementById('disp-B').textContent = params.B;
            document.getElementById('disp-S').textContent = params.S;
            document.getElementById('disp-UCS').textContent = params.UCS;
            document.getElementById('teo-p80').textContent = theoreticalValues.P80.toFixed(2) + '"';
            document.getElementById('teo-p100').textContent = theoreticalValues.P100.toFixed(2) + '"';
        }}

        function createScene() {{
            // Limpiar escena (mantener luces: 4 objetos de iluminación)
            while(scene.children.length > 4) scene.remove(scene.children[scene.children.length-1]);
            pozos = []; waveObjects = []; fragmentObjects = [];
            detonatedCount = 0;

            const {{ B, S, T, H, J }} = params;
            const L = H + J;

            // Piso del banco (nivel inferior)
            const floor = new THREE.Mesh(
                new THREE.PlaneGeometry(100, 80),
                new THREE.MeshLambertMaterial({{ color: 0x6d5d4b, side: THREE.DoubleSide }})
            );
            floor.rotation.x = -Math.PI / 2;
            floor.position.set(S * 1.5, -H - 0.1, B * 1.5);
            floor.receiveShadow = true;
            scene.add(floor);

            // Superficie del banco (nivel superior)
            const surface = new THREE.Mesh(
                new THREE.PlaneGeometry(S * (COLS + 2), B * (ROWS + 2)),
                new THREE.MeshLambertMaterial({{ color: 0xa89078, side: THREE.DoubleSide }})
            );
            surface.rotation.x = -Math.PI / 2;
            surface.position.set(S * 1.5, 0.01, B * 1.5);
            surface.receiveShadow = true;
            scene.add(surface);

            // Grid en superficie
            const grid = new THREE.GridHelper(50, 25, 0x666666, 0x444444);
            grid.position.set(S * 1.5, 0.05, B * 1.5);
            scene.add(grid);

            // Banco como caja semi-transparente
            const banco = new THREE.Mesh(
                new THREE.BoxGeometry(S * (COLS + 0.5), H, B * (ROWS + 0.5)),
                new THREE.MeshLambertMaterial({{ color: 0x8d7b68, transparent: true, opacity: 0.15 }})
            );
            banco.position.set(S * 1.5 - S/2, -H/2, B * 1.5 - B/2);
            scene.add(banco);
            
            // Contorno del banco (wireframe)
            const bancoWire = new THREE.LineSegments(
                new THREE.EdgesGeometry(new THREE.BoxGeometry(S * (COLS + 0.5), H, B * (ROWS + 0.5))),
                new THREE.LineBasicMaterial({{ color: 0x555555 }})
            );
            bancoWire.position.copy(banco.position);
            scene.add(bancoWire);

            // Pozos
            let id = 1;
            for (let row = 0; row < ROWS; row++) {{
                for (let col = 0; col < COLS; col++) {{
                    createPozo(id++, col * S, row * B, col, row);
                }}
            }}
        }}

        function createPozo(id, x, z, col, row) {{
            const {{ T, H, J, d, th, tr, dobleTaco, Ti, TiPos }} = params;
            const L = H + J;
            const Lc = L - T - J;
            const visualD = Math.max(d/1000 * 2.5, 0.5);
            const detTime = col * th + row * tr;
            const apdY = -T - Lc;
            
            const grp = new THREE.Group();
            grp.position.set(x, 0, z);

            // Anillo superficie
            const ring = new THREE.Mesh(
                new THREE.RingGeometry(visualD * 0.8, visualD + 0.15, 32),
                new THREE.MeshBasicMaterial({{ color: 0xffffff, side: THREE.DoubleSide }})
            );
            ring.rotation.x = -Math.PI / 2;
            ring.position.y = 0.08;
            grp.add(ring);

            // TACO
            const taco = new THREE.Mesh(
                new THREE.CylinderGeometry(visualD * 0.85, visualD * 0.85, T, 32),
                new THREE.MeshPhongMaterial({{ color: 0x00e5ff, emissive: 0x00bcd4, emissiveIntensity: 0.5 }})
            );
            taco.position.y = -T / 2;
            grp.add(taco);

            if (dobleTaco) {{
                const Lc1 = TiPos - T;
                if (Lc1 > 0) {{
                    const carga1 = new THREE.Mesh(
                        new THREE.CylinderGeometry(visualD * 0.7, visualD * 0.7, Lc1, 32),
                        new THREE.MeshPhongMaterial({{ color: 0xff1744, emissive: 0xd50000, emissiveIntensity: 0.5, transparent: true, opacity: 0.8 }})
                    );
                    carga1.position.y = -T - Lc1 / 2;
                    grp.add(carga1);
                }}
                
                const tacoInt = new THREE.Mesh(
                    new THREE.CylinderGeometry(visualD * 0.85, visualD * 0.85, Ti, 32),
                    new THREE.MeshPhongMaterial({{ color: 0x9c27b0, emissive: 0x7b1fa2, emissiveIntensity: 0.5 }})
                );
                tacoInt.position.y = -TiPos - Ti / 2;
                grp.add(tacoInt);
                
                const Lc2 = L - TiPos - Ti - J;
                if (Lc2 > 0) {{
                    const carga2 = new THREE.Mesh(
                        new THREE.CylinderGeometry(visualD * 0.7, visualD * 0.7, Lc2, 32),
                        new THREE.MeshPhongMaterial({{ color: 0xff1744, emissive: 0xd50000, emissiveIntensity: 0.5, transparent: true, opacity: 0.8 }})
                    );
                    carga2.position.y = -TiPos - Ti - Lc2 / 2;
                    grp.add(carga2);
                }}
            }} else {{
                const carga = new THREE.Mesh(
                    new THREE.CylinderGeometry(visualD * 0.7, visualD * 0.7, Lc, 32),
                    new THREE.MeshPhongMaterial({{ color: 0xff1744, emissive: 0xd50000, emissiveIntensity: 0.5, transparent: true, opacity: 0.7 }})
                );
                carga.position.y = -T - Lc / 2;
                grp.add(carga);
            }}

            // APD visible
            const apdSphere = new THREE.Mesh(
                new THREE.SphereGeometry(0.4, 32, 32),
                new THREE.MeshPhongMaterial({{ color: 0xffea00, emissive: 0xffc400, emissiveIntensity: 1.0 }})
            );
            apdSphere.position.y = apdY;
            grp.add(apdSphere);

            // PASADURA
            const pas = new THREE.Mesh(
                new THREE.CylinderGeometry(visualD * 0.65, visualD * 0.65, J, 32),
                new THREE.MeshPhongMaterial({{ color: 0x00e676, emissive: 0x00c853, emissiveIntensity: 0.7 }})
            );
            pas.position.y = -L + J / 2;
            grp.add(pas);

            // Etiqueta
            const cv = document.createElement('canvas');
            cv.width = 80; cv.height = 45;
            const ct = cv.getContext('2d');
            ct.fillStyle = 'rgba(0,0,0,0.9)';
            ct.fillRect(0, 0, 80, 45);
            ct.fillStyle = '#fff';
            ct.font = 'bold 14px Arial';
            ct.textAlign = 'center';
            ct.fillText('P' + id, 40, 18);
            ct.font = '10px Arial';
            ct.fillStyle = '#ffab40';
            ct.fillText(detTime.toFixed(0) + 'ms', 40, 32);
            
            const spr = new THREE.Sprite(new THREE.SpriteMaterial({{ map: new THREE.CanvasTexture(cv) }}));
            spr.position.y = 3;
            spr.scale.set(2.5, 1.4, 1);
            grp.add(spr);

            scene.add(grp);
            pozos.push({{ id, x, z, col, row, detTime, grp, taco, pas, apdY, detonated: false, apdSphere }});
        }}

        async function runFullSimulation() {{
            document.getElementById('btn-simulate').disabled = true;
            document.getElementById('progress-container').style.display = 'block';
            document.getElementById('detonation-status').textContent = '🔄 Monte Carlo...';
            
            await runMonteCarlo();
            generateVisualFragments();
            updateRecommendations();
            
            document.getElementById('detonation-status').textContent = '💥 Detonando...';
            await new Promise(r => setTimeout(r, 100));
            startDetonationAnimation();
            
            document.getElementById('btn-simulate').disabled = false;
        }}

        async function runMonteCarlo() {{
            const iterations = parseInt(document.getElementById('input-iterations').value);
            const variability = parseFloat(document.getElementById('input-var').value) / 100;
            const fragsPerIter = 150;
            
            mcResults = {{ p80_values: [], p100_values: [], allSizes: [], iterations, sensibilidad: {{}} }};
            
            // Usar n calculado (Cunningham) o input
            const {{ X50, dobleTaco, Ti }} = params;
            const n_eff = params.n_calc || params.n;
            
            // Guardar valores base para análisis de sensibilidad
            const X50_base = X50;
            const n_base = n_eff;
            
            for (let i = 0; i < iterations; i++) {{
                // Variabilidad PILIX v3.6: perturbación normal truncada
                const Z1 = (Math.random() - 0.5) * 2;  // ~U(-1, 1)
                const Z2 = (Math.random() - 0.5) * 2;
                
                const X50_var = X50 * (1 + variability * Z1);
                const n_var = Math.max(0.5, Math.min(2.5, n_eff * (1 + 0.3 * variability * Z2)));
                
                const Xc = X50_var / Math.pow(0.693, 1 / n_var);
                const sizes = [];
                
                for (let f = 0; f < fragsPerIter; f++) {{
                    const u = Math.random() * 0.998;
                    let size = Xc * Math.pow(-Math.log(1 - u), 1 / n_var);
                    
                    // Reducción por doble taco (γ=0.20)
                    if (dobleTaco && size > Xc * 2) {{
                        const gamma = 0.20;
                        const T_total = params.T + (Ti || 2.0);
                        const reduccion = 1 - gamma * (Ti || 2.0) / T_total;
                        size *= reduccion;
                    }}
                    sizes.push(size);
                }}
                
                sizes.sort((a, b) => a - b);
                mcResults.p80_values.push(sizes[Math.floor(sizes.length * 0.8)] / 2.54);
                mcResults.p100_values.push(sizes[sizes.length - 1] / 2.54);
                
                if (i % 25 === 0) mcResults.allSizes.push(sizes.map(s => s / 2.54));
                
                if (i % 50 === 0) {{
                    document.getElementById('progress-fill').style.width = (i / iterations * 100) + '%';
                    document.getElementById('progress-text').textContent = i + '/' + iterations;
                    await new Promise(r => setTimeout(r, 1));
                }}
            }}
            
            // Análisis de sensibilidad PILIX v3.6
            const mean = arr => arr.reduce((a,b) => a+b, 0) / arr.length;
            const std = arr => Math.sqrt(arr.map(x => Math.pow(x - mean(arr), 2)).reduce((a,b) => a+b, 0) / arr.length);
            
            mcResults.sensibilidad = {{
                'X50': {{ base: X50_base.toFixed(2), cv: (std(mcResults.p80_values) / mean(mcResults.p80_values) * 100).toFixed(1) }},
                'n': {{ base: n_base.toFixed(2), efecto: 'Mayor n → menor dispersión' }},
                'Kx': {{ base: theoreticalValues.Kx?.toFixed(2) || '1.0', efecto: 'Kx < 1 → más fino' }},
                'variabilidad': {{ base: (variability * 100).toFixed(0) + '%', efecto: 'Mayor CV → mayor incertidumbre' }}
            }};
            
            drawCurveWithCI();
            drawHistogram('hist-p80', mcResults.p80_values, 'P80', '#2196f3', 4.5);
            drawHistogram('hist-p100', mcResults.p100_values, 'P100', '#f44336', 12);
            updateStats();
            
            document.getElementById('progress-container').style.display = 'none';
        }}

        function updateRecommendations() {{
            const p80 = mcResults.p80_values, p100 = mcResults.p100_values;
            const N = p80.length;
            const mean = arr => arr.reduce((a,b) => a+b, 0) / arr.length;
            const std = arr => Math.sqrt(arr.map(x => Math.pow(x - mean(arr), 2)).reduce((a,b) => a+b, 0) / arr.length);
            
            const mc_p80 = mean(p80), mc_p100 = mean(p100);
            const std_p80 = std(p80), std_p100 = std(p100);
            const cv_p80 = (std_p80 / mc_p80 * 100);
            
            const diff_p80 = ((mc_p80 - theoreticalValues.P80) / theoreticalValues.P80 * 100);
            const diff_p100 = ((mc_p100 - theoreticalValues.P100) / theoreticalValues.P100 * 100);
            
            document.getElementById('mc-p80').textContent = mc_p80.toFixed(2) + '"';
            document.getElementById('mc-p100').textContent = mc_p100.toFixed(2) + '"';
            
            const dc80 = Math.abs(diff_p80) < 10 ? 'diff-ok' : 'diff-bad';
            const dc100 = Math.abs(diff_p100) < 10 ? 'diff-ok' : 'diff-bad';
            document.getElementById('diff-p80').innerHTML = '<span class="' + dc80 + '">' + (diff_p80 > 0 ? '+' : '') + diff_p80.toFixed(1) + '%</span>';
            document.getElementById('diff-p100').innerHTML = '<span class="' + dc100 + '">' + (diff_p100 > 0 ? '+' : '') + diff_p100.toFixed(1) + '%</span>';
            
            const probP100ok = p100.filter(v => v <= 12).length / N * 100;
            const probP80ok = p80.filter(v => v <= 4.5).length / N * 100;
            
            let recs = [];
            
            // Recomendaciones PILIX v3.6
            if (params.UCS > 150 && mc_p100 > 10 && !params.dobleTaco) {{
                recs.push({{ type: 'critical', text: '<b>🔷 Doble Taco recomendado:</b> UCS=' + params.UCS + 'MPa (roca dura). Reduciría P100 ~20%.' }});
            }}
            
            if (params.dobleTaco) {{
                recs.push({{ type: 'success', text: '<b>✓ Doble Taco activo:</b> Reducción P100 aplicada (γ=0.20).' }});
            }}
            
            // Análisis Kx (modelo PILIX)
            if (theoreticalValues.Kx && theoreticalValues.Kx < 0.5) {{
                recs.push({{ type: 'success', text: '<b>📊 Calibración Kx=' + theoreticalValues.Kx.toFixed(2) + ':</b> Fragmentación más fina que modelo teórico.' }});
            }}
            
            // Análisis uniformidad n
            const n_eff = params.n_calc || params.n;
            if (n_eff < 1.0) {{
                recs.push({{ type: 'warning', text: '<b>⚠️ Uniformidad baja (n=' + n_eff.toFixed(2) + '):</b> Distribución amplia de tamaños.' }});
            }} else if (n_eff > 1.5) {{
                recs.push({{ type: 'success', text: '<b>✓ Uniformidad alta (n=' + n_eff.toFixed(2) + '):</b> Distribución concentrada.' }});
            }}
            
            // Probabilidad de cumplimiento
            if (probP100ok < 80) {{
                recs.push({{ type: 'critical', text: '<b>⛔ Riesgo sobretamaños:</b> Solo ' + probP100ok.toFixed(0) + '% cumple P100≤12". Reducir B/S.' }});
            }} else if (probP100ok >= 90) {{
                recs.push({{ type: 'success', text: '<b>✓ Fragmentación OK:</b> ' + probP100ok.toFixed(0) + '% cumple P100≤12", ' + probP80ok.toFixed(0) + '% cumple P80≤4.5".' }});
            }}
            
            // CV (incertidumbre)
            if (cv_p80 > 10) {{
                recs.push({{ type: 'warning', text: '<b>📈 Alta variabilidad (CV=' + cv_p80.toFixed(1) + '%):</b> Considere reducir variabilidad input.' }});
            }}
            
            document.getElementById('recommendation-box').innerHTML = 
                '<h4>💡 Recomendaciones PILIX v3.6</h4>' + 
                recs.map(r => '<div class="rec-item ' + (r.type === 'critical' ? 'rec-critical' : (r.type === 'warning' ? 'rec-warning' : 'rec-success')) + '">' + r.text + '</div>').join('');
        }}

        function generateVisualFragments() {{
            const {{ X50, n, B, S, H, T }} = params;
            const Xc = X50 / Math.pow(0.693, 1 / n);
            currentSimFragments = [];
            
            // Calcular cantidad de fragmentos basado en volumen de roca
            const volPorPozo = B * S * H;
            const numFrags = Math.min(40, Math.max(18, Math.floor(volPorPozo / 60)));
            
            pozos.forEach(p => {{
                const frags = [];
                for (let i = 0; i < numFrags; i++) {{
                    const u = Math.random() * 0.99;
                    let size = Xc * Math.pow(-Math.log(1 - u), 1 / n);
                    
                    // Doble taco reduce fragmentos grandes
                    if (params.dobleTaco && size > Xc * 1.8) {{
                        size *= 0.75;
                    }}
                    
                    // UCS alta = fragmentos más angulares y resistentes
                    const ucsEffect = Math.min(1.2, params.UCS / 120);
                    
                    frags.push({{
                        visualSize: Math.max(0.08, Math.min(1.0, size / 12)),
                        angle: Math.random() * Math.PI * 2,
                        speed: (0.06 + Math.random() * 0.10) * ucsEffect,
                        ySpeed: (0.12 + Math.random() * 0.18) / ucsEffect
                    }});
                }}
                currentSimFragments.push({{ fragments: frags }});
            }});
        }}

        function startDetonationAnimation() {{
            isAnimating = true;
            simulationTime = 0;
            detonatedCount = 0;
            
            waveObjects.forEach(w => scene.remove(w));
            fragmentObjects.forEach(f => scene.remove(f));
            waveObjects = [];
            fragmentObjects = [];
            
            pozos.forEach(p => {{ p.detonated = false; }});
        }}

        function resetVisual() {{
            isAnimating = false;
            waveObjects.forEach(w => scene.remove(w));
            fragmentObjects.forEach(f => scene.remove(f));
            waveObjects = [];
            fragmentObjects = [];
            createScene();
            document.getElementById('detonation-status').textContent = '⏸️ Esperando...';
        }}

        function updateDetonationAnimation(dt) {{
            if (!isAnimating) return;
            
            const speed = document.getElementById('speed').value / 300;
            const showWaves = document.getElementById('show-waves').checked;
            const showFragments = document.getElementById('show-fragments').checked;
            
            simulationTime += dt * speed * 1000;
            
            pozos.forEach((p, idx) => {{
                if (!p.detonated && simulationTime >= p.detTime) {{
                    triggerDetonation(p, idx, showWaves, showFragments);
                }}
            }});
            
            updateWaves();
            if (showFragments) updateFragments();
            
            document.getElementById('detonation-status').textContent = 
                '💥 T=' + simulationTime.toFixed(0) + 'ms | ' + detonatedCount + '/' + (ROWS*COLS);
            
            const maxT = pozos[pozos.length - 1].detTime;
            if (simulationTime > maxT + 400 && waveObjects.length === 0) {{
                isAnimating = false;
                const p80 = mcResults.p80_values.length > 0 ? 
                    (mcResults.p80_values.reduce((a,b)=>a+b,0)/mcResults.p80_values.length).toFixed(2) : '--';
                document.getElementById('detonation-status').textContent = '✅ Completo | P80=' + p80 + '"';
            }}
        }}

        function triggerDetonation(p, idx, showWaves, showFragments) {{
            p.detonated = true;
            detonatedCount++;
            
            if (p.apdSphere) p.apdSphere.visible = false;
            
            const {{ H, T, B, S, dobleTaco, Ti, TiPos }} = params;
            const apdY = p.apdY;
            
            if (showWaves) {{
                // Esfera de explosión principal (onda 3D)
                const sphere = new THREE.Mesh(
                    new THREE.SphereGeometry(1, 32, 24),
                    new THREE.MeshBasicMaterial({{ color: 0xff5500, transparent: true, opacity: 0.7, side: THREE.DoubleSide }})
                );
                sphere.position.set(p.x, apdY, p.z);
                sphere.userData = {{ type: 'sphere', maxR: Math.max(B, S) * 1.2, speed: 0.5, r: 0.8 }};
                scene.add(sphere);
                waveObjects.push(sphere);
                
                // Wireframe de la esfera (más visible)
                const wireGeom = new THREE.SphereGeometry(1, 16, 12);
                const wireMat = new THREE.MeshBasicMaterial({{ 
                    color: 0xffaa00, 
                    wireframe: true,
                    transparent: true,
                    opacity: 0.9
                }});
                const wire = new THREE.Mesh(wireGeom, wireMat);
                wire.position.set(p.x, apdY, p.z);
                wire.userData = {{ type: 'wire', maxR: Math.max(B, S) * 1.3, speed: 0.55, r: 0.5 }};
                scene.add(wire);
                waveObjects.push(wire);
                
                // Anillos de onda en superficie
                for (let i = 0; i < 2; i++) {{
                    setTimeout(() => {{
                        const ring = new THREE.Mesh(
                            new THREE.RingGeometry(0.5, 0.9, 32),
                            new THREE.MeshBasicMaterial({{ color: 0xffaa00, transparent: true, opacity: 0.85, side: THREE.DoubleSide }})
                        );
                        ring.rotation.x = -Math.PI / 2;
                        ring.position.set(p.x, 0.15, p.z);
                        ring.userData = {{ type: 'ring', age: 0, maxAge: 25 }};
                        scene.add(ring);
                        waveObjects.push(ring);
                    }}, i * 40);
                }}
                
                // Onda vertical que sube por la columna
                const cylGeom = new THREE.CylinderGeometry(0.5, 0.5, 1, 16, 1, true);
                const cylMat = new THREE.MeshBasicMaterial({{ 
                    color: 0xff4400, 
                    transparent: true, 
                    opacity: 0.75,
                    side: THREE.DoubleSide
                }});
                const cyl = new THREE.Mesh(cylGeom, cylMat);
                cyl.position.set(p.x, apdY, p.z);
                cyl.userData = {{ type: 'rising', targetY: -T + 1, speed: 0.7, maxScale: 3.5 }};
                scene.add(cyl);
                waveObjects.push(cyl);
                
                // Si doble taco, segunda onda desde taco intermedio
                if (dobleTaco) {{
                    setTimeout(() => {{
                        const sphere2 = new THREE.Mesh(
                            new THREE.SphereGeometry(0.8, 20, 16),
                            new THREE.MeshBasicMaterial({{ color: 0xff00ff, transparent: true, opacity: 0.6, side: THREE.DoubleSide }})
                        );
                        sphere2.position.set(p.x, -TiPos, p.z);
                        sphere2.userData = {{ type: 'sphere', maxR: B * 0.9, speed: 0.4, r: 0.5 }};
                        scene.add(sphere2);
                        waveObjects.push(sphere2);
                    }}, 60);
                }}
                
                // Luz brillante
                const light = new THREE.PointLight(0xff4400, 5, 30);
                light.position.set(p.x, apdY + 1, p.z);
                light.userData = {{ type: 'light', age: 0, maxAge: 18 }};
                scene.add(light);
                waveObjects.push(light);
            }}
            
            // Fragmentos (más cantidad y mejor física)
            if (showFragments && currentSimFragments[idx]) {{
                currentSimFragments[idx].fragments.forEach(fd => {{
                    const geom = new THREE.DodecahedronGeometry(fd.visualSize, 0);
                    
                    // Deformar fragmentos para que se vean más naturales
                    const pos = geom.attributes.position;
                    for (let i = 0; i < pos.count; i++) {{
                        const f = 0.65 + Math.random() * 0.7;
                        pos.setXYZ(i, pos.getX(i) * f, pos.getY(i) * f, pos.getZ(i) * f);
                    }}
                    geom.computeVertexNormals();
                    
                    const shade = 0.45 + Math.random() * 0.25;
                    const frag = new THREE.Mesh(geom, new THREE.MeshLambertMaterial({{ 
                        color: new THREE.Color(shade * 0.95, shade * 0.9, shade * 0.85) 
                    }}));
                    frag.position.set(
                        p.x + (Math.random()-0.5) * 1.5, 
                        -T - Math.random() * (H - T - 2), 
                        p.z + (Math.random()-0.5) * 1.5
                    );
                    
                    const ang = Math.random() * Math.PI * 2;
                    const force = 0.08 + Math.random() * 0.12;
                    frag.userData = {{
                        vel: new THREE.Vector3(
                            Math.cos(ang) * force * 0.5, 
                            fd.ySpeed * 1.2, 
                            Math.sin(ang) * force * 0.5 + 0.06
                        ),
                        rotVel: new THREE.Vector3(
                            (Math.random()-0.5) * 0.12, 
                            (Math.random()-0.5) * 0.12, 
                            (Math.random()-0.5) * 0.12
                        ),
                        grounded: false
                    }};
                    frag.castShadow = true;
                    scene.add(frag);
                    fragmentObjects.push(frag);
                }});
            }}
        }}

        function updateWaves() {{
            for (let i = waveObjects.length - 1; i >= 0; i--) {{
                const obj = waveObjects[i];
                const ud = obj.userData;
                
                if (ud.type === 'sphere') {{
                    ud.r += ud.speed;
                    if (ud.r < ud.maxR) {{
                        obj.scale.setScalar(ud.r);
                        obj.material.opacity = 0.7 * (1 - ud.r / ud.maxR);
                    }} else {{
                        scene.remove(obj);
                        waveObjects.splice(i, 1);
                    }}
                }} else if (ud.type === 'wire') {{
                    ud.r += ud.speed;
                    if (ud.r < ud.maxR) {{
                        obj.scale.setScalar(ud.r);
                        obj.material.opacity = 0.9 * (1 - ud.r / ud.maxR);
                    }} else {{
                        scene.remove(obj);
                        waveObjects.splice(i, 1);
                    }}
                }} else if (ud.type === 'ring') {{
                    ud.age++;
                    const progress = ud.age / ud.maxAge;
                    const scale = 1 + progress * 10;
                    obj.scale.set(scale, scale, 1);
                    obj.material.opacity = 0.85 * (1 - progress);
                    if (ud.age >= ud.maxAge) {{
                        scene.remove(obj);
                        waveObjects.splice(i, 1);
                    }}
                }} else if (ud.type === 'rising') {{
                    if (obj.position.y < ud.targetY) {{
                        obj.position.y += ud.speed;
                        obj.scale.x *= 1.06;
                        obj.scale.z *= 1.06;
                        obj.material.opacity *= 0.97;
                    }} else {{
                        scene.remove(obj);
                        waveObjects.splice(i, 1);
                    }}
                }} else if (ud.type === 'light') {{
                    ud.age++;
                    obj.intensity = 5 * (1 - ud.age / ud.maxAge);
                    if (ud.age > ud.maxAge) {{
                        scene.remove(obj);
                        waveObjects.splice(i, 1);
                    }}
                }}
            }}
        }}

        function updateFragments() {{
            const groundY = -params.H;
            const gravity = 0.012;
            const airResist = 0.995;
            
            fragmentObjects.forEach(f => {{
                if (f.userData.grounded) return;
                
                // Aplicar velocidad
                f.position.add(f.userData.vel);
                
                // Gravedad y resistencia del aire
                f.userData.vel.y -= gravity;
                f.userData.vel.x *= airResist;
                f.userData.vel.z *= airResist;
                
                // Rotación activa
                f.rotation.x += f.userData.rotVel.x;
                f.rotation.y += f.userData.rotVel.y;
                f.rotation.z += f.userData.rotVel.z;
                
                // Colisión con el suelo
                if (f.position.y < groundY + 0.12) {{
                    f.position.y = groundY + 0.12;
                    
                    // Rebote con fricción
                    if (f.userData.vel.y < -0.01) {{
                        f.userData.vel.y *= -0.2;
                        f.userData.vel.x *= 0.6;
                        f.userData.vel.z *= 0.6;
                        f.userData.rotVel.multiplyScalar(0.5);
                    }} else {{
                        f.userData.vel.set(0, 0, 0);
                        f.userData.rotVel.multiplyScalar(0.1);
                        f.userData.grounded = true;
                    }}
                }}
            }});
        }}

        function drawEmptyCurve() {{
            const cv = document.getElementById('curve-canvas');
            const ctx = cv.getContext('2d');
            drawCurveBase(ctx, cv.width, cv.height);
            drawRosinRammler(ctx, params.X50, params.n, '#1a1a1a', 2, cv.width, cv.height);
        }}

        function drawCurveWithCI() {{
            const cv = document.getElementById('curve-canvas');
            const ctx = cv.getContext('2d');
            const W = cv.width, H = cv.height;
            const m = {{ t: 15, r: 15, b: 20, l: 28 }};
            
            drawCurveBase(ctx, W, H);
            
            if (mcResults.allSizes.length > 0) {{
                const xMax = 14;
                const toX = v => m.l + (v / xMax) * (W - m.l - m.r);
                const toY = p => m.t + (1 - p / 100) * (H - m.t - m.b);
                
                const pts = [];
                for (let s = 0.5; s <= 35; s += 0.5) {{
                    const s_in = s / 2.54;
                    if (s_in > xMax) break;
                    const pcts = mcResults.allSizes.map(sz => 100 * sz.filter(v => v <= s_in).length / sz.length);
                    pcts.sort((a,b) => a - b);
                    pts.push({{ x: s_in, p5: pcts[Math.floor(pcts.length*0.05)]||0, p95: pcts[Math.floor(pcts.length*0.95)]||100 }});
                }}
                
                ctx.fillStyle = 'rgba(33,150,243,0.2)';
                ctx.beginPath();
                pts.forEach((p,i) => i === 0 ? ctx.moveTo(toX(p.x), toY(p.p5)) : ctx.lineTo(toX(p.x), toY(p.p5)));
                for (let i = pts.length-1; i >= 0; i--) ctx.lineTo(toX(pts[i].x), toY(pts[i].p95));
                ctx.closePath();
                ctx.fill();
            }}
            
            drawRosinRammler(ctx, params.X50, params.n, '#1a1a1a', 2, W, H);
        }}

        function drawCurveBase(ctx, W, H) {{
            const m = {{ t: 15, r: 15, b: 20, l: 28 }};
            const xMax = 14;
            const toX = v => m.l + (v / xMax) * (W - m.l - m.r);
            const toY = p => m.t + (1 - p / 100) * (H - m.t - m.b);
            
            ctx.fillStyle = '#fafafa';
            ctx.fillRect(0, 0, W, H);
            
            ctx.strokeStyle = '#e0e0e0';
            ctx.lineWidth = 1;
            for (let x = 0; x <= xMax; x += 2) {{ ctx.beginPath(); ctx.moveTo(toX(x), m.t); ctx.lineTo(toX(x), H-m.b); ctx.stroke(); }}
            for (let y = 0; y <= 100; y += 20) {{ ctx.beginPath(); ctx.moveTo(m.l, toY(y)); ctx.lineTo(W-m.r, toY(y)); ctx.stroke(); }}
            
            ctx.strokeStyle = '#333';
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.moveTo(m.l, m.t); ctx.lineTo(m.l, H-m.b); ctx.lineTo(W-m.r, H-m.b);
            ctx.stroke();
            
            ctx.fillStyle = '#666';
            ctx.font = '7px Arial';
            ctx.textAlign = 'center';
            for (let x = 0; x <= xMax; x += 2) ctx.fillText(x+'"', toX(x), H-m.b+8);
            ctx.textAlign = 'right';
            for (let y = 0; y <= 100; y += 20) ctx.fillText(y+'%', m.l-2, toY(y)+2);
            
            ctx.setLineDash([2,2]);
            ctx.lineWidth = 1;
            ctx.strokeStyle = '#ff9800';
            ctx.beginPath(); ctx.moveTo(m.l, toY(80)); ctx.lineTo(W-m.r, toY(80)); ctx.stroke();
            ctx.strokeStyle = '#f44336';
            ctx.beginPath(); ctx.moveTo(m.l, toY(99)); ctx.lineTo(W-m.r, toY(99)); ctx.stroke();
            ctx.setLineDash([]);
        }}

        function drawRosinRammler(ctx, X50, n, color, lw, W, H) {{
            const m = {{ t: 15, r: 15, b: 20, l: 28 }};
            const xMax = 14;
            const toX = v => m.l + (v / xMax) * (W - m.l - m.r);
            const toY = p => m.t + (1 - p / 100) * (H - m.t - m.b);
            
            const Xc = X50 / Math.pow(0.693, 1/n);
            ctx.strokeStyle = color;
            ctx.lineWidth = lw;
            ctx.beginPath();
            for (let i = 0; i <= 200; i++) {{
                const sz = (i/200) * 35;
                const sz_in = sz / 2.54;
                if (sz_in > xMax) break;
                const P = 100 * (1 - Math.exp(-Math.pow(sz/Xc, n)));
                i === 0 ? ctx.moveTo(toX(sz_in), toY(P)) : ctx.lineTo(toX(sz_in), toY(P));
            }}
            ctx.stroke();
        }}

        function drawHistogram(canvasId, values, label, color, limit) {{
            const cv = document.getElementById(canvasId);
            const ctx = cv.getContext('2d');
            const W = cv.width, H = cv.height;
            const m = {{ t: 12, r: 4, b: 10, l: 18 }};
            
            ctx.fillStyle = '#fafafa';
            ctx.fillRect(0, 0, W, H);
            
            if (!values.length) {{
                ctx.fillStyle = '#888';
                ctx.font = '8px Arial';
                ctx.textAlign = 'center';
                ctx.fillText(label, W/2, H/2);
                return;
            }}
            
            const min = Math.min(...values), max = Math.max(...values);
            const bins = new Array(12).fill(0);
            const bw = (max - min) / 12;
            values.forEach(v => bins[Math.min(Math.floor((v-min)/bw), 11)]++);
            
            const maxC = Math.max(...bins);
            const pW = W - m.l - m.r, pH = H - m.t - m.b;
            const barW = pW / 12;
            
            bins.forEach((c, i) => {{
                const h = (c/maxC) * pH;
                const val = min + (i+0.5)*bw;
                ctx.fillStyle = val <= limit ? color : '#ffcdd2';
                ctx.fillRect(m.l + i*barW, H-m.b-h, barW-1, h);
            }});
            
            if (limit >= min && limit <= max) {{
                const lx = m.l + ((limit-min)/(max-min))*pW;
                ctx.strokeStyle = '#c62828';
                ctx.lineWidth = 1;
                ctx.setLineDash([2,1]);
                ctx.beginPath(); ctx.moveTo(lx, m.t); ctx.lineTo(lx, H-m.b); ctx.stroke();
                ctx.setLineDash([]);
            }}
            
            ctx.fillStyle = '#333';
            ctx.font = 'bold 7px Arial';
            ctx.textAlign = 'center';
            ctx.fillText(label + ': μ=' + (values.reduce((a,b)=>a+b,0)/values.length).toFixed(2) + '"', W/2, 8);
        }}

        function drawEmptyHistograms() {{
            drawHistogram('hist-p80', [], 'P80', '#2196f3', 4.5);
            drawHistogram('hist-p100', [], 'P100', '#f44336', 12);
        }}

        function updateStats() {{
            if (!mcResults.p80_values.length) return;
            const p80 = mcResults.p80_values, p100 = mcResults.p100_values;
            const mean = a => a.reduce((x,y)=>x+y,0)/a.length;
            const std = a => Math.sqrt(a.reduce((x,y)=>x+Math.pow(y-mean(a),2),0)/a.length);
            const pct = (a,p) => [...a].sort((x,y)=>x-y)[Math.floor(a.length*p)];
            const prob = p100.filter(v => v <= 12).length / p100.length * 100;
            
            document.getElementById('stats-grid').innerHTML = 
                '<div class="stat-box"><div class="stat-title">P80 Medio</div><div class="stat-value ' + (mean(p80)<=4.5?'stat-ok':'stat-bad') + '">' + mean(p80).toFixed(2) + '"</div><div class="stat-sub">±' + std(p80).toFixed(2) + '"</div></div>' +
                '<div class="stat-box"><div class="stat-title">P100 Medio</div><div class="stat-value ' + (mean(p100)<=12?'stat-ok':'stat-bad') + '">' + mean(p100).toFixed(1) + '"</div><div class="stat-sub">±' + std(p100).toFixed(2) + '"</div></div>' +
                '<div class="stat-box"><div class="stat-title">IC 90% P80</div><div class="stat-value" style="font-size:10px;">' + pct(p80,0.05).toFixed(2) + '-' + pct(p80,0.95).toFixed(2) + '"</div></div>' +
                '<div class="stat-box"><div class="stat-title">IC 90% P100</div><div class="stat-value" style="font-size:10px;">' + pct(p100,0.05).toFixed(1) + '-' + pct(p100,0.95).toFixed(1) + '"</div></div>' +
                '<div class="stat-box" style="grid-column:span 2;"><div class="stat-title">P(P100≤12")</div><div class="stat-value ' + (prob>=90?'stat-ok':prob>=70?'stat-warning':'stat-bad') + '">' + prob.toFixed(1) + '%</div></div>';
        }}

        let lastT = 0;
        function animate(t) {{
            requestAnimationFrame(animate);
            const dt = Math.min((t - lastT) / 1000, 0.1);
            lastT = t;
            updateDetonationAnimation(dt);
            renderer.render(scene, camera);
        }}

        init();
    </script>
</body>
</html>
'''
    return html_code


def generar_simulador_pilix_completo(params):
    """
    Genera el HTML del Simulador PILIX v3.6 completo con todas las características.
    
    Incluye:
    - Índice de Lilly
    - Factores Kx/Kn
    - Modelo Doble Taco con 3 APDs
    - Pozos Especiales (Buffer, Borde, Precorte)
    - Monte Carlo con análisis de sensibilidad
    - Zonas de daño (Holmberg-Persson)
    - Visualización 3D con Three.js
    """
    
    html_code = f'''
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PILIX v3.6 - Simulador Completo</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: 'Segoe UI', Arial, sans-serif; 
            background: linear-gradient(135deg, #0d1a1f 0%, #122429 100%);
            color: #e2e8f0;
            min-height: 100vh;
        }}
        
        .header {{
            background: linear-gradient(135deg, #00778B, #005566);
            padding: 12px 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 3px solid #EAAA00;
        }}
        
        .header h1 {{
            color: #fff;
            font-size: 18px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .header .badge {{
            background: #EAAA00;
            color: #000;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 10px;
            font-weight: bold;
        }}
        
        .container {{
            display: grid;
            grid-template-columns: 300px 1fr 350px;
            gap: 10px;
            padding: 10px;
            height: calc(100vh - 60px);
        }}
        
        .panel {{
            background: rgba(26, 48, 56, 0.95);
            border-radius: 8px;
            border: 1px solid #2d4a52;
            overflow: hidden;
        }}
        
        .panel-header {{
            background: linear-gradient(135deg, #00778B, #005566);
            color: #fff;
            padding: 10px 15px;
            font-weight: bold;
            font-size: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .panel-content {{
            padding: 12px;
            max-height: calc(100vh - 120px);
            overflow-y: auto;
        }}
        
        .section-title {{
            color: #64CCC9;
            font-size: 11px;
            font-weight: bold;
            margin: 12px 0 8px 0;
            padding-bottom: 4px;
            border-bottom: 1px solid #2d4a52;
        }}
        
        .input-row {{
            display: grid;
            grid-template-columns: 1fr 70px 40px;
            align-items: center;
            margin: 5px 0;
            gap: 8px;
        }}
        
        .input-row label {{
            color: #9aa0a6;
            font-size: 10px;
        }}
        
        .input-row input, .input-row select {{
            background: #1a3038;
            border: 1px solid #3d5a62;
            color: #fff;
            padding: 5px 8px;
            border-radius: 4px;
            font-size: 10px;
            text-align: right;
        }}
        
        .input-row input:focus {{
            border-color: #64CCC9;
            outline: none;
        }}
        
        .input-row .unit {{
            color: #6b7280;
            font-size: 9px;
        }}
        
        .calc-display {{
            background: linear-gradient(135deg, rgba(100, 204, 201, 0.15), rgba(0, 119, 139, 0.15));
            border: 1px solid #64CCC9;
            border-radius: 6px;
            padding: 10px;
            margin: 10px 0;
        }}
        
        .calc-display .title {{
            color: #64CCC9;
            font-size: 9px;
            margin-bottom: 5px;
        }}
        
        .calc-display .value {{
            color: #fff;
            font-size: 16px;
            font-weight: bold;
        }}
        
        .calc-display .formula {{
            color: #9aa0a6;
            font-size: 8px;
            margin-top: 3px;
            font-family: monospace;
        }}
        
        .btn {{
            padding: 8px 16px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            font-size: 11px;
            transition: all 0.2s;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }}
        
        .btn:hover {{ transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,0,0,0.3); }}
        .btn-primary {{ background: linear-gradient(135deg, #00778B, #005566); color: #fff; }}
        .btn-success {{ background: linear-gradient(135deg, #27ae60, #1e8449); color: #fff; }}
        .btn-warning {{ background: linear-gradient(135deg, #EAAA00, #d4940a); color: #000; }}
        .btn-danger {{ background: linear-gradient(135deg, #D22630, #a81d25); color: #fff; }}
        
        .btn-group {{
            display: flex;
            gap: 6px;
            margin-top: 10px;
            flex-wrap: wrap;
        }}
        
        .checkbox-row {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin: 8px 0;
            padding: 8px;
            background: rgba(100, 204, 201, 0.1);
            border-radius: 4px;
        }}
        
        .checkbox-row input[type="checkbox"] {{
            width: 16px;
            height: 16px;
            accent-color: #64CCC9;
        }}
        
        .checkbox-row label {{
            font-size: 10px;
            color: #e2e8f0;
        }}
        
        #threejs-container {{
            width: 100%;
            height: 100%;
            border-radius: 8px;
            overflow: hidden;
            background: #1a2e38;
        }}
        
        .results-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 8px;
            margin: 10px 0;
        }}
        
        .result-card {{
            background: rgba(0, 119, 139, 0.2);
            border: 1px solid #00778B;
            border-radius: 6px;
            padding: 10px;
            text-align: center;
        }}
        
        .result-card .label {{
            color: #9aa0a6;
            font-size: 9px;
            margin-bottom: 4px;
        }}
        
        .result-card .value {{
            font-size: 18px;
            font-weight: bold;
        }}
        
        .result-card .value.ok {{ color: #64CCC9; }}
        .result-card .value.warning {{ color: #EAAA00; }}
        .result-card .value.bad {{ color: #D22630; }}
        
        .result-card .sub {{
            color: #6b7280;
            font-size: 8px;
            margin-top: 2px;
        }}
        
        .progress-bar {{
            background: #1a3038;
            border-radius: 10px;
            height: 8px;
            overflow: hidden;
            margin: 10px 0;
        }}
        
        .progress-bar .fill {{
            background: linear-gradient(90deg, #64CCC9, #00778B);
            height: 100%;
            transition: width 0.3s;
        }}
        
        table.data-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 9px;
            margin: 10px 0;
        }}
        
        table.data-table th {{
            background: rgba(0, 119, 139, 0.3);
            color: #64CCC9;
            padding: 6px;
            text-align: left;
            border-bottom: 1px solid #3d5a62;
        }}
        
        table.data-table td {{
            padding: 5px 6px;
            border-bottom: 1px solid #2d4a52;
        }}
        
        table.data-table tr:hover {{
            background: rgba(100, 204, 201, 0.1);
        }}
        
        .recommendation-box {{
            background: rgba(234, 170, 0, 0.1);
            border-left: 3px solid #EAAA00;
            padding: 10px;
            margin: 10px 0;
            border-radius: 0 6px 6px 0;
        }}
        
        .recommendation-box h4 {{
            color: #EAAA00;
            font-size: 11px;
            margin-bottom: 8px;
        }}
        
        .rec-item {{
            font-size: 10px;
            padding: 5px 0;
            border-bottom: 1px solid rgba(234, 170, 0, 0.2);
        }}
        
        .rec-item:last-child {{ border-bottom: none; }}
        .rec-critical {{ color: #D22630; }}
        .rec-warning {{ color: #EAAA00; }}
        .rec-success {{ color: #64CCC9; }}
        
        .canvas-container {{
            background: #1a2e38;
            border-radius: 6px;
            padding: 5px;
            margin: 8px 0;
        }}
        
        .lilly-display {{
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 4px;
            margin: 8px 0;
        }}
        
        .lilly-item {{
            background: rgba(0, 119, 139, 0.2);
            padding: 4px;
            border-radius: 4px;
            text-align: center;
        }}
        
        .lilly-item .name {{
            font-size: 8px;
            color: #9aa0a6;
        }}
        
        .lilly-item .val {{
            font-size: 11px;
            color: #64CCC9;
            font-weight: bold;
        }}
        
        .config-loaded {{
            background: rgba(39, 174, 96, 0.2);
            border: 1px solid #27ae60;
            border-radius: 6px;
            padding: 8px 12px;
            margin-bottom: 10px;
            font-size: 10px;
            color: #27ae60;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 PILIX <span class="badge">v3.6</span> Simulador de Fragmentación Kuz-Ram</h1>
        <div style="font-size: 10px; color: #9aa0a6;">
            Modelo Calibrado Los Pelambres | ENAEX
        </div>
    </div>
    
    <div class="container">
        <!-- Panel Izquierdo: Parámetros -->
        <div class="panel">
            <div class="panel-header">⚙️ Parámetros de Entrada</div>
            <div class="panel-content">
                <div id="config-info" class="config-loaded" style="display: {{'block' if params.get('nombre_config') else 'none'}};">
                    📋 Config: <strong>{params.get('nombre_config', 'Manual')}</strong>
                </div>
                
                <div class="section-title">🪨 Propiedades de Roca</div>
                <div class="input-row">
                    <label>UCS (σc)</label>
                    <input type="number" id="input-UCS" value="{params.get('UCS', 100)}" min="20" max="300">
                    <span class="unit">MPa</span>
                </div>
                <div class="input-row">
                    <label>RQD</label>
                    <input type="number" id="input-RQD" value="{params.get('RQD', 70)}" min="0" max="100">
                    <span class="unit">%</span>
                </div>
                
                <div class="lilly-display" id="lilly-display">
                    <div class="lilly-item"><div class="name">RMD</div><div class="val" id="lilly-RMD">-</div></div>
                    <div class="lilly-item"><div class="name">JPS</div><div class="val" id="lilly-JPS">-</div></div>
                    <div class="lilly-item"><div class="name">BI</div><div class="val" id="lilly-BI">-</div></div>
                    <div class="lilly-item"><div class="name">A</div><div class="val" id="lilly-A">-</div></div>
                    <div class="lilly-item"><div class="name">σt</div><div class="val" id="lilly-st">-</div></div>
                </div>
                
                <div class="section-title">📐 Geometría de Malla</div>
                <div class="input-row">
                    <label>Burden (B)</label>
                    <input type="number" id="input-B" value="{params.get('B', 7.5)}" step="0.1" min="3" max="15">
                    <span class="unit">m</span>
                </div>
                <div class="input-row">
                    <label>Espaciamiento (S)</label>
                    <input type="number" id="input-S" value="{params.get('S', 8.6)}" step="0.1" min="3" max="18">
                    <span class="unit">m</span>
                </div>
                <div class="input-row">
                    <label>Taco (T)</label>
                    <input type="number" id="input-T" value="{params.get('T', 5.5)}" step="0.1" min="2" max="10">
                    <span class="unit">m</span>
                </div>
                <div class="input-row">
                    <label>Altura Banco (H)</label>
                    <input type="number" id="input-H" value="{params.get('H', 16.5)}" step="0.5" min="5" max="30">
                    <span class="unit">m</span>
                </div>
                <div class="input-row">
                    <label>Pasadura (J)</label>
                    <input type="number" id="input-J" value="{params.get('J', 1.5)}" step="0.1" min="0" max="5">
                    <span class="unit">m</span>
                </div>
                <div class="input-row">
                    <label>Diámetro (d)</label>
                    <input type="number" id="input-d" value="{params.get('d', 10.625)}" step="0.125" min="4" max="15">
                    <span class="unit">pulg</span>
                </div>
                
                <div class="section-title">🔧 Calibración Kx/Kn</div>
                <div class="input-row">
                    <label>Kx (corrección X50)</label>
                    <input type="number" id="input-Kx" value="{params.get('Kx', 0.5)}" step="0.01" min="0.1" max="2.0">
                    <span class="unit">-</span>
                </div>
                <div class="input-row">
                    <label>Kn (corrección n)</label>
                    <input type="number" id="input-Kn" value="{params.get('Kn', 1.02)}" step="0.01" min="0.5" max="1.5">
                    <span class="unit">-</span>
                </div>
                
                <div class="section-title">💥 Explosivo</div>
                <div class="input-row" style="grid-template-columns: 1fr 110px;">
                    <label>Tipo</label>
                    <select id="input-explosivo">
                        <option value="Energex_70" {"selected" if params.get('explosivo', '') == 'Energex 70' else ''}>Energex 70</option>
                        <option value="Energex_50">Energex 50</option>
                        <option value="Blendex_940">Blendex 940</option>
                        <option value="ANFO">ANFO</option>
                    </select>
                </div>
                
                <div class="section-title">🎲 Monte Carlo</div>
                <div class="input-row">
                    <label>Iteraciones</label>
                    <input type="number" id="input-iterations" value="500" min="100" max="5000" step="100">
                    <span class="unit">-</span>
                </div>
                <div class="input-row">
                    <label>Variabilidad</label>
                    <input type="number" id="input-variability" value="10" min="1" max="30">
                    <span class="unit">%</span>
                </div>
                
                <div class="checkbox-row">
                    <input type="checkbox" id="check-doble-taco" {"checked" if params.get('doble_taco', False) else ''}>
                    <label for="check-doble-taco">🔷 Activar Doble Taco (3 APDs)</label>
                </div>
                
                <div id="doble-taco-params" style="display: {{'block' if params.get('doble_taco', False) else 'none'}};">
                    <div class="input-row">
                        <label>Altura Taco Int. (Ti)</label>
                        <input type="number" id="input-Ti" value="{params.get('Ti', 2.0)}" step="0.1" min="0.5" max="5">
                        <span class="unit">m</span>
                    </div>
                    <div class="input-row">
                        <label>Posición Ti</label>
                        <input type="number" id="input-Ti-pos" value="{params.get('Ti_pos', 8.0)}" step="0.5" min="3" max="15">
                        <span class="unit">m</span>
                    </div>
                </div>
                
                <div class="btn-group">
                    <button class="btn btn-primary" onclick="runSimulation()">🔥 Simular</button>
                    <button class="btn btn-success" onclick="startAnimation()">▶️ Animar 3D</button>
                    <button class="btn btn-warning" onclick="resetSimulation()">🔄 Reset</button>
                </div>
            </div>
        </div>
        
        <!-- Panel Central: Visualización 3D -->
        <div class="panel">
            <div class="panel-header">🎯 Visualización 3D</div>
            <div id="threejs-container"></div>
        </div>
        
        <!-- Panel Derecho: Resultados -->
        <div class="panel">
            <div class="panel-header">📊 Resultados Monte Carlo</div>
            <div class="panel-content">
                <div class="calc-display">
                    <div class="title">X50 Calculado (Kuz-Ram + Kx)</div>
                    <div class="value" id="result-X50">-- cm</div>
                    <div class="formula">X50 = Kx × A × (V₀/Q)^0.8 × Q^(1/6)</div>
                </div>
                
                <div class="results-grid" id="results-grid">
                    <div class="result-card">
                        <div class="label">P80 Teórico</div>
                        <div class="value ok" id="result-P80-teo">--"</div>
                    </div>
                    <div class="result-card">
                        <div class="label">P100 Teórico</div>
                        <div class="value ok" id="result-P100-teo">--"</div>
                    </div>
                    <div class="result-card">
                        <div class="label">P80 Monte Carlo</div>
                        <div class="value" id="result-P80-mc">--"</div>
                        <div class="sub" id="result-P80-ic">IC 90%: --</div>
                    </div>
                    <div class="result-card">
                        <div class="label">P100 Monte Carlo</div>
                        <div class="value" id="result-P100-mc">--"</div>
                        <div class="sub" id="result-P100-ic">IC 90%: --</div>
                    </div>
                </div>
                
                <div class="progress-bar" id="progress-bar" style="display: none;">
                    <div class="fill" id="progress-fill" style="width: 0%;"></div>
                </div>
                <div id="progress-text" style="text-align: center; font-size: 10px; color: #9aa0a6;"></div>
                
                <div class="section-title">📈 Curva Granulométrica</div>
                <div class="canvas-container">
                    <canvas id="canvas-curve" width="300" height="150"></canvas>
                </div>
                
                <div class="section-title">📊 Histogramas</div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 5px;">
                    <div class="canvas-container">
                        <canvas id="canvas-hist-p80" width="145" height="80"></canvas>
                    </div>
                    <div class="canvas-container">
                        <canvas id="canvas-hist-p100" width="145" height="80"></canvas>
                    </div>
                </div>
                
                <div class="section-title">📋 Comparación Teórico vs Calibrado</div>
                <table class="data-table" id="table-comparison">
                    <tr><th>Parámetro</th><th>Teórico</th><th>Calibrado</th><th>Δ%</th></tr>
                    <tr><td>X50</td><td id="cmp-x50-teo">--</td><td id="cmp-x50-cal">--</td><td id="cmp-x50-diff">--</td></tr>
                    <tr><td>P80</td><td id="cmp-p80-teo">--</td><td id="cmp-p80-cal">--</td><td id="cmp-p80-diff">--</td></tr>
                    <tr><td>P100</td><td id="cmp-p100-teo">--</td><td id="cmp-p100-cal">--</td><td id="cmp-p100-diff">--</td></tr>
                </table>
                
                <div class="recommendation-box" id="recommendations">
                    <h4>💡 Recomendaciones</h4>
                    <div class="rec-item">Ejecute la simulación para obtener recomendaciones.</div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // =============================================
        // VARIABLES GLOBALES
        // =============================================
        let scene, camera, renderer, controls;
        let pozos = [], waveObjects = [], fragmentObjects = [];
        let isAnimating = false, simulationTime = 0;
        let mcResults = {{ p80: [], p100: [], sizes: [] }};
        
        const params = {{
            UCS: {params.get('UCS', 100)},
            RQD: {params.get('RQD', 70)},
            B: {params.get('B', 7.5)},
            S: {params.get('S', 8.6)},
            T: {params.get('T', 5.5)},
            H: {params.get('H', 16.5)},
            J: {params.get('J', 1.5)},
            d: {params.get('d', 10.625)},
            Kx: {params.get('Kx', 0.5)},
            Kn: {params.get('Kn', 1.02)},
            dobleTaco: {'true' if params.get('doble_taco', False) else 'false'},
            Ti: {params.get('Ti', 2.0)},
            TiPos: {params.get('Ti_pos', 8.0)}
        }};
        
        const theoreticalValues = {{}};
        
        // =============================================
        // ÍNDICE DE LILLY
        // =============================================
        function calculateLilly() {{
            const UCS = parseFloat(document.getElementById('input-UCS').value);
            const RQD = parseFloat(document.getElementById('input-RQD').value);
            
            const RMD = RQD > 75 ? 50 : (RQD >= 50 ? 20 : 10);
            const JPS = RQD > 80 ? 50 : (RQD >= 50 ? 20 : 10);
            const JPO = 25;
            const SG = Math.min(2.9, Math.max(2.4, 2.5 + UCS / 400));
            const SGI = 25 * SG - 50;
            const HP = 0.05 * UCS;
            const BI = 0.5 * (RMD + JPS + JPO + SGI + HP);
            const A = Math.max(3.0, Math.min(13.0, 0.12 * BI));
            const tensile = UCS / 10;
            
            document.getElementById('lilly-RMD').textContent = RMD;
            document.getElementById('lilly-JPS').textContent = JPS;
            document.getElementById('lilly-BI').textContent = BI.toFixed(1);
            document.getElementById('lilly-A').textContent = A.toFixed(2);
            document.getElementById('lilly-st').textContent = tensile.toFixed(1);
            
            return {{ RMD, JPS, JPO, SG, SGI, HP, BI, A, tensile }};
        }}
        
        // =============================================
        // CÁLCULOS PRINCIPALES
        // =============================================
        function updateCalculations() {{
            // Leer parámetros
            params.UCS = parseFloat(document.getElementById('input-UCS').value);
            params.RQD = parseFloat(document.getElementById('input-RQD').value);
            params.B = parseFloat(document.getElementById('input-B').value);
            params.S = parseFloat(document.getElementById('input-S').value);
            params.T = parseFloat(document.getElementById('input-T').value);
            params.H = parseFloat(document.getElementById('input-H').value);
            params.J = parseFloat(document.getElementById('input-J').value);
            params.d = parseFloat(document.getElementById('input-d').value);
            params.Kx = parseFloat(document.getElementById('input-Kx').value);
            params.Kn = parseFloat(document.getElementById('input-Kn').value);
            params.dobleTaco = document.getElementById('check-doble-taco').checked;
            
            // Lilly
            const lilly = calculateLilly();
            const A = 19.9; // Calibrado LP
            
            // Geometría
            const L = params.H + params.J;
            const Lc = L - params.T - params.J;
            const d_m = params.d * 0.0254;
            const area = Math.PI * Math.pow(d_m/2, 2);
            const Q = Lc * area * 1.2 * 1000; // kg
            const V0 = params.B * params.S * params.H;
            
            // Kuz-Ram con Kx
            const X50 = params.Kx * A * Math.pow(V0/Q, 0.8) * Math.pow(Q, 1/6) * Math.pow(1.15, 19/30);
            theoreticalValues.X50 = Math.max(5, Math.min(60, X50));
            
            // n de Cunningham
            const d_mm = params.d * 25.4;
            const n_calc = (2.2 - 14 * params.B / d_mm) * Math.sqrt((1 + params.S/params.B)/2) * (1 - 0.5/params.B) * (Lc/params.H);
            theoreticalValues.n = Math.max(0.5, Math.min(2.5, n_calc * params.Kn));
            
            // Rosin-Rammler
            const Xc = theoreticalValues.X50 / Math.pow(0.693, 1/theoreticalValues.n);
            theoreticalValues.Xc = Xc;
            theoreticalValues.P80 = Xc * Math.pow(-Math.log(0.20), 1/theoreticalValues.n) / 2.54;
            theoreticalValues.P100 = Xc * Math.pow(-Math.log(0.01), 1/theoreticalValues.n) / 2.54;
            
            // Factor de taco
            let f_taco = 1.0;
            if (params.B >= 11) f_taco = 1.0 - 0.30 * (params.T - 5.0);
            else if (params.B < 8) f_taco = 1.0 + 0.30 * Math.pow(params.T - 5.25, 2);
            f_taco = Math.max(0.5, Math.min(2.0, f_taco));
            
            theoreticalValues.P80 *= f_taco;
            theoreticalValues.P100 *= f_taco;
            
            // Doble taco
            if (params.dobleTaco) {{
                theoreticalValues.P100 *= 0.80;
            }}
            
            // Actualizar UI
            document.getElementById('result-X50').textContent = theoreticalValues.X50.toFixed(2) + ' cm';
            document.getElementById('result-P80-teo').textContent = theoreticalValues.P80.toFixed(2) + '"';
            document.getElementById('result-P100-teo').textContent = theoreticalValues.P100.toFixed(2) + '"';
            
            // Clases de color
            const p80El = document.getElementById('result-P80-teo');
            const p100El = document.getElementById('result-P100-teo');
            p80El.className = 'value ' + (theoreticalValues.P80 <= 4.5 ? 'ok' : theoreticalValues.P80 <= 6 ? 'warning' : 'bad');
            p100El.className = 'value ' + (theoreticalValues.P100 <= 12 ? 'ok' : theoreticalValues.P100 <= 15 ? 'warning' : 'bad');
        }}
        
        // =============================================
        // MONTE CARLO
        // =============================================
        async function runMonteCarlo() {{
            const iterations = parseInt(document.getElementById('input-iterations').value);
            const variability = parseFloat(document.getElementById('input-variability').value) / 100;
            
            document.getElementById('progress-bar').style.display = 'block';
            mcResults = {{ p80: [], p100: [], sizes: [] }};
            
            const X50_base = theoreticalValues.X50;
            const n_base = theoreticalValues.n;
            
            for (let i = 0; i < iterations; i++) {{
                const X50_var = X50_base * (1 + variability * (Math.random() - 0.5) * 2);
                const n_var = Math.max(0.5, n_base * (1 + 0.3 * variability * (Math.random() - 0.5) * 2));
                
                const Xc = X50_var / Math.pow(0.693, 1/n_var);
                const sizes = [];
                
                for (let f = 0; f < 150; f++) {{
                    const u = Math.random() * 0.998;
                    let size = Xc * Math.pow(-Math.log(1-u), 1/n_var);
                    if (params.dobleTaco && size > Xc * 2) size *= 0.80;
                    sizes.push(size);
                }}
                
                sizes.sort((a,b) => a-b);
                mcResults.p80.push(sizes[Math.floor(sizes.length * 0.8)] / 2.54);
                mcResults.p100.push(sizes[sizes.length - 1] / 2.54);
                
                if (i % 25 === 0) mcResults.sizes.push(sizes.map(s => s / 2.54));
                
                if (i % 50 === 0) {{
                    document.getElementById('progress-fill').style.width = (i/iterations*100) + '%';
                    document.getElementById('progress-text').textContent = i + '/' + iterations;
                    await new Promise(r => setTimeout(r, 1));
                }}
            }}
            
            document.getElementById('progress-fill').style.width = '100%';
            document.getElementById('progress-text').textContent = 'Completado';
            
            displayResults();
            drawCurve();
            drawHistograms();
            generateRecommendations();
            
            setTimeout(() => {{
                document.getElementById('progress-bar').style.display = 'none';
            }}, 1000);
        }}
        
        function displayResults() {{
            const mean = arr => arr.reduce((a,b) => a+b, 0) / arr.length;
            const std = arr => Math.sqrt(arr.map(x => Math.pow(x - mean(arr), 2)).reduce((a,b) => a+b, 0) / arr.length);
            const pct = (arr, p) => [...arr].sort((a,b) => a-b)[Math.floor(arr.length * p)];
            
            const p80_mean = mean(mcResults.p80);
            const p100_mean = mean(mcResults.p100);
            
            document.getElementById('result-P80-mc').textContent = p80_mean.toFixed(2) + '"';
            document.getElementById('result-P100-mc').textContent = p100_mean.toFixed(2) + '"';
            
            document.getElementById('result-P80-ic').textContent = 'IC 90%: ' + pct(mcResults.p80, 0.05).toFixed(2) + '-' + pct(mcResults.p80, 0.95).toFixed(2) + '"';
            document.getElementById('result-P100-ic').textContent = 'IC 90%: ' + pct(mcResults.p100, 0.05).toFixed(1) + '-' + pct(mcResults.p100, 0.95).toFixed(1) + '"';
            
            // Colores
            const p80mcEl = document.getElementById('result-P80-mc');
            const p100mcEl = document.getElementById('result-P100-mc');
            p80mcEl.className = 'value ' + (p80_mean <= 4.5 ? 'ok' : p80_mean <= 6 ? 'warning' : 'bad');
            p100mcEl.className = 'value ' + (p100_mean <= 12 ? 'ok' : p100_mean <= 15 ? 'warning' : 'bad');
            
            // Tabla comparación
            const diff_p80 = ((p80_mean - theoreticalValues.P80) / theoreticalValues.P80 * 100);
            const diff_p100 = ((p100_mean - theoreticalValues.P100) / theoreticalValues.P100 * 100);
            
            document.getElementById('cmp-x50-teo').textContent = theoreticalValues.X50.toFixed(2) + ' cm';
            document.getElementById('cmp-x50-cal').textContent = theoreticalValues.X50.toFixed(2) + ' cm';
            document.getElementById('cmp-x50-diff').textContent = '0%';
            document.getElementById('cmp-p80-teo').textContent = theoreticalValues.P80.toFixed(2) + '"';
            document.getElementById('cmp-p80-cal').textContent = p80_mean.toFixed(2) + '"';
            document.getElementById('cmp-p80-diff').textContent = (diff_p80 > 0 ? '+' : '') + diff_p80.toFixed(1) + '%';
            document.getElementById('cmp-p100-teo').textContent = theoreticalValues.P100.toFixed(2) + '"';
            document.getElementById('cmp-p100-cal').textContent = p100_mean.toFixed(2) + '"';
            document.getElementById('cmp-p100-diff').textContent = (diff_p100 > 0 ? '+' : '') + diff_p100.toFixed(1) + '%';
        }}
        
        function drawCurve() {{
            const canvas = document.getElementById('canvas-curve');
            const ctx = canvas.getContext('2d');
            const W = canvas.width, H = canvas.height;
            
            ctx.fillStyle = '#1a2e38';
            ctx.fillRect(0, 0, W, H);
            
            // Ejes
            ctx.strokeStyle = '#3d5a62';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(40, 10);
            ctx.lineTo(40, H - 25);
            ctx.lineTo(W - 10, H - 25);
            ctx.stroke();
            
            // Curva teórica
            ctx.strokeStyle = '#64CCC9';
            ctx.lineWidth = 2;
            ctx.beginPath();
            
            const Xc = theoreticalValues.Xc / 2.54;
            const n = theoreticalValues.n;
            
            for (let i = 0; i <= 100; i++) {{
                const x_size = i * 0.3; // 0 a 30 pulgadas
                const passing = (1 - Math.exp(-Math.pow(x_size * 2.54 / (Xc * 2.54), n))) * 100;
                
                const px = 40 + (x_size / 30) * (W - 50);
                const py = H - 25 - (passing / 100) * (H - 40);
                
                if (i === 0) ctx.moveTo(px, py);
                else ctx.lineTo(px, py);
            }}
            ctx.stroke();
            
            // Líneas P80 y P100
            ctx.setLineDash([4, 4]);
            ctx.strokeStyle = '#EAAA00';
            ctx.beginPath();
            const p80x = 40 + (theoreticalValues.P80 / 30) * (W - 50);
            ctx.moveTo(p80x, H - 25);
            ctx.lineTo(p80x, 10);
            ctx.stroke();
            
            ctx.strokeStyle = '#D22630';
            ctx.beginPath();
            const p100x = 40 + (Math.min(theoreticalValues.P100, 30) / 30) * (W - 50);
            ctx.moveTo(p100x, H - 25);
            ctx.lineTo(p100x, 10);
            ctx.stroke();
            
            ctx.setLineDash([]);
            
            // Labels
            ctx.fillStyle = '#9aa0a6';
            ctx.font = '8px Arial';
            ctx.fillText('0', 38, H - 12);
            ctx.fillText('15"', W/2 - 10, H - 12);
            ctx.fillText('30"', W - 20, H - 12);
            ctx.fillText('100%', 5, 15);
            ctx.fillText('0%', 15, H - 28);
            
            ctx.fillStyle = '#EAAA00';
            ctx.fillText('P80', p80x - 8, 20);
            ctx.fillStyle = '#D22630';
            ctx.fillText('P100', p100x - 10, 30);
        }}
        
        function drawHistograms() {{
            drawHistogram('canvas-hist-p80', mcResults.p80, 'P80', '#2196f3', 4.5);
            drawHistogram('canvas-hist-p100', mcResults.p100, 'P100', '#f44336', 12);
        }}
        
        function drawHistogram(canvasId, data, label, color, limit) {{
            const canvas = document.getElementById(canvasId);
            const ctx = canvas.getContext('2d');
            const W = canvas.width, H = canvas.height;
            
            ctx.fillStyle = '#1a2e38';
            ctx.fillRect(0, 0, W, H);
            
            if (!data.length) return;
            
            const bins = 15;
            const min = Math.min(...data);
            const max = Math.max(...data);
            const range = max - min || 1;
            const binWidth = range / bins;
            
            const counts = new Array(bins).fill(0);
            data.forEach(v => {{
                const idx = Math.min(bins - 1, Math.floor((v - min) / binWidth));
                counts[idx]++;
            }});
            
            const maxCount = Math.max(...counts);
            const barW = (W - 10) / bins;
            
            counts.forEach((c, i) => {{
                const barH = (c / maxCount) * (H - 20);
                const x = 5 + i * barW;
                const y = H - 15 - barH;
                
                ctx.fillStyle = color;
                ctx.fillRect(x, y, barW - 1, barH);
            }});
            
            // Línea límite
            const limitX = 5 + ((limit - min) / range) * (W - 10);
            if (limitX > 5 && limitX < W - 5) {{
                ctx.strokeStyle = '#27ae60';
                ctx.lineWidth = 2;
                ctx.setLineDash([3, 3]);
                ctx.beginPath();
                ctx.moveTo(limitX, 0);
                ctx.lineTo(limitX, H - 15);
                ctx.stroke();
                ctx.setLineDash([]);
            }}
            
            ctx.fillStyle = '#fff';
            ctx.font = 'bold 8px Arial';
            ctx.fillText(label, 5, 10);
        }}
        
        function generateRecommendations() {{
            const mean = arr => arr.reduce((a,b) => a+b, 0) / arr.length;
            const p80_mean = mean(mcResults.p80);
            const p100_mean = mean(mcResults.p100);
            const probP100 = mcResults.p100.filter(v => v <= 12).length / mcResults.p100.length * 100;
            
            let recs = [];
            
            if (params.UCS > 150 && p100_mean > 10 && !params.dobleTaco) {{
                recs.push({{ type: 'critical', text: '🔷 <b>Doble Taco recomendado</b>: UCS=' + params.UCS + 'MPa. Reduciría P100 ~20%.' }});
            }}
            
            if (params.dobleTaco) {{
                recs.push({{ type: 'success', text: '✓ <b>Doble Taco activo</b>: Reducción P100 aplicada.' }});
            }}
            
            if (probP100 < 80) {{
                recs.push({{ type: 'critical', text: '⛔ <b>Riesgo alto</b>: Solo ' + probP100.toFixed(0) + '% cumple P100≤12".' }});
            }} else if (probP100 >= 90) {{
                recs.push({{ type: 'success', text: '✓ <b>Fragmentación OK</b>: ' + probP100.toFixed(0) + '% cumple P100≤12".' }});
            }}
            
            if (theoreticalValues.n < 1.0) {{
                recs.push({{ type: 'warning', text: '⚠️ <b>Uniformidad baja</b> (n=' + theoreticalValues.n.toFixed(2) + '): Distribución amplia.' }});
            }}
            
            if (params.Kx < 0.4) {{
                recs.push({{ type: 'success', text: '📊 <b>Kx=' + params.Kx.toFixed(2) + '</b>: Fragmentación más fina que modelo base.' }});
            }}
            
            document.getElementById('recommendations').innerHTML = 
                '<h4>💡 Recomendaciones PILIX v3.6</h4>' +
                recs.map(r => '<div class="rec-item rec-' + r.type + '">' + r.text + '</div>').join('');
        }}
        
        // =============================================
        // THREE.JS - VISUALIZACIÓN 3D
        // =============================================
        function initThreeJS() {{
            const container = document.getElementById('threejs-container');
            const W = container.clientWidth;
            const H = container.clientHeight;
            
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x1a2e38);
            
            camera = new THREE.PerspectiveCamera(45, W/H, 0.1, 1000);
            camera.position.set(30, 25, 40);
            camera.lookAt(0, 0, 0);
            
            renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(W, H);
            renderer.shadowMap.enabled = true;
            container.appendChild(renderer.domElement);
            
            // Luces
            const ambient = new THREE.AmbientLight(0x404040, 0.5);
            scene.add(ambient);
            
            const sun = new THREE.DirectionalLight(0xffffff, 0.8);
            sun.position.set(50, 50, 30);
            sun.castShadow = true;
            scene.add(sun);
            
            // Controles simples
            let isDragging = false;
            let prevX = 0, prevY = 0;
            
            container.addEventListener('mousedown', e => {{
                isDragging = true;
                prevX = e.clientX;
                prevY = e.clientY;
            }});
            
            container.addEventListener('mousemove', e => {{
                if (!isDragging) return;
                const dx = e.clientX - prevX;
                const dy = e.clientY - prevY;
                
                camera.position.x += dx * 0.1;
                camera.position.y -= dy * 0.1;
                camera.lookAt(0, 0, 0);
                
                prevX = e.clientX;
                prevY = e.clientY;
            }});
            
            container.addEventListener('mouseup', () => isDragging = false);
            container.addEventListener('mouseleave', () => isDragging = false);
            
            container.addEventListener('wheel', e => {{
                const delta = e.deltaY > 0 ? 1.1 : 0.9;
                camera.position.multiplyScalar(delta);
            }});
            
            createScene3D();
            animate3D();
        }}
        
        function createScene3D() {{
            // Limpiar escena
            while(scene.children.length > 2) scene.remove(scene.children[scene.children.length - 1]);
            pozos = [];
            
            const {{ B, S, T, H, J }} = params;
            const ROWS = 3, COLS = 4;
            
            // Terreno
            const floor = new THREE.Mesh(
                new THREE.PlaneGeometry(60, 50),
                new THREE.MeshLambertMaterial({{ color: 0x5d4d3d }})
            );
            floor.rotation.x = -Math.PI / 2;
            floor.position.y = -H;
            floor.receiveShadow = true;
            scene.add(floor);
            
            // Superficie
            const surface = new THREE.Mesh(
                new THREE.PlaneGeometry(S * (COLS + 1), B * (ROWS + 1)),
                new THREE.MeshLambertMaterial({{ color: 0x8d7b68 }})
            );
            surface.rotation.x = -Math.PI / 2;
            surface.position.set(S * COLS / 2, 0.01, B * ROWS / 2);
            scene.add(surface);
            
            // Banco
            const banco = new THREE.Mesh(
                new THREE.BoxGeometry(S * COLS, H, B * ROWS),
                new THREE.MeshLambertMaterial({{ color: 0x6d5d4d, transparent: true, opacity: 0.3 }})
            );
            banco.position.set(S * COLS / 2, -H/2, B * ROWS / 2);
            scene.add(banco);
            
            // Pozos
            for (let row = 0; row < ROWS; row++) {{
                for (let col = 0; col < COLS; col++) {{
                    createPozo3D(col * S + S/2, row * B + B/2, col, row);
                }}
            }}
        }}
        
        function createPozo3D(x, z, col, row) {{
            const {{ T, H, J, d }} = params;
            const L = H + J;
            const Lc = L - T - J;
            const visualD = 0.5;
            
            const group = new THREE.Group();
            group.position.set(x, 0, z);
            
            // Taco
            const taco = new THREE.Mesh(
                new THREE.CylinderGeometry(visualD, visualD, T, 16),
                new THREE.MeshPhongMaterial({{ color: 0x00e5ff, emissive: 0x006688 }})
            );
            taco.position.y = -T/2;
            group.add(taco);
            
            // Carga
            const carga = new THREE.Mesh(
                new THREE.CylinderGeometry(visualD * 0.8, visualD * 0.8, Lc, 16),
                new THREE.MeshPhongMaterial({{ color: 0xff1744, emissive: 0x880000 }})
            );
            carga.position.y = -T - Lc/2;
            group.add(carga);
            
            // Pasadura
            const pasadura = new THREE.Mesh(
                new THREE.CylinderGeometry(visualD * 0.6, visualD * 0.6, J, 16),
                new THREE.MeshPhongMaterial({{ color: 0xffeb3b }})
            );
            pasadura.position.y = -T - Lc - J/2;
            group.add(pasadura);
            
            scene.add(group);
            pozos.push({{ group, detonated: false, time: col * 30 + row * 70 }});
        }}
        
        function animate3D() {{
            requestAnimationFrame(animate3D);
            renderer.render(scene, camera);
        }}
        
        // =============================================
        // FUNCIONES AUXILIARES
        // =============================================
        function runSimulation() {{
            updateCalculations();
            runMonteCarlo();
            createScene3D();
        }}
        
        function startAnimation() {{
            isAnimating = true;
            simulationTime = 0;
            // Implementar animación de detonación
        }}
        
        function resetSimulation() {{
            mcResults = {{ p80: [], p100: [], sizes: [] }};
            document.getElementById('result-P80-mc').textContent = '--"';
            document.getElementById('result-P100-mc').textContent = '--"';
            document.getElementById('result-P80-ic').textContent = 'IC 90%: --';
            document.getElementById('result-P100-ic').textContent = 'IC 90%: --';
            document.getElementById('progress-fill').style.width = '0%';
            document.getElementById('progress-text').textContent = '';
            createScene3D();
        }}
        
        // Toggle doble taco
        document.getElementById('check-doble-taco').addEventListener('change', function() {{
            document.getElementById('doble-taco-params').style.display = this.checked ? 'block' : 'none';
        }});
        
        // Inicializar
        window.onload = function() {{
            initThreeJS();
            updateCalculations();
        }};
        
        // Actualizar en cambio de inputs
        document.querySelectorAll('input, select').forEach(el => {{
            el.addEventListener('change', updateCalculations);
        }});
    </script>
</body>
</html>
'''
    
    return html_code


# Determinar si usar doble taco (recomendado para roca dura y P100 alto)
usar_doble_taco_default = ucs_input > 150 and params_multiobj.get('P100_estimado', 10) > 10

# Generar el HTML del simulador con los parámetros de la app
simulador_mc_html = generar_simulador_montecarlo_html(
    ucs=ucs_input,
    burden=params_multiobj['burden_optimo'],
    espaciamiento=params_multiobj['espaciamiento_optimo'],
    taco=params_multiobj['taco_optimo'],
    altura=altura_banco_est,
    pasadura=1.5,
    diametro_pulg=diametro_input,
    x50_estimado=params_multiobj.get('X50_estimado', 12.0),
    n_uniformidad=1.15,
    tp_ms=params_multiobj.get('timing_pozos', 33.75),
    tf_ms=params_multiobj.get('timing_filas', 74.75),
    usar_doble_taco=usar_doble_taco_default
)

# Renderizar el simulador Monte Carlo
components.html(simulador_mc_html, height=680, scrolling=False)

# Explicación del simulador Monte Carlo
with st.expander("📖 Guía del Simulador Monte Carlo", expanded=False):
    st.markdown("""
    ### ¿Qué es la Simulación Monte Carlo?
    
    La simulación Monte Carlo genera **miles de escenarios** variando aleatoriamente los parámetros 
    dentro de un rango de incertidumbre (σ). Esto permite:
    
    - **Estimar la distribución** de P80 y P100 (no solo un valor puntual)
    - **Calcular intervalos de confianza** (IC 90%)
    - **Evaluar probabilidades** de cumplir especificaciones (ej: P(P100 ≤ 12"))
    
    ### Efecto del UCS
    
    El UCS (Resistencia a Compresión Uniaxial) afecta directamente la fragmentación:
    
    | UCS (MPa) | Tipo de Roca | Efecto en X50 |
    |-----------|--------------|---------------|
    | < 60 | Blanda | X50 bajo → fragmentación fina |
    | 60-120 | Media | X50 moderado |
    | 120-180 | Dura | X50 alto → fragmentación gruesa |
    | > 180 | Muy dura | X50 muy alto → sobretamaños |
    
    ### Efecto del Doble Taco
    
    El **taco intermedio** divide la columna explosiva, generando:
    - ✅ Reducción de P100 en ~20% (menos sobretamaños)
    - ✅ Mejor distribución de energía en el banco
    - ⚠️ Recomendado cuando UCS > 150 MPa
    
    ### Modelo Kuz-Ram Calibrado (Los Pelambres)
    
    ```
    X50 = 0.038 × UCS^0.5 × B^0.75    [cm]
    P(x) = 100 × [1 - exp(-(x/Xc)^n)]
    Xc = X50 / 0.693^(1/n)
    ```
    
    ### Controles
    
    - 🖱️ **Arrastrar**: Rotar vista 3D
    - 🔄 **Scroll**: Zoom
    - 🔥 **Simular MC**: Ejecuta Monte Carlo + detonación visual
    - ☑️ **Doble Taco**: Activa taco intermedio
    """)


# ===== ANÁLISIS Y SIMULACIÓN DE TACO INTERMEDIO =====
st.markdown("---")
st.markdown("### 📏 Análisis y Simulación de Taco Intermedio")

# Información del tipo de malla
B_opt = params_multiobj['burden_optimo']
tipo_malla_info = params_multiobj.get('tipo_malla', 'INTERMEDIA')
necesita_taco_int = params_multiobj.get('necesita_taco_intermedio', False)
taco_optimo_calc = params_multiobj.get('taco_optimo', 5.25)

# Cards informativas
col_taco1, col_taco2, col_taco3 = st.columns(3)

with col_taco1:
    st.markdown(f"""
    <div style="background-color:#e1f5fe; padding:12px; border-radius:8px; border-left:4px solid #0288d1;">
    <h5 style="margin:0; color:#01579b;">📐 Tipo de Malla</h5>
    <h3 style="margin:5px 0; color:#0277bd;">{tipo_malla_info}</h3>
    <p style="margin:0; font-size:12px; color:#0288d1;">
    B = {B_opt} m<br>
    {'B < 8m → Malla cerrada' if B_opt < 8 else ('B ≥ 11m → Malla abierta' if B_opt >= 11 else '8m ≤ B < 11m')}
    </p>
    </div>
    """, unsafe_allow_html=True)

with col_taco2:
    st.markdown(f"""
    <div style="background-color:#{'b3e5fc' if necesita_taco_int else 'e8f5e9'}; padding:12px; border-radius:8px; border-left:4px solid #{'0288d1' if necesita_taco_int else '4caf50'};">
    <h5 style="margin:0; color:#{'01579b' if necesita_taco_int else '2e7d32'};">🎯 Taco Óptimo</h5>
    <h3 style="margin:5px 0; color:#{'0277bd' if necesita_taco_int else '388e3c'};">{taco_optimo_calc:.2f} m</h3>
    <p style="margin:0; font-size:12px; color:#{'0288d1' if necesita_taco_int else '43a047'};">
    {'✅ Taco intermedio recomendado' if necesita_taco_int else '✓ Taco estándar suficiente'}<br>
    f_taco = {params_multiobj.get('f_taco', 1.0):.2f}
    </p>
    </div>
    """, unsafe_allow_html=True)

with col_taco3:
    carga_kg = params_multiobj.get('carga_pozo_kg', 0)
    st.markdown(f"""
    <div style="background-color:#e1f5fe; padding:12px; border-radius:8px; border-left:4px solid #0288d1;">
    <h5 style="margin:0; color:#01579b;">💥 Carga por Pozo</h5>
    <h3 style="margin:5px 0; color:#0277bd;">{carga_kg:.0f} kg</h3>
    <p style="margin:0; font-size:12px; color:#0288d1;">
    Q = 68.7 × (H - T)<br>
    Q = 68.7 × ({altura_banco_est:.1f} - {taco_optimo_calc:.2f})
    </p>
    </div>
    """, unsafe_allow_html=True)

# Tarjeta adicional: Timing óptimo
st.markdown("")  # Espacio
col_timing1, col_timing2, col_timing3 = st.columns(3)

tp_opt = params_multiobj.get('timing_pozos', 33.75)
tf_opt = params_multiobj.get('timing_filas', 74.75)
Th_usado = params_multiobj.get('Th_usado', 4.5)
f_timing = params_multiobj.get('f_timing', 1.0)

with col_timing1:
    st.markdown(f"""
    <div style="background-color:#e3f2fd; padding:12px; border-radius:8px; border-left:4px solid #1976d2;">
    <h5 style="margin:0; color:#1565c0;">⏱️ Timing Pozos (tp)</h5>
    <h3 style="margin:5px 0; color:#1976d2;">{tp_opt:.1f} ms</h3>
    <p style="margin:0; font-size:12px; color:#1565c0;">
    tp = Th × S = {Th_usado} × {params_multiobj.get('espaciamiento_optimo', 7.5)}<br>
    Th = {Th_usado} ms/m (según UCS)
    </p>
    </div>
    """, unsafe_allow_html=True)

with col_timing2:
    st.markdown(f"""
    <div style="background-color:#e3f2fd; padding:12px; border-radius:8px; border-left:4px solid #1976d2;">
    <h5 style="margin:0; color:#1565c0;">⏱️ Timing Filas (tf)</h5>
    <h3 style="margin:5px 0; color:#1976d2;">{tf_opt:.1f} ms</h3>
    <p style="margin:0; font-size:12px; color:#1565c0;">
    tf = 11.5 × B = 11.5 × {B_opt}<br>
    Konya timing óptimo
    </p>
    </div>
    """, unsafe_allow_html=True)

with col_timing3:
    timing_status = "✅ Óptimo" if f_timing <= 1.02 else ("⚠️ Aceptable" if f_timing <= 1.1 else "⚠️ Revisar")
    st.markdown(f"""
    <div style="background-color:#{'e8f5e9' if f_timing <= 1.02 else 'fff3e0'}; padding:12px; border-radius:8px; border-left:4px solid #{'4caf50' if f_timing <= 1.02 else 'ff9800'};">
    <h5 style="margin:0; color:#{'2e7d32' if f_timing <= 1.02 else 'e65100'};">📊 Factor Timing</h5>
    <h3 style="margin:5px 0; color:#{'388e3c' if f_timing <= 1.02 else 'ef6c00'};">{f_timing:.2f}</h3>
    <p style="margin:0; font-size:12px; color:#{'43a047' if f_timing <= 1.02 else 'f57c00'};">
    {timing_status}<br>
    f_timing = 1.0 + penalizaciones
    </p>
    </div>
    """, unsafe_allow_html=True)

# Simulación de tacos (si hay datos de simulación)
if 'simulacion_tacos' in params_multiobj and len(params_multiobj['simulacion_tacos']) > 0:
    with st.expander("🔬 Ver simulación completa de tacos", expanded=False):
        st.markdown("""
        **Simulación del efecto del taco en la fragmentación**
        
        Esta tabla muestra cómo varía la fragmentación (P80, P100) según el valor del taco.
        El taco óptimo minimiza la función objetivo y cumple con los límites de fragmentación.
        """)
        
        # Crear DataFrame de simulación
        df_sim = pd.DataFrame(params_multiobj['simulacion_tacos'])
        
        # Formatear columnas
        df_sim['taco'] = df_sim['taco'].apply(lambda x: f"{x:.2f} m")
        df_sim['f_taco'] = df_sim['f_taco'].apply(lambda x: f"{x:.2f}")
        if 'f_timing' in df_sim.columns:
            df_sim['f_timing'] = df_sim['f_timing'].apply(lambda x: f"{x:.2f}")
        df_sim['Q_kg'] = df_sim['Q_kg'].apply(lambda x: f"{x:.0f}")
        df_sim['P80'] = df_sim['P80'].apply(lambda x: f'{x:.2f}"')
        df_sim['P100'] = df_sim['P100'].apply(lambda x: f'{x:.2f}"')
        if 'tp_opt' in df_sim.columns:
            df_sim['tp_opt'] = df_sim['tp_opt'].apply(lambda x: f"{x:.1f}")
        if 'tf_opt' in df_sim.columns:
            df_sim['tf_opt'] = df_sim['tf_opt'].apply(lambda x: f"{x:.1f}")
        df_sim['cumple'] = df_sim['cumple'].apply(lambda x: '✅' if x else '❌')
        
        # Renombrar columnas
        df_sim = df_sim.rename(columns={
            'taco': 'Taco',
            'f_taco': 'f_taco',
            'f_timing': 'f_timing',
            'Q_kg': 'Carga (kg)',
            'P80': 'P80',
            'P100': 'P100',
            'tp_opt': 'tp (ms)',
            'tf_opt': 'tf (ms)',
            'J': 'f_objetivo',
            'cumple': 'Cumple'
        })
        
        # Mostrar solo columnas relevantes
        cols_show = ['Taco', 'f_taco', 'f_timing', 'Carga (kg)', 'tp (ms)', 'tf (ms)', 'P80', 'P100', 'Cumple']
        cols_show = [c for c in cols_show if c in df_sim.columns]
        
        st.dataframe(df_sim[cols_show], use_container_width=True, hide_index=True)
        
        # Explicación de fórmulas
        st.markdown(f"""
        ---
        **Fórmulas del Factor de Taco (calibradas Los Pelambres):**
        
        | Tipo Malla | Condición | Fórmula |
        |------------|-----------|---------|
        | CERRADA | B < 8m | f = 1.0 + 0.30×(T-5.25)² |
        | ABIERTA | B ≥ 11m | f = 1.0 - 0.30×(T-5.0) |
        
        **Tu malla:** B = {B_opt}m → **{tipo_malla_info}**
        
        {'🎯 **El taco óptimo para mallas cerradas es ~5.25m**, donde el factor de taco es mínimo (f=1.0).' if B_opt < 8 else ''}
        """)

# Mostrar recomendación según tipo de malla
if tipo_malla_info == "CERRADA" or necesita_taco_int:
    st.info(f"""
    ✅ **Se recomienda taco intermedio de {taco_optimo_calc:.2f}m** para esta malla cerrada.
    
    **Beneficios del taco óptimo (5.0-5.5m):**
    - Reduce P100 en ~50% comparado con taco alto (6.5m)
    - Factor de taco mínimo (f≈1.0) → mejor fragmentación
    - Datos validados con 87 registros reales: P80=4.03", P100=8.6"
    
    **Comparación con taco alto:**
    | Taco | P80 real | P100 real | Resultado |
    |------|----------|-----------|-----------|
    | 5.0-5.5m | 4.03" | 8.6" | ✅ Óptimo |
    | 6.0-6.5m | 5.50" | 17.1" | ❌ Empeora 50% |
    """)
else:
    st.success(f"""
    ✓ **Taco estándar de {taco_optimo_calc:.2f}m es adecuado** para esta malla.
    
    Con malla {'abierta' if tipo_malla_info == 'ABIERTA' else 'intermedia'} (B={B_opt}m), 
    el taco calculado como 0.85×B = {0.85*B_opt:.2f}m es apropiado.
    """)

# Mostrar histórico de tacos intermedios si existen
if 'taco_intermedio' in df_filtrado.columns:
    df_con_taco_int = df_filtrado[df_filtrado['taco_intermedio'].notna() & (df_filtrado['taco_intermedio'] > 0)]
    if len(df_con_taco_int) > 0:
        st.markdown("**📊 Tacos intermedios usados históricamente:**")
        taco_int_stats = df_con_taco_int['taco_intermedio'].describe()
        col_ti1, col_ti2, col_ti3 = st.columns(3)
        with col_ti1:
            st.metric("Mínimo usado", f"{taco_int_stats['min']:.2f} m")
        with col_ti2:
            st.metric("Promedio", f"{taco_int_stats['mean']:.2f} m")
        with col_ti3:
            st.metric("Máximo usado", f"{taco_int_stats['max']:.2f} m")

# ===== EXPLOSIVO RECOMENDADO =====
st.markdown("---")
st.markdown(f"""
### 💥 Explosivo Recomendado

Para **UCS = {ucs_input} MPa** se recomienda: **`{params_teoricos['explosivo_recomendado']}`**

| Rango UCS | Explosivos recomendados |
|-----------|------------------------|
| < 50 MPa | ANFO, Blendex 920-930 |
| 50-80 MPa | Blendex 940-950, Vertex ALR |
| 80-120 MPa | Emultex BN, Energex 50 |
| 120-180 MPa | Energex 70, Pirex S |
| > 180 MPa | Pirex S Plus, Energex 70 Plus |
""")

# ===== FÓRMULAS (COLAPSADAS) =====
with st.expander("📚 Ver fórmulas y explicaciones - Modelo PILIX v3.6", expanded=False):
    st.markdown(f"""
    ## 📘 Modelo PILIX v3.6 - Simulador de Fragmentación Kuz-Ram
    
    **Integra:** Índice de Lilly, Calibración Kx/Kn, Cunningham, Rosin-Rammler, Monte Carlo
    
    ---
    
    ### 1. ÍNDICE DE LILLY (Factor de Roca A)
    
    **Basado en:** Lilly, P.A. (1986) - "An Empirical Method of Assessing Rock Mass Blastability"
    
    ```
    SGI = 25 × SG - 50          (Influencia densidad)
    HP = 0.05 × UCS             (Hardness Parameter)
    BI = 0.5 × (RMD + JPS + JPO + SGI + HP)
    A = 0.12 × BI               (Factor de roca, típico 3-13)
    ```
    
    | Parámetro | Significado | Rango típico |
    |-----------|-------------|--------------|
    | RMD | Rock Mass Description (desde RQD) | 10-50 |
    | JPS | Joint Plane Spacing | 10-50 |
    | JPO | Joint Plane Orientation | 10-40 |
    | SGI | Specific Gravity Influence | 10-22.5 |
    | HP | Hardness Parameter | 2.5-10 |
    
    ---
    
    ### 2. FACTORES DE CALIBRACIÓN Kx/Kn (Mundaca)
    
    **Referencia:** Mundaca et al. (2015) - "Calibración modelo Kuz-Ram en UGT-4 Chuquicamata"
    
    ```
    Kx ≈ 0.25 + 0.75 × (UCS/200) × (RQD/100)    [Rango: 0.2-1.5]
    Kn ≈ 0.95 + 0.10 × (RQD/100)                [Rango: 0.85-1.15]
    ```
    
    | Factor | Interpretación |
    |--------|----------------|
    | **Kx < 1** | Fragmentación real más fina que predicción teórica |
    | **Kx > 1** | Fragmentación real más gruesa que predicción teórica |
    | **Kn < 1** | Distribución menos uniforme |
    | **Kn > 1** | Distribución más uniforme |
    
    **Referencia UGT-4 Chuquicamata:** Kx=0.31, Kn=1.04 (calibrado con Split)
    
    ---
    
    ### 3. MODELO KUZ-RAM CON Kx (PILIX v3.6)
    
    **Fórmula completa:**
    ```
    X50 = Kx × A × (V₀/Q)^0.8 × Q^(1/6) × (1.15/Er)^(19/30)   [cm]
    ```
    
    Donde:
    - **Kx** = Factor de calibración Mundaca
    - **A** = Factor de roca (Lilly o calibrado 19.9 para Los Pelambres)
    - **V₀** = Volumen de roca por pozo (B × S × H) [m³]
    - **Q** = Masa de explosivo por pozo [kg]
    - **Er** = RWS/100 (Relative Weight Strength, ANFO=1.0)
    
    ---
    
    ### 4. ÍNDICE DE UNIFORMIDAD n (Cunningham)
    
    **IMPORTANTE:** El diámetro debe estar en MILÍMETROS
    
    ```
    n = (2.2 - 14×B/D_mm) × √[(1+S/B)/2] × (1-W/B) × (Lc/H) × Kn
    ```
    
    Donde:
    - D_mm = Diámetro en milímetros (ej: 10.625" = 270mm)
    - W = Desviación estándar de perforación (~0.5m)
    - Lc = Longitud de carga [m]
    
    ---
    
    ### 5. DISTRIBUCIÓN ROSIN-RAMMLER
    
    **Característica constante Xc:**
    ```
    Xc = X50 / (ln2)^(1/n)
    ```
    
    **Percentiles:**
    ```
    P(x) = 1 - exp[-(x/Xc)^n]
    Px = Xc × [-ln(1-P/100)]^(1/n)
    ```
    
    | Percentil | Fórmula | Significado |
    |-----------|---------|-------------|
    | P10 | Xc × [-ln(0.90)]^(1/n) | 10% más fino |
    | P50 (X50) | - | Tamaño medio |
    | P80 | Xc × [-ln(0.20)]^(1/n) | Objetivo chancado |
    | P100 | Xc × [-ln(0.01)]^(1/n) | Sobretamaño máximo |
    
    ---
    
    ### 6. MODELO DOBLE TACO (3 APDs)
    
    **Factor de distribución η:**
    ```
    balance = 1 - |Lc1 - Lc2| / (Lc1 + Lc2)
    η_dist = 1 + α × balance    [α ≈ 0.10-0.15]
    ```
    
    **Índice n efectivo:**
    ```
    n_eff = n × Kn × [1 + β×(η-1)]    [β = 0.5]
    ```
    
    **Reducción P100:**
    ```
    P100_reducido = P100_base × (1 - γ × Ti/(T+Ti))    [γ = 0.20]
    ```
    
    **Beneficio:** El doble taco reduce típicamente el P100 en 15-25%
    
    ---
    
    ### 7. ZONAS DE DAÑO (Holmberg-Persson)
    
    ```
    Pd = ρe × VOD² / 4            [GPa] - Presión de detonación
    rc = r₀ × (Pd/σc)^0.6         [m] - Radio de trituración
    rf = r₀ × (Pd/σt)^0.45        [m] - Radio de fractura
    rd = rf × 2.0                 [m] - Radio de daño
    ```
    
    ---
    
    ### 8. FUNCIÓN OBJETIVO PILIX v3.6
    
    **Minimización multi-objetivo:**
    ```
    min f = w₁×(Metros/1500) + w₂×(P80/4.5) + w₃×(P100/12) + w₄×(1-n/1.5)
    ```
    
    | Componente | Peso | Normalización | Objetivo |
    |------------|------|---------------|----------|
    | Metros perforados | 0.30 | /1500 m/ha | Minimizar perforación |
    | P80 | 0.20 | /4.5" | Fragmentación operativa |
    | P100 | 0.35 | /12.0" | Minimizar sobretamaño |
    | Uniformidad (n) | 0.15 | /1.5 | Mayor n = mejor distribución |
    
    ---
    
    ### 9. PARÁMETROS DE MALLA (ENAEX/Ash)
    
    **Burden (Ash modificada):**
    ```
    B = (Kb × De × (ρe/ρr)^0.33 × (VOD/4000)^0.5) / 39.37
    B = ({params_teoricos['Kb']} × {diametro_input} × ({densidad_exp}/2.65)^0.33 × ({vod_exp}/4000)^0.5) / 39.37
    B = {params_teoricos['burden_optimo']} m
    ```
    
    **Timing (Konya):**
    ```
    Timing pozos = Th × S = {params_teoricos['Th']} × {params_teoricos['espaciamiento_optimo']} = {params_teoricos['timing_pozos_optimo']} ms
    Timing filas = 11.5 × B = 11.5 × {params_teoricos['burden_optimo']} = {params_teoricos['timing_filas_optimo']} ms
    ```
    
    ---
    
    ### 10. CONSTANTES SEGÚN DUREZA DE ROCA
    
    | UCS (MPa) | Tipo | Kb | Ks (S/B) | Th (ms/m) | FC óptimo |
    |-----------|------|-----|----------|-----------|-----------|
    | < 50 | Blanda | 35 | 1.40 | 6.5 | 0.35 |
    | 50-100 | Media | 30 | 1.30 | 5.5 | 0.50 |
    | 100-150 | Dura | 28 | 1.20 | 4.5 | 0.75 |
    | > 150 | Muy dura | 25 | 1.15 | 3.5 | 1.00 |
    
    ---
    
    ### Referencias Bibliográficas
    
    - **Cunningham, C.V.B.** (1983). "The Kuz-Ram Model for Prediction of Fragmentation from Blasting". ISEE.
    - **Lilly, P.A.** (1986). "An Empirical Method of Assessing Rock Mass Blastability". AusIMM.
    - **Holmberg, R. & Persson, P.A.** (1980). "Design of Tunnel Perimeter Blasthole Patterns".
    - **Mundaca et al.** (2015). "Calibración modelo Kuz-Ram en UGT-4 Chuquicamata".
    - **ENAEX** (2020). "Manual Técnico de Explosivos".
    - **Konya, C.J.** (1995). "Blast Design".
    - **Manual de Diseño P&T Los Pelambres** (Sep-Nov 2025).
    - **Datos calibración:** Los Pelambres, 12,456 registros
    """)

# ===== ANÁLISIS DE SENSIBILIDAD =====
with st.expander("📈 Análisis de Sensibilidad (¿Qué variable influye más?)", expanded=False):
    
    st.markdown("""
    Este análisis muestra **cuánto cambia el P80 estimado** al variar cada parámetro ±20%.
    Las barras más largas indican las variables más influyentes para tu configuración actual.
    """)
    
    # Parámetros base
    B_base = params_multiobj['burden_optimo']
    S_base = params_multiobj['espaciamiento_optimo']
    FC_base = params_multiobj['fc_ajustado']
    
    # Función para calcular P80 con parámetros específicos
    def calcular_p80_sensibilidad(burden, espaciamiento, fc, ucs_val, vod_val):
        area = burden * espaciamiento
        Area_ref = 20.0
        FC_ref = 0.75
        VOD_ref = 4500
        K_base = 5.5
        
        f_malla = (area / Area_ref) ** 0.40
        f_roca = (ucs_val / 100) ** 0.35
        f_energia = (FC_ref / max(fc, 0.3)) ** 0.30
        f_vod = (VOD_ref / max(vod_val, 3000)) ** 0.15
        
        P80 = K_base * f_malla * f_roca * f_energia * f_vod
        return max(4.0, min(15.0, P80))
    
    # P80 base
    p80_base = calcular_p80_sensibilidad(B_base, S_base, FC_base, ucs_input, vod_exp)
    
    # Calcular sensibilidad de cada variable
    variacion = 0.20  # ±20%
    
    sensibilidades = {}
    
    # Sensibilidad a Burden
    p80_burden_up = calcular_p80_sensibilidad(B_base * (1 + variacion), S_base, FC_base, ucs_input, vod_exp)
    p80_burden_down = calcular_p80_sensibilidad(B_base * (1 - variacion), S_base, FC_base, ucs_input, vod_exp)
    sensibilidades['Burden'] = {
        'cambio_up': ((p80_burden_up - p80_base) / p80_base) * 100,
        'cambio_down': ((p80_burden_down - p80_base) / p80_base) * 100,
        'rango': abs(p80_burden_up - p80_burden_down)
    }
    
    # Sensibilidad a Espaciamiento
    p80_esp_up = calcular_p80_sensibilidad(B_base, S_base * (1 + variacion), FC_base, ucs_input, vod_exp)
    p80_esp_down = calcular_p80_sensibilidad(B_base, S_base * (1 - variacion), FC_base, ucs_input, vod_exp)
    sensibilidades['Espaciamiento'] = {
        'cambio_up': ((p80_esp_up - p80_base) / p80_base) * 100,
        'cambio_down': ((p80_esp_down - p80_base) / p80_base) * 100,
        'rango': abs(p80_esp_up - p80_esp_down)
    }
    
    # Sensibilidad a Factor de Carga
    p80_fc_up = calcular_p80_sensibilidad(B_base, S_base, FC_base * (1 + variacion), ucs_input, vod_exp)
    p80_fc_down = calcular_p80_sensibilidad(B_base, S_base, FC_base * (1 - variacion), ucs_input, vod_exp)
    sensibilidades['Factor de Carga'] = {
        'cambio_up': ((p80_fc_up - p80_base) / p80_base) * 100,
        'cambio_down': ((p80_fc_down - p80_base) / p80_base) * 100,
        'rango': abs(p80_fc_up - p80_fc_down)
    }
    
    # Sensibilidad a UCS
    p80_ucs_up = calcular_p80_sensibilidad(B_base, S_base, FC_base, ucs_input * (1 + variacion), vod_exp)
    p80_ucs_down = calcular_p80_sensibilidad(B_base, S_base, FC_base, ucs_input * (1 - variacion), vod_exp)
    sensibilidades['UCS (dureza)'] = {
        'cambio_up': ((p80_ucs_up - p80_base) / p80_base) * 100,
        'cambio_down': ((p80_ucs_down - p80_base) / p80_base) * 100,
        'rango': abs(p80_ucs_up - p80_ucs_down)
    }
    
    # Sensibilidad a VOD
    p80_vod_up = calcular_p80_sensibilidad(B_base, S_base, FC_base, ucs_input, vod_exp * (1 + variacion))
    p80_vod_down = calcular_p80_sensibilidad(B_base, S_base, FC_base, ucs_input, vod_exp * (1 - variacion))
    sensibilidades['VOD (explosivo)'] = {
        'cambio_up': ((p80_vod_up - p80_base) / p80_base) * 100,
        'cambio_down': ((p80_vod_down - p80_base) / p80_base) * 100,
        'rango': abs(p80_vod_up - p80_vod_down)
    }
    
    # Sensibilidad a Timing (usando el factor de timing)
    # f_timing = 1.0 + 0.08×|tp/tp_opt - 1| + 0.12×|tf/tf_opt - 1|
    f_timing_base = 1.0
    f_timing_up = 1.0 + 0.08 * variacion  # tp 20% desviado del óptimo
    f_timing_down = 1.0 + 0.08 * variacion  # Penalización simétrica
    
    # Efecto en P80: P80_real = P80_estimado × f_timing
    p80_timing_up = p80_base * f_timing_up
    p80_timing_down = p80_base  # Sin penalización
    sensibilidades['Timing pozos'] = {
        'cambio_up': ((p80_timing_up - p80_base) / p80_base) * 100,
        'cambio_down': 0,  # Timing óptimo no penaliza
        'rango': abs(p80_timing_up - p80_base)
    }
    
    # Timing filas tiene mayor impacto (coef 0.12 vs 0.08)
    f_timing_filas_up = 1.0 + 0.12 * variacion
    p80_tf_up = p80_base * f_timing_filas_up
    sensibilidades['Timing filas'] = {
        'cambio_up': ((p80_tf_up - p80_base) / p80_base) * 100,
        'cambio_down': 0,
        'rango': abs(p80_tf_up - p80_base)
    }
    
    # Sensibilidad a Área de Malla (B×S combinado)
    factor_area = np.sqrt(1 + variacion)  # Para mantener proporcionalidad
    p80_area_up = calcular_p80_sensibilidad(B_base * factor_area, S_base * factor_area, FC_base, ucs_input, vod_exp)
    factor_area_down = np.sqrt(1 - variacion)
    p80_area_down = calcular_p80_sensibilidad(B_base * factor_area_down, S_base * factor_area_down, FC_base, ucs_input, vod_exp)
    sensibilidades['Área Malla (B×S)'] = {
        'cambio_up': ((p80_area_up - p80_base) / p80_base) * 100,
        'cambio_down': ((p80_area_down - p80_base) / p80_base) * 100,
        'rango': abs(p80_area_up - p80_area_down)
    }
    
    # Ordenar por rango (mayor influencia primero)
    vars_ordenadas = sorted(sensibilidades.items(), key=lambda x: x[1]['rango'], reverse=True)
    
    # Crear gráfico de tornado con mejor dimensionamiento
    fig_sens, ax_sens = plt.subplots(figsize=(10, 6))
    
    variables = [v[0] for v in vars_ordenadas]
    cambios_up = [v[1]['cambio_up'] for v in vars_ordenadas]
    cambios_down = [v[1]['cambio_down'] for v in vars_ordenadas]
    
    y_pos = np.arange(len(variables))
    
    # Barras hacia la derecha (aumento del parámetro)
    bars_up = ax_sens.barh(y_pos, cambios_up, height=0.35, label='+20% parámetro', color='#0288d1', alpha=0.85)
    # Barras hacia la izquierda (disminución del parámetro)
    bars_down = ax_sens.barh(y_pos, cambios_down, height=0.35, label='-20% parámetro', color='#81d4fa', alpha=0.85)
    
    ax_sens.set_yticks(y_pos)
    ax_sens.set_yticklabels(variables, fontsize=11)
    ax_sens.set_xlabel('Cambio en P80 (%)', fontsize=11)
    ax_sens.set_title(f'Sensibilidad del P80 a variaciones de ±20%\n(P80 base = {p80_base:.2f}")', fontsize=12, fontweight='bold')
    ax_sens.axvline(x=0, color='black', linewidth=0.8)
    ax_sens.legend(loc='lower right', fontsize=10)
    ax_sens.grid(axis='x', alpha=0.3)
    ax_sens.tick_params(axis='x', labelsize=10)
    
    # Añadir valores en las barras con mejor posicionamiento
    for i, (up, down) in enumerate(zip(cambios_up, cambios_down)):
        if abs(up) > 0.5:
            offset = 0.3 if up > 0 else -0.3
            ax_sens.text(up + offset, i, f'{up:+.2f}%', 
                        va='center', ha='left' if up > 0 else 'right', fontsize=10, fontweight='bold')
        if abs(down) > 0.5:
            offset = 0.3 if down > 0 else -0.3
            ax_sens.text(down + offset, i, f'{down:+.2f}%', 
                        va='center', ha='left' if down > 0 else 'right', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig_sens, use_container_width=True)
    plt.close(fig_sens)
    
    # Tabla de sensibilidad
    st.markdown("### 📋 Tabla de Sensibilidad")
    
    df_sens = pd.DataFrame({
        'Variable': [v[0] for v in vars_ordenadas],
        'Si aumenta 20%': [f"P80 {v[1]['cambio_up']:+.2f}%" for v in vars_ordenadas],
        'Si disminuye 20%': [f"P80 {v[1]['cambio_down']:+.2f}%" for v in vars_ordenadas],
        'Rango de impacto': [f"{v[1]['rango']:.2f}\"" for v in vars_ordenadas],
        'Influencia': ['🔴 Alta' if v[1]['rango'] > 0.5 else '🟡 Media' if v[1]['rango'] > 0.2 else '🟢 Baja' 
                      for v in vars_ordenadas]
    })
    
    st.dataframe(df_sens, use_container_width=True, hide_index=True)
    
    # Interpretación
    var_mas_influyente = vars_ordenadas[0][0]
    var_menos_influyente = vars_ordenadas[-1][0]
    
    st.markdown(f"""
    ### 💡 Interpretación para UCS = {ucs_input} MPa
    
    **Variable más influyente:** `{var_mas_influyente}`
    - Un cambio de ±20% en {var_mas_influyente} produce un cambio de hasta **{vars_ordenadas[0][1]['rango']:.2f}"** en P80
    - Esta es la variable donde debes poner más atención en el control de calidad
    
    **Variable menos influyente:** `{var_menos_influyente}`
    - Cambios en {var_menos_influyente} tienen menor impacto en la fragmentación
    - Hay más flexibilidad para ajustar esta variable según otras restricciones
    
    ---
    
    **Recomendaciones operacionales:**
    """)
    
    # Recomendaciones específicas
    if sensibilidades['Área Malla (B×S)']['rango'] > sensibilidades['Factor de Carga']['rango']:
        st.success("""
        ✅ **La geometría de malla (B×S) es más crítica que el factor de carga.**
        - Prioriza el control preciso del burden y espaciamiento
        - Pequeñas desviaciones en perforación afectan significativamente la fragmentación
        """)
    else:
        st.success("""
        ✅ **El factor de carga es más crítico que la geometría.**
        - Prioriza la dosificación precisa de explosivo
        - Hay algo más de tolerancia en la precisión de perforación
        """)
    
    if sensibilidades['UCS (dureza)']['rango'] > 0.3:
        st.warning("""
        ⚠️ **Alta sensibilidad a la dureza de roca.**
        - Variaciones en UCS dentro del polígono afectarán la fragmentación
        - Considera ajustar parámetros por zonas de diferente dureza
        """)

# ===== DATOS HISTÓRICOS (si existen) =====
if col_ucs is not None and col_ucs in df_filtrado.columns and 'P80TRON' in df_filtrado.columns:
    with st.expander("📊 Comparar con datos históricos", expanded=False):
        # Filtrar datos cerca del UCS objetivo
        ucs_tolerance = 20
        df_historico = df_filtrado[
            (df_filtrado[col_ucs] >= ucs_input - ucs_tolerance) &
            (df_filtrado[col_ucs] <= ucs_input + ucs_tolerance)
        ].copy()
        
        if len(df_historico) > 0:
            st.write(f"**{len(df_historico)} tronaduras con UCS ≈ {ucs_input} MPa (±{ucs_tolerance}):**")
            
            # Mejor configuración histórica
            if 'P80TRON' in df_historico.columns:
                mejor_hist = df_historico.nsmallest(3, 'P80TRON')
                cols_mostrar = ['BxS', 'Burden', 'Espaciamiento', 'FC', 'Tipo_Explosivo',
                               'P80TRON', 'P100TRON', 'UCS_MPA']
                cols_mostrar = [c for c in cols_mostrar if c in mejor_hist.columns]
                
                st.dataframe(mejor_hist[cols_mostrar], use_container_width=True, hide_index=True)
        else:
            st.info(f"No hay datos históricos para UCS ≈ {ucs_input} MPa")
    
    # Filtrar datos cerca del UCS objetivo
    ucs_tolerance = 20  # ±20 MPa
    if col_ucs is not None and col_ucs in df_filtrado.columns:
        df_historico = df_filtrado[
            (df_filtrado[col_ucs] >= ucs_input - ucs_tolerance) &
            (df_filtrado[col_ucs] <= ucs_input + ucs_tolerance)
        ].copy()
        
        if len(df_historico) > 0:
            # Encontrar mejor configuración histórica (menor P80)
            if 'P80TRON' in df_historico.columns:
                mejor_hist = df_historico.nsmallest(1, 'P80TRON')
            elif 'P100TRON' in df_historico.columns:
                mejor_hist = df_historico.nsmallest(1, 'P100TRON')
            else:
                mejor_hist = pd.DataFrame()
            
            if len(mejor_hist) > 0:
                cols_mostrar = ['BxS', 'Burden', 'Espaciamiento', 'FC', 'Tipo_Explosivo',
                               'tpozos_ms', 'tfilas_ms', 'taco_gravilla', 
                               'P80TRON', 'P100TRON', 'UCS_MPA']
                cols_mostrar = [c for c in cols_mostrar if c in mejor_hist.columns]
                
                st.write(f"Mejor configuración observada para UCS ≈ {ucs_input} MPa (±{ucs_tolerance}):")
                st.dataframe(mejor_hist[cols_mostrar], use_container_width=True)
                
                # Extraer valores históricos para comparación
                if 'Burden' in mejor_hist.columns and 'Espaciamiento' in mejor_hist.columns:
                    b_hist = mejor_hist['Burden'].values[0]
                    s_hist = mejor_hist['Espaciamiento'].values[0]
                    if pd.notna(b_hist) and pd.notna(s_hist):
                        st.info(f"**Malla histórica (BxS):** {b_hist}x{s_hist}")
            else:
                st.info(f"No hay suficientes datos históricos para UCS ≈ {ucs_input} MPa")
        else:
            st.info(f"No hay datos históricos disponibles para UCS ≈ {ucs_input} MPa (±{ucs_tolerance})")
    else:
        st.info("No se puede mostrar recomendación histórica sin datos de UCS")

# ============================================
# SECCIÓN 2: VISUALIZADOR DE HEATMAPS
# ============================================

st.markdown("---")
st.header("Sección 2: Visualizador de Heatmaps")

st.markdown(
    """
Configura la métrica y la variable del eje Y para generar el heatmap.
"""
)

metrica_sel = st.selectbox(
    "Métrica (color del heatmap):",
    metricas_posibles,
    help="Selecciona P100TRON, P80TRON, P50TRON o P20TRON."
)

tipo_var = st.radio(
    "Tipo de variable en eje Y:",
    options=["Numérica", "Categórica"],
    index=0
)

if tipo_var == "Numérica":
    if not vars_numericas:
        st.warning("No se encontraron variables numéricas configuradas en este dataset.")
        st.stop()
    var_y_sel = st.selectbox(
        "Variable numérica (eje Y):",
        vars_numericas
    )
else:
    if not vars_categoricas:
        st.warning("No se encontraron variables categóricas configuradas en este dataset.")
        st.stop()
    var_y_sel = st.selectbox(
        "Variable categórica (eje Y):",
        vars_categoricas
    )

st.markdown(
    """
**Ejemplo de uso**: para analizar el desempeño en **Fase 7**, 
elige en la barra lateral `Fase → F7`, selecciona la métrica 
(P100TRON, P80TRON, P50TRON o P20TRON) y luego la variable que quieras estudiar 
(ej. `BxS_cat`, `FC`, `tpozos_ms`, etc.).
"""
)

if "mostrar_heatmap" not in st.session_state:
    st.session_state["mostrar_heatmap"] = False

def activar_heatmap():
    st.session_state["mostrar_heatmap"] = True

st.button("Generar Heatmap", on_click=activar_heatmap)

if st.session_state["mostrar_heatmap"]:
    nombre_subset = f"Filtro actual ({df_filtrado.shape[0]} filas)"
    generar_heatmap_una_variable(
        df_filtrado,
        nombre_subset,
        var_y_sel,
        metrica_sel,
        altura_min,
        altura_por_fila
    )

    st.subheader("Tronaduras incluidas en el Heatmap")

    columnas_descriptivas = [
        col_fecha,
        col_fase,
        col_banco,
        "UGT_tron",
        col_tipo_tron,
        col_tipo_explo,
        col_ucs,
        "P100TRON",
        "P80TRON",
        "P50TRON",
        "P20TRON",
        "tpozos_ms",
        "tfilas_ms",
        "BxS",
        "taco_gravilla",
        "taco_intermedio",
        "FC",
        "fc1",
        "fc2",
        col_lito_moda,
        col_minzon_moda,
        col_lito,
        col_zonmin,
        col_alt,
        col_cat,
        "Tiempo_entre_Pozos_Filas"
    ]
    columnas_descriptivas = [c for c in columnas_descriptivas if c is not None and c in df_filtrado.columns]

    if not columnas_descriptivas:
        st.info(
            "No se encontraron columnas descriptivas (Fecha_tronadura, Fase_cat, Banco, "
            "UCS_MPA, LITO, MINZON, ALTERACION, CAT, P100TRON, P80TRON, P50TRON, "
            "P20TRON, FC, fc1, fc2, tpozos_ms, tfilas_ms, Tipo_Explosivo, LITO_moda, MINZON_moda)."
        )
    else:
        df_tron = (
            df_filtrado[columnas_descriptivas]
            .drop_duplicates()
            .sort_values(by=[c for c in columnas_descriptivas if c in df_filtrado.columns])
        )
        st.write(
            "Cada fila corresponde a una combinación única de las variables "
            "`Fecha_tronadura`, `Fase_cat`, `Banco`, `Tipo_de_tronadura`, `Tipo_Explosivo`, "
            "`UCS_MPA`, `P100TRON`, `P80TRON`, `P50TRON`, `P20TRON`, `tpozos_ms`, `FC`, `fc1`, `fc2`, "
            "`tfilas_ms`, `LITO`, `MINZON`, `ALTERACION`, `CAT`, `LITO_moda`, `MINZON_moda` "
            "presente en los datos del heatmap."
        )
        st.dataframe(df_tron, use_container_width=True)

# ============================================
# SECCIÓN 3: EXPLOSIVOS
# ============================================

st.markdown("---")
st.header("Sección 3: Explosivos")

st.subheader("Ficha técnica de los explosivos")

explosivo_sel = st.selectbox(
    "Selecciona un explosivo para ver su ficha técnica:",
    options=sorted(EXPLOSIVOS_PROPIEDADES.keys())
)

prop = EXPLOSIVOS_PROPIEDADES[explosivo_sel]

st.markdown(f"""
**Explosivo seleccionado:** `{explosivo_sel}`  

- **Densidad aproximada:** `{prop['densidad']}` g/cc  
- **VOD (Velocidad de detonación):** `{prop['VOD']}` m/s  
- **RWS (Relative Weight Strength):** `{prop['RWS']}` % (referencia ANFO)  
- **Energía:** `{prop['energia']}` MJ/kg (aprox.)
- **Resistencia al agua:** `{prop.get('Resistencia_Agua', 'N/D')}`
""")

st.table(pd.DataFrame([prop], index=[explosivo_sel]))

st.subheader("Gráfico interactivo de propiedades técnicas de explosivos")

# Cargar datos de explosivos
df_explo = pd.DataFrame.from_dict(EXPLOSIVOS_CSV, orient='index').reset_index().rename(columns={'index': 'Producto'})
# Asegura nombres de columnas limpios (por compatibilidad con selectores)
df_explo.columns = df_explo.columns.str.strip()

vars_numericas_explosivos = [
    "Densidad_g_cc",
    "VoD_Tipica_m_s",
    "Presion_Det_Kbar",
    "Energia_KJ_Kg",
    "Vol_Gases_l_Kg"
]

def escalar_variable(serie, escala='log', rango=(10, 1000)):
    if escala == 'log':
        serie = np.log10(serie.replace(0, np.nan).dropna())
    else:
        serie = serie.copy()
    min_val, max_val = serie.min(), serie.max()
    scaled = (serie - min_val) / (max_val - min_val)
    return scaled * (rango[1] - rango[0]) + rango[0]

st.write("Selecciona las variables para visualizar el gráfico.")

col1, col2, col3 = st.columns(3)
with col1:
    x = st.selectbox("Variable en X", vars_numericas_explosivos)
with col2:
    y = st.selectbox("Variable en Y", vars_numericas_explosivos)
with col3:
    size = st.selectbox("Variable para tamaño", vars_numericas_explosivos)

if (x == y) or (x == size) or (y == size):
    st.warning("⚠️ Las variables X, Y y la variable de tamaño deben ser distintas.")
    #st.stop()

else:
    df_plot = df_explo[[x, y, size, "Producto"]].copy()

    # Convertir todo a numérico
    df_plot[x] = pd.to_numeric(df_plot[x], errors="coerce")
    df_plot[y] = pd.to_numeric(df_plot[y], errors="coerce")
    df_plot[size] = pd.to_numeric(df_plot[size], errors="coerce")

    df_plot = df_plot.dropna()

    df_plot["scaled_size"] = escalar_variable(df_plot[size], escala='log', rango=(10, 1000))

    fig = px.scatter(
        df_plot,
        x=x,
        y=y,
        size="scaled_size",
        color="Producto",
        hover_name="Producto",
        hover_data={x: True, y: True, size: True, "scaled_size": False},
        title=f"{y} vs {x} (Tamaño por {size})",
        size_max=40
    )
    
    # Mejorar layout del gráfico
    fig.update_layout(
        title=dict(font=dict(size=14)),
        xaxis=dict(title=dict(font=dict(size=12)), tickfont=dict(size=10)),
        yaxis=dict(title=dict(font=dict(size=12)), tickfont=dict(size=10)),
        legend=dict(font=dict(size=10)),
        margin=dict(l=60, r=40, t=50, b=60)
    )

    st.plotly_chart(fig, use_container_width=True)

# ============================================
# 3.1) FILTROS PARA EL GRAFICO 2 DE EXPLOSIVOS
# ============================================

st.sidebar.header("4. Filtros para grafico de explosivos")

df_base_explo = df_raw.copy()

# --- Filtro por Tipo_de_tronadura ---
if col_tipo_tron is not None:
    tipos_tron = sorted(df_base_explo[col_tipo_tron].dropna().unique())
    tipo_tron_sel = st.sidebar.multiselect(
        "Tipo de tronadura",
        tipos_tron,
        default=tipos_tron,
        key='filter_tipotron_plot2'
    )
    if tipo_tron_sel:
        df_base_explo = df_base_explo[df_base_explo[col_tipo_tron].isin(tipo_tron_sel)]
    st.sidebar.write(f"Filas después de filtro Tipo_de_tronadura: {len(df_base_explo)}")

# --- Filtro M ---
if col_m is not None:
    opciones_m = sorted(df_base_explo[col_m].dropna().unique())
    m_sel = st.sidebar.multiselect(
        "M (por ejemplo M4, M3, etc.)",
        opciones_m,
        default=opciones_m,
        key='filter_m_plot2'
    )
    if m_sel:
        df_base_explo = df_base_explo[df_base_explo[col_m].isin(m_sel)]
    st.sidebar.write(f"Filas después de filtro M: {len(df_base_explo)}")

# --- Filtro por Fase ---
if col_fase is not None:
    fases = sorted(df_base_explo[col_fase].dropna().unique())
    fase_sel = st.sidebar.multiselect("Fase", fases, default=fases, key='filter_fase_plot2')
    if fase_sel:
        df_base_explo = df_base_explo[df_base_explo[col_fase].isin(fase_sel)]

# --- Filtro por Banco (RANGO NUMÉRICO) ---
if col_banco is not None:
    serie_banco_base = pd.to_numeric(df_base_explo[col_banco], errors="coerce")
    if serie_banco_base.notna().any():
        banco_min = float(serie_banco_base.min())
        banco_max = float(serie_banco_base.max())
        banco_rango = st.sidebar.slider(
            "Rango Banco",
            min_value=float(np.floor(banco_min)),
            max_value=float(np.ceil(banco_max)),
            value=(float(np.floor(banco_min)), float(np.ceil(banco_max))),
            step=1.0,
            key='filter_banco_plot2'
        )
        mascara_banco = serie_banco_base.between(banco_rango[0], banco_rango[1])
        df_base_explo = df_base_explo[mascara_banco]
        st.sidebar.write(f"Filas después de filtro Banco: {len(df_base_explo)}")
    else:
        st.sidebar.info("No hay valores numéricos válidos en Banco para aplicar filtro de rango.")


# --- Filtro por UGT_tron (CATEGÓRICO) ---
if col_ugt_moda is not None:
    ugts = sorted(df_base_explo[col_ugt_moda].dropna().unique())

    ugt_sel = st.sidebar.multiselect(
        "UGT_tron",
        ugts,
        default=ugts,
        key='filter_ugt_plot2'
    )

    if ugt_sel:
        df_base_explo = df_base_explo[df_base_explo[col_ugt_moda].isin(ugt_sel)]

    st.sidebar.write(f"Filas después de filtro UGT_tron: {len(df_base_explo)}")

# --- Filtro Malla (Categórico) ---
# if col_malla is not None:
#     mallas = sorted(df_base_explo[col_malla].dropna().unique())
#     malla_sel = st.sidebar.multiselect("Malla", mallas, default=mallas, key='filter_malla_plot2')
#     if malla_sel:
#         df_base_explo = df_base_explo[df_base_explo[col_malla].isin(malla_sel)]

#     st.sidebar.write(f"Filas después de filtro Malla: {len(df_base_explo)}")

# --- Filtro Burden (Rango numérico) ---
if col_burden is not None:
    serie = pd.to_numeric(df_base_explo[col_burden], errors="coerce")
    if serie.notna().any():
        val_min = float(serie.min())
        val_max = float(serie.max())
        rango = st.sidebar.slider(
            "Rango Burden",
            min_value=float(np.floor(val_min)),
            max_value=float(np.ceil(val_max)),
            value=(float(np.floor(val_min)), float(np.ceil(val_max))),
            step=.5,
            key='filter_burden_plot2'
        )
        mascara = serie.between(rango[0], rango[1])
        df_base_explo = df_base_explo[mascara]
        st.sidebar.write(f"Filas después de filtro Burden: {len(df_base_explo)}")
    else:
        st.sidebar.info("No hay valores numéricos válidos en Burden para aplicar filtro de rango.")


# --- Filtro Espaciamiento (Rango numérico) ---
if col_espaciamiento is not None:
    serie = pd.to_numeric(df_base_explo[col_espaciamiento], errors="coerce")
    if serie.notna().any():
        val_min = float(serie.min())
        val_max = float(serie.max())
        rango = st.sidebar.slider(
            "Rango Espaciamiento",
            min_value=float(np.floor(val_min)),
            max_value=float(np.ceil(val_max)),
            value=(float(np.floor(val_min)), float(np.ceil(val_max))),
            step=.5,
            key='filter_espaciamiento_plot2'
        )
        mascara = serie.between(rango[0], rango[1])
        df_base_explo = df_base_explo[mascara]
        st.sidebar.write(f"Filas después de filtro Espaciamiento: {len(df_base_explo)}")
    else:
        st.sidebar.info("No hay valores numéricos válidos en Espaciamiento para aplicar filtro de rango.")

# ============================================
# 3.2) GRAFICO 2 DE EXPLOSIVOS COMPLETO
# ============================================

st.subheader("Gráfico interactivo de propiedades técnicas de explosivos v/s características de la tronadura")
st.write("Selecciona las variables para visualizar el gráfico.")

vars_geologia = [
    "ANHIDRITA",
    "UG",
    "CUEQ",
    "CU",
    "MO",
    "AS",
    "CUS",
    "AU",
    "AG",
    "PY",
    "CPY",
    "CSCV",
    "BO",
    "AXB",
    "LEYCONC",
    "RECCU",
    "CONCALIN",
    "CONCAL",
    "WI",
    "FINOC",
    "FINOR",
    "TONCON",
    "CEESAG",
    "SPI",
    "BAI",
    "DWI",
    "UCS_MPA",
    "RQD",
    "RECCU.1"
]

vars_tronadura = [
    "BxS",
    "Tipo_Explosivo",
    "Diametro",
    "FC",
    "tpozos_ms",
    "tfilas_ms",
    "taco_gravilla",
    "taco_intermedio",
    "taco_aire",
    "Burden",
    "Espaciamiento",
    "AreaMalla",
    "Densidad_g_cc",
    "VoD_Minima_m_s",
    "VoD_Tipica_m_s",
    "Presion_Det_Kbar",
    "Energia_KJ_Kg",
    "Vol_Gases_l_Kg",
    "Diam_Min_pulg",
    "Pot_Rel_ANFO_Peso",
    "Pot_Rel_ANFO_Vol"
]

vars_rendimiento = [
    "P80_geo",
    "RDTO_PLANTA",
    "RENDIMIENTOXSAG",
    "F80",
    "P20TRON",
    "P50TRON",
    "P80TRON",
    "P100TRON",
    "P20",
    "P50",
    "P80_tron",
    "P100"
]

col1, col2, col3 = st.columns(3)
with col1:
    x1 = st.selectbox("Variable Geológica", vars_geologia)
with col2:
    y1 = st.selectbox("Variable de Tronadura", vars_tronadura)
with col3:
    size1 = st.selectbox("Variable de Rendimiento", vars_rendimiento)

if (x1 == y1) or (x1 == size1) or (y1 == size1):
    st.warning("⚠️ Las variables X, Y y la variable de tamaño deben ser distintas.")
    #st.stop()

else:
    df_plot = df_base_explo[[x1, y1, size1]].copy()

    #df_plot = df_plot.dropna()

    df_plot["scaled_size"] = escalar_variable(df_plot[size1], escala='log', rango=(10, 1000))

    #st.dataframe(df_plot, use_container_width=True)

    fig = px.scatter(
        df_plot,
        x=x1,
        y=y1,
        size="scaled_size",
        hover_data={x1: True, y1: True, size1: True, "scaled_size": False},
        title=f"{y1} vs {x1} (Tamaño por {size1})",
        size_max=40
    )
    
    # Mejorar layout del gráfico
    fig.update_layout(
        title=dict(font=dict(size=14)),
        xaxis=dict(title=dict(font=dict(size=12)), tickfont=dict(size=10)),
        yaxis=dict(title=dict(font=dict(size=12)), tickfont=dict(size=10)),
        margin=dict(l=60, r=40, t=50, b=60)
    )

    st.plotly_chart(fig, use_container_width=True)
# ============================================
# TAB SIMULADOR PILIX v3.6 - CONTENIDO
# ============================================

with tab_simulador:
    st.markdown("## 🔬 Simulador PILIX v3.6 - Fragmentación Kuz-Ram")
    
    st.info("""
    **Características del Simulador PILIX v3.6:**
    - 📊 Índice de Lilly para Factor de Roca A
    - 🔧 Calibración Kx/Kn (Mundaca)
    - 📐 Índice n de Cunningham
    - 🎲 Simulación Monte Carlo con análisis de sensibilidad
    - 🔷 Modelo Doble Taco con 3 APDs
    - 🎯 Visualización 3D interactiva
    """)
    
    # Mostrar configuraciones disponibles para cargar
    st.markdown("### 📋 Cargar Configuración desde Recomendaciones TOP")
    
    col_configs = st.columns(4)
    
    with col_configs[0]:
        if st.button("🎯 Cargar Teórico ENAEX", key="load_teorico"):
            st.session_state.pilix_params.update({
                'nombre_config': 'Teórico ENAEX',
                'B': st.session_state.get('params_teoricos', {}).get('burden_optimo', 7.5),
                'S': st.session_state.get('params_teoricos', {}).get('espaciamiento_optimo', 8.6),
                'T': st.session_state.get('params_teoricos', {}).get('taco_optimo', 5.5),
            })
            st.rerun()
    
    with col_configs[1]:
        if st.button("⚡ Cargar Optimizado", key="load_optim"):
            st.session_state.pilix_params.update({
                'nombre_config': 'Optimizado Multiobjetivo',
                'B': st.session_state.get('params_multiobj', {}).get('burden_optimo', 7.5),
                'S': st.session_state.get('params_multiobj', {}).get('espaciamiento_optimo', 8.6),
                'T': st.session_state.get('params_multiobj', {}).get('taco_optimo', 5.5),
            })
            st.rerun()
    
    with col_configs[2]:
        if st.button("📉 Cargar Min. Metros", key="load_minm"):
            st.session_state.pilix_params.update({
                'nombre_config': 'Mínimos Metros',
                'B': st.session_state.get('params_minmetros', {}).get('burden_minmetros', 8.0),
                'S': st.session_state.get('params_minmetros', {}).get('espaciamiento_minmetros', 9.2),
                'T': st.session_state.get('params_minmetros', {}).get('taco_ajustado', 5.5),
            })
            st.rerun()
    
    with col_configs[3]:
        if st.button("🔄 Reset Manual", key="reset_params"):
            st.session_state.pilix_params = {
                'UCS': 100,
                'RQD': 70,
                'B': 7.5,
                'S': 8.6,
                'T': 5.5,
                'H': 16.5,
                'J': 1.5,
                'd': 10.625,
                'n': 1.2,
                'Kx': 0.5,
                'Kn': 1.02,
                'explosivo': 'Energex 70',
                'doble_taco': False,
                'Ti': 2.0,
                'Ti_pos': 8.0,
                'nombre_config': 'Configuración Manual'
            }
            st.rerun()
    
    # Mostrar configuración actual
    if st.session_state.pilix_params.get('nombre_config'):
        st.success(f"📋 Configuración cargada: **{st.session_state.pilix_params['nombre_config']}**")
    
    # Panel de ajuste de parámetros antes de simular
    with st.expander("⚙️ Ajustar parámetros antes de simular", expanded=True):
        col_p1, col_p2, col_p3 = st.columns(3)
        
        with col_p1:
            st.markdown("**🪨 Roca**")
            pilix_ucs = st.number_input("UCS (MPa)", 20, 300, 
                                        int(st.session_state.pilix_params.get('UCS', 100)), 
                                        key="pilix_ucs")
            pilix_rqd = st.number_input("RQD (%)", 0, 100, 
                                        int(st.session_state.pilix_params.get('RQD', 70)), 
                                        key="pilix_rqd")
        
        with col_p2:
            st.markdown("**📐 Geometría**")
            pilix_B = st.number_input("Burden (m)", 3.0, 15.0, 
                                      float(st.session_state.pilix_params.get('B', 7.5)), 
                                      step=0.1, key="pilix_B")
            pilix_S = st.number_input("Espaciamiento (m)", 3.0, 18.0, 
                                      float(st.session_state.pilix_params.get('S', 8.6)), 
                                      step=0.1, key="pilix_S")
            pilix_T = st.number_input("Taco (m)", 2.0, 10.0, 
                                      float(st.session_state.pilix_params.get('T', 5.5)), 
                                      step=0.1, key="pilix_T")
            pilix_H = st.number_input("Altura Banco (m)", 5.0, 30.0, 
                                      float(st.session_state.pilix_params.get('H', 16.5)), 
                                      step=0.5, key="pilix_H")
        
        with col_p3:
            st.markdown("**🔧 Calibración**")
            pilix_Kx = st.number_input("Kx", 0.1, 2.0, 
                                       float(st.session_state.pilix_params.get('Kx', 0.5)), 
                                       step=0.01, key="pilix_Kx")
            pilix_Kn = st.number_input("Kn", 0.5, 1.5, 
                                       float(st.session_state.pilix_params.get('Kn', 1.02)), 
                                       step=0.01, key="pilix_Kn")
            pilix_doble_taco = st.checkbox("🔷 Doble Taco", 
                                           st.session_state.pilix_params.get('doble_taco', False),
                                           key="pilix_doble_taco")
        
        # Actualizar parámetros
        st.session_state.pilix_params.update({
            'UCS': pilix_ucs,
            'RQD': pilix_rqd,
            'B': pilix_B,
            'S': pilix_S,
            'T': pilix_T,
            'H': pilix_H,
            'J': 1.5,
            'd': 10.625,
            'Kx': pilix_Kx,
            'Kn': pilix_Kn,
            'doble_taco': pilix_doble_taco,
        })
    
    # Generar y mostrar el simulador PILIX v3.6 completo
    st.markdown("### 🔬 Simulador Interactivo")
    
    simulador_pilix_html = generar_simulador_pilix_completo(st.session_state.pilix_params)
    components.html(simulador_pilix_html, height=750, scrolling=False)
    
    # Guía del simulador
    with st.expander("📖 Guía del Simulador PILIX v3.6", expanded=False):
        st.markdown("""
        ### 🎯 ¿Qué es PILIX v3.6?
        
        PILIX es un simulador de fragmentación basado en el modelo **Kuz-Ram** con las siguientes capacidades:
        
        | Característica | Descripción |
        |----------------|-------------|
        | **Índice de Lilly** | Calcula Factor A desde UCS y RQD |
        | **Calibración Kx/Kn** | Ajusta predicciones con datos reales |
        | **Monte Carlo** | Cuantifica incertidumbre y rangos de confianza |
        | **Doble Taco** | Simula configuración con taco intermedio |
        | **Visualización 3D** | Animación de secuencia de detonación |
        
        ### 📐 Fórmulas Principales
        
        **Kuz-Ram con Kx:**
        ```
        X50 = Kx × A × (V₀/Q)^0.8 × Q^(1/6) × (1.15/Er)^(19/30)
        ```
        
        **Índice n (Cunningham):**
        ```
        n = (2.2 - 14×B/D_mm) × √[(1+S/B)/2] × (1-W/B) × (Lc/H) × Kn
        ```
        
        **Rosin-Rammler:**
        ```
        Xc = X50 / (ln2)^(1/n)
        P80 = Xc × [-ln(0.20)]^(1/n)
        P100 = Xc × [-ln(0.01)]^(1/n)
        ```
        
        ### 🎲 Interpretación Monte Carlo
        
        | CV | Interpretación | Acción |
        |----|----------------|--------|
        | <5% | Baja incertidumbre | Resultados confiables |
        | 5-10% | Moderada | Considere reducir variabilidad |
        | >10% | Alta incertidumbre | Revise parámetros |
        
        ### 📚 Referencias
        
        - Cunningham (1983) - Kuz-Ram Model
        - Lilly (1986) - Blastability Index
        - Mundaca et al. (2015) - Calibración Kx/Kn
        - Manual ENAEX (2020)
        """)
