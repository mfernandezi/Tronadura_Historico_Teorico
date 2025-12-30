import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import os

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
        'timing_pozos_optimo': round(timing_pozos_optimo, 1),
        'timing_filas_optimo': round(timing_filas_optimo, 1),
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
        'timing_pozos_ajustado': round(timing_pozos_ajustado, 1),
        'timing_filas_ajustado': round(timing_filas_ajustado, 1),
        'fc_compensado': round(fc_compensado, 2),
        'factor_expansion_usado': round(factor_expansion, 2),
        'reduccion_pozos_pct': round(reduccion_pozos * 100, 1),
        'explosivo_recomendado': params_teoricos['explosivo_recomendado'],
        'p80_objetivo': p80_objetivo,
        'p100_objetivo': p100_objetivo,
        'advertencias': []
    }

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

st.markdown(
    """
Esta app permite:
1. Ver las **mejores 3 configuraciones** observadas según una métrica (Top 3).
2. Explorar el desempeño de **P100TRON, P80TRON, P50TRON, P20TRON** en distintos **heatmaps**.
3. Consultar la **ficha técnica** y comparar propiedades de **explosivos**.
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
        ax=ax
    )

    ax.set_title(f"{metrica} medio — UCS vs {var_y} — {nombre_subset}", pad=20)
    ax.set_xlabel("UCS (bins)")
    ax.set_ylabel(var_y)

    ax.tick_params(axis="x", labelrotation=0)
    ax.tick_params(axis="y", labelrotation=0)

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
# SECCIÓN 1.1: RECOMENDACIONES DE MALLA
# ============================================

st.markdown("---")
st.header("Sección 1.1: Recomendaciones de Mejor Malla")

st.markdown("""
Esta sección presenta **tres tipos de recomendaciones** para optimizar la malla de perforación:

1. **📊 Histórica**: Basada en las mejores configuraciones observadas en los datos reales
2. **📐 Teórica ENAEX**: Basada en fórmulas del Manual de Tronadura ENAEX
3. **🎯 Mínimo Metros**: Optimizada para reducir perforación manteniendo control de fragmentación

""")

# Inputs para los cálculos teóricos
st.subheader("Parámetros de entrada para cálculos teóricos")

col_input1, col_input2, col_input3 = st.columns(3)

with col_input1:
    ucs_input = st.number_input(
        "UCS objetivo (MPa)",
        min_value=20.0,
        max_value=300.0,
        value=100.0,
        step=10.0,
        help="Resistencia a compresión uniaxial de la roca objetivo"
    )

with col_input2:
    diametro_input = st.selectbox(
        "Diámetro de perforación (pulgadas)",
        options=[5.0, 5.5, 6.0, 6.5, 6.75, 7.0, 7.875, 9.0, 10.625, 12.25],
        index=4,
        help="Diámetro del pozo de perforación"
    )

with col_input3:
    explosivo_input = st.selectbox(
        "Tipo de explosivo",
        options=list(EXPLOSIVOS_PROPIEDADES.keys()),
        index=0,
        help="Selecciona el explosivo para obtener densidad y VOD"
    )

# Obtener propiedades del explosivo seleccionado
prop_explosivo = EXPLOSIVOS_PROPIEDADES.get(explosivo_input, {})
densidad_exp = prop_explosivo.get('densidad', 1.2)
vod_exp = prop_explosivo.get('VOD', 4500)

col_exp1, col_exp2 = st.columns(2)
with col_exp1:
    st.info(f"**Densidad**: {densidad_exp} g/cc")
with col_exp2:
    st.info(f"**VOD**: {vod_exp} m/s")

# Opciones adicionales
col_opt1, col_opt2 = st.columns(2)
with col_opt1:
    p80_objetivo = st.number_input(
        "P80 objetivo máximo (pulgadas)",
        min_value=3.0,
        max_value=15.0,
        value=8.0,
        step=0.5,
        help="P80 objetivo para recomendación de mínimo metros"
    )
with col_opt2:
    p100_objetivo = st.number_input(
        "P100 objetivo máximo (pulgadas)",
        min_value=8.0,
        max_value=30.0,
        value=15.0,
        step=1.0,
        help="P100 objetivo para recomendación de mínimo metros"
    )

factor_expansion = st.slider(
    "Factor de expansión de malla (para recomendación mínimo metros)",
    min_value=1.00,
    max_value=1.25,
    value=1.15,
    step=0.05,
    help="Factor para expandir la malla teórica. Mayor = menos pozos pero más riesgo de fragmentación gruesa"
)

# Botón para calcular recomendaciones
if st.button("🔄 Calcular Recomendaciones", type="primary"):
    
    # ====== RECOMENDACIÓN TEÓRICA ENAEX ======
    st.markdown("---")
    st.subheader("📐 Recomendación Teórica (ENAEX)")
    
    params_teoricos = calcular_parametros_teoricos_enaex(
        ucs=ucs_input,
        densidad_explosivo=densidad_exp,
        vod=vod_exp,
        diametro_pulg=diametro_input
    )
    
    st.markdown(f"""
    **Parámetros calculados según fórmulas ENAEX para UCS = {ucs_input} MPa:**
    
    | Parámetro | Valor Óptimo | Descripción |
    |-----------|--------------|-------------|
    | **Burden** | {params_teoricos['burden_optimo']} m | Fórmula Ash modificada (Kb={params_teoricos['Kb']}) |
    | **Espaciamiento** | {params_teoricos['espaciamiento_optimo']} m | Ratio S/B = {params_teoricos['ratio_SB']} |
    | **Área de malla** | {params_teoricos['area_malla']} m² | B × S |
    | **Taco óptimo** | {params_teoricos['taco_optimo']} m | 0.85 × Burden |
    | **Taco mínimo** | {params_teoricos['taco_minimo']} m | 0.70 × Burden |
    | **Timing pozos** | {params_teoricos['timing_pozos_optimo']} ms | Th={params_teoricos['Th']} ms/m × S |
    | **Timing filas** | {params_teoricos['timing_filas_optimo']} ms | 11.5 ms/m × B |
    | **Factor de carga** | {params_teoricos['fc_optimo']} kg/m³ | Según dureza de roca |
    | **Pasadura** | {params_teoricos['pasadura_optima']} m | 0.3 × Burden |
    | **Explosivo recomendado** | {params_teoricos['explosivo_recomendado']} | Según UCS |
    """)
    
    # Mostrar BxS como string
    bxs_teorico = f"{params_teoricos['burden_optimo']}x{params_teoricos['espaciamiento_optimo']}"
    st.success(f"**Malla teórica recomendada (BxS):** {bxs_teorico}")
    
    # ====== RECOMENDACIÓN MÍNIMO METROS ======
    st.markdown("---")
    st.subheader("🎯 Recomendación Mínimo Metros de Perforación")
    
    params_minmetros = calcular_malla_minimos_metros(
        ucs=ucs_input,
        densidad_explosivo=densidad_exp,
        vod=vod_exp,
        diametro_pulg=diametro_input,
        p80_objetivo=p80_objetivo,
        p100_objetivo=p100_objetivo,
        factor_expansion=factor_expansion
    )
    
    st.markdown(f"""
    **Parámetros optimizados para minimizar metros de perforación:**
    
    Esta recomendación expande la malla teórica un **{params_minmetros['factor_expansion_usado']*100 - 100:.0f}%** 
    para reducir la cantidad de pozos, compensando con mayor factor de carga y ajuste de timing.
    
    | Parámetro | Valor | Descripción |
    |-----------|-------|-------------|
    | **Burden** | {params_minmetros['burden_minmetros']} m | Expandido {params_minmetros['factor_expansion_usado']}x |
    | **Espaciamiento** | {params_minmetros['espaciamiento_minmetros']} m | Expandido {params_minmetros['factor_expansion_usado']}x |
    | **Área de malla** | {params_minmetros['area_malla_expandida']} m² | ↑ Mayor área = menos pozos |
    | **Taco ajustado** | {params_minmetros['taco_ajustado']} m | 0.85 × Burden expandido |
    | **Timing pozos** | {params_minmetros['timing_pozos_ajustado']} ms | Reducido 10% para compensar |
    | **Timing filas** | {params_minmetros['timing_filas_ajustado']} ms | Reducido 5% para compensar |
    | **Factor de carga** | {params_minmetros['fc_compensado']} kg/m³ | Aumentado para mantener energía |
    | **Reducción de pozos** | {params_minmetros['reduccion_pozos_pct']}% | vs. malla teórica |
    """)
    
    bxs_minmetros = f"{params_minmetros['burden_minmetros']}x{params_minmetros['espaciamiento_minmetros']}"
    st.success(f"**Malla mínimo metros (BxS):** {bxs_minmetros}")
    
    st.warning(f"""
    ⚠️ **Consideraciones importantes:**
    - Esta malla expandida puede aumentar P80 y P100 respecto a la teórica
    - Se recomienda monitorear fragmentación en las primeras tronaduras
    - Objetivos establecidos: P80 ≤ {p80_objetivo}" y P100 ≤ {p100_objetivo}"
    - Si no se cumplen objetivos, reducir factor de expansión
    """)
    
    # ====== COMPARATIVA DE RECOMENDACIONES ======
    st.markdown("---")
    st.subheader("📊 Comparativa de Recomendaciones")
    
    # Crear DataFrame comparativo
    df_comparativa = pd.DataFrame({
        'Parámetro': [
            'Burden (m)',
            'Espaciamiento (m)',
            'Área malla (m²)',
            'Ratio S/B',
            'Taco (m)',
            'Timing pozos (ms)',
            'Timing filas (ms)',
            'FC (kg/m³)',
            'Metros perf. por m² (*)',
        ],
        'Teórica ENAEX': [
            params_teoricos['burden_optimo'],
            params_teoricos['espaciamiento_optimo'],
            params_teoricos['area_malla'],
            params_teoricos['ratio_SB'],
            params_teoricos['taco_optimo'],
            params_teoricos['timing_pozos_optimo'],
            params_teoricos['timing_filas_optimo'],
            params_teoricos['fc_optimo'],
            round(1 / params_teoricos['area_malla'], 3),
        ],
        'Mínimo Metros': [
            params_minmetros['burden_minmetros'],
            params_minmetros['espaciamiento_minmetros'],
            params_minmetros['area_malla_expandida'],
            params_minmetros['ratio_SB'],
            params_minmetros['taco_ajustado'],
            params_minmetros['timing_pozos_ajustado'],
            params_minmetros['timing_filas_ajustado'],
            params_minmetros['fc_compensado'],
            round(1 / params_minmetros['area_malla_expandida'], 3),
        ]
    })
    
    st.dataframe(df_comparativa, use_container_width=True, hide_index=True)
    
    st.caption("(*) Metros de perforación por m² de área = 1 / Área malla. Menor es mejor para rendimiento de perforadoras.")
    
    # ====== GRÁFICO COMPARATIVO ======
    st.markdown("---")
    st.subheader("📈 Visualización Comparativa")
    
    fig_comp, axes_comp = plt.subplots(1, 3, figsize=(14, 4))
    
    # Gráfico 1: Burden vs Espaciamiento
    ax1 = axes_comp[0]
    ax1.scatter([params_teoricos['burden_optimo']], [params_teoricos['espaciamiento_optimo']], 
               s=200, c='blue', marker='o', label='Teórica ENAEX', zorder=5)
    ax1.scatter([params_minmetros['burden_minmetros']], [params_minmetros['espaciamiento_minmetros']], 
               s=200, c='green', marker='^', label='Mínimo Metros', zorder=5)
    ax1.set_xlabel('Burden (m)')
    ax1.set_ylabel('Espaciamiento (m)')
    ax1.set_title('Burden vs Espaciamiento')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: Área de malla
    ax2 = axes_comp[1]
    categorias = ['Teórica\nENAEX', 'Mínimo\nMetros']
    areas = [params_teoricos['area_malla'], params_minmetros['area_malla_expandida']]
    colors = ['blue', 'green']
    bars = ax2.bar(categorias, areas, color=colors, alpha=0.7)
    ax2.set_ylabel('Área de malla (m²)')
    ax2.set_title('Área de Malla')
    for bar, area in zip(bars, areas):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{area:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Gráfico 3: Metros de perforación por m²
    ax3 = axes_comp[2]
    metros_teorica = round(1 / params_teoricos['area_malla'], 3)
    metros_minmetros = round(1 / params_minmetros['area_malla_expandida'], 3)
    metros_vals = [metros_teorica, metros_minmetros]
    bars3 = ax3.bar(categorias, metros_vals, color=colors, alpha=0.7)
    ax3.set_ylabel('Pozos por m²')
    ax3.set_title('Densidad de Perforación')
    for bar, m in zip(bars3, metros_vals):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{m:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig_comp)
    plt.close(fig_comp)
    
    # ====== ANÁLISIS DE VARIACIÓN P80 BASADO EN DATOS HISTÓRICOS ======
    st.markdown("---")
    st.subheader("📉 Análisis de Variación de Fragmentación (datos históricos)")
    
    if col_ucs is not None and col_ucs in df_filtrado.columns and 'P80TRON' in df_filtrado.columns:
        # Analizar variación de P80 según área de malla
        df_analisis = df_filtrado.copy()
        if 'Burden' in df_analisis.columns and 'Espaciamiento' in df_analisis.columns:
            df_analisis['Area_malla_calc'] = df_analisis['Burden'] * df_analisis['Espaciamiento']
            
            # Crear bins de área de malla
            df_analisis['Area_bin'] = pd.cut(
                df_analisis['Area_malla_calc'],
                bins=[0, 40, 50, 60, 70, 80, 100, 200],
                labels=['<40', '40-50', '50-60', '60-70', '70-80', '80-100', '>100']
            )
            
            # Estadísticas por bin de área
            stats_area = df_analisis.groupby('Area_bin').agg({
                'P80TRON': ['mean', 'std', 'count'],
                'P100TRON': ['mean', 'std']
            }).round(2)
            
            if len(stats_area) > 0:
                st.write("**Fragmentación promedio según área de malla (datos históricos):**")
                
                # Reformatear para mostrar
                stats_display = pd.DataFrame({
                    'Área malla (m²)': stats_area.index.astype(str),
                    'P80 medio (pulg)': stats_area[('P80TRON', 'mean')].values,
                    'P80 desv. std': stats_area[('P80TRON', 'std')].values,
                    'P100 medio (pulg)': stats_area[('P100TRON', 'mean')].values,
                    'n registros': stats_area[('P80TRON', 'count')].values.astype(int)
                })
                st.dataframe(stats_display, use_container_width=True, hide_index=True)
                
                # Indicar en qué rango cae cada recomendación
                area_teorica = params_teoricos['area_malla']
                area_minmetros = params_minmetros['area_malla_expandida']
                
                st.info(f"""
                📊 **Ubicación de las recomendaciones:**
                - Malla teórica ({area_teorica:.1f} m²): Esperar P80 y variación según datos históricos del rango correspondiente
                - Malla mínimo metros ({area_minmetros:.1f} m²): Con área mayor, posible incremento en P80
                """)
    
    # ====== RECOMENDACIÓN HISTÓRICA (si hay datos) ======
    st.markdown("---")
    st.subheader("📊 Mejor Configuración Histórica (datos observados)")
    
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
        title=f"{y} vs {x} (Tamaño por {size}, escalado)",
        size_max=40
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
        #color="Producto",
        #hover_name="Producto",
        hover_data={x1: True, y1: True, size1: True, "scaled_size": False},
        title=f"{y1} vs {x1} (Tamaño por {size1}, escalado)",
        size_max=40
    )

    st.plotly_chart(fig, use_container_width=True)