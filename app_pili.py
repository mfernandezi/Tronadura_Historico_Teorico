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


def calcular_malla_optimizada_multiobjetivo(ucs, densidad_explosivo=1.2, vod=4500, diametro_pulg=6.75,
                                             p80_objetivo=8.0, p100_objetivo=15.0,
                                             peso_metros=0.4, peso_p80=0.35, peso_p100=0.25,
                                             altura_banco=15.0):
    """
    ================================================================================
    FÓRMULA DE OPTIMIZACIÓN MULTI-OBJETIVO
    ================================================================================
    
    Minimiza simultáneamente:
    1. Metros de perforación por tonelada (menos pozos = mejor rendimiento perforadoras)
    2. P80 esperado (fragmentación más fina = mejor molienda)
    3. P100 esperado (menos sobretamaño = menos problemas en chancado)
    
    FUNCIÓN OBJETIVO:
    ==================
    
    J(B, S, FC, tp, tf) = w₁·f_metros + w₂·f_P80 + w₃·f_P100
    
    Donde:
    - w₁, w₂, w₃ = pesos de cada objetivo (suman 1.0)
    - f_metros = 1 / (B × S)  [metros perforados por m²]
    - f_P80 = P80_estimado / P80_objetivo
    - f_P100 = P100_estimado / P100_objetivo
    
    MODELO DE FRAGMENTACIÓN (Kuz-Ram simplificado + ajuste empírico):
    ================================================================
    
    P80_estimado = k₁ × (B × S)^α × UCS^β × (1/FC)^γ × (1/VOD)^δ × f_timing
    
    Donde:
    - k₁ = constante empírica = 0.015
    - α = 0.45 (sensibilidad al área de malla)
    - β = 0.25 (sensibilidad a la dureza de roca)
    - γ = 0.35 (sensibilidad al factor de carga)
    - δ = 0.15 (sensibilidad a la velocidad de detonación)
    - f_timing = factor de corrección por timing
    
    P100_estimado = P80_estimado × k_ratio
    
    Donde k_ratio depende del ratio S/B y uniformidad de la tronadura:
    - k_ratio = 1.8 para S/B ≈ 1.15 (óptimo)
    - k_ratio = 2.0 para S/B < 1.0 o S/B > 1.5 (fuera de óptimo)
    
    FACTOR DE TIMING:
    =================
    
    f_timing = 1 + 0.1×|tp/tp_opt - 1| + 0.15×|tf/tf_opt - 1|
    
    - Penaliza desviaciones del timing óptimo
    - tp_opt = Th × S (timing pozos óptimo)
    - tf_opt = 11.5 × B (timing filas óptimo)
    
    OPTIMIZACIÓN:
    =============
    
    Se busca el factor de expansión óptimo (1.0 a 1.25) que minimiza J.
    
    Parámetros:
    - ucs: Resistencia a compresión uniaxial (MPa)
    - densidad_explosivo: Densidad del explosivo (g/cc)
    - vod: Velocidad de detonación (m/s)
    - diametro_pulg: Diámetro de perforación (pulgadas)
    - p80_objetivo: P80 objetivo máximo (pulgadas)
    - p100_objetivo: P100 objetivo máximo (pulgadas)
    - peso_metros: Peso del objetivo de minimizar metros (0-1)
    - peso_p80: Peso del objetivo de minimizar P80 (0-1)
    - peso_p100: Peso del objetivo de minimizar P100 (0-1)
    - altura_banco: Altura del banco (m)
    
    Retorna:
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
    
    # Constantes del modelo de fragmentación (calibradas con datos típicos de minería)
    # Basado en modelo Kuz-Ram y datos empíricos de fragmentación
    # P80 típico en minería cielo abierto: 5-12 pulgadas
    
    # Función para estimar P80 usando modelo Kuz-Ram modificado
    def estimar_P80(B, S, FC, tp, tf, tp_opt, tf_opt):
        """
        Modelo de fragmentación empírico para P80.
        
        FÓRMULA CALIBRADA:
        ==================
        
        P80 = K_base × f_malla × f_roca × f_energia × f_timing × f_vod
        
        Donde:
        - K_base = 5.5 (constante base para P80 ≈ 6" con malla óptima)
        - f_malla = (B×S / Area_ref)^0.40  [sensibilidad a área de malla]
        - f_roca = (UCS/100)^0.35  [sensibilidad a dureza]
        - f_energia = (FC_ref / FC)^0.30  [sensibilidad a factor de carga]
        - f_timing = 1 + pen_timing  [penalización por timing subóptimo]
        - f_vod = (VOD_ref / VOD)^0.15  [sensibilidad a velocidad detonación]
        
        Calibración objetivo:
        - Malla 4x5m, UCS=100, FC=0.75 → P80 ≈ 5.5"
        - Malla 5x6m, UCS=100, FC=0.75 → P80 ≈ 6.5"
        - Malla 6x7m, UCS=100, FC=0.75 → P80 ≈ 7.5"
        """
        area = B * S
        Area_ref = 20.0  # Área de referencia (m²)
        FC_ref = 0.75    # FC de referencia (kg/m³)
        VOD_ref = 4500   # VOD de referencia (m/s)
        
        # Constante base calibrada
        K_base = 5.5
        
        # Factor de malla - área mayor = fragmentación más gruesa
        f_malla = (area / Area_ref) ** 0.40
        
        # Factor de roca - UCS mayor = fragmentación más gruesa
        f_roca = (ucs / 100) ** 0.35
        
        # Factor de energía - FC mayor = fragmentación más fina
        f_energia = (FC_ref / max(FC, 0.3)) ** 0.30
        
        # Factor de timing - penaliza desviaciones del óptimo
        if tp_opt > 0 and tf_opt > 0:
            pen_pozos = 0.08 * abs(tp/tp_opt - 1)
            pen_filas = 0.12 * abs(tf/tf_opt - 1)
            f_timing = 1 + pen_pozos + pen_filas
        else:
            f_timing = 1.0
        
        # Factor de VOD - mayor VOD = mejor fragmentación
        f_vod = (VOD_ref / max(vod, 3000)) ** 0.15
        
        # Calcular P80
        P80_est = K_base * f_malla * f_roca * f_energia * f_timing * f_vod
        
        # Limitar a rango realista (4-15 pulgadas para P80)
        P80_est = max(4.0, min(15.0, P80_est))
        
        return P80_est
    
    # Función para estimar P100
    def estimar_P100(P80, S_B_ratio):
        if 1.10 <= S_B_ratio <= 1.30:
            k_ratio = 1.8  # Óptimo
        elif 1.0 <= S_B_ratio < 1.10 or 1.30 < S_B_ratio <= 1.40:
            k_ratio = 1.9  # Aceptable
        else:
            k_ratio = 2.1  # Fuera de óptimo
        
        return P80 * k_ratio
    
    # Función objetivo
    def funcion_objetivo(factor_exp):
        B = B_base * factor_exp
        S = S_base * factor_exp
        area = B * S
        
        # Ajustar FC para compensar (aumentar energía)
        FC = FC_base * (factor_exp ** 1.5)
        
        # Ajustar timing
        tp = tp_base * 0.95  # Reducir ligeramente
        tf = tf_base * 0.97
        
        # Timing óptimo para esta malla
        tp_opt = Th * S
        tf_opt = 11.5 * B
        
        # Estimar fragmentación
        P80_est = estimar_P80(B, S, FC, tp, tf, tp_opt, tf_opt)
        P100_est = estimar_P100(P80_est, S/B)
        
        # Calcular componentes normalizados de la función objetivo
        f_metros = 1 / area  # Metros por m² (menor es mejor)
        f_metros_norm = f_metros / (1 / (B_base * S_base))  # Normalizado vs base
        
        f_P80_norm = P80_est / p80_objetivo
        f_P100_norm = P100_est / p100_objetivo
        
        # Función objetivo ponderada
        J = peso_metros * f_metros_norm + peso_p80 * f_P80_norm + peso_p100 * f_P100_norm
        
        return J, B, S, FC, tp, tf, P80_est, P100_est, area
    
    # Buscar factor óptimo (búsqueda en grid)
    mejor_J = float('inf')
    mejor_resultado = None
    
    # Limitar factor según dureza de roca
    if pd.isna(ucs) or ucs >= 120:
        max_factor = 1.10
    elif ucs >= 80:
        max_factor = 1.15
    else:
        max_factor = 1.20
    
    for factor in np.arange(1.00, max_factor + 0.01, 0.01):
        J, B, S, FC, tp, tf, P80_est, P100_est, area = funcion_objetivo(factor)
        
        # Verificar restricciones
        if P80_est <= p80_objetivo and P100_est <= p100_objetivo:
            if J < mejor_J:
                mejor_J = J
                mejor_resultado = {
                    'factor_optimo': round(factor, 2),
                    'burden_optimo': round(B, 2),
                    'espaciamiento_optimo': round(S, 2),
                    'area_malla': round(area, 2),
                    'ratio_SB': round(S/B, 2),
                    'fc_ajustado': round(FC, 2),
                    'timing_pozos': round(tp, 1),
                    'timing_filas': round(tf, 1),
                    'taco_optimo': round(0.85 * B, 2),
                    'P80_estimado': round(P80_est, 2),
                    'P100_estimado': round(P100_est, 2),
                    'reduccion_metros_pct': round((1 - (B_base * S_base) / area) * 100, 1),
                    'funcion_objetivo': round(J, 4),
                    'cumple_P80': P80_est <= p80_objetivo,
                    'cumple_P100': P100_est <= p100_objetivo,
                    'explosivo_recomendado': params_base['explosivo_recomendado']
                }
    
    # Si no se encontró solución factible, usar la base
    if mejor_resultado is None:
        P80_base = estimar_P80(B_base, S_base, FC_base, tp_base, tf_base, tp_base, tf_base)
        P100_base = estimar_P100(P80_base, S_base/B_base)
        
        mejor_resultado = {
            'factor_optimo': 1.0,
            'burden_optimo': B_base,
            'espaciamiento_optimo': S_base,
            'area_malla': round(B_base * S_base, 2),
            'ratio_SB': round(S_base/B_base, 2),
            'fc_ajustado': FC_base,
            'timing_pozos': tp_base,
            'timing_filas': tf_base,
            'taco_optimo': round(0.85 * B_base, 2),
            'P80_estimado': round(P80_base, 2),
            'P100_estimado': round(P100_base, 2),
            'reduccion_metros_pct': 0.0,
            'funcion_objetivo': 1.0,
            'cumple_P80': P80_base <= p80_objetivo,
            'cumple_P100': P100_base <= p100_objetivo,
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
# SECCIÓN 1.1: RECOMENDACIONES DE MALLA (SIMPLIFICADA)
# ============================================

st.markdown("---")
st.header("Sección 1.1: Calculadora de Malla Óptima")

# ===== INPUT PRINCIPAL: SOLO UCS =====
st.markdown("### 🎯 Ingresa el UCS de tu zona de trabajo")

col_main1, col_main2 = st.columns([2, 1])

with col_main1:
    ucs_input = st.slider(
        "UCS objetivo (MPa)",
        min_value=30,
        max_value=200,
        value=100,
        step=10,
        help="Resistencia a compresión uniaxial de la roca"
    )

with col_main2:
    # Clasificación automática de roca
    if ucs_input < 50:
        tipo_roca = "🟢 Blanda"
        color_roca = "green"
    elif ucs_input < 100:
        tipo_roca = "🟡 Media"
        color_roca = "orange"
    elif ucs_input < 150:
        tipo_roca = "🟠 Dura"
        color_roca = "red"
    else:
        tipo_roca = "🔴 Muy Dura"
        color_roca = "darkred"
    
    st.metric("Tipo de Roca", tipo_roca)

# ===== OPCIONES AVANZADAS (COLAPSADAS) =====
with st.expander("⚙️ Opciones avanzadas (opcional)", expanded=False):
    col_adv1, col_adv2 = st.columns(2)
    
    with col_adv1:
        diametro_input = st.selectbox(
            "Diámetro perforación (pulg)",
            options=[5.0, 5.5, 6.0, 6.5, 6.75, 7.0, 7.875, 9.0, 10.625, 12.25],
            index=4
        )
        
        p80_objetivo = st.number_input("P80 objetivo máx (pulg)", 3.0, 15.0, 8.0, 0.5)
        
    with col_adv2:
        explosivo_input = st.selectbox(
            "Tipo de explosivo",
            options=list(EXPLOSIVOS_PROPIEDADES.keys()),
            index=0
        )
        
        p100_objetivo = st.number_input("P100 objetivo máx (pulg)", 8.0, 30.0, 15.0, 1.0)
    
    st.markdown("**Prioridades de optimización:**")
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        peso_metros = st.slider("Reducir metros", 0.0, 1.0, 0.40, 0.1)
    with col_p2:
        peso_p80 = st.slider("Reducir P80", 0.0, 1.0, 0.35, 0.1)
    with col_p3:
        peso_p100 = st.slider("Reducir P100", 0.0, 1.0, 0.25, 0.1)

# Valores por defecto si no se abrió el expander
if 'diametro_input' not in dir():
    diametro_input = 6.75
if 'explosivo_input' not in dir():
    explosivo_input = list(EXPLOSIVOS_PROPIEDADES.keys())[0]
if 'p80_objetivo' not in dir():
    p80_objetivo = 8.0
if 'p100_objetivo' not in dir():
    p100_objetivo = 15.0
if 'peso_metros' not in dir():
    peso_metros = 0.40
if 'peso_p80' not in dir():
    peso_p80 = 0.35
if 'peso_p100' not in dir():
    peso_p100 = 0.25

# Obtener propiedades del explosivo
prop_explosivo = EXPLOSIVOS_PROPIEDADES.get(explosivo_input, {})
densidad_exp = prop_explosivo.get('densidad', 1.2)
vod_exp = prop_explosivo.get('VOD', 4500)

# Normalizar pesos
suma_pesos = peso_metros + peso_p80 + peso_p100
peso_metros_norm = peso_metros / suma_pesos if suma_pesos > 0 else 1/3
peso_p80_norm = peso_p80 / suma_pesos if suma_pesos > 0 else 1/3
peso_p100_norm = peso_p100 / suma_pesos if suma_pesos > 0 else 1/3

# ===== CÁLCULOS AUTOMÁTICOS =====
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

# Calcular taco intermedio teórico (si aplica)
# Taco intermedio se recomienda cuando la columna explosiva es larga (>10m)
# Típicamente se coloca a 1/3 de la altura de la columna
altura_banco_est = 15.0  # metros estimados
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
# PANEL PRINCIPAL DE RECOMENDACIONES
# ============================================
st.markdown("---")
st.markdown("## 🎯 RECOMENDACIONES DE DISEÑO DE TRONADURA")
st.markdown(f"**Para UCS = {ucs_input} MPa ({tipo_roca})**")

# ===== CARDS DE RECOMENDACIONES =====
st.markdown("### 📋 Comparativa de Configuraciones")

# Crear tabla principal clara
col_param = [
    "**PARÁMETRO**",
    "━━━━━━━━━━━━━━",
    "**Malla (B × S)**",
    "**Área malla**",
    "**Factor de carga**",
    "━━━━━━━━━━━━━━",
    "**Taco superior**",
    "**Taco intermedio**",
    "━━━━━━━━━━━━━━",
    "**Timing pozos**",
    "**Timing filas**",
    "━━━━━━━━━━━━━━",
    "**P80 esperado**",
    "**P100 esperado**",
    "━━━━━━━━━━━━━━",
    "**Reducción metros**",
    "**Explosivo**"
]

# Valores para cada columna
col_teorica = [
    "📐 **TEÓRICA**",
    "",
    f"**{params_teoricos['burden_optimo']} × {params_teoricos['espaciamiento_optimo']} m**",
    f"{params_teoricos['area_malla']} m²",
    f"{params_teoricos['fc_optimo']} kg/m³",
    "",
    f"{params_teoricos['taco_optimo']} m",
    f"{taco_intermedio_teorico} m" if usar_taco_intermedio else "No requerido",
    "",
    f"{params_teoricos['timing_pozos_optimo']} ms",
    f"{params_teoricos['timing_filas_optimo']} ms",
    "",
    "~ 5-7\"",
    "~ 10-13\"",
    "",
    "0% (base)",
    params_teoricos['explosivo_recomendado']
]

col_optimizada = [
    "⚡ **OPTIMIZADA**",
    "",
    f"**{params_multiobj['burden_optimo']} × {params_multiobj['espaciamiento_optimo']} m**",
    f"{params_multiobj['area_malla']} m²",
    f"{params_multiobj['fc_ajustado']} kg/m³",
    "",
    f"{params_multiobj['taco_optimo']} m",
    f"{round(taco_intermedio_teorico * factor_expansion_auto, 2)} m" if usar_taco_intermedio else "No requerido",
    "",
    f"{params_multiobj['timing_pozos']} ms",
    f"{params_multiobj['timing_filas']} ms",
    "",
    f"**{params_multiobj['P80_estimado']}\"** {'✅' if params_multiobj['cumple_P80'] else '⚠️'}",
    f"**{params_multiobj['P100_estimado']}\"** {'✅' if params_multiobj['cumple_P100'] else '⚠️'}",
    "",
    f"**-{params_multiobj['reduccion_metros_pct']}%**",
    params_teoricos['explosivo_recomendado']
]

# Columna histórica
if mejor_historico is not None:
    burden_hist = mejor_historico.get('Burden', '-')
    esp_hist = mejor_historico.get('Espaciamiento', '-')
    fc_hist = mejor_historico.get('FC', mejor_historico.get('fc1', '-'))
    taco_hist = mejor_historico.get('taco_gravilla', '-')
    taco_int_hist = mejor_historico.get('taco_intermedio', '-')
    tp_hist = mejor_historico.get('tpozos_ms', '-')
    tf_hist = mejor_historico.get('tfilas_ms', '-')
    p80_hist = mejor_historico.get('P80TRON', '-')
    p100_hist = mejor_historico.get('P100TRON', '-')
    exp_hist = mejor_historico.get('Tipo_Explosivo', '-')
    
    # Calcular área histórica
    if pd.notna(burden_hist) and pd.notna(esp_hist):
        area_hist = round(float(burden_hist) * float(esp_hist), 1)
        malla_hist = f"**{burden_hist} × {esp_hist} m**"
    else:
        area_hist = '-'
        malla_hist = '-'
    
    col_historica = [
        "📊 **HISTÓRICA**",
        "",
        malla_hist,
        f"{area_hist} m²",
        f"{fc_hist} kg/m³" if pd.notna(fc_hist) else "-",
        "",
        f"{taco_hist} m" if pd.notna(taco_hist) else "-",
        f"{taco_int_hist} m" if pd.notna(taco_int_hist) and taco_int_hist != 0 else "No usado",
        "",
        f"{tp_hist} ms" if pd.notna(tp_hist) else "-",
        f"{tf_hist} ms" if pd.notna(tf_hist) else "-",
        "",
        f"**{round(p80_hist, 2)}\"**" if pd.notna(p80_hist) else "-",
        f"**{round(p100_hist, 2)}\"**" if pd.notna(p100_hist) else "-",
        "",
        "Dato real",
        str(exp_hist) if pd.notna(exp_hist) else "-"
    ]
else:
    col_historica = [
        "📊 **HISTÓRICA**",
        "",
        "Sin datos",
        "-",
        "-",
        "",
        "-",
        "-",
        "",
        "-",
        "-",
        "",
        "-",
        "-",
        "",
        "-",
        "-"
    ]

# Crear DataFrame para mostrar
df_recomendaciones = pd.DataFrame({
    'Parámetro': col_param,
    'Teórica ENAEX': col_teorica,
    'Optimizada': col_optimizada,
    'Mejor Histórica': col_historica
})

# Mostrar sin índice y con formato
st.dataframe(
    df_recomendaciones,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Parámetro": st.column_config.TextColumn(width="medium"),
        "Teórica ENAEX": st.column_config.TextColumn(width="medium"),
        "Optimizada": st.column_config.TextColumn(width="medium"),
        "Mejor Histórica": st.column_config.TextColumn(width="medium"),
    }
)

# ===== RESUMEN VISUAL EN CARDS =====
st.markdown("### 🏆 Resumen Rápido")

col_card1, col_card2, col_card3 = st.columns(3)

with col_card1:
    st.markdown(f"""
    <div style="background-color:#d4edda; padding:15px; border-radius:10px; border-left:5px solid #28a745;">
    <h4 style="margin:0; color:#155724;">📐 TEÓRICA</h4>
    <h2 style="margin:5px 0; color:#155724;">{params_teoricos['burden_optimo']} × {params_teoricos['espaciamiento_optimo']}</h2>
    <p style="margin:0; color:#155724;">
    Taco: {params_teoricos['taco_optimo']}m<br>
    Timing: {params_teoricos['timing_pozos_optimo']}/{params_teoricos['timing_filas_optimo']} ms<br>
    FC: {params_teoricos['fc_optimo']} kg/m³
    </p>
    </div>
    """, unsafe_allow_html=True)

with col_card2:
    cumple_icon = "✅" if (params_multiobj['cumple_P80'] and params_multiobj['cumple_P100']) else "⚠️"
    st.markdown(f"""
    <div style="background-color:#fff3cd; padding:15px; border-radius:10px; border-left:5px solid #ffc107;">
    <h4 style="margin:0; color:#856404;">⚡ OPTIMIZADA {cumple_icon}</h4>
    <h2 style="margin:5px 0; color:#856404;">{params_multiobj['burden_optimo']} × {params_multiobj['espaciamiento_optimo']}</h2>
    <p style="margin:0; color:#856404;">
    P80: {params_multiobj['P80_estimado']}" | P100: {params_multiobj['P100_estimado']}"<br>
    Timing: {params_multiobj['timing_pozos']}/{params_multiobj['timing_filas']} ms<br>
    <b>Ahorro: {params_multiobj['reduccion_metros_pct']}% metros</b>
    </p>
    </div>
    """, unsafe_allow_html=True)

with col_card3:
    if mejor_historico is not None and pd.notna(burden_hist) and pd.notna(esp_hist):
        st.markdown(f"""
        <div style="background-color:#cce5ff; padding:15px; border-radius:10px; border-left:5px solid #004085;">
        <h4 style="margin:0; color:#004085;">📊 HISTÓRICA</h4>
        <h2 style="margin:5px 0; color:#004085;">{burden_hist} × {esp_hist}</h2>
        <p style="margin:0; color:#004085;">
        P80: {round(p80_hist, 1) if pd.notna(p80_hist) else '-'}" | P100: {round(p100_hist, 1) if pd.notna(p100_hist) else '-'}"<br>
        Timing: {tp_hist}/{tf_hist} ms<br>
        <b>Resultado real probado</b>
        </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background-color:#e2e3e5; padding:15px; border-radius:10px; border-left:5px solid #6c757d;">
        <h4 style="margin:0; color:#383d41;">📊 HISTÓRICA</h4>
        <h2 style="margin:5px 0; color:#383d41;">Sin datos</h2>
        <p style="margin:0; color:#383d41;">
        No hay tronaduras históricas<br>
        para UCS ≈ {ucs_input} MPa<br>
        en los datos cargados
        </p>
        </div>
        """, unsafe_allow_html=True)

# ===== TOP 3 HISTÓRICOS =====
if top3_historico is not None and len(top3_historico) > 0:
    st.markdown("---")
    st.markdown("### 🏅 Top 3 Mejores Tronaduras Históricas")
    st.markdown(f"*Tronaduras con UCS entre {ucs_input - ucs_tolerance} y {ucs_input + ucs_tolerance} MPa, ordenadas por mejor P80*")
    
    # Seleccionar columnas relevantes
    cols_top3 = ['BxS', 'Burden', 'Espaciamiento', 'FC', 'taco_gravilla', 'taco_intermedio',
                 'tpozos_ms', 'tfilas_ms', 'Tipo_Explosivo', 'P80TRON', 'P100TRON', 'UCS_MPA',
                 'Fase_cat', 'Banco']
    cols_top3 = [c for c in cols_top3 if c in top3_historico.columns]
    
    # Renombrar columnas para claridad
    df_top3_display = top3_historico[cols_top3].copy()
    rename_map = {
        'BxS': 'Malla',
        'Burden': 'B (m)',
        'Espaciamiento': 'S (m)',
        'FC': 'FC',
        'taco_gravilla': 'Taco (m)',
        'taco_intermedio': 'Taco Int.',
        'tpozos_ms': 'T.Pozos',
        'tfilas_ms': 'T.Filas',
        'Tipo_Explosivo': 'Explosivo',
        'P80TRON': 'P80',
        'P100TRON': 'P100',
        'UCS_MPA': 'UCS',
        'Fase_cat': 'Fase',
        'Banco': 'Banco'
    }
    df_top3_display = df_top3_display.rename(columns={k: v for k, v in rename_map.items() if k in df_top3_display.columns})
    
    # Redondear valores numéricos
    for col in df_top3_display.select_dtypes(include=[np.number]).columns:
        df_top3_display[col] = df_top3_display[col].round(2)
    
    st.dataframe(df_top3_display, use_container_width=True, hide_index=True)
    
    # Estadísticas de los top 3
    st.markdown("**📈 Estadísticas de las mejores tronaduras:**")
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    
    with col_stat1:
        if 'P80TRON' in top3_historico.columns:
            st.metric("P80 promedio", f"{top3_historico['P80TRON'].mean():.2f}\"")
    with col_stat2:
        if 'P100TRON' in top3_historico.columns:
            st.metric("P100 promedio", f"{top3_historico['P100TRON'].mean():.2f}\"")
    with col_stat3:
        if 'Burden' in top3_historico.columns and 'Espaciamiento' in top3_historico.columns:
            area_prom = (top3_historico['Burden'] * top3_historico['Espaciamiento']).mean()
            st.metric("Área malla prom.", f"{area_prom:.1f} m²")
    with col_stat4:
        if 'FC' in top3_historico.columns:
            st.metric("FC promedio", f"{top3_historico['FC'].mean():.2f}")

# ===== INFORMACIÓN SOBRE TACO INTERMEDIO =====
st.markdown("---")
st.markdown("### 📏 Recomendación de Taco Intermedio")

if usar_taco_intermedio:
    st.info(f"""
    **Se recomienda usar taco intermedio** para esta configuración.
    
    📐 **Teórico (ENAEX):**
    - Longitud de columna explosiva estimada: **{longitud_carga:.1f} m** (> 10m)
    - Taco intermedio recomendado: **{taco_intermedio_teorico} m**
    - Posición desde fondo del pozo: **{posicion_taco_int} m**
    - Material: Gravilla 3/4" o detritus de perforación
    
    💡 **Beneficios del taco intermedio:**
    - Mejor distribución de energía en la columna
    - Reduce la presión en el taco superior
    - Mejora la fragmentación en la parte superior del banco
    """)
else:
    st.success(f"""
    **No se requiere taco intermedio** para esta configuración.
    
    - Longitud de columna explosiva estimada: **{longitud_carga:.1f} m** (< 10m)
    - Con columnas cortas, un solo taco superior es suficiente
    """)

# Mostrar histórico de tacos intermedios si existen
if 'taco_intermedio' in df_filtrado.columns:
    df_con_taco_int = df_filtrado[df_filtrado['taco_intermedio'].notna() & (df_filtrado['taco_intermedio'] > 0)]
    if len(df_con_taco_int) > 0:
        st.markdown("**📊 Tacos intermedios usados históricamente:**")
        taco_int_stats = df_con_taco_int['taco_intermedio'].describe()
        col_ti1, col_ti2, col_ti3 = st.columns(3)
        with col_ti1:
            st.metric("Mínimo usado", f"{taco_int_stats['min']:.1f} m")
        with col_ti2:
            st.metric("Promedio", f"{taco_int_stats['mean']:.1f} m")
        with col_ti3:
            st.metric("Máximo usado", f"{taco_int_stats['max']:.1f} m")

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
with st.expander("📚 Ver fórmulas y explicaciones", expanded=False):
    st.markdown(f"""
    ### Fórmulas utilizadas para UCS = {ucs_input} MPa
    
    **1. Burden (Ash modificada):**
    ```
    B = (Kb × De × (ρe/ρr)^0.33 × (VOD/4000)^0.5) / 39.37
    B = ({params_teoricos['Kb']} × {diametro_input} × ({densidad_exp}/2.65)^0.33 × ({vod_exp}/4000)^0.5) / 39.37
    B = {params_teoricos['burden_optimo']} m
    ```
    
    **2. Espaciamiento:**
    ```
    S = Ks × B = {params_teoricos['ratio_SB']} × {params_teoricos['burden_optimo']} = {params_teoricos['espaciamiento_optimo']} m
    ```
    
    **3. Timing (Konya):**
    ```
    Timing pozos = Th × S = {params_teoricos['Th']} × {params_teoricos['espaciamiento_optimo']} = {params_teoricos['timing_pozos_optimo']} ms
    Timing filas = 11.5 × B = 11.5 × {params_teoricos['burden_optimo']} = {params_teoricos['timing_filas_optimo']} ms
    ```
    
    **4. Taco:**
    ```
    Taco = 0.85 × B = 0.85 × {params_teoricos['burden_optimo']} = {params_teoricos['taco_optimo']} m
    ```
    
    **5. Modelo de fragmentación (P80):**
    ```
    P80 = 5.5 × (Área/20)^0.40 × (UCS/100)^0.35 × (0.75/FC)^0.30 × f_timing × f_vod
    ```
    
    ---
    
    ### Constantes según dureza de roca
    
    | UCS (MPa) | Tipo | Kb | Ks (S/B) | Th (ms/m) | FC óptimo |
    |-----------|------|-----|----------|-----------|-----------|
    | < 50 | Blanda | 35 | 1.40 | 6.5 | 0.35 |
    | 50-100 | Media | 30 | 1.30 | 5.5 | 0.50 |
    | 100-150 | Dura | 28 | 1.20 | 4.5 | 0.75 |
    | > 150 | Muy dura | 25 | 1.15 | 3.5 | 1.00 |
    
    ---
    
    ### Referencias
    - Manual de Tronadura ENAEX
    - Ash (1963), Konya (1995), Cunningham (1983)
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
    
    # Crear gráfico de tornado
    fig_sens, ax_sens = plt.subplots(figsize=(10, 5))
    
    variables = [v[0] for v in vars_ordenadas]
    cambios_up = [v[1]['cambio_up'] for v in vars_ordenadas]
    cambios_down = [v[1]['cambio_down'] for v in vars_ordenadas]
    
    y_pos = np.arange(len(variables))
    
    # Barras hacia la derecha (aumento del parámetro)
    bars_up = ax_sens.barh(y_pos, cambios_up, height=0.4, label='+20% parámetro', color='#e74c3c', alpha=0.8)
    # Barras hacia la izquierda (disminución del parámetro)
    bars_down = ax_sens.barh(y_pos, cambios_down, height=0.4, label='-20% parámetro', color='#3498db', alpha=0.8)
    
    ax_sens.set_yticks(y_pos)
    ax_sens.set_yticklabels(variables)
    ax_sens.set_xlabel('Cambio en P80 (%)')
    ax_sens.set_title(f'Sensibilidad del P80 a variaciones de ±20%\n(P80 base = {p80_base:.2f}")', fontweight='bold')
    ax_sens.axvline(x=0, color='black', linewidth=0.5)
    ax_sens.legend(loc='lower right')
    ax_sens.grid(axis='x', alpha=0.3)
    
    # Añadir valores en las barras
    for i, (up, down) in enumerate(zip(cambios_up, cambios_down)):
        if abs(up) > 1:
            ax_sens.text(up + 0.5 if up > 0 else up - 0.5, i, f'{up:+.1f}%', 
                        va='center', ha='left' if up > 0 else 'right', fontsize=9)
        if abs(down) > 1:
            ax_sens.text(down + 0.5 if down > 0 else down - 0.5, i, f'{down:+.1f}%', 
                        va='center', ha='left' if down > 0 else 'right', fontsize=9)
    
    plt.tight_layout()
    st.pyplot(fig_sens)
    plt.close(fig_sens)
    
    # Tabla de sensibilidad
    st.markdown("### 📋 Tabla de Sensibilidad")
    
    df_sens = pd.DataFrame({
        'Variable': [v[0] for v in vars_ordenadas],
        'Si aumenta 20%': [f"P80 {v[1]['cambio_up']:+.1f}%" for v in vars_ordenadas],
        'Si disminuye 20%': [f"P80 {v[1]['cambio_down']:+.1f}%" for v in vars_ordenadas],
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