# -*- coding: utf-8 -*-
"""
================================================================================
ANALISIS DETALLADO FASES F7R1 vs F9SE
================================================================================

Análisis específico de explosivos, propiedades físicas (densidad, VOD, energía),
fragmentación (P100, P80, P50), taco y sensibilidad según fórmulas ENAEX.

Autor: Análisis para María Ignacia
Fecha: Diciembre 2025

================================================================================
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')

# Configuración de gráficos
plt.rcParams.update({
    'figure.figsize': (14, 10),
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'axes.titleweight': 'bold',
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight'
})
sns.set_style("whitegrid")

# Directorio de salida
OUTPUT_DIR = 'analisis_f7_f9'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# PROPIEDADES DE EXPLOSIVOS (Datos ENAEX actualizados)
# ==============================================================================
# Fuente: Tabla de características técnicas ENAEX
# VOD: Velocidad de detonación típica (m/s)
# RWS: Potencia relativa ANFO peso (×100)
# Energia: MJ/kg (KJ/kg ÷ 1000)
# Presion: Presión de detonación (Kbar)

EXPLOSIVOS_PROPIEDADES = {
    # PIREX Series - Emulsiones encartuchadas y bombeables
    'Pirex S': {'densidad': 1.29, 'VOD': 4500, 'RWS': 81, 'energia': 2.865, 'presion': 65.0},
    'Pirex S plus': {'densidad': 1.27, 'VOD': 5400, 'RWS': 81, 'energia': 2.864, 'presion': 93.0},
    'Pirex S Plus': {'densidad': 1.27, 'VOD': 5400, 'RWS': 81, 'energia': 2.864, 'presion': 93.0},
    'Pirex S HP': {'densidad': 1.25, 'VOD': 6000, 'RWS': 81, 'energia': 2.836, 'presion': 113.0},
    'Pirex P-20': {'densidad': 0.88, 'VOD': 3756, 'RWS': 95, 'energia': 3.599, 'presion': 31.0},
    'Pirex 20': {'densidad': 0.88, 'VOD': 3756, 'RWS': 95, 'energia': 3.599, 'presion': 31.0},
    'Pirex P-30': {'densidad': 1.00, 'VOD': 3949, 'RWS': 93, 'energia': 3.486, 'presion': 39.0},
    'Pirex 30': {'densidad': 1.00, 'VOD': 3949, 'RWS': 93, 'energia': 3.486, 'presion': 39.0},
    'Pirex P-40': {'densidad': 1.20, 'VOD': 4018, 'RWS': 91, 'energia': 3.377, 'presion': 48.0},
    'Pirex 40': {'densidad': 1.20, 'VOD': 4018, 'RWS': 91, 'energia': 3.377, 'presion': 48.0},
    'Pirex P-45': {'densidad': 1.30, 'VOD': 4006, 'RWS': 90, 'energia': 3.318, 'presion': 52.0},
    'Pirex 45': {'densidad': 1.30, 'VOD': 4006, 'RWS': 90, 'energia': 3.318, 'presion': 52.0},
    'Pirex P-50': {'densidad': 1.30, 'VOD': 4101, 'RWS': 89, 'energia': 3.264, 'presion': 55.0},
    'Pirex 50': {'densidad': 1.30, 'VOD': 4101, 'RWS': 89, 'energia': 3.264, 'presion': 55.0},
    'Pirex 950': {'densidad': 1.30, 'VOD': 4101, 'RWS': 89, 'energia': 3.264, 'presion': 55.0},
    'Pirex-950': {'densidad': 1.30, 'VOD': 4101, 'RWS': 89, 'energia': 3.264, 'presion': 55.0},
    'Pirex P-65': {'densidad': 1.32, 'VOD': 3200, 'RWS': 82, 'energia': 3.028, 'presion': 33.8},
    'Pirex 65': {'densidad': 1.32, 'VOD': 3200, 'RWS': 82, 'energia': 3.028, 'presion': 33.8},
    'Pirex P-70': {'densidad': 1.32, 'VOD': 3161, 'RWS': 82, 'energia': 3.207, 'presion': 33.0},
    'Pirex 70': {'densidad': 1.32, 'VOD': 3161, 'RWS': 82, 'energia': 3.207, 'presion': 33.0},

    # HIDREX Series - Emulsiones de baja densidad
    'Hidrex LD500': {'densidad': 0.50, 'VOD': 2500, 'RWS': 82, 'energia': 3.027, 'presion': 7.81},
    'Hidrex LD600': {'densidad': 0.60, 'VOD': 2800, 'RWS': 81, 'energia': 2.974, 'presion': 12.0},
    'Hidrex LD700': {'densidad': 0.70, 'VOD': 2900, 'RWS': 82, 'energia': 3.027, 'presion': 14.72},
    'Hidrex LD800': {'densidad': 0.80, 'VOD': 3300, 'RWS': 81, 'energia': 2.974, 'presion': 22.0},
    'Hidrex LD900': {'densidad': 0.90, 'VOD': 3400, 'RWS': 82, 'energia': 3.027, 'presion': 26.01},

    # ENERGEX Series - Emulsiones sensibilizadas in-situ
    'Energex': {'densidad': 1.29, 'VOD': 5300, 'RWS': 88, 'energia': 3.232, 'presion': 91.0},
    'Energex 50': {'densidad': 1.34, 'VOD': 3800, 'RWS': 88, 'energia': 3.252, 'presion': 48.0},
    'Energex 50 Plus': {'densidad': 1.30, 'VOD': 4900, 'RWS': 90, 'energia': 3.361, 'presion': 78.0},
    'Energex 50 plus': {'densidad': 1.30, 'VOD': 4900, 'RWS': 90, 'energia': 3.361, 'presion': 78.0},
    'Energex 70': {'densidad': 1.29, 'VOD': 5300, 'RWS': 88, 'energia': 3.232, 'presion': 91.0},
    'Energex 70 Plus': {'densidad': 1.28, 'VOD': 5800, 'RWS': 89, 'energia': 3.299, 'presion': 108.0},
    'Energex 70 plus': {'densidad': 1.28, 'VOD': 5800, 'RWS': 89, 'energia': 3.299, 'presion': 108.0},

    # EMULTEX Series - Emulsiones bombeables
    'Emultex BN': {'densidad': 1.32, 'VOD': 4680, 'RWS': 84, 'energia': 3.017, 'presion': 50.0},
    'EMULTEX BN': {'densidad': 1.32, 'VOD': 4680, 'RWS': 84, 'energia': 3.017, 'presion': 50.0},
    'Emultex BN 1600': {'densidad': 1.32, 'VOD': 4680, 'RWS': 84, 'energia': 3.017, 'presion': 50.0},
    'Emultex BN1600': {'densidad': 1.32, 'VOD': 4680, 'RWS': 84, 'energia': 3.017, 'presion': 50.0},

    # BLENDEX Series - Mezclas emulsión/ANFO
    'Blendex 920': {'densidad': 0.90, 'VOD': 3900, 'RWS': 95, 'energia': 3.590, 'presion': 35.0},
    'Blendex 930': {'densidad': 1.00, 'VOD': 3920, 'RWS': 93, 'energia': 3.473, 'presion': 40.0},
    'Blendex 940': {'densidad': 1.20, 'VOD': 3950, 'RWS': 91, 'energia': 3.360, 'presion': 52.0},
    'Blendex 945': {'densidad': 1.32, 'VOD': 4200, 'RWS': 89, 'energia': 3.305, 'presion': 62.0},
    'Blendex 950': {'densidad': 1.32, 'VOD': 4150, 'RWS': 88, 'energia': 3.247, 'presion': 59.0},
    'BLENDEX 950': {'densidad': 1.32, 'VOD': 4150, 'RWS': 88, 'energia': 3.247, 'presion': 59.0},

    # VERTEX ALR Series - Agentes de voladura de baja resistencia al agua
    'Vertex ALR 920': {'densidad': 0.86, 'VOD': 3333, 'RWS': 95, 'energia': 3.559, 'presion': 24.0},
    'Vertex 920': {'densidad': 0.86, 'VOD': 3333, 'RWS': 95, 'energia': 3.559, 'presion': 24.0},
    'Vertex ALR 930': {'densidad': 1.00, 'VOD': 3390, 'RWS': 92, 'energia': 3.423, 'presion': 29.0},
    'Vertex 930': {'densidad': 1.00, 'VOD': 3390, 'RWS': 92, 'energia': 3.423, 'presion': 29.0},
    'Vertex ALR 940': {'densidad': 1.23, 'VOD': 4103, 'RWS': 89, 'energia': 3.297, 'presion': 52.0},
    'Vertex 940': {'densidad': 1.23, 'VOD': 4103, 'RWS': 89, 'energia': 3.297, 'presion': 52.0},
    'Vertex 40': {'densidad': 1.23, 'VOD': 4103, 'RWS': 89, 'energia': 3.297, 'presion': 52.0},
    'Vertex ALR 945': {'densidad': 1.30, 'VOD': 3408, 'RWS': 88, 'energia': 3.226, 'presion': 38.0},
    'Vertex 945': {'densidad': 1.30, 'VOD': 3408, 'RWS': 88, 'energia': 3.226, 'presion': 38.0},
    'Vertex ALR 950': {'densidad': 1.32, 'VOD': 3363, 'RWS': 87, 'energia': 3.163, 'presion': 37.0},
    'Vertex 950': {'densidad': 1.32, 'VOD': 3363, 'RWS': 87, 'energia': 3.163, 'presion': 37.0},
    'Vertex 50': {'densidad': 1.32, 'VOD': 3363, 'RWS': 87, 'energia': 3.163, 'presion': 37.0},
    'Vertex ALR 970': {'densidad': 1.34, 'VOD': 3293, 'RWS': 82, 'energia': 2.920, 'presion': 36.0},
    'Vertex 970': {'densidad': 1.34, 'VOD': 3293, 'RWS': 82, 'energia': 2.920, 'presion': 36.0},
    'VERTEX 970': {'densidad': 1.34, 'VOD': 3293, 'RWS': 82, 'energia': 2.920, 'presion': 36.0},
    'Vertex 70': {'densidad': 1.34, 'VOD': 3293, 'RWS': 82, 'energia': 2.920, 'presion': 36.0},
    'Vertex70': {'densidad': 1.34, 'VOD': 3293, 'RWS': 82, 'energia': 2.920, 'presion': 36.0},

    # VERTEX S Series - Emulsiones sensibilizadas
    'Vertex S': {'densidad': 1.31, 'VOD': 4250, 'RWS': 83, 'energia': 3.032, 'presion': 45.0},
    'Vertex-S': {'densidad': 1.31, 'VOD': 4250, 'RWS': 83, 'energia': 3.032, 'presion': 45.0},
    'Vertex S Plus': {'densidad': 1.29, 'VOD': 5400, 'RWS': 82, 'energia': 2.994, 'presion': 87.0},
    'Vertex S plus': {'densidad': 1.29, 'VOD': 5400, 'RWS': 82, 'energia': 2.994, 'presion': 87.0},

    # Otros
    'TITAN 1000': {'densidad': 1.30, 'VOD': 6000, 'RWS': 100, 'energia': 4.0, 'presion': 120.0},
    'Differential Energy': {'densidad': 1.29, 'VOD': 5300, 'RWS': 88, 'energia': 3.232, 'presion': 91.0},
    'Vistan 250': {'densidad': 1.25, 'VOD': 5600, 'RWS': 90, 'energia': 3.3, 'presion': 100.0},
}


# ==============================================================================
# FUNCIONES DE CARGA Y PREPARACIÓN DE DATOS
# ==============================================================================

def cargar_datos():
    """Carga y combina los archivos de datos."""
    print("=" * 70)
    print("CARGANDO DATOS")
    print("=" * 70)

    archivos = [
        'merge_detalle_p1 1.csv',
        'merge_detalle_p2 1.csv',
        'merge_detalle_p3.csv'
    ]

    dfs = []
    for archivo in archivos:
        if os.path.exists(archivo):
            df = pd.read_csv(archivo)
            dfs.append(df)
            print(f"  Cargado: {archivo} ({len(df):,} registros)")

    df = pd.concat(dfs, ignore_index=True).drop_duplicates()
    print(f"\nTotal registros: {len(df):,}")
    return df


def preparar_datos(df):
    """Prepara y filtra los datos para análisis."""
    print("\n" + "=" * 70)
    print("PREPARANDO DATOS")
    print("=" * 70)

    df = df.copy()

    # Convertir columnas numéricas
    cols_numericas = ['P80', 'P100', 'P50', 'P20', 'Burden', 'Espaciamiento',
                      'Fc', 'UCS_MPA', 'tpozos_ms', 'tfilas_ms', 'taco', 'gravilla']
    for col in cols_numericas:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Categorizar fase
    def categorizar_fase(fase):
        if pd.isna(fase):
            return 'Otros'
        fase_str = str(fase).upper()
        if 'F7R1' in fase_str or fase_str == 'F7':
            return 'F7R1'
        elif 'F9SE' in fase_str or 'F9' in fase_str:
            return 'F9SE'
        else:
            return 'Otros'

    df['Fase_categoria'] = df['Fase_tron'].apply(categorizar_fase)

    # Filtrar solo F7R1 y F9SE
    df = df[df['Fase_categoria'].isin(['F7R1', 'F9SE'])].copy()

    # Variables calculadas
    df['S_B_ratio'] = df['Espaciamiento'] / df['Burden']
    df['Area_malla'] = df['Burden'] * df['Espaciamiento']
    df['taco_efectivo'] = df['taco'].fillna(0)
    df['taco_burden_ratio'] = df['taco_efectivo'] / df['Burden']
    df['taco_total'] = df['taco'].fillna(0) + df['gravilla'].fillna(0)

    # Asignar propiedades de explosivos
    def obtener_propiedad(explosivo, prop):
        if pd.isna(explosivo):
            return np.nan
        exp_str = str(explosivo).strip()
        if exp_str in EXPLOSIVOS_PROPIEDADES:
            return EXPLOSIVOS_PROPIEDADES[exp_str].get(prop, np.nan)
        return np.nan

    df['rho_explosivo'] = df['Tipo_Explosivo'].apply(lambda x: obtener_propiedad(x, 'densidad'))
    df['VOD'] = df['Tipo_Explosivo'].apply(lambda x: obtener_propiedad(x, 'VOD'))
    df['RWS'] = df['Tipo_Explosivo'].apply(lambda x: obtener_propiedad(x, 'RWS'))
    df['Energia_MJ'] = df['Tipo_Explosivo'].apply(lambda x: obtener_propiedad(x, 'energia'))

    # Gamma lineal (kg/m) según ENAEX
    diametro_m = 0.27  # 270mm típico
    df['gamma_lineal'] = np.pi / 4 * diametro_m**2 * df['rho_explosivo'] * 1000

    # Presión de detonación (GPa)
    df['P_detonacion'] = df['rho_explosivo'] * (df['VOD'] / 1000)**2 / 4

    # Categorías de UCS
    df['UCS_categoria'] = pd.cut(
        df['UCS_MPA'],
        bins=[0, 60, 80, 100, 120, 140, 500],
        labels=['<60', '60-80', '80-100', '100-120', '120-140', '>140']
    )

    # Filtrar registros válidos
    df_clean = df[
        (df['P80'].notna()) &
        (df['P100'].notna()) &
        (df['Burden'].notna()) &
        (df['Burden'] > 0) &
        (df['Espaciamiento'].notna()) &
        (df['Espaciamiento'] > 0)
    ].copy()

    print(f"  F7R1: {len(df_clean[df_clean['Fase_categoria'] == 'F7R1']):,} registros")
    print(f"  F9SE: {len(df_clean[df_clean['Fase_categoria'] == 'F9SE']):,} registros")
    print(f"  Total válidos: {len(df_clean):,}")

    return df_clean


# ==============================================================================
# CÁLCULOS ENAEX
# ==============================================================================

def calcular_timing_enaex(df):
    """
    Calcula timing óptimo según Manual ENAEX Cap. 7.

    Fórmulas:
    - Konya: th = Th × S (Th = 3.5-6.5 ms/m según roca)
    - Richard Ash: td = 3.28×B + PC/VOD + x/vb
    - Velocidad burden: vb = 0.0055 × VOD² / Vp
    """
    print("\n" + "=" * 70)
    print("CALCULANDO TIMING ÓPTIMO (ENAEX)")
    print("=" * 70)

    df = df.copy()

    # Parámetros
    Vp_default = 4500  # Velocidad onda P (m/s)
    H_banco = 15  # Altura de banco (m)
    ratio_carga = 0.70  # PC/H
    x_cara_libre = 0.25  # m

    # Constante Th según UCS (Manual ENAEX Tabla 7.1)
    def obtener_Th(ucs):
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

    df['Th_konya'] = df['UCS_MPA'].apply(obtener_Th)

    # Timing óptimo entre pozos (Konya)
    df['tpozos_optimo'] = df['Th_konya'] * df['Espaciamiento']

    # Timing óptimo entre filas (Konya)
    Tr_optimo = 11.5  # ms/m para fragmentación óptima
    df['tfilas_optimo'] = Tr_optimo * df['Burden']

    # Velocidad de desplazamiento del burden
    df['vb_enaex'] = np.where(
        df['VOD'].notna() & (df['VOD'] > 0),
        0.0055 * (df['VOD']**2) / Vp_default,
        0.0055 * (5000**2) / Vp_default
    )

    # Timing Richard Ash
    df['PC'] = H_banco * ratio_carga
    df['tm_ash'] = 3.28 * df['Burden']  # ms
    df['te_ash'] = df['PC'] / df['VOD'].fillna(5000) * 1000  # ms
    df['tb_ash'] = x_cara_libre / df['vb_enaex'] * 1000  # ms
    df['tfilas_ash'] = df['tm_ash'] + df['te_ash'] + df['tb_ash']

    # Desviación del timing real vs óptimo
    df['desv_tpozos'] = df['tpozos_ms'] - df['tpozos_optimo']
    df['desv_tfilas'] = df['tfilas_ms'] - df['tfilas_optimo']

    # Clasificación del timing
    def clasificar_timing(row):
        if pd.isna(row['tpozos_ms']) or pd.isna(row['tpozos_optimo']):
            return 'Sin datos'
        ratio = row['tpozos_ms'] / row['tpozos_optimo']
        if ratio < 0.5:
            return 'Muy bajo'
        elif ratio < 0.8:
            return 'Bajo'
        elif ratio <= 1.2:
            return 'Óptimo'
        elif ratio <= 1.5:
            return 'Alto'
        else:
            return 'Muy alto'

    df['timing_eval'] = df.apply(clasificar_timing, axis=1)

    print(f"  Timing pozos real promedio: {df['tpozos_ms'].mean():.1f} ms")
    print(f"  Timing pozos óptimo ENAEX: {df['tpozos_optimo'].mean():.1f} ms")
    print(f"  Desviación promedio: {df['desv_tpozos'].mean():+.1f} ms")

    return df


def calcular_taco_enaex(df):
    """
    Calcula parámetros de taco según Manual ENAEX.

    Recomendaciones ENAEX:
    - Taco mínimo: 0.7 × Burden
    - Taco óptimo: 0.8-1.0 × Burden
    - Taco/Burden < 0.5 → Riesgo de flyrock
    """
    print("\n" + "=" * 70)
    print("CALCULANDO PARÁMETROS DE TACO (ENAEX)")
    print("=" * 70)

    df = df.copy()

    # Taco óptimo según ENAEX
    df['taco_minimo_enaex'] = 0.7 * df['Burden']
    df['taco_optimo_enaex'] = 0.85 * df['Burden']

    # Evaluación del taco
    def evaluar_taco(row):
        if pd.isna(row['taco_efectivo']) or pd.isna(row['Burden']):
            return 'Sin datos'
        ratio = row['taco_efectivo'] / row['Burden']
        if ratio < 0.5:
            return 'Crítico (flyrock)'
        elif ratio < 0.7:
            return 'Bajo'
        elif ratio <= 1.0:
            return 'Óptimo'
        else:
            return 'Alto'

    df['taco_eval'] = df.apply(evaluar_taco, axis=1)

    # Desviación del taco óptimo
    df['desv_taco'] = df['taco_efectivo'] - df['taco_optimo_enaex']

    print(f"  Taco real promedio: {df['taco_efectivo'].mean():.2f} m")
    print(f"  Taco óptimo ENAEX: {df['taco_optimo_enaex'].mean():.2f} m")
    print(f"  Ratio T/B promedio: {df['taco_burden_ratio'].mean():.2f}")

    return df


def calcular_parametros_enaex(df):
    """
    Calcula todos los parámetros óptimos según teoría ENAEX.

    Fórmulas ENAEX (Manual de Tronadura):
    =====================================

    1. BURDEN ÓPTIMO (Ash modificado):
       B = Kb × De × (ρe/ρr)^0.33 × (VOD/4000)^0.5
       Donde:
       - Kb = 25-40 (constante según tipo de roca)
       - De = diámetro de perforación (pulg)
       - ρe = densidad explosivo (g/cc)
       - ρr = densidad roca (g/cc) ≈ 2.6-2.8
       - VOD = velocidad detonación (m/s)

    2. ESPACIAMIENTO ÓPTIMO:
       S = Ks × B
       Donde Ks = 1.15 (roca dura) a 1.40 (roca blanda)

    3. PASADURA (Subdrilling):
       J = 0.2 a 0.5 × B (típico 0.3×B)

    4. TACO ÓPTIMO:
       T = 0.7 a 1.0 × B

    5. FACTOR DE CARGA ÓPTIMO (kg/m³):
       FC = f(UCS, tipo explosivo, geometría)
       - Roca blanda (UCS<50): 0.2-0.4 kg/m³
       - Roca media (50-100): 0.4-0.6 kg/m³
       - Roca dura (100-150): 0.6-0.9 kg/m³
       - Roca muy dura (>150): 0.9-1.2 kg/m³

    6. ENERGÍA ESPECÍFICA REQUERIDA (MJ/m³):
       E_req = Ce × UCS^0.63
       Donde Ce = 0.004-0.006 (constante empírica)

    7. SELECCIÓN DE EXPLOSIVO según dureza:
       - UCS < 50 MPa: ANFO, Blendex baja densidad
       - UCS 50-100: Blendex, Vertex ALR
       - UCS 100-150: Emultex, Energex
       - UCS > 150: Pirex S, Energex 70 Plus
    """
    print("\n" + "=" * 70)
    print("CALCULANDO PARÁMETROS ÓPTIMOS (ENAEX)")
    print("=" * 70)

    df = df.copy()

    # Parámetros de roca (estimados)
    rho_roca = 2.65  # g/cc densidad típica

    # 1. BURDEN ÓPTIMO
    # ================
    def calcular_Kb(ucs):
        """Constante de burden según dureza de roca."""
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

    df['Kb_enaex'] = df['UCS_MPA'].apply(calcular_Kb)

    # Diámetro asumido (convertir de mm a pulgadas si existe)
    if 'Diametro_mm' in df.columns:
        df['De_pulg'] = df['Diametro_mm'] / 25.4
    else:
        df['De_pulg'] = 6.75  # 6 3/4" típico en minería a cielo abierto

    # Burden óptimo Ash modificado
    df['burden_optimo_enaex'] = (
        df['Kb_enaex'] * df['De_pulg'] *
        ((df['rho_explosivo'].fillna(1.2) / rho_roca) ** 0.33) *
        ((df['VOD'].fillna(4500) / 4000) ** 0.5)
    ) / 39.37  # Convertir a metros

    df['desv_burden'] = df['Burden'] - df['burden_optimo_enaex']
    df['burden_ratio'] = df['Burden'] / df['burden_optimo_enaex']

    # 2. ESPACIAMIENTO ÓPTIMO
    # =======================
    def calcular_Ks(ucs):
        """Constante S/B según dureza de roca."""
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

    df['Ks_enaex'] = df['UCS_MPA'].apply(calcular_Ks)
    df['espaciamiento_optimo_enaex'] = df['Ks_enaex'] * df['burden_optimo_enaex']
    df['SB_ratio_real'] = df['Espaciamiento'] / df['Burden']
    df['SB_ratio_optimo'] = df['Ks_enaex']
    df['desv_espaciamiento'] = df['Espaciamiento'] - df['espaciamiento_optimo_enaex']

    # 3. PASADURA (Subdrilling)
    # =========================
    df['pasadura_optima_enaex'] = 0.3 * df['burden_optimo_enaex']

    # 4. FACTOR DE CARGA ÓPTIMO
    # =========================
    def calcular_fc_optimo(ucs):
        """Factor de carga óptimo según UCS (kg/m³)."""
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

    df['fc_optimo_enaex'] = df['UCS_MPA'].apply(calcular_fc_optimo)
    df['desv_fc'] = df['Fc'].fillna(0.5) - df['fc_optimo_enaex']

    # 5. ENERGÍA ESPECÍFICA REQUERIDA
    # ================================
    Ce = 0.005  # Constante empírica
    df['energia_requerida_MJ'] = Ce * (df['UCS_MPA'].fillna(100) ** 0.63)

    # Energía entregada por el explosivo
    df['energia_entregada_MJ'] = (
        df['Energia_MJ'].fillna(3.0) *
        df['rho_explosivo'].fillna(1.2) *
        df['Fc'].fillna(0.5) / 1000
    )

    df['ratio_energia'] = df['energia_entregada_MJ'] / df['energia_requerida_MJ']

    # 6. SELECCIÓN DE EXPLOSIVO RECOMENDADO
    # =====================================
    def recomendar_explosivo(row):
        """Recomienda explosivo según UCS y condiciones."""
        ucs = row.get('UCS_MPA', 100)
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

    df['explosivo_recomendado'] = df.apply(recomendar_explosivo, axis=1)

    # 7. EVALUACIÓN GLOBAL DE DISEÑO
    # ==============================
    def evaluar_diseno(row):
        """Evaluación global del diseño de voladura."""
        score = 0
        issues = []

        # Evaluar burden
        if pd.notna(row.get('burden_ratio')):
            if 0.85 <= row['burden_ratio'] <= 1.15:
                score += 1
            else:
                issues.append('Burden')

        # Evaluar S/B
        if pd.notna(row.get('SB_ratio_real')) and pd.notna(row.get('SB_ratio_optimo')):
            if 0.9 * row['SB_ratio_optimo'] <= row['SB_ratio_real'] <= 1.1 * row['SB_ratio_optimo']:
                score += 1
            else:
                issues.append('S/B')

        # Evaluar taco
        if pd.notna(row.get('taco_burden_ratio')):
            if 0.7 <= row['taco_burden_ratio'] <= 1.0:
                score += 1
            else:
                issues.append('Taco')

        # Evaluar timing
        if pd.notna(row.get('timing_eval')) and row['timing_eval'] == 'Óptimo':
            score += 1
        else:
            issues.append('Timing')

        # Evaluar factor de carga
        if pd.notna(row.get('desv_fc')):
            if abs(row['desv_fc']) < 0.15:
                score += 1
            else:
                issues.append('FC')

        if score >= 4:
            return 'Óptimo'
        elif score >= 3:
            return 'Aceptable'
        elif score >= 2:
            return 'Mejorable'
        else:
            return 'Deficiente'

    df['evaluacion_diseno_enaex'] = df.apply(evaluar_diseno, axis=1)

    # Resumen
    print(f"  Burden óptimo ENAEX: {df['burden_optimo_enaex'].mean():.2f} m")
    print(f"  Burden real promedio: {df['Burden'].mean():.2f} m")
    print(f"  Ratio S/B óptimo: {df['SB_ratio_optimo'].mean():.2f}")
    print(f"  Ratio S/B real: {df['SB_ratio_real'].mean():.2f}")
    print(f"  Factor carga óptimo: {df['fc_optimo_enaex'].mean():.2f} kg/m³")

    return df


# ==============================================================================
# GRÁFICOS DE EXPLOSIVOS
# ==============================================================================

def graficar_explosivos_por_fase(df):
    """Genera heatmaps de explosivos por fase."""
    print("\n" + "=" * 70)
    print("GENERANDO GRÁFICOS DE EXPLOSIVOS")
    print("=" * 70)

    # Top explosivos con suficientes datos
    top_explosivos = df.groupby('Tipo_Explosivo').filter(lambda x: len(x) >= 30)['Tipo_Explosivo'].unique()
    df_exp = df[df['Tipo_Explosivo'].isin(top_explosivos)]

    if len(df_exp) < 100:
        print("  ADVERTENCIA: Datos insuficientes de explosivos")
        return

    # =========================================================================
    # GRÁFICO 1: Heatmap P100 por Fase × Explosivo
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, metrica in zip(axes, ['P100', 'P80']):
        pivot = df_exp.pivot_table(values=metrica, index='Fase_categoria',
                                   columns='Tipo_Explosivo', aggfunc='mean').round(1)
        counts = df_exp.pivot_table(values=metrica, index='Fase_categoria',
                                    columns='Tipo_Explosivo', aggfunc='count')
        mask = (counts.reindex_like(pivot) < 20) | pivot.isna()

        vmin = 8 if metrica == 'P100' else 3
        vmax = 25 if metrica == 'P100' else 12

        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn_r',
                    mask=mask, ax=ax, vmin=vmin, vmax=vmax,
                    linewidths=0.5, cbar_kws={'label': f'{metrica} (pulgadas)'})
        ax.set_title(f'{metrica} por Fase y Explosivo', fontweight='bold')
        ax.set_xlabel('Tipo de Explosivo')
        ax.set_ylabel('Fase')

    plt.suptitle('Fragmentación por Fase y Tipo de Explosivo', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/01_heatmap_fase_explosivo.png')
    plt.close()
    print("  ✓ 01_heatmap_fase_explosivo.png")

    # =========================================================================
    # GRÁFICO 2: Heatmap propiedades (Densidad, VOD, Energía) por explosivo
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # 2a. P100 vs Densidad por fase
    ax = axes[0, 0]
    df_rho = df_exp[df_exp['rho_explosivo'].notna()]
    df_rho['rho_cat'] = pd.cut(df_rho['rho_explosivo'],
                               bins=[0, 1.10, 1.15, 1.20, 1.25, 2.0],
                               labels=['<1.10', '1.10-1.15', '1.15-1.20', '1.20-1.25', '>1.25'])

    pivot = df_rho.pivot_table(values='P100', index='Fase_categoria',
                               columns='rho_cat', aggfunc='mean').round(1)
    counts = df_rho.pivot_table(values='P100', index='Fase_categoria',
                                columns='rho_cat', aggfunc='count')
    mask = (counts.reindex_like(pivot) < 15) | pivot.isna()

    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn_r',
                mask=mask, ax=ax, vmin=8, vmax=25, linewidths=0.5)
    ax.set_title('P100 por Fase × Densidad Explosivo', fontweight='bold')
    ax.set_xlabel('Densidad (g/cm³)')
    ax.set_ylabel('Fase')

    # 2b. P100 vs VOD por fase
    ax = axes[0, 1]
    df_vod = df_exp[df_exp['VOD'].notna()]
    df_vod['VOD_cat'] = pd.cut(df_vod['VOD'],
                               bins=[0, 4800, 5200, 5500, 5800, 7000],
                               labels=['<4800', '4800-5200', '5200-5500', '5500-5800', '>5800'])

    pivot = df_vod.pivot_table(values='P100', index='Fase_categoria',
                               columns='VOD_cat', aggfunc='mean').round(1)
    counts = df_vod.pivot_table(values='P100', index='Fase_categoria',
                                columns='VOD_cat', aggfunc='count')
    mask = (counts.reindex_like(pivot) < 15) | pivot.isna()

    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn_r',
                mask=mask, ax=ax, vmin=8, vmax=25, linewidths=0.5)
    ax.set_title('P100 por Fase × VOD', fontweight='bold')
    ax.set_xlabel('VOD (m/s)')
    ax.set_ylabel('Fase')

    # 2c. P100 vs Energía por fase
    ax = axes[1, 0]
    df_energia = df_exp[df_exp['Energia_MJ'].notna()]
    df_energia['Energia_cat'] = pd.cut(df_energia['Energia_MJ'],
                                       bins=[0, 3.2, 3.5, 3.8, 4.0, 5.0],
                                       labels=['<3.2', '3.2-3.5', '3.5-3.8', '3.8-4.0', '>4.0'])

    pivot = df_energia.pivot_table(values='P100', index='Fase_categoria',
                                   columns='Energia_cat', aggfunc='mean').round(1)
    counts = df_energia.pivot_table(values='P100', index='Fase_categoria',
                                    columns='Energia_cat', aggfunc='count')
    mask = (counts.reindex_like(pivot) < 15) | pivot.isna()

    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn_r',
                mask=mask, ax=ax, vmin=8, vmax=25, linewidths=0.5)
    ax.set_title('P100 por Fase × Energía', fontweight='bold')
    ax.set_xlabel('Energía (MJ/kg)')
    ax.set_ylabel('Fase')

    # 2d. P100 vs Presión detonación por fase
    ax = axes[1, 1]
    df_pdet = df_exp[df_exp['P_detonacion'].notna()]
    df_pdet['Pdet_cat'] = pd.cut(df_pdet['P_detonacion'],
                                 bins=[0, 6, 7, 8, 9, 15],
                                 labels=['<6', '6-7', '7-8', '8-9', '>9'])

    pivot = df_pdet.pivot_table(values='P100', index='Fase_categoria',
                                columns='Pdet_cat', aggfunc='mean').round(1)
    counts = df_pdet.pivot_table(values='P100', index='Fase_categoria',
                                 columns='Pdet_cat', aggfunc='count')
    mask = (counts.reindex_like(pivot) < 15) | pivot.isna()

    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn_r',
                mask=mask, ax=ax, vmin=8, vmax=25, linewidths=0.5)
    ax.set_title('P100 por Fase × Presión Detonación', fontweight='bold')
    ax.set_xlabel('Presión Det. (GPa)')
    ax.set_ylabel('Fase')

    plt.suptitle('Propiedades de Explosivos vs Fragmentación por Fase',
                 fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/02_heatmap_propiedades_explosivos.png', bbox_inches='tight')
    plt.close()
    print("  ✓ 02_heatmap_propiedades_explosivos.png")

    # =========================================================================
    # GRÁFICO 3: P50 por Fase × Explosivo
    # =========================================================================
    if 'P50' in df_exp.columns and df_exp['P50'].notna().sum() > 100:
        fig, ax = plt.subplots(figsize=(14, 6))

        pivot = df_exp.pivot_table(values='P50', index='Fase_categoria',
                                   columns='Tipo_Explosivo', aggfunc='mean').round(1)
        counts = df_exp.pivot_table(values='P50', index='Fase_categoria',
                                    columns='Tipo_Explosivo', aggfunc='count')
        mask = (counts.reindex_like(pivot) < 20) | pivot.isna()

        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn_r',
                    mask=mask, ax=ax, vmin=2, vmax=10, linewidths=0.5,
                    cbar_kws={'label': 'P50 (pulgadas)'})
        ax.set_title('P50 por Fase y Tipo de Explosivo', fontweight='bold', fontsize=14)
        ax.set_xlabel('Tipo de Explosivo')
        ax.set_ylabel('Fase')

        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/03_heatmap_p50_explosivo.png')
        plt.close()
        print("  ✓ 03_heatmap_p50_explosivo.png")

    # =========================================================================
    # GRÁFICO 4: Comparación detallada F7R1 vs F9SE por explosivo
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    metricas = ['P100', 'P80', 'P50']
    for col, metrica in enumerate(metricas):
        if metrica not in df_exp.columns:
            continue

        for row, fase in enumerate(['F7R1', 'F9SE']):
            ax = axes[row, col]
            df_f = df_exp[df_exp['Fase_categoria'] == fase]

            # Ordenar explosivos por P100
            orden = df_f.groupby('Tipo_Explosivo')[metrica].mean().sort_values().index

            sns.boxplot(data=df_f, y='Tipo_Explosivo', x=metrica,
                       order=orden, ax=ax, palette='RdYlGn_r')

            # Línea de target
            if metrica == 'P100':
                ax.axvline(x=12, color='red', linestyle='--', linewidth=2)
            elif metrica == 'P80':
                ax.axvline(x=5, color='red', linestyle='--', linewidth=2)

            ax.set_title(f'{fase}: {metrica}', fontweight='bold')
            ax.set_xlabel(f'{metrica} (pulgadas)')
            ax.set_ylabel('Explosivo' if col == 0 else '')

    plt.suptitle('Comparación F7R1 vs F9SE por Explosivo y Métrica',
                 fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/04_comparacion_explosivos_fase.png', bbox_inches='tight')
    plt.close()
    print("  ✓ 04_comparacion_explosivos_fase.png")


# ==============================================================================
# GRÁFICOS DE TACO
# ==============================================================================

def graficar_taco_por_fase(df):
    """Genera gráficos de análisis de taco por fase."""
    print("\n" + "=" * 70)
    print("GENERANDO GRÁFICOS DE TACO")
    print("=" * 70)

    df_taco = df[df['taco_efectivo'] > 0].copy()

    if len(df_taco) < 100:
        print("  ADVERTENCIA: Datos insuficientes de taco")
        return

    # Categorías de taco
    df_taco['taco_cat'] = pd.cut(df_taco['taco_efectivo'],
                                 bins=[0, 4, 5, 6, 7, 15],
                                 labels=['<4m', '4-5m', '5-6m', '6-7m', '>7m'])

    df_taco['tb_cat'] = pd.cut(df_taco['taco_burden_ratio'],
                               bins=[0, 0.5, 0.6, 0.7, 0.8, 1.0, 2.0],
                               labels=['<0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-1.0', '>1.0'])

    # =========================================================================
    # GRÁFICO 5: Heatmaps de Taco por Fase
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # 5a. P100 por Fase × Longitud Taco
    ax = axes[0, 0]
    pivot = df_taco.pivot_table(values='P100', index='Fase_categoria',
                                columns='taco_cat', aggfunc='mean').round(1)
    counts = df_taco.pivot_table(values='P100', index='Fase_categoria',
                                 columns='taco_cat', aggfunc='count')
    mask = (counts.reindex_like(pivot) < 15) | pivot.isna()

    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn_r',
                mask=mask, ax=ax, vmin=8, vmax=25, linewidths=0.5)
    ax.set_title('P100 por Fase × Longitud de Taco', fontweight='bold')
    ax.set_xlabel('Longitud de Taco')
    ax.set_ylabel('Fase')

    # 5b. P100 por Fase × Ratio T/B
    ax = axes[0, 1]
    pivot = df_taco.pivot_table(values='P100', index='Fase_categoria',
                                columns='tb_cat', aggfunc='mean').round(1)
    counts = df_taco.pivot_table(values='P100', index='Fase_categoria',
                                 columns='tb_cat', aggfunc='count')
    mask = (counts.reindex_like(pivot) < 15) | pivot.isna()

    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn_r',
                mask=mask, ax=ax, vmin=8, vmax=25, linewidths=0.5)
    ax.set_title('P100 por Fase × Ratio Taco/Burden', fontweight='bold')
    ax.set_xlabel('Ratio T/B')
    ax.set_ylabel('Fase')

    # 5c. P80 por Fase × Longitud Taco
    ax = axes[1, 0]
    pivot = df_taco.pivot_table(values='P80', index='Fase_categoria',
                                columns='taco_cat', aggfunc='mean').round(1)
    counts = df_taco.pivot_table(values='P80', index='Fase_categoria',
                                 columns='taco_cat', aggfunc='count')
    mask = (counts.reindex_like(pivot) < 15) | pivot.isna()

    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn_r',
                mask=mask, ax=ax, vmin=3, vmax=12, linewidths=0.5)
    ax.set_title('P80 por Fase × Longitud de Taco', fontweight='bold')
    ax.set_xlabel('Longitud de Taco')
    ax.set_ylabel('Fase')

    # 5d. Heatmap Taco × Timing
    ax = axes[1, 1]
    df_taco['timing_cat'] = pd.cut(df_taco['tpozos_ms'],
                                   bins=[0, 2, 4, 6, 10, 50],
                                   labels=['<2', '2-4', '4-6', '6-10', '>10'])

    pivot = df_taco.pivot_table(values='P100', index='taco_cat',
                                columns='timing_cat', aggfunc='mean').round(1)
    counts = df_taco.pivot_table(values='P100', index='taco_cat',
                                 columns='timing_cat', aggfunc='count')
    mask = (counts.reindex_like(pivot) < 15) | pivot.isna()

    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn_r',
                mask=mask, ax=ax, vmin=8, vmax=25, linewidths=0.5)
    ax.set_title('P100: Taco × Timing (ambas fases)', fontweight='bold')
    ax.set_xlabel('Timing (ms)')
    ax.set_ylabel('Taco')

    plt.suptitle('Análisis de Taco por Fase', fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/05_heatmap_taco_fase.png', bbox_inches='tight')
    plt.close()
    print("  ✓ 05_heatmap_taco_fase.png")

    # =========================================================================
    # GRÁFICO 6: Comparación Taco F7R1 vs F9SE
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, fase in zip(axes, ['F7R1', 'F9SE']):
        df_f = df_taco[df_taco['Fase_categoria'] == fase]

        pivot = df_f.pivot_table(values='P100', index='taco_cat',
                                 columns='UCS_categoria', aggfunc='mean').round(1)
        counts = df_f.pivot_table(values='P100', index='taco_cat',
                                  columns='UCS_categoria', aggfunc='count')
        mask = (counts.reindex_like(pivot) < 10) | pivot.isna()

        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn_r',
                    mask=mask, ax=ax, vmin=8, vmax=30, linewidths=0.5)
        ax.set_title(f'{fase}: P100 por Taco × UCS', fontweight='bold')
        ax.set_xlabel('UCS (MPa)')
        ax.set_ylabel('Taco')

    plt.suptitle('Comparación Taco × UCS por Fase', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/06_heatmap_taco_ucs_fase.png')
    plt.close()
    print("  ✓ 06_heatmap_taco_ucs_fase.png")

    # =========================================================================
    # GRÁFICO 7: Evaluación de Taco vs ENAEX
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 7a. Distribución de evaluación de taco
    ax = axes[0, 0]
    eval_counts = df_taco.groupby(['Fase_categoria', 'taco_eval']).size().unstack(fill_value=0)
    eval_pct = eval_counts.div(eval_counts.sum(axis=1), axis=0) * 100

    eval_pct.plot(kind='bar', stacked=True, ax=ax,
                  color=['#E74C3C', '#F39C12', '#27AE60', '#3498DB'])
    ax.set_ylabel('% de registros')
    ax.set_xlabel('Fase')
    ax.set_title('7a. Evaluación de Taco según ENAEX', fontweight='bold')
    ax.legend(title='Evaluación', bbox_to_anchor=(1.02, 1))
    ax.tick_params(axis='x', rotation=0)

    # 7b. Taco real vs óptimo ENAEX
    ax = axes[0, 1]
    for fase, color in [('F7R1', '#27AE60'), ('F9SE', '#E74C3C')]:
        df_f = df_taco[df_taco['Fase_categoria'] == fase]
        ax.scatter(df_f['taco_optimo_enaex'], df_f['taco_efectivo'],
                   alpha=0.3, s=20, label=fase, c=color)

    # Línea de igualdad
    max_val = max(df_taco['taco_optimo_enaex'].max(), df_taco['taco_efectivo'].max())
    ax.plot([0, max_val], [0, max_val], 'k--', label='Óptimo')
    ax.set_xlabel('Taco óptimo ENAEX (m)')
    ax.set_ylabel('Taco real (m)')
    ax.set_title('7b. Taco Real vs Óptimo ENAEX', fontweight='bold')
    ax.legend()

    # 7c. P100 vs desviación de taco
    ax = axes[1, 0]
    df_plot = df_taco[df_taco['desv_taco'].notna()]
    ax.scatter(df_plot['desv_taco'], df_plot['P100'], alpha=0.3, s=15, c='steelblue')
    ax.axvline(x=0, color='green', linestyle='--', linewidth=2, label='Óptimo ENAEX')
    ax.axhline(y=12, color='red', linestyle='--', linewidth=2, label='Target P100')
    ax.set_xlabel('Desviación del taco óptimo (m)')
    ax.set_ylabel('P100 (pulgadas)')
    ax.set_title('7c. P100 vs Desviación de Taco', fontweight='bold')
    ax.legend()

    # 7d. Tabla resumen
    ax = axes[1, 1]
    ax.axis('off')

    resumen = df_taco.groupby('Fase_categoria').agg({
        'taco_efectivo': 'mean',
        'taco_optimo_enaex': 'mean',
        'taco_burden_ratio': 'mean',
        'desv_taco': 'mean',
        'P100': 'mean'
    }).round(2)

    resumen.columns = ['Taco real', 'Taco ENAEX', 'Ratio T/B', 'Desv.', 'P100']

    tabla = ax.table(
        cellText=resumen.values,
        rowLabels=resumen.index,
        colLabels=resumen.columns,
        cellLoc='center', loc='center'
    )
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(11)
    tabla.scale(1.2, 2)
    ax.set_title('7d. Resumen Taco por Fase', fontweight='bold', pad=20)

    plt.suptitle('Evaluación de Taco según ENAEX', fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/07_evaluacion_taco_enaex.png', bbox_inches='tight')
    plt.close()
    print("  ✓ 07_evaluacion_taco_enaex.png")


# ==============================================================================
# ANÁLISIS DE SENSIBILIDAD
# ==============================================================================

def analisis_sensibilidad(df):
    """Análisis de sensibilidad de variables según ENAEX."""
    print("\n" + "=" * 70)
    print("GENERANDO ANÁLISIS DE SENSIBILIDAD")
    print("=" * 70)

    variables = [
        ('tpozos_ms', 'Timing pozos (ms)', 'timing'),
        ('tfilas_ms', 'Timing filas (ms)', 'timing'),
        ('taco_efectivo', 'Taco (m)', 'taco'),
        ('taco_burden_ratio', 'Ratio T/B', 'taco'),
        ('Area_malla', 'Área malla (m²)', 'geometria'),
        ('S_B_ratio', 'Ratio S/B', 'geometria'),
        ('Fc', 'Factor carga', 'energia'),
        ('gamma_lineal', 'γ lineal (kg/m)', 'energia'),
        ('rho_explosivo', 'Densidad exp.', 'explosivo'),
        ('VOD', 'VOD (m/s)', 'explosivo'),
        ('UCS_MPA', 'UCS (MPa)', 'roca'),
    ]

    # =========================================================================
    # GRÁFICO 8: Tornado de sensibilidad por fase
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))

    for ax, fase in zip(axes, ['F7R1', 'F9SE']):
        df_f = df[df['Fase_categoria'] == fase]
        resultados = []

        for var, label, categoria in variables:
            if var not in df_f.columns:
                continue

            df_var = df_f[[var, 'P100']].copy()
            df_var[var] = pd.to_numeric(df_var[var], errors='coerce')
            df_var = df_var.dropna()

            if len(df_var) < 50:
                continue

            q25 = df_var[var].quantile(0.25)
            q75 = df_var[var].quantile(0.75)

            p100_q1 = df_var[df_var[var] <= q25]['P100'].mean()
            p100_q3 = df_var[df_var[var] >= q75]['P100'].mean()

            resultados.append({
                'Variable': label,
                'Categoria': categoria,
                'Delta': p100_q3 - p100_q1,
                'P100_Q1': p100_q1,
                'P100_Q3': p100_q3
            })

        if not resultados:
            continue

        df_res = pd.DataFrame(resultados).sort_values('Delta', key=abs, ascending=True)

        # Colores por categoría
        color_map = {
            'timing': '#E74C3C',
            'taco': '#F39C12',
            'geometria': '#3498DB',
            'energia': '#9B59B6',
            'explosivo': '#27AE60',
            'roca': '#7F8C8D'
        }
        colors = [color_map.get(cat, '#95A5A6') for cat in df_res['Categoria']]

        y_pos = np.arange(len(df_res))
        ax.barh(y_pos, df_res['Delta'], color=colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_res['Variable'])
        ax.axvline(x=0, color='black', linewidth=1)
        ax.set_xlabel('Cambio en P100 (Q3 - Q1)')
        ax.set_title(f'{fase}: Sensibilidad de P100', fontweight='bold')

        # Anotaciones
        for i, (_, row) in enumerate(df_res.iterrows()):
            x = row['Delta']
            ax.text(x + 0.2 if x > 0 else x - 0.2, i,
                   f'{row["Delta"]:+.1f}"', va='center', fontsize=9,
                   ha='left' if x > 0 else 'right')

    # Leyenda
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color_map[cat], label=cat.title())
                       for cat in color_map.keys()]
    fig.legend(handles=legend_elements, loc='upper center', ncol=6, bbox_to_anchor=(0.5, 0.02))

    plt.suptitle('Análisis de Sensibilidad por Fase (Q1 vs Q3)', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/08_sensibilidad_tornado.png', bbox_inches='tight')
    plt.close()
    print("  ✓ 08_sensibilidad_tornado.png")

    # =========================================================================
    # GRÁFICO 8b: Sensibilidad detallada - P100 en Q1 y Q3 por separado
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(18, 12))

    for ax, fase in zip(axes, ['F7R1', 'F9SE']):
        df_f = df[df['Fase_categoria'] == fase]
        resultados = []

        for var, label, categoria in variables:
            if var not in df_f.columns:
                continue

            df_var = df_f[[var, 'P100']].copy()
            df_var[var] = pd.to_numeric(df_var[var], errors='coerce')
            df_var = df_var.dropna()

            if len(df_var) < 50:
                continue

            q25_val = df_var[var].quantile(0.25)
            q75_val = df_var[var].quantile(0.75)

            p100_q1 = df_var[df_var[var] <= q25_val]['P100'].mean()
            p100_q3 = df_var[df_var[var] >= q75_val]['P100'].mean()
            n_q1 = len(df_var[df_var[var] <= q25_val])
            n_q3 = len(df_var[df_var[var] >= q75_val])

            resultados.append({
                'Variable': label,
                'Categoria': categoria,
                'Q1_valor': q25_val,
                'Q3_valor': q75_val,
                'P100_Q1': p100_q1,
                'P100_Q3': p100_q3,
                'n_Q1': n_q1,
                'n_Q3': n_q3,
                'Delta': p100_q3 - p100_q1
            })

        if not resultados:
            continue

        df_res = pd.DataFrame(resultados).sort_values('Delta', key=abs, ascending=False)

        # Gráfico de barras pareadas
        y_pos = np.arange(len(df_res))
        height = 0.35

        bars1 = ax.barh(y_pos - height/2, df_res['P100_Q1'], height,
                        label='P100 en Q1 (valores bajos)', color='#3498DB', alpha=0.8)
        bars2 = ax.barh(y_pos + height/2, df_res['P100_Q3'], height,
                        label='P100 en Q3 (valores altos)', color='#E74C3C', alpha=0.8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_res['Variable'])
        ax.axvline(x=12, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Target 12"')
        ax.set_xlabel('P100 (pulgadas)')
        ax.set_title(f'{fase}: P100 en Q1 vs Q3 por Variable', fontweight='bold')
        ax.legend(loc='lower right')

        # Anotaciones con valores
        for i, (_, row) in enumerate(df_res.iterrows()):
            ax.text(row['P100_Q1'] + 0.3, i - height/2,
                   f'{row["P100_Q1"]:.1f}" (n={row["n_Q1"]})', va='center', fontsize=8)
            ax.text(row['P100_Q3'] + 0.3, i + height/2,
                   f'{row["P100_Q3"]:.1f}" (n={row["n_Q3"]})', va='center', fontsize=8)

    plt.suptitle('P100 Detallado: Cuartil 1 vs Cuartil 3 por Variable',
                 fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/08b_sensibilidad_detallada_q1_q3.png', bbox_inches='tight')
    plt.close()
    print("  ✓ 08b_sensibilidad_detallada_q1_q3.png")

    # =========================================================================
    # GRÁFICO 8c: Tabla detallada con todos los valores Q1/Q3
    # =========================================================================
    fig, axes = plt.subplots(2, 1, figsize=(16, 14))

    for idx, fase in enumerate(['F7R1', 'F9SE']):
        ax = axes[idx]
        ax.axis('off')

        df_f = df[df['Fase_categoria'] == fase]
        resultados = []

        for var, label, categoria in variables:
            if var not in df_f.columns:
                continue

            df_var = df_f[[var, 'P100']].copy()
            df_var[var] = pd.to_numeric(df_var[var], errors='coerce')
            df_var = df_var.dropna()

            if len(df_var) < 50:
                continue

            q25_val = df_var[var].quantile(0.25)
            q75_val = df_var[var].quantile(0.75)

            p100_q1 = df_var[df_var[var] <= q25_val]['P100'].mean()
            p100_q3 = df_var[df_var[var] >= q75_val]['P100'].mean()
            n_q1 = len(df_var[df_var[var] <= q25_val])
            n_q3 = len(df_var[df_var[var] >= q75_val])

            resultados.append({
                'Variable': label,
                'Q1 (≤25%)': f'{q25_val:.2f}',
                'Q3 (≥75%)': f'{q75_val:.2f}',
                'P100 en Q1': f'{p100_q1:.1f}"',
                'n Q1': n_q1,
                'P100 en Q3': f'{p100_q3:.1f}"',
                'n Q3': n_q3,
                'Delta': f'{p100_q3 - p100_q1:+.1f}"'
            })

        df_tabla = pd.DataFrame(resultados)

        tabla = ax.table(
            cellText=df_tabla.values,
            colLabels=df_tabla.columns,
            cellLoc='center',
            loc='center',
            colColours=['#E8E8E8'] * len(df_tabla.columns)
        )
        tabla.auto_set_font_size(False)
        tabla.set_fontsize(9)
        tabla.scale(1.2, 1.8)

        # Colorear celdas según Delta
        for i, row in enumerate(resultados):
            delta_val = float(row['Delta'].replace('"', '').replace('+', ''))
            if delta_val > 1:
                tabla[(i+1, 7)].set_facecolor('#FFCCCC')  # Rojo claro
            elif delta_val < -1:
                tabla[(i+1, 7)].set_facecolor('#CCFFCC')  # Verde claro

        ax.set_title(f'{fase}: Tabla Detallada Q1 vs Q3', fontweight='bold', fontsize=12, pad=20)

    plt.suptitle('ANÁLISIS DE SENSIBILIDAD DETALLADO - VALORES Q1 Y Q3',
                 fontweight='bold', fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/08c_tabla_sensibilidad_detallada.png', bbox_inches='tight')
    plt.close()
    print("  ✓ 08c_tabla_sensibilidad_detallada.png")

    # =========================================================================
    # GRÁFICO 8d: Scatter plots individuales para cada variable
    # =========================================================================
    n_vars = len(variables)
    n_cols = 4
    n_rows = (n_vars + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
    axes = axes.flatten()

    for i, (var, label, categoria) in enumerate(variables):
        ax = axes[i]

        if var not in df.columns:
            ax.axis('off')
            continue

        for fase, color in [('F7R1', '#27AE60'), ('F9SE', '#E74C3C')]:
            df_f = df[df['Fase_categoria'] == fase][[var, 'P100']].copy()
            df_f[var] = pd.to_numeric(df_f[var], errors='coerce')
            df_f = df_f.dropna()

            if len(df_f) > 30:
                ax.scatter(df_f[var], df_f['P100'], alpha=0.2, s=10, c=color, label=fase)

                # Línea de tendencia
                slope, intercept, r_value, _, _ = stats.linregress(
                    df_f[var].astype(float), df_f['P100'].astype(float)
                )
                x_line = np.linspace(df_f[var].min(), df_f[var].max(), 100)
                y_line = slope * x_line + intercept
                ax.plot(x_line, y_line, color=color, linestyle='--', linewidth=2)

                # Marcar Q1 y Q3
                q1 = df_f[var].quantile(0.25)
                q3 = df_f[var].quantile(0.75)
                ax.axvline(x=q1, color=color, linestyle=':', alpha=0.5)
                ax.axvline(x=q3, color=color, linestyle=':', alpha=0.5)

        ax.axhline(y=12, color='red', linestyle='--', linewidth=1, alpha=0.7)
        ax.set_xlabel(label)
        ax.set_ylabel('P100')
        ax.set_title(f'{label}', fontweight='bold', fontsize=10)
        ax.legend(fontsize=7)

    # Ocultar ejes vacíos
    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.suptitle('P100 vs Cada Variable (líneas punteadas = Q1 y Q3)',
                 fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/08d_scatter_todas_variables.png', bbox_inches='tight')
    plt.close()
    print("  ✓ 08d_scatter_todas_variables.png")

    # =========================================================================
    # GRÁFICO 8e: Ranking de sensibilidad (mayor a menor)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))

    for ax, fase in zip(axes, ['F7R1', 'F9SE']):
        df_f = df[df['Fase_categoria'] == fase]
        resultados = []

        for var, label, categoria in variables:
            if var not in df_f.columns:
                continue

            df_var = df_f[[var, 'P100']].copy()
            df_var[var] = pd.to_numeric(df_var[var], errors='coerce')
            df_var = df_var.dropna()

            if len(df_var) < 50:
                continue

            q25_val = df_var[var].quantile(0.25)
            q75_val = df_var[var].quantile(0.75)

            p100_q1 = df_var[df_var[var] <= q25_val]['P100'].mean()
            p100_q3 = df_var[df_var[var] >= q75_val]['P100'].mean()
            n_q1 = len(df_var[df_var[var] <= q25_val])
            n_q3 = len(df_var[df_var[var] >= q75_val])

            resultados.append({
                'Variable': label,
                'Categoria': categoria,
                'Q1_valor': q25_val,
                'Q3_valor': q75_val,
                'P100_Q1': p100_q1,
                'P100_Q3': p100_q3,
                'n_Q1': n_q1,
                'n_Q3': n_q3,
                'Delta': p100_q3 - p100_q1,
                'Abs_Delta': abs(p100_q3 - p100_q1)
            })

        if not resultados:
            continue

        # Ordenar de MAYOR a MENOR sensibilidad (abs delta)
        df_res = pd.DataFrame(resultados).sort_values('Abs_Delta', ascending=False)

        # Colores según dirección del efecto
        colors = ['#E74C3C' if d > 0 else '#27AE60' for d in df_res['Delta']]

        y_pos = np.arange(len(df_res))

        bars = ax.barh(y_pos, df_res['Delta'], color=colors, alpha=0.8, edgecolor='black')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_res['Variable'])
        ax.axvline(x=0, color='black', linewidth=1.5)
        ax.set_xlabel('Cambio en P100 (pulgadas): Q3 - Q1', fontsize=11)
        ax.set_title(f'{fase}: Ranking de Sensibilidad\n(Mayor a Menor Impacto)', fontweight='bold', fontsize=12)

        # Anotaciones detalladas
        for i, (_, row) in enumerate(df_res.iterrows()):
            delta = row['Delta']
            if delta > 0:
                ax.text(delta + 0.15, i, f'{delta:+.1f}"', va='center', fontsize=9, fontweight='bold')
            else:
                ax.text(delta - 0.15, i, f'{delta:+.1f}"', va='center', fontsize=9, fontweight='bold', ha='right')

        # Agregar número de ranking
        for i, (_, row) in enumerate(df_res.iterrows()):
            ax.text(-0.1, i, f'#{i+1}', va='center', ha='right', fontsize=9,
                   fontweight='bold', color='gray', transform=ax.get_yaxis_transform())

        ax.set_xlim(ax.get_xlim()[0] - 1, ax.get_xlim()[1] + 1.5)

        # Leyenda
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#E74C3C', label='↑ Variable → ↑ P100 (peor)'),
            Patch(facecolor='#27AE60', label='↑ Variable → ↓ P100 (mejor)')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.suptitle('RANKING DE SENSIBILIDAD: Variables ordenadas por Impacto en P100',
                 fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/08e_ranking_sensibilidad.png', bbox_inches='tight')
    plt.close()
    print("  ✓ 08e_ranking_sensibilidad.png")

    # =========================================================================
    # GRÁFICO 8f: Tabla de ranking con todos los detalles
    # =========================================================================
    fig, axes = plt.subplots(2, 1, figsize=(18, 12))

    for idx, fase in enumerate(['F7R1', 'F9SE']):
        ax = axes[idx]
        ax.axis('off')

        df_f = df[df['Fase_categoria'] == fase]
        resultados = []

        for var, label, categoria in variables:
            if var not in df_f.columns:
                continue

            df_var = df_f[[var, 'P100']].copy()
            df_var[var] = pd.to_numeric(df_var[var], errors='coerce')
            df_var = df_var.dropna()

            if len(df_var) < 50:
                continue

            q25_val = df_var[var].quantile(0.25)
            q75_val = df_var[var].quantile(0.75)

            p100_q1 = df_var[df_var[var] <= q25_val]['P100'].mean()
            p100_q3 = df_var[df_var[var] >= q75_val]['P100'].mean()
            n_q1 = len(df_var[df_var[var] <= q25_val])
            n_q3 = len(df_var[df_var[var] >= q75_val])
            delta = p100_q3 - p100_q1

            resultados.append({
                'Rank': 0,
                'Variable': label,
                'Q1': f'{q25_val:.2f}',
                'P100@Q1': f'{p100_q1:.1f}"',
                'n(Q1)': n_q1,
                'Q3': f'{q75_val:.2f}',
                'P100@Q3': f'{p100_q3:.1f}"',
                'n(Q3)': n_q3,
                'Delta': delta,
                'Impacto': 'ALTO' if abs(delta) > 2 else ('MEDIO' if abs(delta) > 1 else 'BAJO')
            })

        # Ordenar por abs(delta) descendente y asignar ranking
        resultados = sorted(resultados, key=lambda x: abs(x['Delta']), reverse=True)
        for i, r in enumerate(resultados):
            r['Rank'] = i + 1
            r['Delta'] = f'{r["Delta"]:+.2f}"'

        df_tabla = pd.DataFrame(resultados)

        tabla = ax.table(
            cellText=df_tabla.values,
            colLabels=df_tabla.columns,
            cellLoc='center',
            loc='center',
            colColours=['#D5D5D5'] * len(df_tabla.columns)
        )
        tabla.auto_set_font_size(False)
        tabla.set_fontsize(9)
        tabla.scale(1.1, 1.8)

        # Colorear filas según ranking
        for i, row in enumerate(resultados):
            if row['Rank'] <= 3:
                bg_color = '#FFE5E5'  # Top 3 - rojo claro
            elif row['Rank'] <= 6:
                bg_color = '#FFF5E5'  # 4-6 - naranja claro
            else:
                bg_color = '#E5FFE5'  # resto - verde claro

            for j in range(len(df_tabla.columns)):
                tabla[(i+1, j)].set_facecolor(bg_color)

            # Color especial para columna Impacto
            impacto = row['Impacto']
            if impacto == 'ALTO':
                tabla[(i+1, 9)].set_facecolor('#FF6B6B')
                tabla[(i+1, 9)].set_text_props(color='white', fontweight='bold')
            elif impacto == 'MEDIO':
                tabla[(i+1, 9)].set_facecolor('#FFB347')
            else:
                tabla[(i+1, 9)].set_facecolor('#90EE90')

        ax.set_title(f'{fase}: Ranking de Variables por Sensibilidad (Mayor a Menor)',
                     fontweight='bold', fontsize=12, pad=20)

    plt.suptitle('RANKING COMPLETO DE SENSIBILIDAD - TODAS LAS VARIABLES',
                 fontweight='bold', fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/08f_tabla_ranking_sensibilidad.png', bbox_inches='tight')
    plt.close()
    print("  ✓ 08f_tabla_ranking_sensibilidad.png")

    # =========================================================================
    # GRÁFICO 9: Heatmap de correlaciones por fase
    # =========================================================================
    vars_corr = ['P100', 'P80', 'tpozos_ms', 'taco_efectivo', 'Area_malla',
                 'S_B_ratio', 'Fc', 'UCS_MPA', 'rho_explosivo', 'VOD']
    vars_corr = [v for v in vars_corr if v in df.columns]

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    for ax, fase in zip(axes, ['F7R1', 'F9SE']):
        df_f = df[df['Fase_categoria'] == fase][vars_corr].copy()
        for col in vars_corr:
            df_f[col] = pd.to_numeric(df_f[col], errors='coerce')

        corr = df_f.corr()

        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                    center=0, ax=ax, vmin=-1, vmax=1, linewidths=0.5,
                    cbar_kws={'label': 'Correlación'})
        ax.set_title(f'{fase}: Matriz de Correlaciones', fontweight='bold')

    plt.suptitle('Correlaciones entre Variables por Fase', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/09_correlaciones_fase.png')
    plt.close()
    print("  ✓ 09_correlaciones_fase.png")

    # =========================================================================
    # GRÁFICO 10: Timing real vs ENAEX por fase
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 10a. Timing pozos: real vs óptimo
    ax = axes[0, 0]
    for fase, color in [('F7R1', '#27AE60'), ('F9SE', '#E74C3C')]:
        df_f = df[df['Fase_categoria'] == fase]
        ax.scatter(df_f['tpozos_optimo'], df_f['tpozos_ms'],
                   alpha=0.3, s=20, label=fase, c=color)

    max_val = max(df['tpozos_optimo'].max(), df['tpozos_ms'].max())
    ax.plot([0, max_val], [0, max_val], 'k--', label='Óptimo')
    ax.set_xlabel('Timing óptimo ENAEX (ms)')
    ax.set_ylabel('Timing real (ms)')
    ax.set_title('10a. Timing Pozos: Real vs ENAEX', fontweight='bold')
    ax.legend()

    # 10b. Heatmap timing vs P100 por fase
    ax = axes[0, 1]
    df['timing_cat'] = pd.cut(df['tpozos_ms'], bins=[0, 2, 4, 6, 8, 12, 50],
                              labels=['<2', '2-4', '4-6', '6-8', '8-12', '>12'])

    pivot = df.pivot_table(values='P100', index='Fase_categoria',
                           columns='timing_cat', aggfunc='mean').round(1)
    counts = df.pivot_table(values='P100', index='Fase_categoria',
                            columns='timing_cat', aggfunc='count')
    mask = (counts.reindex_like(pivot) < 20) | pivot.isna()

    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn_r',
                mask=mask, ax=ax, vmin=8, vmax=25, linewidths=0.5)
    ax.set_title('10b. P100 por Fase × Timing', fontweight='bold')
    ax.set_xlabel('Timing (ms)')
    ax.set_ylabel('Fase')

    # 10c. Distribución de evaluación timing
    ax = axes[1, 0]
    eval_counts = df.groupby(['Fase_categoria', 'timing_eval']).size().unstack(fill_value=0)
    orden = ['Muy bajo', 'Bajo', 'Óptimo', 'Alto', 'Muy alto', 'Sin datos']
    eval_counts = eval_counts.reindex(columns=[c for c in orden if c in eval_counts.columns])
    eval_pct = eval_counts.div(eval_counts.sum(axis=1), axis=0) * 100

    eval_pct.plot(kind='bar', stacked=True, ax=ax,
                  color=['#E74C3C', '#F39C12', '#27AE60', '#3498DB', '#9B59B6', '#95A5A6'])
    ax.set_ylabel('% de registros')
    ax.set_xlabel('Fase')
    ax.set_title('10c. Evaluación Timing según ENAEX', fontweight='bold')
    ax.legend(title='Evaluación', bbox_to_anchor=(1.02, 1))
    ax.tick_params(axis='x', rotation=0)

    # 10d. Tabla resumen timing
    ax = axes[1, 1]
    ax.axis('off')

    resumen = df.groupby('Fase_categoria').agg({
        'tpozos_ms': 'mean',
        'tpozos_optimo': 'mean',
        'desv_tpozos': 'mean',
        'tfilas_ms': 'mean',
        'tfilas_optimo': 'mean',
        'P100': 'mean'
    }).round(1)

    resumen.columns = ['t_pozos real', 't_pozos ENAEX', 'Desv. pozos',
                       't_filas real', 't_filas ENAEX', 'P100']

    tabla = ax.table(
        cellText=resumen.values,
        rowLabels=resumen.index,
        colLabels=resumen.columns,
        cellLoc='center', loc='center'
    )
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(10)
    tabla.scale(1.1, 2)
    ax.set_title('10d. Resumen Timing por Fase', fontweight='bold', pad=20)

    plt.suptitle('Análisis de Timing según Fórmulas ENAEX', fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/10_timing_enaex.png', bbox_inches='tight')
    plt.close()
    print("  ✓ 10_timing_enaex.png")


# ==============================================================================
# RESUMEN EJECUTIVO
# ==============================================================================

def generar_resumen(df):
    """Genera tabla resumen ejecutivo."""
    print("\n" + "=" * 70)
    print("GENERANDO RESUMEN EJECUTIVO")
    print("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # 11a. Tabla resumen general
    ax = axes[0, 0]
    ax.axis('off')

    resumen = []
    for fase in ['F7R1', 'F9SE']:
        df_f = df[df['Fase_categoria'] == fase]
        resumen.append({
            'Fase': fase,
            'n': len(df_f),
            'P100': df_f['P100'].mean(),
            'P80': df_f['P80'].mean(),
            'P50': df_f['P50'].mean() if 'P50' in df_f.columns else np.nan,
            '% >12"': (df_f['P100'] > 12).sum() / len(df_f) * 100,
            'Timing': df_f['tpozos_ms'].mean(),
            'Taco': df_f['taco_efectivo'].mean(),
            'T/B': df_f['taco_burden_ratio'].mean(),
            'UCS': df_f['UCS_MPA'].mean()
        })

    df_resumen = pd.DataFrame(resumen).set_index('Fase').round(1)

    tabla = ax.table(
        cellText=df_resumen.values,
        rowLabels=df_resumen.index,
        colLabels=df_resumen.columns,
        cellLoc='center', loc='center'
    )
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(10)
    tabla.scale(1.1, 2)
    ax.set_title('Resumen General por Fase', fontweight='bold', fontsize=12, pad=20)

    # 11b. Comparación visual P100, P80, P50
    ax = axes[0, 1]
    metricas = ['P100', 'P80']
    if 'P50' in df.columns and df['P50'].notna().sum() > 100:
        metricas.append('P50')

    x = np.arange(len(metricas))
    width = 0.35

    f7 = [df[df['Fase_categoria'] == 'F7R1'][m].mean() for m in metricas]
    f9 = [df[df['Fase_categoria'] == 'F9SE'][m].mean() for m in metricas]

    bars1 = ax.bar(x - width/2, f7, width, label='F7R1', color='#27AE60')
    bars2 = ax.bar(x + width/2, f9, width, label='F9SE', color='#E74C3C')

    ax.axhline(y=12, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax.axhline(y=5, color='green', linestyle='--', linewidth=1, alpha=0.7)

    ax.set_ylabel('Pulgadas')
    ax.set_xticks(x)
    ax.set_xticklabels(metricas)
    ax.set_title('Comparación de Fragmentación', fontweight='bold')
    ax.legend()

    # Anotaciones
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords='offset points',
                       ha='center', va='bottom', fontsize=10)

    # 11c. Distribución de P100
    ax = axes[1, 0]
    for fase, color in [('F7R1', '#27AE60'), ('F9SE', '#E74C3C')]:
        df_f = df[df['Fase_categoria'] == fase]
        sns.kdeplot(df_f['P100'], ax=ax, label=fase, color=color, fill=True, alpha=0.3)

    ax.axvline(x=12, color='red', linestyle='--', linewidth=2, label='Target 12"')
    ax.set_xlabel('P100 (pulgadas)')
    ax.set_ylabel('Densidad')
    ax.set_title('Distribución de P100 por Fase', fontweight='bold')
    ax.legend()

    # 11d. Recomendaciones
    ax = axes[1, 1]
    ax.axis('off')

    # Calcular métricas para recomendaciones
    f9_timing = df[df['Fase_categoria'] == 'F9SE']['tpozos_ms'].mean()
    f9_timing_opt = df[df['Fase_categoria'] == 'F9SE']['tpozos_optimo'].mean()
    f9_taco = df[df['Fase_categoria'] == 'F9SE']['taco_burden_ratio'].mean()

    texto = f"""
    RECOMENDACIONES PARA MEJORAR F9SE
    ══════════════════════════════════════════════════════════════

    1. TIMING ENTRE POZOS
       ► Actual: {f9_timing:.1f} ms
       ► Óptimo ENAEX: {f9_timing_opt:.1f} ms
       ► Acción: Aumentar timing a 4-8 ms
       ► Impacto esperado: Reducción P100 de 5-8"

    2. TACO
       ► Ratio T/B actual: {f9_taco:.2f}
       ► Óptimo ENAEX: 0.7-1.0
       ► Acción: Ajustar taco según Burden
       ► Impacto esperado: Reducción P100 de 2-3"

    3. EXPLOSIVO
       ► Evaluar uso de explosivos con mayor VOD
       ► Considerar densidad > 1.20 g/cm³ para roca dura
       ► Impacto esperado: Reducción P100 de 1-2"

    4. GEOMETRÍA
       ► Mantener ratio S/B entre 1.15-1.25
       ► Evitar mallas muy cerradas (interacción ondas)
    """

    ax.text(0.05, 0.95, texto, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.set_title('Recomendaciones', fontweight='bold', fontsize=12)

    plt.suptitle('RESUMEN EJECUTIVO: F7R1 vs F9SE', fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/11_resumen_ejecutivo.png', bbox_inches='tight')
    plt.close()
    print("  ✓ 11_resumen_ejecutivo.png")


# ==============================================================================
# ANÁLISIS DE UCS Y RQD (DUREZA DE ROCA)
# ==============================================================================

def analisis_dureza_roca(df):
    """Análisis detallado de UCS y RQD y su impacto en fragmentación."""
    print("\n" + "=" * 70)
    print("GENERANDO ANÁLISIS DE DUREZA DE ROCA (UCS/RQD)")
    print("=" * 70)

    # Verificar que existen las columnas necesarias
    if 'UCS_MPA' not in df.columns:
        print("  ADVERTENCIA: Columna UCS_MPA no encontrada")
        return

    # Crear categorías de RQD si existe
    if 'RQD' in df.columns:
        df['RQD_categoria'] = pd.cut(
            df['RQD'],
            bins=[0, 25, 50, 75, 90, 100],
            labels=['Muy pobre (<25)', 'Pobre (25-50)', 'Regular (50-75)',
                    'Buena (75-90)', 'Excelente (>90)']
        )

    # =========================================================================
    # GRÁFICO 12: Análisis de dureza de roca
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # 12a. P100 vs UCS por fase (scatter con regresión)
    ax = axes[0, 0]
    for fase, color in [('F7R1', '#27AE60'), ('F9SE', '#E74C3C')]:
        df_f = df[df['Fase_categoria'] == fase]
        df_ucs = df_f[['UCS_MPA', 'P100']].dropna()

        if len(df_ucs) > 30:
            ax.scatter(df_ucs['UCS_MPA'], df_ucs['P100'],
                       alpha=0.3, s=20, label=fase, c=color)

            # Regresión lineal
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                df_ucs['UCS_MPA'].astype(float), df_ucs['P100'].astype(float)
            )
            x_line = np.linspace(df_ucs['UCS_MPA'].min(), df_ucs['UCS_MPA'].max(), 100)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, color=color, linestyle='--', linewidth=2,
                    label=f'{fase}: R²={r_value**2:.2f}')

    ax.axhline(y=12, color='red', linestyle=':', linewidth=2, alpha=0.7)
    ax.set_xlabel('UCS (MPa)')
    ax.set_ylabel('P100 (pulgadas)')
    ax.set_title('12a. P100 vs UCS por Fase', fontweight='bold')
    ax.legend()

    # 12b. Heatmap P100 por UCS × Fase
    ax = axes[0, 1]
    pivot = df.pivot_table(values='P100', index='Fase_categoria',
                           columns='UCS_categoria', aggfunc='mean').round(1)
    counts = df.pivot_table(values='P100', index='Fase_categoria',
                            columns='UCS_categoria', aggfunc='count')
    mask = (counts.reindex_like(pivot) < 15) | pivot.isna()

    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn_r',
                mask=mask, ax=ax, vmin=8, vmax=30, linewidths=0.5,
                cbar_kws={'label': 'P100 (pulgadas)'})
    ax.set_title('12b. P100 por Fase × UCS', fontweight='bold')
    ax.set_xlabel('UCS (MPa)')
    ax.set_ylabel('Fase')

    # 12c. Distribución de UCS por fase
    ax = axes[1, 0]
    for fase, color in [('F7R1', '#27AE60'), ('F9SE', '#E74C3C')]:
        df_f = df[df['Fase_categoria'] == fase]['UCS_MPA'].dropna()
        if len(df_f) > 30:
            sns.kdeplot(df_f, ax=ax, label=fase, color=color, fill=True, alpha=0.3)

    ax.axvline(x=80, color='orange', linestyle='--', linewidth=2, label='Roca media (80 MPa)')
    ax.axvline(x=120, color='red', linestyle='--', linewidth=2, label='Roca dura (120 MPa)')
    ax.set_xlabel('UCS (MPa)')
    ax.set_ylabel('Densidad')
    ax.set_title('12c. Distribución de UCS por Fase', fontweight='bold')
    ax.legend()

    # 12d. Tabla resumen UCS
    ax = axes[1, 1]
    ax.axis('off')

    resumen_ucs = df.groupby('Fase_categoria').agg({
        'UCS_MPA': ['mean', 'std', 'min', 'max', 'count']
    }).round(1)
    resumen_ucs.columns = ['Media', 'Std', 'Min', 'Max', 'n']

    # Agregar P100 promedio por rango UCS
    resumen_adicional = []
    for fase in ['F7R1', 'F9SE']:
        df_f = df[df['Fase_categoria'] == fase]
        p100_baja = df_f[df_f['UCS_MPA'] < 80]['P100'].mean()
        p100_media = df_f[(df_f['UCS_MPA'] >= 80) & (df_f['UCS_MPA'] < 120)]['P100'].mean()
        p100_alta = df_f[df_f['UCS_MPA'] >= 120]['P100'].mean()
        resumen_adicional.append({
            'Fase': fase,
            'P100 UCS<80': f'{p100_baja:.1f}"' if not pd.isna(p100_baja) else 'N/A',
            'P100 80-120': f'{p100_media:.1f}"' if not pd.isna(p100_media) else 'N/A',
            'P100 UCS>120': f'{p100_alta:.1f}"' if not pd.isna(p100_alta) else 'N/A'
        })

    tabla_texto = f"""
    RESUMEN UCS POR FASE
    ════════════════════════════════════════════════════════════

    Fase     Media    Std     Min     Max      n
    ──────────────────────────────────────────────────────────
    F7R1    {resumen_ucs.loc['F7R1', 'Media']:6.1f}   {resumen_ucs.loc['F7R1', 'Std']:5.1f}   {resumen_ucs.loc['F7R1', 'Min']:5.0f}   {resumen_ucs.loc['F7R1', 'Max']:5.0f}   {int(resumen_ucs.loc['F7R1', 'n']):5d}
    F9SE    {resumen_ucs.loc['F9SE', 'Media']:6.1f}   {resumen_ucs.loc['F9SE', 'Std']:5.1f}   {resumen_ucs.loc['F9SE', 'Min']:5.0f}   {resumen_ucs.loc['F9SE', 'Max']:5.0f}   {int(resumen_ucs.loc['F9SE', 'n']):5d}

    P100 PROMEDIO POR RANGO DE UCS
    ──────────────────────────────────────────────────────────
    Fase     UCS<80    80-120    UCS>120
    F7R1    {resumen_adicional[0]['P100 UCS<80']:>8}  {resumen_adicional[0]['P100 80-120']:>8}  {resumen_adicional[0]['P100 UCS>120']:>8}
    F9SE    {resumen_adicional[1]['P100 UCS<80']:>8}  {resumen_adicional[1]['P100 80-120']:>8}  {resumen_adicional[1]['P100 UCS>120']:>8}

    INTERPRETACIÓN:
    ► Mayor UCS → Mayor resistencia → Mayor P100 esperado
    ► F9SE tiene mayor UCS promedio → Requiere más energía
    """

    ax.text(0.05, 0.95, tabla_texto, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.set_title('12d. Resumen UCS', fontweight='bold')

    plt.suptitle('ANÁLISIS DE DUREZA DE ROCA (UCS)', fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/12_analisis_dureza_roca.png', bbox_inches='tight')
    plt.close()
    print("  ✓ 12_analisis_dureza_roca.png")

    # =========================================================================
    # GRÁFICO 13: Interacción UCS con diseño de voladura
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # 13a. P100 por UCS × Explosivo
    ax = axes[0, 0]
    df_exp = df[df['rho_explosivo'].notna()].copy()
    df_exp['rho_cat'] = pd.cut(df_exp['rho_explosivo'],
                               bins=[0, 1.15, 1.20, 2.0],
                               labels=['Baja (<1.15)', 'Media (1.15-1.20)', 'Alta (>1.20)'])

    pivot = df_exp.pivot_table(values='P100', index='UCS_categoria',
                               columns='rho_cat', aggfunc='mean').round(1)
    counts = df_exp.pivot_table(values='P100', index='UCS_categoria',
                                columns='rho_cat', aggfunc='count')
    mask = (counts.reindex_like(pivot) < 10) | pivot.isna()

    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn_r',
                mask=mask, ax=ax, vmin=8, vmax=30, linewidths=0.5)
    ax.set_title('13a. P100: UCS × Densidad Explosivo', fontweight='bold')
    ax.set_xlabel('Densidad Explosivo')
    ax.set_ylabel('UCS (MPa)')

    # 13b. P100 por UCS × Timing
    ax = axes[0, 1]
    df['timing_cat2'] = pd.cut(df['tpozos_ms'],
                               bins=[0, 3, 6, 10, 50],
                               labels=['Bajo (<3)', 'Medio (3-6)', 'Alto (6-10)', 'Muy alto (>10)'])

    pivot = df.pivot_table(values='P100', index='UCS_categoria',
                           columns='timing_cat2', aggfunc='mean').round(1)
    counts = df.pivot_table(values='P100', index='UCS_categoria',
                            columns='timing_cat2', aggfunc='count')
    mask = (counts.reindex_like(pivot) < 10) | pivot.isna()

    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn_r',
                mask=mask, ax=ax, vmin=8, vmax=30, linewidths=0.5)
    ax.set_title('13b. P100: UCS × Timing', fontweight='bold')
    ax.set_xlabel('Timing (ms)')
    ax.set_ylabel('UCS (MPa)')

    # 13c. P100 por UCS × Taco
    ax = axes[1, 0]
    pivot = df.pivot_table(values='P100', index='UCS_categoria',
                           columns='taco_eval', aggfunc='mean').round(1)
    counts = df.pivot_table(values='P100', index='UCS_categoria',
                            columns='taco_eval', aggfunc='count')
    mask = (counts.reindex_like(pivot) < 10) | pivot.isna()

    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn_r',
                mask=mask, ax=ax, vmin=8, vmax=30, linewidths=0.5)
    ax.set_title('13c. P100: UCS × Evaluación Taco', fontweight='bold')
    ax.set_xlabel('Evaluación Taco (ENAEX)')
    ax.set_ylabel('UCS (MPa)')

    # 13d. P100 por UCS × Área malla
    ax = axes[1, 1]
    df['malla_cat'] = pd.cut(df['Area_malla'],
                             bins=[0, 50, 60, 70, 80, 150],
                             labels=['<50', '50-60', '60-70', '70-80', '>80'])

    pivot = df.pivot_table(values='P100', index='UCS_categoria',
                           columns='malla_cat', aggfunc='mean').round(1)
    counts = df.pivot_table(values='P100', index='UCS_categoria',
                            columns='malla_cat', aggfunc='count')
    mask = (counts.reindex_like(pivot) < 10) | pivot.isna()

    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn_r',
                mask=mask, ax=ax, vmin=8, vmax=30, linewidths=0.5)
    ax.set_title('13d. P100: UCS × Área Malla (m²)', fontweight='bold')
    ax.set_xlabel('Área Malla (m²)')
    ax.set_ylabel('UCS (MPa)')

    plt.suptitle('INTERACCIÓN UCS CON DISEÑO DE VOLADURA', fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/13_interaccion_ucs_diseno.png', bbox_inches='tight')
    plt.close()
    print("  ✓ 13_interaccion_ucs_diseno.png")

    # =========================================================================
    # GRÁFICO 14: Comparación F7R1 vs F9SE por categoría UCS
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # 14a. Boxplot P100 por UCS y Fase
    ax = axes[0, 0]
    df_ucs = df[df['UCS_categoria'].notna()]
    sns.boxplot(data=df_ucs, x='UCS_categoria', y='P100', hue='Fase_categoria',
                ax=ax, palette={'F7R1': '#27AE60', 'F9SE': '#E74C3C'})
    ax.axhline(y=12, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('UCS (MPa)')
    ax.set_ylabel('P100 (pulgadas)')
    ax.set_title('14a. P100 por UCS y Fase', fontweight='bold')
    ax.legend(title='Fase')
    ax.tick_params(axis='x', rotation=45)

    # 14b. Boxplot P80 por UCS y Fase
    ax = axes[0, 1]
    sns.boxplot(data=df_ucs, x='UCS_categoria', y='P80', hue='Fase_categoria',
                ax=ax, palette={'F7R1': '#27AE60', 'F9SE': '#E74C3C'})
    ax.axhline(y=5, color='green', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('UCS (MPa)')
    ax.set_ylabel('P80 (pulgadas)')
    ax.set_title('14b. P80 por UCS y Fase', fontweight='bold')
    ax.legend(title='Fase')
    ax.tick_params(axis='x', rotation=45)

    # 14c. Violin plot P100 por UCS (ambas fases)
    ax = axes[1, 0]
    sns.violinplot(data=df_ucs, x='UCS_categoria', y='P100', hue='Fase_categoria',
                   split=True, ax=ax, palette={'F7R1': '#27AE60', 'F9SE': '#E74C3C'})
    ax.axhline(y=12, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('UCS (MPa)')
    ax.set_ylabel('P100 (pulgadas)')
    ax.set_title('14c. Distribución P100 por UCS (Violin)', fontweight='bold')
    ax.legend(title='Fase')
    ax.tick_params(axis='x', rotation=45)

    # 14d. Diferencia P100 entre fases por UCS
    ax = axes[1, 1]
    diferencias = []
    for ucs_cat in df_ucs['UCS_categoria'].dropna().unique():
        df_cat = df_ucs[df_ucs['UCS_categoria'] == ucs_cat]
        p100_f7 = df_cat[df_cat['Fase_categoria'] == 'F7R1']['P100'].mean()
        p100_f9 = df_cat[df_cat['Fase_categoria'] == 'F9SE']['P100'].mean()
        n_f7 = len(df_cat[df_cat['Fase_categoria'] == 'F7R1'])
        n_f9 = len(df_cat[df_cat['Fase_categoria'] == 'F9SE'])
        if n_f7 >= 20 and n_f9 >= 20:
            diferencias.append({
                'UCS': str(ucs_cat),
                'Diferencia': p100_f9 - p100_f7,
                'P100_F7R1': p100_f7,
                'P100_F9SE': p100_f9
            })

    if diferencias:
        df_diff = pd.DataFrame(diferencias)
        colors = ['#E74C3C' if d > 0 else '#27AE60' for d in df_diff['Diferencia']]
        bars = ax.barh(df_diff['UCS'], df_diff['Diferencia'], color=colors, alpha=0.8)
        ax.axvline(x=0, color='black', linewidth=1)
        ax.set_xlabel('Diferencia P100 (F9SE - F7R1)')
        ax.set_ylabel('UCS (MPa)')
        ax.set_title('14d. Diferencia P100 entre Fases por UCS', fontweight='bold')

        for i, (_, row) in enumerate(df_diff.iterrows()):
            ax.text(row['Diferencia'] + 0.2 if row['Diferencia'] > 0 else row['Diferencia'] - 0.2,
                    i, f'{row["Diferencia"]:+.1f}"', va='center', fontsize=9,
                    ha='left' if row['Diferencia'] > 0 else 'right')
    else:
        ax.text(0.5, 0.5, 'Datos insuficientes', transform=ax.transAxes,
                ha='center', va='center', fontsize=14)

    plt.suptitle('COMPARACIÓN F7R1 vs F9SE POR CATEGORÍA UCS', fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/14_comparacion_por_ucs.png', bbox_inches='tight')
    plt.close()
    print("  ✓ 14_comparacion_por_ucs.png")

    # =========================================================================
    # GRÁFICO 15: Análisis RQD (si existe)
    # =========================================================================
    if 'RQD' in df.columns and df['RQD'].notna().sum() > 50:
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        # 15a. P100 vs RQD por fase
        ax = axes[0, 0]
        for fase, color in [('F7R1', '#27AE60'), ('F9SE', '#E74C3C')]:
            df_f = df[df['Fase_categoria'] == fase]
            df_rqd = df_f[['RQD', 'P100']].dropna()

            if len(df_rqd) > 30:
                ax.scatter(df_rqd['RQD'], df_rqd['P100'],
                           alpha=0.3, s=20, label=fase, c=color)

                # Regresión lineal
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    df_rqd['RQD'].astype(float), df_rqd['P100'].astype(float)
                )
                x_line = np.linspace(df_rqd['RQD'].min(), df_rqd['RQD'].max(), 100)
                y_line = slope * x_line + intercept
                ax.plot(x_line, y_line, color=color, linestyle='--', linewidth=2,
                        label=f'{fase}: R²={r_value**2:.2f}')

        ax.axhline(y=12, color='red', linestyle=':', linewidth=2, alpha=0.7)
        ax.set_xlabel('RQD (%)')
        ax.set_ylabel('P100 (pulgadas)')
        ax.set_title('15a. P100 vs RQD por Fase', fontweight='bold')
        ax.legend()

        # 15b. Heatmap P100 por RQD × Fase
        ax = axes[0, 1]
        pivot = df.pivot_table(values='P100', index='Fase_categoria',
                               columns='RQD_categoria', aggfunc='mean').round(1)
        counts = df.pivot_table(values='P100', index='Fase_categoria',
                                columns='RQD_categoria', aggfunc='count')
        mask = (counts.reindex_like(pivot) < 10) | pivot.isna()

        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn_r',
                    mask=mask, ax=ax, vmin=8, vmax=30, linewidths=0.5)
        ax.set_title('15b. P100 por Fase × RQD', fontweight='bold')
        ax.set_xlabel('RQD')
        ax.set_ylabel('Fase')
        ax.tick_params(axis='x', rotation=45)

        # 15c. Distribución RQD por fase
        ax = axes[1, 0]
        for fase, color in [('F7R1', '#27AE60'), ('F9SE', '#E74C3C')]:
            df_f = df[df['Fase_categoria'] == fase]['RQD'].dropna()
            if len(df_f) > 30:
                sns.kdeplot(df_f, ax=ax, label=fase, color=color, fill=True, alpha=0.3)

        ax.axvline(x=50, color='orange', linestyle='--', linewidth=2, label='RQD pobre (50)')
        ax.axvline(x=75, color='green', linestyle='--', linewidth=2, label='RQD buena (75)')
        ax.set_xlabel('RQD (%)')
        ax.set_ylabel('Densidad')
        ax.set_title('15c. Distribución de RQD por Fase', fontweight='bold')
        ax.legend()

        # 15d. Boxplot RQD por fase
        ax = axes[1, 1]
        df_rqd = df[df['RQD'].notna()]
        sns.boxplot(data=df_rqd, x='Fase_categoria', y='RQD',
                    palette={'F7R1': '#27AE60', 'F9SE': '#E74C3C'}, ax=ax)
        ax.set_xlabel('Fase')
        ax.set_ylabel('RQD (%)')
        ax.set_title('15d. Distribución RQD por Fase', fontweight='bold')

        # Agregar estadísticas
        for i, fase in enumerate(['F7R1', 'F9SE']):
            df_f = df_rqd[df_rqd['Fase_categoria'] == fase]['RQD']
            media = df_f.mean()
            ax.text(i, df_f.max() + 2, f'μ={media:.1f}', ha='center', fontsize=10)

        plt.suptitle('ANÁLISIS DE RQD (Rock Quality Designation)', fontweight='bold', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/15_analisis_rqd.png', bbox_inches='tight')
        plt.close()
        print("  ✓ 15_analisis_rqd.png")
    else:
        print("  ! Datos de RQD insuficientes, omitiendo gráfico 15")

    # =========================================================================
    # GRÁFICO 16: Matriz UCS × RQD
    # =========================================================================
    if 'RQD' in df.columns and df['RQD'].notna().sum() > 50:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # 16a. Heatmap P100: UCS × RQD
        ax = axes[0]
        pivot = df.pivot_table(values='P100', index='UCS_categoria',
                               columns='RQD_categoria', aggfunc='mean').round(1)
        counts = df.pivot_table(values='P100', index='UCS_categoria',
                                columns='RQD_categoria', aggfunc='count')
        mask = (counts.reindex_like(pivot) < 8) | pivot.isna()

        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn_r',
                    mask=mask, ax=ax, vmin=8, vmax=30, linewidths=0.5,
                    cbar_kws={'label': 'P100 (pulgadas)'})
        ax.set_title('16a. P100: UCS × RQD (todas las fases)', fontweight='bold')
        ax.set_xlabel('RQD')
        ax.set_ylabel('UCS (MPa)')
        ax.tick_params(axis='x', rotation=45)

        # 16b. Scatter UCS vs RQD coloreado por P100
        ax = axes[1]
        df_ucs_rqd = df[['UCS_MPA', 'RQD', 'P100', 'Fase_categoria']].dropna()

        if len(df_ucs_rqd) > 30:
            scatter = ax.scatter(df_ucs_rqd['UCS_MPA'], df_ucs_rqd['RQD'],
                                 c=df_ucs_rqd['P100'], cmap='RdYlGn_r',
                                 alpha=0.6, s=30, vmin=8, vmax=25)
            plt.colorbar(scatter, ax=ax, label='P100 (pulgadas)')

            # Zonas de referencia
            ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
            ax.axhline(y=75, color='gray', linestyle='--', alpha=0.5)
            ax.axvline(x=80, color='gray', linestyle='--', alpha=0.5)
            ax.axvline(x=120, color='gray', linestyle='--', alpha=0.5)

            # Etiquetas de zonas
            ax.text(50, 85, 'Roca media\nBuen RQD', fontsize=8, ha='center', alpha=0.7)
            ax.text(140, 35, 'Roca dura\nRQD pobre', fontsize=8, ha='center', alpha=0.7)

        ax.set_xlabel('UCS (MPa)')
        ax.set_ylabel('RQD (%)')
        ax.set_title('16b. Relación UCS-RQD coloreado por P100', fontweight='bold')

        plt.suptitle('MATRIZ UCS × RQD Y FRAGMENTACIÓN', fontweight='bold', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/16_matriz_ucs_rqd.png', bbox_inches='tight')
        plt.close()
        print("  ✓ 16_matriz_ucs_rqd.png")
    else:
        print("  ! Datos de RQD insuficientes, omitiendo gráfico 16")


# ==============================================================================
# ANÁLISIS DE SENSIBILIDAD DETALLADO POR FACTOR
# ==============================================================================

def analisis_sensibilidad_por_factor(df):
    """Análisis de sensibilidad detallado para cada factor: explosivo, taco, geometría, timing."""
    print("\n" + "=" * 70)
    print("GENERANDO ANÁLISIS DE SENSIBILIDAD POR FACTOR")
    print("=" * 70)

    # =========================================================================
    # GRÁFICO 21: Sensibilidad factores de EXPLOSIVO
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    factores_explosivo = [
        ('rho_explosivo', 'Densidad (g/cc)', [0, 1.1, 1.2, 1.25, 1.3, 1.35, 2]),
        ('VOD', 'VOD (m/s)', [0, 3000, 3500, 4000, 4500, 5000, 7000]),
        ('Energia_MJ', 'Energía (MJ/kg)', [0, 2.8, 3.0, 3.2, 3.4, 3.6, 5]),
        ('RWS', 'RWS (%ANFO)', [0, 80, 85, 88, 92, 95, 100]),
    ]

    for idx, (var, label, bins) in enumerate(factores_explosivo):
        ax = axes[idx // 2, idx % 2]

        if var not in df.columns or df[var].notna().sum() < 100:
            ax.text(0.5, 0.5, f'Datos insuficientes\npara {label}',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'21{chr(97+idx)}. P100 vs {label}', fontweight='bold')
            continue

        # Crear categorías
        labels_cat = [f'{bins[i]}-{bins[i+1]}' for i in range(len(bins)-1)]
        df[f'{var}_cat'] = pd.cut(df[var], bins=bins, labels=labels_cat)

        # Calcular P100 promedio por categoría y fase
        for fase, color in [('F7R1', '#27AE60'), ('F9SE', '#E74C3C')]:
            df_f = df[df['Fase_categoria'] == fase]
            grouped = df_f.groupby(f'{var}_cat').agg({
                'P100': ['mean', 'std', 'count']
            }).reset_index()
            grouped.columns = ['cat', 'mean', 'std', 'n']
            grouped = grouped[grouped['n'] >= 20]

            if len(grouped) > 0:
                x = range(len(grouped))
                ax.errorbar(x, grouped['mean'], yerr=grouped['std']/2,
                           marker='o', markersize=8, capsize=5,
                           label=f'{fase} (n={grouped["n"].sum():,})',
                           color=color, linewidth=2)

                # Anotaciones
                for i, row in grouped.iterrows():
                    ax.annotate(f'{row["mean"]:.1f}"', (list(x)[list(grouped.index).index(i)], row['mean']),
                               textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

        ax.axhline(y=12, color='red', linestyle='--', alpha=0.7, label='Target 12"')
        ax.set_xticks(range(len(labels_cat)))
        ax.set_xticklabels(labels_cat, rotation=45)
        ax.set_xlabel(label)
        ax.set_ylabel('P100 (pulgadas)')
        ax.set_title(f'21{chr(97+idx)}. P100 vs {label}', fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle('SENSIBILIDAD: FACTORES DE EXPLOSIVO', fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/21_sensibilidad_explosivo.png', bbox_inches='tight')
    plt.close()
    print("  ✓ 21_sensibilidad_explosivo.png")

    # =========================================================================
    # GRÁFICO 22: Sensibilidad TACO (stemming)
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # 22a. P100 vs Taco absoluto
    ax = axes[0, 0]
    if 'taco_efectivo' in df.columns:
        bins_taco = [0, 2, 3, 4, 5, 6, 7, 15]
        labels_taco = ['<2', '2-3', '3-4', '4-5', '5-6', '6-7', '>7']
        df['taco_cat2'] = pd.cut(df['taco_efectivo'], bins=bins_taco, labels=labels_taco)

        for fase, color in [('F7R1', '#27AE60'), ('F9SE', '#E74C3C')]:
            df_f = df[df['Fase_categoria'] == fase]
            grouped = df_f.groupby('taco_cat2')['P100'].agg(['mean', 'count']).reset_index()
            grouped = grouped[grouped['count'] >= 20]

            if len(grouped) > 0:
                ax.plot(range(len(grouped)), grouped['mean'], marker='o', markersize=10,
                       label=f'{fase}', color=color, linewidth=2)

        ax.axhline(y=12, color='red', linestyle='--', alpha=0.7)
        ax.set_xticks(range(len(labels_taco)))
        ax.set_xticklabels(labels_taco)
        ax.set_xlabel('Taco (m)')
        ax.set_ylabel('P100 (pulgadas)')
        ax.set_title('22a. P100 vs Longitud de Taco', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

    # 22b. P100 vs Ratio Taco/Burden
    ax = axes[0, 1]
    if 'taco_burden_ratio' in df.columns:
        bins_tb = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 2.0]
        labels_tb = ['<0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-1.0', '>1.0']
        df['tb_cat2'] = pd.cut(df['taco_burden_ratio'], bins=bins_tb, labels=labels_tb)

        for fase, color in [('F7R1', '#27AE60'), ('F9SE', '#E74C3C')]:
            df_f = df[df['Fase_categoria'] == fase]
            grouped = df_f.groupby('tb_cat2')['P100'].agg(['mean', 'count']).reset_index()
            grouped = grouped[grouped['count'] >= 20]

            if len(grouped) > 0:
                ax.plot(range(len(grouped)), grouped['mean'], marker='s', markersize=10,
                       label=f'{fase}', color=color, linewidth=2)

        ax.axhline(y=12, color='red', linestyle='--', alpha=0.7)
        ax.axvline(x=4.5, color='green', linestyle='--', alpha=0.7, label='Óptimo ENAEX (0.7)')
        ax.set_xticks(range(len(labels_tb)))
        ax.set_xticklabels(labels_tb, rotation=45)
        ax.set_xlabel('Ratio Taco/Burden')
        ax.set_ylabel('P100 (pulgadas)')
        ax.set_title('22b. P100 vs Ratio Taco/Burden', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

    # 22c. Heatmap Taco × Fase detallado
    ax = axes[1, 0]
    if 'taco_efectivo' in df.columns:
        pivot = df.pivot_table(values='P100', index='taco_cat2',
                               columns='Fase_categoria', aggfunc='mean').round(1)
        counts = df.pivot_table(values='P100', index='taco_cat2',
                                columns='Fase_categoria', aggfunc='count')
        mask = (counts.reindex_like(pivot) < 15) | pivot.isna()

        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn_r',
                    mask=mask, ax=ax, vmin=8, vmax=25, linewidths=0.5)
        ax.set_title('22c. Heatmap P100: Taco × Fase', fontweight='bold')
        ax.set_xlabel('Fase')
        ax.set_ylabel('Taco (m)')

    # 22d. Tabla resumen taco
    ax = axes[1, 1]
    ax.axis('off')

    resumen_taco = []
    for fase in ['F7R1', 'F9SE']:
        df_f = df[df['Fase_categoria'] == fase]
        if 'taco_efectivo' in df.columns:
            # Mejor y peor taco
            grouped = df_f.groupby('taco_cat2')['P100'].mean()
            mejor_taco = grouped.idxmin() if len(grouped) > 0 else 'N/A'
            peor_taco = grouped.idxmax() if len(grouped) > 0 else 'N/A'
            mejor_p100 = grouped.min() if len(grouped) > 0 else 0
            peor_p100 = grouped.max() if len(grouped) > 0 else 0

            resumen_taco.append({
                'Fase': fase,
                'Taco prom': f'{df_f["taco_efectivo"].mean():.1f}m',
                'T/B prom': f'{df_f["taco_burden_ratio"].mean():.2f}',
                'Mejor taco': f'{mejor_taco} ({mejor_p100:.1f}")',
                'Peor taco': f'{peor_taco} ({peor_p100:.1f}")',
                'Diferencia': f'{peor_p100 - mejor_p100:.1f}"'
            })

    if resumen_taco:
        tabla_texto = "RESUMEN SENSIBILIDAD TACO\n" + "="*60 + "\n\n"
        for r in resumen_taco:
            tabla_texto += f"  {r['Fase']}:\n"
            tabla_texto += f"    Taco promedio: {r['Taco prom']}\n"
            tabla_texto += f"    Ratio T/B promedio: {r['T/B prom']}\n"
            tabla_texto += f"    Mejor configuración: {r['Mejor taco']}\n"
            tabla_texto += f"    Peor configuración: {r['Peor taco']}\n"
            tabla_texto += f"    Rango de impacto: {r['Diferencia']}\n\n"

        ax.text(0.1, 0.9, tabla_texto, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.set_title('22d. Resumen Taco', fontweight='bold')

    plt.suptitle('SENSIBILIDAD: TACO (STEMMING)', fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/22_sensibilidad_taco.png', bbox_inches='tight')
    plt.close()
    print("  ✓ 22_sensibilidad_taco.png")

    # =========================================================================
    # GRÁFICO 23: Sensibilidad BURDEN × ESPACIAMIENTO
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # 23a. P100 vs Burden
    ax = axes[0, 0]
    bins_burden = [0, 6, 7, 8, 9, 10, 11, 15]
    labels_burden = ['<6', '6-7', '7-8', '8-9', '9-10', '10-11', '>11']
    df['burden_cat'] = pd.cut(df['Burden'], bins=bins_burden, labels=labels_burden)

    for fase, color in [('F7R1', '#27AE60'), ('F9SE', '#E74C3C')]:
        df_f = df[df['Fase_categoria'] == fase]
        grouped = df_f.groupby('burden_cat')['P100'].agg(['mean', 'count']).reset_index()
        grouped = grouped[grouped['count'] >= 20]

        if len(grouped) > 0:
            ax.plot(range(len(grouped)), grouped['mean'], marker='o', markersize=10,
                   label=f'{fase}', color=color, linewidth=2)

    ax.axhline(y=12, color='red', linestyle='--', alpha=0.7)
    ax.set_xticks(range(len(labels_burden)))
    ax.set_xticklabels(labels_burden)
    ax.set_xlabel('Burden (m)')
    ax.set_ylabel('P100 (pulgadas)')
    ax.set_title('23a. P100 vs Burden', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # 23b. P100 vs Espaciamiento
    ax = axes[0, 1]
    bins_esp = [0, 7, 8, 9, 10, 11, 12, 15]
    labels_esp = ['<7', '7-8', '8-9', '9-10', '10-11', '11-12', '>12']
    df['esp_cat2'] = pd.cut(df['Espaciamiento'], bins=bins_esp, labels=labels_esp)

    for fase, color in [('F7R1', '#27AE60'), ('F9SE', '#E74C3C')]:
        df_f = df[df['Fase_categoria'] == fase]
        grouped = df_f.groupby('esp_cat2')['P100'].agg(['mean', 'count']).reset_index()
        grouped = grouped[grouped['count'] >= 20]

        if len(grouped) > 0:
            ax.plot(range(len(grouped)), grouped['mean'], marker='s', markersize=10,
                   label=f'{fase}', color=color, linewidth=2)

    ax.axhline(y=12, color='red', linestyle='--', alpha=0.7)
    ax.set_xticks(range(len(labels_esp)))
    ax.set_xticklabels(labels_esp)
    ax.set_xlabel('Espaciamiento (m)')
    ax.set_ylabel('P100 (pulgadas)')
    ax.set_title('23b. P100 vs Espaciamiento', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # 23c. P100 vs Área malla (B×S)
    ax = axes[1, 0]
    bins_area = [0, 50, 60, 70, 80, 90, 100, 120, 200]
    labels_area = ['<50', '50-60', '60-70', '70-80', '80-90', '90-100', '100-120', '>120']
    df['area_cat2'] = pd.cut(df['Area_malla'], bins=bins_area, labels=labels_area)

    for fase, color in [('F7R1', '#27AE60'), ('F9SE', '#E74C3C')]:
        df_f = df[df['Fase_categoria'] == fase]
        grouped = df_f.groupby('area_cat2')['P100'].agg(['mean', 'count']).reset_index()
        grouped = grouped[grouped['count'] >= 20]

        if len(grouped) > 0:
            ax.plot(range(len(grouped)), grouped['mean'], marker='^', markersize=10,
                   label=f'{fase}', color=color, linewidth=2)

    ax.axhline(y=12, color='red', linestyle='--', alpha=0.7)
    ax.set_xticks(range(len(labels_area)))
    ax.set_xticklabels(labels_area, rotation=45)
    ax.set_xlabel('Área malla B×S (m²)')
    ax.set_ylabel('P100 (pulgadas)')
    ax.set_title('23c. P100 vs Área Malla (B×S)', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # 23d. P100 vs Ratio S/B
    ax = axes[1, 1]
    bins_sb = [0, 0.9, 1.0, 1.1, 1.15, 1.2, 1.25, 1.3, 2.0]
    labels_sb = ['<0.9', '0.9-1.0', '1.0-1.1', '1.1-1.15', '1.15-1.2', '1.2-1.25', '1.25-1.3', '>1.3']
    df['sb_cat2'] = pd.cut(df['S_B_ratio'], bins=bins_sb, labels=labels_sb)

    for fase, color in [('F7R1', '#27AE60'), ('F9SE', '#E74C3C')]:
        df_f = df[df['Fase_categoria'] == fase]
        grouped = df_f.groupby('sb_cat2')['P100'].agg(['mean', 'count']).reset_index()
        grouped = grouped[grouped['count'] >= 20]

        if len(grouped) > 0:
            ax.plot(range(len(grouped)), grouped['mean'], marker='d', markersize=10,
                   label=f'{fase}', color=color, linewidth=2)

    ax.axhline(y=12, color='red', linestyle='--', alpha=0.7)
    ax.axvline(x=3.5, color='green', linestyle='--', alpha=0.7, label='Óptimo (1.15)')
    ax.set_xticks(range(len(labels_sb)))
    ax.set_xticklabels(labels_sb, rotation=45)
    ax.set_xlabel('Ratio S/B')
    ax.set_ylabel('P100 (pulgadas)')
    ax.set_title('23d. P100 vs Ratio S/B', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle('SENSIBILIDAD: BURDEN × ESPACIAMIENTO', fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/23_sensibilidad_geometria.png', bbox_inches='tight')
    plt.close()
    print("  ✓ 23_sensibilidad_geometria.png")

    # =========================================================================
    # GRÁFICO 24: Sensibilidad TIMING ENTRE POZOS
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # 24a. P100 vs Timing pozos (detallado)
    ax = axes[0, 0]
    bins_tp = [0, 1, 2, 3, 4, 5, 6, 8, 10, 50]
    labels_tp = ['<1', '1-2', '2-3', '3-4', '4-5', '5-6', '6-8', '8-10', '>10']
    df['tp_cat2'] = pd.cut(df['tpozos_ms'], bins=bins_tp, labels=labels_tp)

    for fase, color in [('F7R1', '#27AE60'), ('F9SE', '#E74C3C')]:
        df_f = df[df['Fase_categoria'] == fase]
        grouped = df_f.groupby('tp_cat2')['P100'].agg(['mean', 'std', 'count']).reset_index()
        grouped = grouped[grouped['count'] >= 20]

        if len(grouped) > 0:
            x = range(len(grouped))
            ax.errorbar(x, grouped['mean'], yerr=grouped['std']/2,
                       marker='o', markersize=10, capsize=5,
                       label=f'{fase}', color=color, linewidth=2)

    ax.axhline(y=12, color='red', linestyle='--', alpha=0.7)
    ax.set_xticks(range(len(labels_tp)))
    ax.set_xticklabels(labels_tp)
    ax.set_xlabel('Timing entre pozos (ms)')
    ax.set_ylabel('P100 (pulgadas)')
    ax.set_title('24a. P100 vs Timing Pozos (detallado)', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # 24b. P100 vs Timing filas
    ax = axes[0, 1]
    bins_tf = [0, 25, 50, 75, 100, 125, 150, 200, 500]
    labels_tf = ['<25', '25-50', '50-75', '75-100', '100-125', '125-150', '150-200', '>200']
    df['tf_cat2'] = pd.cut(df['tfilas_ms'], bins=bins_tf, labels=labels_tf)

    for fase, color in [('F7R1', '#27AE60'), ('F9SE', '#E74C3C')]:
        df_f = df[df['Fase_categoria'] == fase]
        grouped = df_f.groupby('tf_cat2')['P100'].agg(['mean', 'count']).reset_index()
        grouped = grouped[grouped['count'] >= 20]

        if len(grouped) > 0:
            ax.plot(range(len(grouped)), grouped['mean'], marker='s', markersize=10,
                   label=f'{fase}', color=color, linewidth=2)

    ax.axhline(y=12, color='red', linestyle='--', alpha=0.7)
    ax.set_xticks(range(len(labels_tf)))
    ax.set_xticklabels(labels_tf, rotation=45)
    ax.set_xlabel('Timing entre filas (ms)')
    ax.set_ylabel('P100 (pulgadas)')
    ax.set_title('24b. P100 vs Timing Filas', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # 24c. Heatmap Timing pozos × filas
    ax = axes[1, 0]
    pivot = df.pivot_table(values='P100', index='tp_cat2',
                           columns='tf_cat2', aggfunc='mean').round(1)
    counts = df.pivot_table(values='P100', index='tp_cat2',
                            columns='tf_cat2', aggfunc='count')
    mask = (counts.reindex_like(pivot) < 15) | pivot.isna()

    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn_r',
                mask=mask, ax=ax, vmin=8, vmax=25, linewidths=0.5)
    ax.set_title('24c. P100: Timing Pozos × Timing Filas', fontweight='bold')
    ax.set_xlabel('Timing filas (ms)')
    ax.set_ylabel('Timing pozos (ms)')

    # 24d. Resumen timing
    ax = axes[1, 1]
    ax.axis('off')

    resumen_timing = []
    for fase in ['F7R1', 'F9SE']:
        df_f = df[df['Fase_categoria'] == fase]
        grouped_tp = df_f.groupby('tp_cat2')['P100'].mean()
        grouped_tf = df_f.groupby('tf_cat2')['P100'].mean()

        mejor_tp = grouped_tp.idxmin() if len(grouped_tp) > 0 else 'N/A'
        mejor_tf = grouped_tf.idxmin() if len(grouped_tf) > 0 else 'N/A'

        resumen_timing.append({
            'Fase': fase,
            'T. pozos prom': f'{df_f["tpozos_ms"].mean():.1f} ms',
            'T. filas prom': f'{df_f["tfilas_ms"].mean():.1f} ms',
            'Mejor t. pozos': f'{mejor_tp} ms → {grouped_tp.min():.1f}"' if len(grouped_tp) > 0 else 'N/A',
            'Mejor t. filas': f'{mejor_tf} ms → {grouped_tf.min():.1f}"' if len(grouped_tf) > 0 else 'N/A',
            'Rango pozos': f'{grouped_tp.max() - grouped_tp.min():.1f}"' if len(grouped_tp) > 0 else 'N/A',
        })

    tabla_texto = "RESUMEN SENSIBILIDAD TIMING\n" + "="*60 + "\n\n"
    for r in resumen_timing:
        tabla_texto += f"  {r['Fase']}:\n"
        tabla_texto += f"    Timing pozos promedio: {r['T. pozos prom']}\n"
        tabla_texto += f"    Timing filas promedio: {r['T. filas prom']}\n"
        tabla_texto += f"    Mejor timing pozos: {r['Mejor t. pozos']}\n"
        tabla_texto += f"    Mejor timing filas: {r['Mejor t. filas']}\n"
        tabla_texto += f"    Rango impacto pozos: {r['Rango pozos']}\n\n"

    ax.text(0.1, 0.9, tabla_texto, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.set_title('24d. Resumen Timing', fontweight='bold')

    plt.suptitle('SENSIBILIDAD: TIMING ENTRE POZOS Y FILAS', fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/24_sensibilidad_timing.png', bbox_inches='tight')
    plt.close()
    print("  ✓ 24_sensibilidad_timing.png")

    # =========================================================================
    # GRÁFICO 25: Resumen comparativo de sensibilidad
    # =========================================================================
    fig, ax = plt.subplots(figsize=(16, 10))

    # Calcular sensibilidad (rango P100) para cada factor
    factores = []

    for fase in ['F7R1', 'F9SE']:
        df_f = df[df['Fase_categoria'] == fase]

        # Timing pozos
        grouped = df_f.groupby('tp_cat2')['P100'].mean()
        if len(grouped) >= 3:
            factores.append({'Fase': fase, 'Factor': 'Timing pozos', 'Rango': grouped.max() - grouped.min()})

        # Timing filas
        grouped = df_f.groupby('tf_cat2')['P100'].mean()
        if len(grouped) >= 3:
            factores.append({'Fase': fase, 'Factor': 'Timing filas', 'Rango': grouped.max() - grouped.min()})

        # Taco
        if 'taco_cat2' in df.columns:
            grouped = df_f.groupby('taco_cat2')['P100'].mean()
            if len(grouped) >= 3:
                factores.append({'Fase': fase, 'Factor': 'Taco', 'Rango': grouped.max() - grouped.min()})

        # Ratio T/B
        if 'tb_cat2' in df.columns:
            grouped = df_f.groupby('tb_cat2')['P100'].mean()
            if len(grouped) >= 3:
                factores.append({'Fase': fase, 'Factor': 'Ratio T/B', 'Rango': grouped.max() - grouped.min()})

        # Burden
        grouped = df_f.groupby('burden_cat')['P100'].mean()
        if len(grouped) >= 3:
            factores.append({'Fase': fase, 'Factor': 'Burden', 'Rango': grouped.max() - grouped.min()})

        # Espaciamiento
        grouped = df_f.groupby('esp_cat2')['P100'].mean()
        if len(grouped) >= 3:
            factores.append({'Fase': fase, 'Factor': 'Espaciamiento', 'Rango': grouped.max() - grouped.min()})

        # Área malla
        grouped = df_f.groupby('area_cat2')['P100'].mean()
        if len(grouped) >= 3:
            factores.append({'Fase': fase, 'Factor': 'Área malla', 'Rango': grouped.max() - grouped.min()})

        # Ratio S/B
        grouped = df_f.groupby('sb_cat2')['P100'].mean()
        if len(grouped) >= 3:
            factores.append({'Fase': fase, 'Factor': 'Ratio S/B', 'Rango': grouped.max() - grouped.min()})

        # Densidad explosivo
        if 'rho_explosivo_cat' in df.columns:
            grouped = df_f.groupby('rho_explosivo_cat')['P100'].mean()
            if len(grouped) >= 3:
                factores.append({'Fase': fase, 'Factor': 'Densidad exp.', 'Rango': grouped.max() - grouped.min()})

        # VOD
        if 'VOD_cat' in df.columns:
            grouped = df_f.groupby('VOD_cat')['P100'].mean()
            if len(grouped) >= 3:
                factores.append({'Fase': fase, 'Factor': 'VOD', 'Rango': grouped.max() - grouped.min()})

    if factores:
        df_factores = pd.DataFrame(factores)

        # Pivot para gráfico
        pivot = df_factores.pivot(index='Factor', columns='Fase', values='Rango')
        pivot = pivot.sort_values('F9SE', ascending=True)

        y_pos = np.arange(len(pivot))
        height = 0.35

        if 'F7R1' in pivot.columns:
            ax.barh(y_pos - height/2, pivot['F7R1'].fillna(0), height,
                   label='F7R1', color='#27AE60', alpha=0.8)
        if 'F9SE' in pivot.columns:
            ax.barh(y_pos + height/2, pivot['F9SE'].fillna(0), height,
                   label='F9SE', color='#E74C3C', alpha=0.8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel('Rango de P100 (pulgadas)')
        ax.set_title('Comparación de Sensibilidad por Factor\n(Mayor rango = Mayor impacto)', fontweight='bold')
        ax.legend()
        ax.grid(axis='x', alpha=0.3)

        # Anotaciones
        for i, factor in enumerate(pivot.index):
            if 'F7R1' in pivot.columns and not pd.isna(pivot.loc[factor, 'F7R1']):
                ax.text(pivot.loc[factor, 'F7R1'] + 0.1, i - height/2,
                       f'{pivot.loc[factor, "F7R1"]:.1f}"', va='center', fontsize=9)
            if 'F9SE' in pivot.columns and not pd.isna(pivot.loc[factor, 'F9SE']):
                ax.text(pivot.loc[factor, 'F9SE'] + 0.1, i + height/2,
                       f'{pivot.loc[factor, "F9SE"]:.1f}"', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/25_resumen_sensibilidad_factores.png', bbox_inches='tight')
    plt.close()
    print("  ✓ 25_resumen_sensibilidad_factores.png")


# ==============================================================================
# ANÁLISIS DETALLADO DE TIMING (POZOS Y FILAS)
# ==============================================================================

def analisis_timing_detallado(df):
    """Análisis detallado de timing entre pozos y filas según ENAEX."""
    print("\n" + "=" * 70)
    print("GENERANDO ANÁLISIS DETALLADO DE TIMING (POZOS/FILAS)")
    print("=" * 70)

    # Verificar columnas necesarias
    if 'tpozos_ms' not in df.columns or 'tfilas_ms' not in df.columns:
        print("  ADVERTENCIA: Columnas de timing no encontradas")
        return

    # =========================================================================
    # GRÁFICO 17: Análisis timing pozos
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # 17a. Distribución timing pozos por fase
    ax = axes[0, 0]
    for fase, color in [('F7R1', '#27AE60'), ('F9SE', '#E74C3C')]:
        df_f = df[df['Fase_categoria'] == fase]['tpozos_ms'].dropna()
        if len(df_f) > 30:
            sns.kdeplot(df_f, ax=ax, label=f'{fase} (μ={df_f.mean():.1f}ms)',
                        color=color, fill=True, alpha=0.3)

    # Zonas ENAEX
    ax.axvline(x=2, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Mín. ENAEX (2ms)')
    ax.axvline(x=6, color='blue', linestyle='--', linewidth=2, alpha=0.7, label='Óptimo ENAEX (~6ms)')
    ax.axvline(x=10, color='purple', linestyle='--', linewidth=2, alpha=0.7, label='Máx. recomendado (10ms)')
    ax.set_xlabel('Timing entre pozos (ms)')
    ax.set_ylabel('Densidad')
    ax.set_title('17a. Distribución Timing Pozos por Fase', fontweight='bold')
    ax.legend(fontsize=8)

    # 17b. Boxplot timing pozos por fase y UCS
    ax = axes[0, 1]
    df_timing = df[df['tpozos_ms'].notna() & df['UCS_categoria'].notna()]
    if len(df_timing) > 50:
        sns.boxplot(data=df_timing, x='UCS_categoria', y='tpozos_ms', hue='Fase_categoria',
                    ax=ax, palette={'F7R1': '#27AE60', 'F9SE': '#E74C3C'})
        ax.axhline(y=6, color='blue', linestyle='--', linewidth=2, alpha=0.7)
        ax.set_xlabel('UCS (MPa)')
        ax.set_ylabel('Timing pozos (ms)')
        ax.set_title('17b. Timing Pozos por UCS y Fase', fontweight='bold')
        ax.legend(title='Fase')
        ax.tick_params(axis='x', rotation=45)

    # 17c. P100 vs Timing pozos con regresión
    ax = axes[1, 0]
    for fase, color in [('F7R1', '#27AE60'), ('F9SE', '#E74C3C')]:
        df_f = df[df['Fase_categoria'] == fase][['tpozos_ms', 'P100']].dropna()

        if len(df_f) > 30:
            ax.scatter(df_f['tpozos_ms'], df_f['P100'],
                       alpha=0.3, s=20, label=fase, c=color)

            # Regresión
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                df_f['tpozos_ms'].astype(float), df_f['P100'].astype(float)
            )
            x_line = np.linspace(df_f['tpozos_ms'].min(), df_f['tpozos_ms'].max(), 100)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, color=color, linestyle='--', linewidth=2,
                    label=f'{fase}: R²={r_value**2:.2f}, pendiente={slope:.2f}')

    ax.axhline(y=12, color='red', linestyle=':', linewidth=2, alpha=0.7)
    ax.set_xlabel('Timing entre pozos (ms)')
    ax.set_ylabel('P100 (pulgadas)')
    ax.set_title('17c. P100 vs Timing Pozos', fontweight='bold')
    ax.legend(fontsize=9)

    # 17d. Heatmap P100 por timing pozos y fase
    ax = axes[1, 1]
    df['tpozos_cat'] = pd.cut(df['tpozos_ms'],
                              bins=[0, 2, 4, 6, 8, 10, 50],
                              labels=['<2', '2-4', '4-6', '6-8', '8-10', '>10'])

    pivot = df.pivot_table(values='P100', index='Fase_categoria',
                           columns='tpozos_cat', aggfunc='mean').round(1)
    counts = df.pivot_table(values='P100', index='Fase_categoria',
                            columns='tpozos_cat', aggfunc='count')
    mask = (counts.reindex_like(pivot) < 15) | pivot.isna()

    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn_r',
                mask=mask, ax=ax, vmin=8, vmax=25, linewidths=0.5,
                cbar_kws={'label': 'P100 (pulgadas)'})
    ax.set_title('17d. P100 por Fase × Timing Pozos', fontweight='bold')
    ax.set_xlabel('Timing pozos (ms)')
    ax.set_ylabel('Fase')

    plt.suptitle('ANÁLISIS DETALLADO TIMING ENTRE POZOS', fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/17_analisis_timing_pozos.png', bbox_inches='tight')
    plt.close()
    print("  ✓ 17_analisis_timing_pozos.png")

    # =========================================================================
    # GRÁFICO 18: Análisis timing filas
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # 18a. Distribución timing filas por fase
    ax = axes[0, 0]
    for fase, color in [('F7R1', '#27AE60'), ('F9SE', '#E74C3C')]:
        df_f = df[df['Fase_categoria'] == fase]['tfilas_ms'].dropna()
        if len(df_f) > 30:
            sns.kdeplot(df_f, ax=ax, label=f'{fase} (μ={df_f.mean():.1f}ms)',
                        color=color, fill=True, alpha=0.3)

    # Zonas ENAEX (timing filas es típicamente mayor)
    ax.axvline(x=50, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Mín. ENAEX (50ms)')
    ax.axvline(x=100, color='blue', linestyle='--', linewidth=2, alpha=0.7, label='Óptimo (~100ms)')
    ax.set_xlabel('Timing entre filas (ms)')
    ax.set_ylabel('Densidad')
    ax.set_title('18a. Distribución Timing Filas por Fase', fontweight='bold')
    ax.legend(fontsize=8)

    # 18b. Timing filas vs Burden (debe ser proporcional)
    ax = axes[0, 1]
    for fase, color in [('F7R1', '#27AE60'), ('F9SE', '#E74C3C')]:
        df_f = df[df['Fase_categoria'] == fase][['Burden', 'tfilas_ms']].dropna()

        if len(df_f) > 30:
            ax.scatter(df_f['Burden'], df_f['tfilas_ms'],
                       alpha=0.3, s=20, label=fase, c=color)

    # Línea ENAEX: tfilas = 11.5 × Burden
    x_burden = np.linspace(df['Burden'].min(), df['Burden'].max(), 100)
    y_enaex = 11.5 * x_burden
    ax.plot(x_burden, y_enaex, 'b--', linewidth=2, label='ENAEX: 11.5×B')

    ax.set_xlabel('Burden (m)')
    ax.set_ylabel('Timing filas (ms)')
    ax.set_title('18b. Timing Filas vs Burden', fontweight='bold')
    ax.legend()

    # 18c. P100 vs Timing filas
    ax = axes[1, 0]
    for fase, color in [('F7R1', '#27AE60'), ('F9SE', '#E74C3C')]:
        df_f = df[df['Fase_categoria'] == fase][['tfilas_ms', 'P100']].dropna()

        if len(df_f) > 30:
            ax.scatter(df_f['tfilas_ms'], df_f['P100'],
                       alpha=0.3, s=20, label=fase, c=color)

            # Regresión
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                df_f['tfilas_ms'].astype(float), df_f['P100'].astype(float)
            )
            x_line = np.linspace(df_f['tfilas_ms'].min(), df_f['tfilas_ms'].max(), 100)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, color=color, linestyle='--', linewidth=2,
                    label=f'{fase}: R²={r_value**2:.2f}')

    ax.axhline(y=12, color='red', linestyle=':', linewidth=2, alpha=0.7)
    ax.set_xlabel('Timing entre filas (ms)')
    ax.set_ylabel('P100 (pulgadas)')
    ax.set_title('18c. P100 vs Timing Filas', fontweight='bold')
    ax.legend(fontsize=9)

    # 18d. Heatmap P100 por timing filas y fase
    ax = axes[1, 1]
    df['tfilas_cat'] = pd.cut(df['tfilas_ms'],
                              bins=[0, 50, 75, 100, 125, 150, 500],
                              labels=['<50', '50-75', '75-100', '100-125', '125-150', '>150'])

    pivot = df.pivot_table(values='P100', index='Fase_categoria',
                           columns='tfilas_cat', aggfunc='mean').round(1)
    counts = df.pivot_table(values='P100', index='Fase_categoria',
                            columns='tfilas_cat', aggfunc='count')
    mask = (counts.reindex_like(pivot) < 15) | pivot.isna()

    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn_r',
                mask=mask, ax=ax, vmin=8, vmax=25, linewidths=0.5,
                cbar_kws={'label': 'P100 (pulgadas)'})
    ax.set_title('18d. P100 por Fase × Timing Filas', fontweight='bold')
    ax.set_xlabel('Timing filas (ms)')
    ax.set_ylabel('Fase')

    plt.suptitle('ANÁLISIS DETALLADO TIMING ENTRE FILAS', fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/18_analisis_timing_filas.png', bbox_inches='tight')
    plt.close()
    print("  ✓ 18_analisis_timing_filas.png")

    # =========================================================================
    # GRÁFICO 19: Relación timing pozos vs timing filas
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # 19a. Scatter timing pozos vs filas
    ax = axes[0, 0]
    for fase, color in [('F7R1', '#27AE60'), ('F9SE', '#E74C3C')]:
        df_f = df[df['Fase_categoria'] == fase][['tpozos_ms', 'tfilas_ms', 'P100']].dropna()

        if len(df_f) > 30:
            scatter = ax.scatter(df_f['tpozos_ms'], df_f['tfilas_ms'],
                                 c=df_f['P100'], cmap='RdYlGn_r', alpha=0.6, s=30,
                                 vmin=8, vmax=25)
            ax.annotate(fase, xy=(df_f['tpozos_ms'].mean(), df_f['tfilas_ms'].mean()),
                        fontsize=12, fontweight='bold', color=color,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.colorbar(scatter, ax=ax, label='P100 (pulgadas)')
    ax.set_xlabel('Timing pozos (ms)')
    ax.set_ylabel('Timing filas (ms)')
    ax.set_title('19a. Timing Pozos vs Filas (color=P100)', fontweight='bold')

    # 19b. Ratio tfilas/tpozos vs P100
    ax = axes[0, 1]
    df['ratio_timing'] = df['tfilas_ms'] / df['tpozos_ms']
    df_ratio = df[df['ratio_timing'].notna() & (df['ratio_timing'] < 100)]

    for fase, color in [('F7R1', '#27AE60'), ('F9SE', '#E74C3C')]:
        df_f = df_ratio[df_ratio['Fase_categoria'] == fase]
        ax.scatter(df_f['ratio_timing'], df_f['P100'], alpha=0.3, s=20, label=fase, c=color)

    ax.axvline(x=10, color='blue', linestyle='--', linewidth=2, alpha=0.7, label='Ratio óptimo (~10)')
    ax.axhline(y=12, color='red', linestyle=':', linewidth=2, alpha=0.7)
    ax.set_xlabel('Ratio (t_filas / t_pozos)')
    ax.set_ylabel('P100 (pulgadas)')
    ax.set_title('19b. P100 vs Ratio Timing', fontweight='bold')
    ax.legend()

    # 19c. Heatmap timing pozos × timing filas
    ax = axes[1, 0]
    pivot = df.pivot_table(values='P100', index='tpozos_cat',
                           columns='tfilas_cat', aggfunc='mean').round(1)
    counts = df.pivot_table(values='P100', index='tpozos_cat',
                            columns='tfilas_cat', aggfunc='count')
    mask = (counts.reindex_like(pivot) < 10) | pivot.isna()

    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn_r',
                mask=mask, ax=ax, vmin=8, vmax=25, linewidths=0.5,
                cbar_kws={'label': 'P100 (pulgadas)'})
    ax.set_title('19c. P100: Timing Pozos × Timing Filas', fontweight='bold')
    ax.set_xlabel('Timing filas (ms)')
    ax.set_ylabel('Timing pozos (ms)')

    # 19d. Tabla resumen timing
    ax = axes[1, 1]
    ax.axis('off')

    resumen_timing = []
    for fase in ['F7R1', 'F9SE']:
        df_f = df[df['Fase_categoria'] == fase]
        resumen_timing.append({
            'Fase': fase,
            't_pozos real': f'{df_f["tpozos_ms"].mean():.1f}',
            't_pozos ENAEX': f'{df_f["tpozos_optimo"].mean():.1f}',
            'Desv. pozos': f'{df_f["desv_tpozos"].mean():+.1f}',
            't_filas real': f'{df_f["tfilas_ms"].mean():.1f}',
            't_filas ENAEX': f'{df_f["tfilas_optimo"].mean():.1f}',
            'Desv. filas': f'{df_f["desv_tfilas"].mean():+.1f}',
            'P100': f'{df_f["P100"].mean():.1f}"'
        })

    tabla_texto = f"""
    RESUMEN TIMING POR FASE
    ══════════════════════════════════════════════════════════════════════════

                          TIMING POZOS                     TIMING FILAS
    Fase     Real    ENAEX   Desv.       Real    ENAEX   Desv.       P100
    ─────────────────────────────────────────────────────────────────────────
    F7R1    {resumen_timing[0]['t_pozos real']:>5}   {resumen_timing[0]['t_pozos ENAEX']:>5}  {resumen_timing[0]['Desv. pozos']:>6}     {resumen_timing[0]['t_filas real']:>6}  {resumen_timing[0]['t_filas ENAEX']:>6}  {resumen_timing[0]['Desv. filas']:>6}    {resumen_timing[0]['P100']:>5}
    F9SE    {resumen_timing[1]['t_pozos real']:>5}   {resumen_timing[1]['t_pozos ENAEX']:>5}  {resumen_timing[1]['Desv. pozos']:>6}     {resumen_timing[1]['t_filas real']:>6}  {resumen_timing[1]['t_filas ENAEX']:>6}  {resumen_timing[1]['Desv. filas']:>6}    {resumen_timing[1]['P100']:>5}

    FÓRMULAS ENAEX (Manual Cap. 7):
    ─────────────────────────────────────────────────────────────────────────
    ► Timing pozos (Konya): th = Th × S
      - Th según UCS: 3.5-6.5 ms/m

    ► Timing filas (Richard Ash): td = 3.28×B + PC/VOD + x/vb
      - Alternativa simplificada: tfilas = 11.5 × Burden

    RECOMENDACIONES:
    ─────────────────────────────────────────────────────────────────────────
    ► Timing pozos muy bajo (<2ms) → Interacción de ondas, mala fragmentación
    ► Timing filas muy bajo → Confinamiento excesivo, bloques grandes
    ► Ratio óptimo tfilas/tpozos ≈ 8-12
    """

    ax.text(0.02, 0.98, tabla_texto, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.set_title('19d. Resumen Timing', fontweight='bold')

    plt.suptitle('RELACIÓN TIMING POZOS vs TIMING FILAS', fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/19_relacion_timing_pozos_filas.png', bbox_inches='tight')
    plt.close()
    print("  ✓ 19_relacion_timing_pozos_filas.png")

    # =========================================================================
    # GRÁFICO 20: Timing vs diseño de voladura
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # 20a. Timing pozos vs Espaciamiento (debe ser proporcional según Konya)
    ax = axes[0, 0]
    for fase, color in [('F7R1', '#27AE60'), ('F9SE', '#E74C3C')]:
        df_f = df[df['Fase_categoria'] == fase][['Espaciamiento', 'tpozos_ms']].dropna()

        if len(df_f) > 30:
            ax.scatter(df_f['Espaciamiento'], df_f['tpozos_ms'],
                       alpha=0.3, s=20, label=fase, c=color)

    # Línea ENAEX: tpozos = Th × S (usando Th promedio = 4.5)
    x_esp = np.linspace(df['Espaciamiento'].min(), df['Espaciamiento'].max(), 100)
    y_enaex = 4.5 * x_esp
    ax.plot(x_esp, y_enaex, 'b--', linewidth=2, label='ENAEX: 4.5×S')

    ax.set_xlabel('Espaciamiento (m)')
    ax.set_ylabel('Timing pozos (ms)')
    ax.set_title('20a. Timing Pozos vs Espaciamiento', fontweight='bold')
    ax.legend()

    # 20b. Heatmap timing pozos × Espaciamiento
    ax = axes[0, 1]
    df['esp_cat'] = pd.cut(df['Espaciamiento'],
                           bins=[0, 7, 8, 9, 10, 15],
                           labels=['<7', '7-8', '8-9', '9-10', '>10'])

    pivot = df.pivot_table(values='P100', index='tpozos_cat',
                           columns='esp_cat', aggfunc='mean').round(1)
    counts = df.pivot_table(values='P100', index='tpozos_cat',
                            columns='esp_cat', aggfunc='count')
    mask = (counts.reindex_like(pivot) < 10) | pivot.isna()

    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn_r',
                mask=mask, ax=ax, vmin=8, vmax=25, linewidths=0.5)
    ax.set_title('20b. P100: Timing Pozos × Espaciamiento', fontweight='bold')
    ax.set_xlabel('Espaciamiento (m)')
    ax.set_ylabel('Timing pozos (ms)')

    # 20c. Timing vs VOD (explosivos con mayor VOD necesitan timing diferente)
    ax = axes[1, 0]
    df_vod = df[df['VOD'].notna()].copy()
    df_vod['VOD_cat'] = pd.cut(df_vod['VOD'],
                               bins=[0, 5000, 5400, 5800, 7000],
                               labels=['<5000', '5000-5400', '5400-5800', '>5800'])

    pivot = df_vod.pivot_table(values='P100', index='tpozos_cat',
                               columns='VOD_cat', aggfunc='mean').round(1)
    counts = df_vod.pivot_table(values='P100', index='tpozos_cat',
                                columns='VOD_cat', aggfunc='count')
    mask = (counts.reindex_like(pivot) < 10) | pivot.isna()

    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn_r',
                mask=mask, ax=ax, vmin=8, vmax=25, linewidths=0.5)
    ax.set_title('20c. P100: Timing Pozos × VOD', fontweight='bold')
    ax.set_xlabel('VOD (m/s)')
    ax.set_ylabel('Timing pozos (ms)')

    # 20d. Timing vs UCS (roca más dura necesita timing diferente)
    ax = axes[1, 1]
    pivot = df.pivot_table(values='P100', index='tpozos_cat',
                           columns='UCS_categoria', aggfunc='mean').round(1)
    counts = df.pivot_table(values='P100', index='tpozos_cat',
                            columns='UCS_categoria', aggfunc='count')
    mask = (counts.reindex_like(pivot) < 10) | pivot.isna()

    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn_r',
                mask=mask, ax=ax, vmin=8, vmax=25, linewidths=0.5)
    ax.set_title('20d. P100: Timing Pozos × UCS', fontweight='bold')
    ax.set_xlabel('UCS (MPa)')
    ax.set_ylabel('Timing pozos (ms)')

    plt.suptitle('TIMING vs DISEÑO DE VOLADURA', fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/20_timing_vs_diseno.png', bbox_inches='tight')
    plt.close()
    print("  ✓ 20_timing_vs_diseno.png")


def graficar_evaluacion_enaex(df):
    """
    Genera gráficos de evaluación según teoría ENAEX (Gráfico 26).
    Compara parámetros reales vs óptimos teóricos.
    """
    print("\n" + "=" * 70)
    print("GENERANDO EVALUACIÓN ENAEX INTEGRAL")
    print("=" * 70)

    # =========================================================================
    # GRÁFICO 26: Evaluación integral ENAEX
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))

    # 26a. Burden real vs óptimo ENAEX
    ax = axes[0, 0]
    for fase, color in [('F7R1', '#27AE60'), ('F9SE', '#E74C3C')]:
        df_f = df[df['Fase_categoria'] == fase]
        if 'burden_optimo_enaex' in df.columns:
            ax.scatter(df_f['burden_optimo_enaex'], df_f['Burden'],
                       alpha=0.4, s=30, label=fase, c=color)

    if 'burden_optimo_enaex' in df.columns:
        max_val = max(df['burden_optimo_enaex'].max(), df['Burden'].max()) * 1.1
        ax.plot([0, max_val], [0, max_val], 'k--', linewidth=2, label='Óptimo (1:1)')
        ax.plot([0, max_val], [0, max_val * 0.85], 'g:', linewidth=1, alpha=0.7, label='±15%')
        ax.plot([0, max_val], [0, max_val * 1.15], 'g:', linewidth=1, alpha=0.7)
    ax.set_xlabel('Burden óptimo ENAEX (m)')
    ax.set_ylabel('Burden real (m)')
    ax.set_title('26a. Burden Real vs Óptimo ENAEX', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # 26b. Ratio S/B real vs óptimo
    ax = axes[0, 1]
    for fase, color in [('F7R1', '#27AE60'), ('F9SE', '#E74C3C')]:
        df_f = df[df['Fase_categoria'] == fase]
        if 'SB_ratio_optimo' in df.columns and 'SB_ratio_real' in df.columns:
            ax.scatter(df_f['SB_ratio_optimo'], df_f['SB_ratio_real'],
                       alpha=0.4, s=30, label=fase, c=color)

    ax.axhline(y=1.25, color='blue', linestyle='--', linewidth=2, alpha=0.7, label='S/B = 1.25')
    ax.axvspan(1.15, 1.40, alpha=0.2, color='green', label='Rango óptimo')
    ax.set_xlabel('S/B óptimo según UCS')
    ax.set_ylabel('S/B real')
    ax.set_title('26b. Ratio S/B Real vs Óptimo', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # 26c. Factor de carga real vs óptimo
    ax = axes[0, 2]
    if 'fc_optimo_enaex' in df.columns and 'Fc' in df.columns:
        for fase, color in [('F7R1', '#27AE60'), ('F9SE', '#E74C3C')]:
            df_f = df[df['Fase_categoria'] == fase]
            ax.scatter(df_f['fc_optimo_enaex'], df_f['Fc'].fillna(0.5),
                       alpha=0.4, s=30, label=fase, c=color)

        max_val = max(df['fc_optimo_enaex'].max(), df['Fc'].fillna(0.5).max()) * 1.1
        ax.plot([0, max_val], [0, max_val], 'k--', linewidth=2, label='Óptimo (1:1)')
    ax.set_xlabel('FC óptimo ENAEX (kg/m³)')
    ax.set_ylabel('FC real (kg/m³)')
    ax.set_title('26c. Factor Carga Real vs Óptimo', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # 26d. Evaluación global de diseño
    ax = axes[1, 0]
    if 'evaluacion_diseno_enaex' in df.columns:
        eval_counts = df.groupby(['Fase_categoria', 'evaluacion_diseno_enaex']).size().unstack(fill_value=0)
        eval_counts = eval_counts.reindex(columns=['Óptimo', 'Aceptable', 'Mejorable', 'Deficiente'], fill_value=0)

        eval_counts.plot(kind='bar', ax=ax, color=['#27AE60', '#F1C40F', '#E67E22', '#E74C3C'],
                         width=0.7, edgecolor='black')
        ax.set_xlabel('Fase')
        ax.set_ylabel('Cantidad de registros')
        ax.set_title('26d. Evaluación Global de Diseño ENAEX', fontweight='bold')
        ax.legend(title='Evaluación')
        ax.tick_params(axis='x', rotation=0)

    # 26e. Explosivo recomendado vs usado
    ax = axes[1, 1]
    ax.axis('off')

    # Tabla de recomendaciones por fase
    tabla_texto = "RECOMENDACIONES ENAEX POR FASE\n" + "="*55 + "\n\n"

    for fase in ['F7R1', 'F9SE']:
        df_f = df[df['Fase_categoria'] == fase]
        tabla_texto += f"  {fase}:\n"
        tabla_texto += f"    UCS promedio: {df_f['UCS_MPA'].mean():.0f} MPa\n"

        if 'explosivo_recomendado' in df.columns:
            rec = df_f['explosivo_recomendado'].mode()
            rec_str = rec.iloc[0] if len(rec) > 0 else 'N/A'
            tabla_texto += f"    Explosivo recomendado: {rec_str}\n"

        usado = df_f['Tipo_Explosivo'].mode()
        usado_str = usado.iloc[0] if len(usado) > 0 else 'N/A'
        tabla_texto += f"    Explosivo usado: {usado_str}\n"

        if 'burden_optimo_enaex' in df.columns:
            tabla_texto += f"    Burden óptimo: {df_f['burden_optimo_enaex'].mean():.2f} m\n"
            tabla_texto += f"    Burden real: {df_f['Burden'].mean():.2f} m\n"

        if 'fc_optimo_enaex' in df.columns:
            tabla_texto += f"    FC óptimo: {df_f['fc_optimo_enaex'].mean():.2f} kg/m³\n"
            tabla_texto += f"    FC real: {df_f['Fc'].fillna(0.5).mean():.2f} kg/m³\n"

        tabla_texto += "\n"

    # Agregar fórmulas ENAEX
    tabla_texto += "-"*55 + "\n"
    tabla_texto += "FÓRMULAS ENAEX APLICADAS:\n"
    tabla_texto += "  • B = Kb×De×(ρe/ρr)^0.33×(VOD/4000)^0.5\n"
    tabla_texto += "  • S = Ks × B (Ks: 1.15-1.40)\n"
    tabla_texto += "  • T = 0.7-1.0 × B\n"
    tabla_texto += "  • th = Th × S (Konya)\n"
    tabla_texto += "  • tf = 11.5 × B\n"

    ax.text(0.05, 0.95, tabla_texto, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.set_title('26e. Resumen Recomendaciones ENAEX', fontweight='bold')

    # 26f. Desviaciones de parámetros clave
    ax = axes[1, 2]

    parametros = []
    for fase in ['F7R1', 'F9SE']:
        df_f = df[df['Fase_categoria'] == fase]

        if 'desv_burden' in df.columns:
            parametros.append({'Fase': fase, 'Parámetro': 'Burden', 'Desviación': df_f['desv_burden'].mean()})
        if 'desv_espaciamiento' in df.columns:
            parametros.append({'Fase': fase, 'Parámetro': 'Espaciamiento', 'Desviación': df_f['desv_espaciamiento'].mean()})
        if 'desv_taco' in df.columns:
            parametros.append({'Fase': fase, 'Parámetro': 'Taco', 'Desviación': df_f['desv_taco'].mean()})
        if 'desv_fc' in df.columns:
            parametros.append({'Fase': fase, 'Parámetro': 'Factor Carga', 'Desviación': df_f['desv_fc'].mean()})
        if 'desv_tpozos' in df.columns:
            parametros.append({'Fase': fase, 'Parámetro': 'Timing pozos', 'Desviación': df_f['desv_tpozos'].mean() / 10})  # Escalar

    if parametros:
        df_params = pd.DataFrame(parametros)
        pivot = df_params.pivot(index='Parámetro', columns='Fase', values='Desviación')

        x = np.arange(len(pivot.index))
        width = 0.35

        bars1 = ax.bar(x - width/2, pivot['F7R1'], width, label='F7R1', color='#27AE60', edgecolor='black')
        bars2 = ax.bar(x + width/2, pivot['F9SE'], width, label='F9SE', color='#E74C3C', edgecolor='black')

        ax.axhline(y=0, color='black', linewidth=1)
        ax.set_xlabel('Parámetro')
        ax.set_ylabel('Desviación vs óptimo ENAEX')
        ax.set_title('26f. Desviación de Parámetros vs ENAEX', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=45, ha='right')
        ax.legend()
        ax.grid(alpha=0.3, axis='y')

    plt.suptitle('EVALUACIÓN INTEGRAL SEGÚN TEORÍA ENAEX', fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/26_evaluacion_enaex.png', bbox_inches='tight')
    plt.close()
    print("  ✓ 26_evaluacion_enaex.png")

    # =========================================================================
    # GRÁFICO 27: Selección de explosivo según ENAEX
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # 27a. Recomendación de explosivo por UCS
    ax = axes[0, 0]
    ucs_ranges = [
        (0, 50, 'ANFO / Blendex 920-930', '#3498DB'),
        (50, 80, 'Blendex 940-950 / Vertex ALR', '#2ECC71'),
        (80, 120, 'Emultex BN / Energex 50', '#F1C40F'),
        (120, 180, 'Energex 70 / Pirex S', '#E67E22'),
        (180, 300, 'Pirex S Plus / Energex 70+', '#E74C3C'),
    ]

    for ucs_min, ucs_max, exp, color in ucs_ranges:
        ax.barh(exp, ucs_max - ucs_min, left=ucs_min, color=color, edgecolor='black', alpha=0.8)

    # Marcar UCS de cada fase
    for fase, marker in [('F7R1', 'o'), ('F9SE', 's')]:
        ucs_mean = df[df['Fase_categoria'] == fase]['UCS_MPA'].mean()
        ax.axvline(x=ucs_mean, color='black' if fase == 'F7R1' else 'red',
                   linestyle='--', linewidth=2, label=f'{fase}: {ucs_mean:.0f} MPa')

    ax.set_xlabel('UCS (MPa)')
    ax.set_ylabel('Explosivo recomendado')
    ax.set_title('27a. Selección de Explosivo según UCS (ENAEX)', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3, axis='x')

    # 27b. Propiedades recomendadas por dureza
    ax = axes[0, 1]
    dureza_props = {
        'Blanda\n(<50 MPa)': {'Densidad': 0.9, 'VOD': 3500, 'RWS': 95, 'FC': 0.35},
        'Media\n(50-100)': {'Densidad': 1.1, 'VOD': 4000, 'RWS': 90, 'FC': 0.50},
        'Dura\n(100-150)': {'Densidad': 1.25, 'VOD': 4500, 'RWS': 85, 'FC': 0.75},
        'Muy dura\n(>150)': {'Densidad': 1.30, 'VOD': 5500, 'RWS': 82, 'FC': 1.00},
    }

    x = np.arange(len(dureza_props))
    width = 0.2

    ax2 = ax.twinx()
    bars1 = ax.bar(x - width*1.5, [v['Densidad'] for v in dureza_props.values()], width,
                   label='Densidad (g/cc)', color='#3498DB')
    bars2 = ax.bar(x - width*0.5, [v['FC'] for v in dureza_props.values()], width,
                   label='FC (kg/m³)', color='#E74C3C')
    bars3 = ax2.bar(x + width*0.5, [v['VOD']/1000 for v in dureza_props.values()], width,
                    label='VOD (km/s)', color='#2ECC71')
    bars4 = ax2.bar(x + width*1.5, [v['RWS']/100 for v in dureza_props.values()], width,
                    label='RWS/100', color='#F1C40F')

    ax.set_ylabel('Densidad (g/cc) / FC (kg/m³)')
    ax2.set_ylabel('VOD (km/s) / RWS/100')
    ax.set_xticks(x)
    ax.set_xticklabels(dureza_props.keys())
    ax.set_title('27b. Propiedades Recomendadas por Dureza', fontweight='bold')

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)

    # 27c. Energía entregada vs requerida
    ax = axes[1, 0]
    if 'energia_entregada_MJ' in df.columns and 'energia_requerida_MJ' in df.columns:
        for fase, color in [('F7R1', '#27AE60'), ('F9SE', '#E74C3C')]:
            df_f = df[df['Fase_categoria'] == fase]
            ax.scatter(df_f['energia_requerida_MJ'], df_f['energia_entregada_MJ'],
                       alpha=0.4, s=30, label=fase, c=color)

        max_e = max(df['energia_requerida_MJ'].max(), df['energia_entregada_MJ'].max()) * 1.1
        ax.plot([0, max_e], [0, max_e], 'k--', linewidth=2, label='Equilibrio (1:1)')
        ax.fill_between([0, max_e], [0, max_e], [0, max_e*1.2], alpha=0.2, color='green', label='Exceso energía')
        ax.fill_between([0, max_e], [0, max_e*0.8], [0, max_e], alpha=0.2, color='red', label='Déficit energía')

    ax.set_xlabel('Energía requerida (MJ/m³)')
    ax.set_ylabel('Energía entregada (MJ/m³)')
    ax.set_title('27c. Balance de Energía', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # 27d. Resumen de cumplimiento ENAEX
    ax = axes[1, 1]
    ax.axis('off')

    cumplimiento = []
    for fase in ['F7R1', 'F9SE']:
        df_f = df[df['Fase_categoria'] == fase]
        n_total = len(df_f)

        cum = {'Fase': fase}

        # Burden ±15%
        if 'burden_ratio' in df.columns:
            n_ok = len(df_f[(df_f['burden_ratio'] >= 0.85) & (df_f['burden_ratio'] <= 1.15)])
            cum['Burden ±15%'] = f'{100*n_ok/n_total:.0f}%' if n_total > 0 else 'N/A'

        # S/B correcto
        if 'SB_ratio_real' in df.columns and 'SB_ratio_optimo' in df.columns:
            n_ok = len(df_f[(df_f['SB_ratio_real'] >= df_f['SB_ratio_optimo']*0.9) &
                           (df_f['SB_ratio_real'] <= df_f['SB_ratio_optimo']*1.1)])
            cum['S/B ±10%'] = f'{100*n_ok/n_total:.0f}%' if n_total > 0 else 'N/A'

        # Taco óptimo
        if 'taco_burden_ratio' in df.columns:
            n_ok = len(df_f[(df_f['taco_burden_ratio'] >= 0.7) & (df_f['taco_burden_ratio'] <= 1.0)])
            cum['Taco T/B'] = f'{100*n_ok/n_total:.0f}%' if n_total > 0 else 'N/A'

        # Timing óptimo
        if 'timing_eval' in df.columns:
            n_ok = len(df_f[df_f['timing_eval'] == 'Óptimo'])
            cum['Timing'] = f'{100*n_ok/n_total:.0f}%' if n_total > 0 else 'N/A'

        cumplimiento.append(cum)

    tabla_texto = "CUMPLIMIENTO DE PARÁMETROS ENAEX\n" + "="*50 + "\n\n"
    tabla_texto += f"{'Parámetro':<20} {'F7R1':>10} {'F9SE':>10}\n"
    tabla_texto += "-"*50 + "\n"

    if len(cumplimiento) >= 2:
        for key in ['Burden ±15%', 'S/B ±10%', 'Taco T/B', 'Timing']:
            if key in cumplimiento[0]:
                tabla_texto += f"{key:<20} {cumplimiento[0].get(key, 'N/A'):>10} {cumplimiento[1].get(key, 'N/A'):>10}\n"

    tabla_texto += "\n" + "="*50 + "\n"
    tabla_texto += "\nINTERPRETACIÓN:\n"
    tabla_texto += "  >80%: Excelente cumplimiento\n"
    tabla_texto += "  60-80%: Cumplimiento aceptable\n"
    tabla_texto += "  <60%: Requiere optimización\n"

    ax.text(0.1, 0.9, tabla_texto, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    ax.set_title('27d. Cumplimiento Parámetros ENAEX', fontweight='bold')

    plt.suptitle('SELECCIÓN DE EXPLOSIVO Y CUMPLIMIENTO ENAEX', fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/27_seleccion_explosivo_enaex.png', bbox_inches='tight')
    plt.close()
    print("  ✓ 27_seleccion_explosivo_enaex.png")

    # =========================================================================
    # GRÁFICO 28: Evaluación de TIMING según ENAEX
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))

    # 28a. Timing pozos: Real vs Óptimo ENAEX
    ax = axes[0, 0]
    for fase, color in [('F7R1', '#27AE60'), ('F9SE', '#E74C3C')]:
        df_f = df[df['Fase_categoria'] == fase]
        if 'tpozos_optimo' in df.columns and 'tpozos_ms' in df.columns:
            ax.scatter(df_f['tpozos_optimo'], df_f['tpozos_ms'],
                       alpha=0.4, s=30, label=fase, c=color)

    if 'tpozos_optimo' in df.columns:
        max_val = max(df['tpozos_optimo'].max(), df['tpozos_ms'].max()) * 1.1
        ax.plot([0, max_val], [0, max_val], 'k--', linewidth=2, label='Óptimo (1:1)')
        ax.fill_between([0, max_val], [0, max_val*0.8], [0, max_val*1.2],
                        alpha=0.2, color='green', label='Rango ±20%')
    ax.set_xlabel('Timing pozos óptimo ENAEX (ms)')
    ax.set_ylabel('Timing pozos real (ms)')
    ax.set_title('28a. Timing Pozos: Real vs Óptimo', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # 28b. Timing filas: Real vs Óptimo ENAEX
    ax = axes[0, 1]
    for fase, color in [('F7R1', '#27AE60'), ('F9SE', '#E74C3C')]:
        df_f = df[df['Fase_categoria'] == fase]
        if 'tfilas_optimo' in df.columns and 'tfilas_ms' in df.columns:
            ax.scatter(df_f['tfilas_optimo'], df_f['tfilas_ms'],
                       alpha=0.4, s=30, label=fase, c=color)

    if 'tfilas_optimo' in df.columns:
        max_val = max(df['tfilas_optimo'].max(), df['tfilas_ms'].max()) * 1.1
        ax.plot([0, max_val], [0, max_val], 'k--', linewidth=2, label='Óptimo (1:1)')
        ax.fill_between([0, max_val], [0, max_val*0.8], [0, max_val*1.2],
                        alpha=0.2, color='green', label='Rango ±20%')
    ax.set_xlabel('Timing filas óptimo ENAEX (ms)')
    ax.set_ylabel('Timing filas real (ms)')
    ax.set_title('28b. Timing Filas: Real vs Óptimo', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # 28c. Desviación de timing por fase
    ax = axes[0, 2]
    timing_data = []
    for fase in ['F7R1', 'F9SE']:
        df_f = df[df['Fase_categoria'] == fase]
        timing_data.append({
            'Fase': fase,
            'Desv. t_pozos (ms)': df_f['desv_tpozos'].mean(),
            'Desv. t_filas (ms)': df_f['desv_tfilas'].mean(),
        })

    x = np.arange(2)
    width = 0.35
    colors_pozos = ['#27AE60', '#E74C3C']
    colors_filas = ['#2ECC71', '#F39C12']

    bars1 = ax.bar(x - width/2, [t['Desv. t_pozos (ms)'] for t in timing_data], width,
                   label='Desv. timing pozos', color=colors_pozos, edgecolor='black')
    bars2 = ax.bar(x + width/2, [t['Desv. t_filas (ms)'] for t in timing_data], width,
                   label='Desv. timing filas', color=colors_filas, edgecolor='black')

    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_ylabel('Desviación vs óptimo ENAEX (ms)')
    ax.set_xticks(x)
    ax.set_xticklabels(['F7R1', 'F9SE'])
    ax.set_title('28c. Desviación de Timing vs ENAEX', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    # Añadir valores en las barras
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -10),
                    textcoords="offset points",
                    ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -10),
                    textcoords="offset points",
                    ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)

    # 28d. Clasificación de timing (distribución)
    ax = axes[1, 0]
    if 'timing_eval' in df.columns:
        eval_order = ['Muy bajo', 'Bajo', 'Óptimo', 'Alto', 'Muy alto']
        eval_colors = ['#E74C3C', '#F39C12', '#27AE60', '#3498DB', '#9B59B6']

        eval_counts = df.groupby(['Fase_categoria', 'timing_eval']).size().unstack(fill_value=0)
        eval_counts = eval_counts.reindex(columns=eval_order, fill_value=0)

        eval_counts.plot(kind='bar', ax=ax, color=eval_colors, width=0.7, edgecolor='black')
        ax.set_xlabel('Fase')
        ax.set_ylabel('Cantidad de registros')
        ax.set_title('28d. Clasificación de Timing ENAEX', fontweight='bold')
        ax.legend(title='Evaluación', fontsize=9)
        ax.tick_params(axis='x', rotation=0)

    # 28e. Fórmulas y recomendaciones de timing
    ax = axes[1, 1]
    ax.axis('off')

    # Calcular métricas de timing
    tabla_texto = "EVALUACIÓN DE TIMING SEGÚN ENAEX\n" + "="*55 + "\n\n"
    tabla_texto += "FÓRMULAS APLICADAS:\n"
    tabla_texto += "-"*55 + "\n"
    tabla_texto += "  TIMING ENTRE POZOS (Konya):\n"
    tabla_texto += "    th = Th × S\n"
    tabla_texto += "    Donde Th (ms/m) según UCS:\n"
    tabla_texto += "      • UCS < 50 MPa:  Th = 6.5 (roca blanda)\n"
    tabla_texto += "      • UCS 50-80:     Th = 5.5 (calizas)\n"
    tabla_texto += "      • UCS 80-120:    Th = 4.5 (granitos)\n"
    tabla_texto += "      • UCS > 120:     Th = 3.5 (muy dura)\n\n"
    tabla_texto += "  TIMING ENTRE FILAS:\n"
    tabla_texto += "    tf = 11.5 × B (ms)\n"
    tabla_texto += "    Donde B = Burden (m)\n\n"
    tabla_texto += "-"*55 + "\n"
    tabla_texto += "RESULTADOS POR FASE:\n"
    tabla_texto += "-"*55 + "\n"

    for fase in ['F7R1', 'F9SE']:
        df_f = df[df['Fase_categoria'] == fase]
        tabla_texto += f"\n  {fase}:\n"
        tabla_texto += f"    T. pozos real: {df_f['tpozos_ms'].mean():.1f} ms\n"
        tabla_texto += f"    T. pozos ENAEX: {df_f['tpozos_optimo'].mean():.1f} ms\n"
        tabla_texto += f"    Desviación: {df_f['desv_tpozos'].mean():+.1f} ms\n"
        tabla_texto += f"    T. filas real: {df_f['tfilas_ms'].mean():.1f} ms\n"
        tabla_texto += f"    T. filas ENAEX: {df_f['tfilas_optimo'].mean():.1f} ms\n"
        tabla_texto += f"    Desviación: {df_f['desv_tfilas'].mean():+.1f} ms\n"

    ax.text(0.05, 0.95, tabla_texto, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.set_title('28e. Fórmulas y Resultados de Timing', fontweight='bold')

    # 28f. Cumplimiento de timing y recomendaciones
    ax = axes[1, 2]
    ax.axis('off')

    tabla_texto = "CUMPLIMIENTO DE TIMING ENAEX\n" + "="*50 + "\n\n"

    for fase in ['F7R1', 'F9SE']:
        df_f = df[df['Fase_categoria'] == fase]
        n_total = len(df_f)

        # Timing pozos en rango ±20%
        if 'tpozos_ms' in df.columns and 'tpozos_optimo' in df.columns:
            ratio_tp = df_f['tpozos_ms'] / df_f['tpozos_optimo']
            n_ok_tp = len(ratio_tp[(ratio_tp >= 0.8) & (ratio_tp <= 1.2)])
            pct_tp = 100 * n_ok_tp / n_total if n_total > 0 else 0

        # Timing filas en rango ±20%
        if 'tfilas_ms' in df.columns and 'tfilas_optimo' in df.columns:
            ratio_tf = df_f['tfilas_ms'] / df_f['tfilas_optimo']
            n_ok_tf = len(ratio_tf[(ratio_tf >= 0.8) & (ratio_tf <= 1.2)])
            pct_tf = 100 * n_ok_tf / n_total if n_total > 0 else 0

        # Timing óptimo según clasificación
        if 'timing_eval' in df.columns:
            n_optimo = len(df_f[df_f['timing_eval'] == 'Óptimo'])
            pct_optimo = 100 * n_optimo / n_total if n_total > 0 else 0

        tabla_texto += f"  {fase}:\n"
        tabla_texto += f"    T. pozos ±20%: {pct_tp:.0f}%\n"
        tabla_texto += f"    T. filas ±20%: {pct_tf:.0f}%\n"
        tabla_texto += f"    Clasificación 'Óptimo': {pct_optimo:.0f}%\n\n"

    tabla_texto += "="*50 + "\n"
    tabla_texto += "RECOMENDACIONES:\n"
    tabla_texto += "-"*50 + "\n"

    # Determinar recomendaciones basadas en los datos
    f7_desv_tp = df[df['Fase_categoria'] == 'F7R1']['desv_tpozos'].mean()
    f9_desv_tp = df[df['Fase_categoria'] == 'F9SE']['desv_tpozos'].mean()
    f7_desv_tf = df[df['Fase_categoria'] == 'F7R1']['desv_tfilas'].mean()
    f9_desv_tf = df[df['Fase_categoria'] == 'F9SE']['desv_tfilas'].mean()

    if f9_desv_tp < -10:
        tabla_texto += "  • F9SE: Aumentar timing pozos\n"
        tabla_texto += f"    (actual muy bajo: {f9_desv_tp:+.0f} ms)\n"
    elif f9_desv_tp > 10:
        tabla_texto += "  • F9SE: Reducir timing pozos\n"

    if f9_desv_tf < -20:
        tabla_texto += "  • F9SE: Aumentar timing filas\n"
        tabla_texto += f"    (actual muy bajo: {f9_desv_tf:+.0f} ms)\n"

    if f7_desv_tp < -10:
        tabla_texto += "  • F7R1: Aumentar timing pozos\n"

    tabla_texto += "\n  Timing óptimo mejora:\n"
    tabla_texto += "  - Fragmentación uniforme\n"
    tabla_texto += "  - Menor sobrequiebre\n"
    tabla_texto += "  - Control de vibraciones\n"

    ax.text(0.05, 0.95, tabla_texto, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    ax.set_title('28f. Cumplimiento y Recomendaciones', fontweight='bold')

    plt.suptitle('EVALUACIÓN DE TIMING SEGÚN TEORÍA ENAEX', fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/28_evaluacion_timing_enaex.png', bbox_inches='tight')
    plt.close()
    print("  ✓ 28_evaluacion_timing_enaex.png")


# ==============================================================================
# FUNCIÓN PRINCIPAL
# ==============================================================================

def main():
    """Ejecuta el análisis completo."""
    print("=" * 70)
    print("ANÁLISIS DETALLADO FASES F7R1 vs F9SE")
    print("=" * 70)

    # Cargar y preparar datos
    df_raw = cargar_datos()
    df = preparar_datos(df_raw)

    # Cálculos ENAEX
    df = calcular_timing_enaex(df)
    df = calcular_taco_enaex(df)
    df = calcular_parametros_enaex(df)

    # Generar gráficos
    graficar_explosivos_por_fase(df)
    graficar_taco_por_fase(df)
    analisis_sensibilidad(df)
    analisis_dureza_roca(df)
    analisis_timing_detallado(df)
    analisis_sensibilidad_por_factor(df)
    graficar_evaluacion_enaex(df)
    generar_resumen(df)

    # Guardar datos procesados
    df.to_csv(f'{OUTPUT_DIR}/datos_f7_f9_procesados.csv', index=False)

    print("\n" + "=" * 70)
    print("ANÁLISIS COMPLETADO!")
    print("=" * 70)
    print(f"\nArchivos generados en: {OUTPUT_DIR}/")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith('.png'):
            print(f"  ✓ {f}")
    print(f"\n  ✓ datos_f7_f9_procesados.csv")

    return df


if __name__ == "__main__":
    df_analizado = main()
