import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import find_peaks

# ==========================================
# CARGA Y LIMPIEZA DE DATOS
# ==========================================
df = pd.read_csv('FH 2.csv', sep=';', decimal=',', skiprows=[1])

# Aseguramos que los datos sean numéricos (corrigiendo posibles comas residuales)
df['Tiempo'] = pd.to_numeric(df['Tiempo'].astype(str).str.replace(',', '.'), errors='coerce')
df['Canal A'] = pd.to_numeric(df['Canal A'].astype(str).str.replace(',', '.'), errors='coerce')
df['Canal B'] = pd.to_numeric(df['Canal B'].astype(str).str.replace(',', '.'), errors='coerce')
df.dropna(inplace=True)

# ==========================================
# AISLAMIENTO DEL PRIMER CICLO (RAMPA)
# ==========================================
# Buscamos dónde ocurre la caída de voltaje para aislar solo el primer barrido
UA_temp = df['Canal A'] * 10
indice_corte = UA_temp.iloc[:4800].idxmax()
df_calc = df.iloc[:indice_corte].copy()

# Reescalado físico: UA (0-60V), IA (0-50mA)
df_calc['UA_real'] = df_calc['Canal A'] * 10
df_calc['IA_real'] = df_calc['Canal B'] * 5

# ==========================================
# FILTRADO MEDIANTE TRANSFORMADA DE FOURIER (FFT)
# ==========================================
# Calculamos el intervalo de tiempo (dt) en segundos
dt_ms = abs(df_calc['Tiempo'].iloc[1] - df_calc['Tiempo'].iloc[0])
dt_s = dt_ms /400.0
N = len(df_calc)

# Transformada de Fourier de la corriente
yf = fft(df_calc['IA_real'].values)
xf = fftfreq(N, dt_s)

# --- Filtro Paso Bajo en el Dominio de las Frecuencias ---
frecuencia_corte = 150  # Hz (Ajustable: menor valor = más suavizado)

# Hacemos 0 las amplitudes de las frecuencias que superen el corte (ruido)
yf_filtrado = yf.copy()
yf_filtrado[np.abs(xf) > frecuencia_corte] = 0

# Transformada Inversa (IFFT) para volver al dominio original (tiempo/voltaje)
df_calc['IA_filtrado'] = np.real(ifft(yf_filtrado))

# ==========================================
# DETECCIÓN DE MÍNIMOS Y CÁLCULO DE ΔU
# ==========================================
# Ajustes clave:
# - prominence: Lo bajamos a 0.01 (o 0.005) para detectar las caídas muy suaves.
# - distance: Lo ajustamos a 50 puntos para asegurar que quepan todos los valles.

valles_idx, _ = find_peaks(-df_calc['IA_filtrado'], distance=25, prominence=0.01)
voltajes_valles = df_calc['UA_real'].iloc[valles_idx].values

# Calculamos las distancias entre valles consecutivos
distancias = np.diff(voltajes_valles)

# Filtro de seguridad (saltos entre 3V y 7V)
distancias_reales = distancias[(distancias > 3) & (distancias < 7)]

if len(distancias_reales) > 0:
    media_distancia = np.mean(distancias_reales)
    error_estandar = np.std(distancias_reales, ddof=1) / np.sqrt(len(distancias_reales))
else:
    media_distancia, error_estandar = 0, 0


# ==========================================
# GRÁFICA DE RESULTADOS
# ==========================================
plt.figure(figsize=(12, 7))

# Curva original para comparar
plt.plot(df_calc['UA_real'], df_calc['IA_real'], color='lightblue', alpha=0.6, label='Datos Originales (Ruido)')

# Curva filtrada por FFT
plt.plot(df_calc['UA_real'], df_calc['IA_filtrado'], color='darkblue', linewidth=2.5, label='Filtro FFT')

# Marcar los valles detectados
plt.plot(df_calc['UA_real'].iloc[valles_idx], df_calc['IA_filtrado'].iloc[valles_idx],
         "rv", markersize=10, label='Mínimos Detectados')

# Líneas verticales y texto para cada voltaje
for v in voltajes_valles:
    plt.axvline(x=v, color='gray', linestyle='--', alpha=0.4)
    plt.text(v, plt.ylim()[0], f'{v:.2f}V', ha='center', va='bottom', color='red', fontsize=10, fontweight='bold')

plt.title(f'Franck-Hertz mediante FFT | $\Delta U$ = {media_distancia:.2f} ± {error_estandar:.2f} V', fontsize=15)
plt.xlabel('Voltaje de Aceleración $U_A$ (V)', fontsize=14)
plt.ylabel('Corriente $I_A$ (mA)', fontsize=14)
plt.legend(loc='upper left', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Grafico FFT.pdf')
plt.show()


plt.figure(figsize=(12, 7))

# ==========================================
# REPORTE EN CONSOLA
# ==========================================
print("-" * 40)
print("RESULTADOS DEL ANÁLISIS DE FRANCK-HERTZ")
print("-" * 40)
print(f"Número de valles detectados: {len(voltajes_valles)}")
print(f"Posiciones de los valles (V): {np.round(voltajes_valles, 2)}")
print(f"Distancias detectadas ΔU (V): {np.round(distancias_reales, 2)}")
print(f"\nPOTENCIAL DE EXCITACIÓN (Media): {media_distancia:.3f} V")
print(f"Incertidumbre (Error Estándar): ± {error_estandar:.3f} V")
print("-" * 40)

