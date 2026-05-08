import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import pandas as pd
from scipy.io import loadmat
from scipy.signal import savgol_filter, find_peaks
from sympy.printing.tree import print_node
from acoplados import matacifras

plt.rcParams.update({
    'font.size': 12,           # Tamaño base para todo el texto
    'axes.titlesize': 16,      # Títulos
    'axes.labelsize': 14,      # Labels de los ejes
    'xtick.labelsize': 12,     # Ticks eje X
    'ytick.labelsize': 12,     # Ticks eje Y
    'legend.fontsize': 14,     # Leyenda
    'figure.titlesize': 20     # Título de la figura 
})

#FUNCIÓN QUE GRAFICA UNA OSCILACIÓN Y SACA SU ESPECTRO DE FRECUENCIAS MEDIANTE FFT

def analizar_fourier(archivo_csv, fila_inicio, fila_fin, ventana=51):

    #Carga de datos
    df = pd.read_csv(archivo_csv, sep=';', decimal=',', skiprows=3, names=['Tiempo', 'Voltaje'])
    segmento = df.iloc[fila_inicio:fila_fin].copy()
    
    t = segmento['Tiempo'].values
    V = segmento['Voltaje'].values
    
    #Suavizado solo para la gráfica (Savitzky-Golay)
    V_g = savgol_filter(V, ventana, 3)
    
    #Centrado de señal original para FFT
    V_fft = V-np.mean(V)
    
    #Cálculo de la FFT sobre datos originales
    n = len(t)
    dt = np.mean(np.diff(t))
    fft_valores = np.fft.rfft(V_fft)
    f = np.fft.rfftfreq(n, d=dt)
    amplitud = np.abs(fft_valores)
    error = 1/(2*t[fila_fin-3])
    
    #Encontrar frecuencia dominante
    f_dom = f[np.argmax(amplitud)]
    T = 1/f_dom
    
    #VISUALIZACIÓN
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    #Gráfica del péndulo
    #ax1.plot(t, V, color='silver', alpha=0.4, label='Datos originales (escalonados)')
    ax1.plot(t, V_g, color='tab:blue', linewidth=1.5, label='Señal suavizada')
    ax1.set_title('Oscilación del péndulo en el tiempo (suavizada)')
    ax1.set_xlabel('Tiempo (s)')
    ax1.set_ylabel('Voltaje (V)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    #Espectro fft
    ax2.plot(f, amplitud, color='tab:red', linewidth=2, label=f'FFT: Frecuencia dominante = ({matacifras(f_dom, error)[0]} $\pm$ ${matacifras(f_dom, error)[1]}$) $Hz$')
    ax2.axvline(f_dom, color='black', linestyle='--', alpha=0.6)
    ax2.set_title(f'Espectro de frecuencias (calculado sobre datos originales)')
    ax2.set_xlabel('Frecuencia (Hz)')
    ax2.set_ylabel('Amplitud')
    ax2.set_xlim(0, 4) 
    ax2.grid(True, alpha=0.3)
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{archivo_csv.replace('.csv','')}.pdf')
    plt.show()
    
    print(f"Análisis completado para {archivo_csv}:")
    print(f"Frecuencia principal: {f_dom:.5f} Hz")
    print(f"Periodo (T): {T:.5f} s")


analizar_fourier('frec carac izq.csv', 0, 10888, ventana=61)
analizar_fourier('frec carac right.csv', 0, 10888, ventana=61)


#FUNCIÓN QUE GRAFICA DOS OSCILACIONES ACOPLADAS Y SACA SU ESPECTRO DE FRECUENCIAS MEDIANTE FFT

def analizar_fourier_2(archivo_csv, fila_inicio, fila_fin, ventana_suave=51):

    #Carga de datos
    df = pd.read_csv(archivo_csv, sep=';', decimal=',', skiprows=3, 
                        names=['Tiempo', 'V1', 'V2'])
    segmento = df.iloc[fila_inicio:fila_fin].copy()
    
    t = segmento['Tiempo'].values
    dt = np.mean(np.diff(t))
    n = len(t)
 
    señales = {
        'Péndulo izquierdo': {'raw': segmento['V1'].values, 'color': 'tab:blue'},
        'Péndulo derecho': {'raw': segmento['V2'].values, 'color': 'tab:orange'}
    }
    
    #Configuración de la gráfica
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    for nombre, data in señales.items():

        #Gráfica suavizada
        y_visual = savgol_filter(data['raw'], ventana_suave, 3)
        ax1.plot(t, y_visual, label=f'{nombre}', color=data['color'], linewidth=1.5)

        
        #FFT sobre datos originales
        y_fft_input = data['raw'] - np.mean(data['raw'])
        fft_valores = np.fft.rfft(y_fft_input)
        frecuencias = np.fft.rfftfreq(n, d=dt)
        amplitud = np.abs(fft_valores)
        print(frecuencias)
        print(amplitud)
        error = 1/(2*t[fila_fin-3])

        #Encontrar frecuencia dominante
        f_dom = frecuencias[np.argmax(amplitud)]

        #Índices que ordenarían el array de menor a mayor
        indices_ordenados = np.argsort(amplitud)

        idx_1ero = indices_ordenados[-1]
        idx_2do = indices_ordenados[-2]

        #Acceso a los valores
        f_1 = frecuencias[idx_1ero]
        a_1 = amplitud[idx_1ero]

        f_2 = frecuencias[idx_2do]
        a_2 = amplitud[idx_2do]

        print(f"---{nombre}---")
        print(f"Máximo 1: Freq = {f_1:.4f} Hz, Amp = {a_1:.2f}")
        print(f"Máximo 2: Freq = {f_2:.4f} Hz, Amp = {a_2:.2f}")
        
        #Gráfica del espectro
        ax2.plot(frecuencias, amplitud, label=f'FFT {nombre}: frec. dominante = $({f_dom:.2f}\pm{error:.2f})$ $Hz$', 
                    color=data['color'], alpha=0.8)
        ax2.axvline(f_dom, color=data['color'], linestyle='--', alpha=0.3)

    #Estética señal
    ax1.set_title('Oscilaciones de los péndulos en el tiempo (suavizadas)')
    ax1.set_xlabel('Tiempo (s)')
    ax1.set_ylabel('Voltaje (V)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    #Estética espectro
    ax2.set_title('Espectros de frecuencias (calculados sobre los datos originales)')
    ax2.set_xlabel('Frecuencia (Hz)')
    ax2.set_ylabel('Amplitud')
    ax2.set_xlim(0, 4) # Rango típico para péndulos
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{archivo_csv.replace('.csv','')}.pdf')
    plt.show()


analizar_fourier_2('Fase.csv', 0, 10000)
analizar_fourier_2('Antifase.csv', 0, 10000)
analizar_fourier_2('Latidos.csv', 0, 10000)


#FUNCIÓN QUE GRAFICA DOS OSCILACIONES ACOPLADAS CADA UNA POR SEPARADO:

def graf_individual(archivo_csv, fila_inicio, fila_fin, ventana_suave=51):

    #Carga de datos
    df = pd.read_csv(archivo_csv, sep=';', decimal=',', skiprows=3, 
                        names=['Tiempo', 'V1', 'V2'])
    segmento = df.iloc[fila_inicio:fila_fin].copy()
    
    t = segmento['Tiempo'].values
    
    #Definición de señales y colores
    señales = [
        {'nombre': 'Péndulo Izquierdo', 'data': segmento['V1'].values, 'color': '#0077b6'},
        {'nombre': 'Péndulo Derecho', 'data': segmento['V2'].values, 'color': '#e67e22'}
    ]

    #Creación de la figura
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f'Oscilaciones individuales de los péndulos en el tiempo (suavizadas)', fontsize=16)

    for i, s in enumerate(señales):

        y_suave = savgol_filter(s['data'], ventana_suave, 3)
        
        axes[i].plot(t, y_suave, color=s['color'], linewidth=1.5, label=f"{s['nombre']}")
        axes[i].set_ylabel('Voltaje (V)')
        axes[i].grid(True, linestyle='--', alpha=0.5)
        axes[i].legend(loc='upper right')
        
        axes[i].autoscale(enable=True, axis='y', tight=False)

    axes[1].set_xlabel('Tiempo (s)')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'{archivo_csv.replace('.csv','')}_sep.pdf')
    plt.show()


graf_individual('Latidos.csv', 0, 10000)