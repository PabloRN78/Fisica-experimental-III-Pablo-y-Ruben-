import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import sympy as sp
from scipy.optimize import curve_fit
import scipy.constants as cte
import pandas as pd
from scipy.io import loadmat
from sympy.printing.tree import print_node

class Experimento:

    def __init__(self, x, y, x_err, y_err, titulo, titulo_x, titulo_y):
        self.x = np.array(x)
        self.y = np.array(y)
        self.x_err = np.array(x_err)
        if np.ndim(y_err) == 0 and y_err == 0:
            self.y_err = None
        else:
            self.y_err = np.array(y_err)
        self.titulo = titulo
        self.titulo_x = titulo_x
        self.titulo_y = titulo_y

        self.x_ajuste = np.linspace(min(self.x), max(self.x), 100) #Vector de puntos en el eje x para graficar la función del ajsute

        self.fig, self.graf = plt.subplots(figsize=(8,5), dpi=200) #Crea una ventana para una figura del tamaño indicado pero no la mostramos aún.

    def puntos(self):
        self.graf.errorbar(self.x, self.y, xerr=self.x_err, yerr=self.y_err, label='Puntos experimentales', fmt='o', elinewidth=1,
                           capsize=3, markersize='3', color='blue', ecolor='red', alpha=1)
        #Plotea los datos x, y con sus errores xerr e yerr. El resto de parámetros personalizan el estilo.

    def formato(self): #
        self.graf.set_xlabel(self.titulo_x)
        self.graf.set_ylabel(self.titulo_y)
        self.graf.set_title(self.titulo)
        self.graf.title.set_fontsize(13)
        self.graf.tick_params(axis='both', which='major', labelsize=9.5) #Personaliza las marcas en los ejes
        self.graf.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.5f}'))
        self.graf.grid(True, linestyle='--', alpha=0.7) #Crea una cuadrícula en la gráfica
        self.graf.legend(fontsize=8)

    #MODELOS PARA EL AJUSTE:

    def recta(self, z, m, n):
        return m*z+n

    def parabola(self, z, a, b, c):
        return a*z**2+b*z+c

    def exp(self, z, a, b, c):
        return a*np.exp(b*z)+c

    #TIPOS DE AJUSTE:

    def lineal(self):
        if self.y_err is None:
            m_n, cov = curve_fit(self.recta, self.x, self.y)
        else:
            m_n, cov = curve_fit(self.recta, self.x, self.y, sigma=self.y_err, absolute_sigma=True)

        errores = np.sqrt(np.diag(cov))

        print('RESULTADOS DEL AJUSTE LINEAL')
        print(rf'La pendiente es $m={m_n[0]} \pm {errores[0]}$')
        print(rf'La recta corta al eje y en $y={m_n[1]} \pm {errores[1]}$')


        y_recta = self.recta(self.x_ajuste, *m_n)

        self.puntos()
        self.graf.plot(self.x_ajuste, y_recta, label=f'Ajuste lineal', color='green', linewidth=1)
        self.formato()

        plt.show()

    def poli_g2(self):
        if self.y_err is None:
            abc, cov = curve_fit(self.parabola, self.x, self.y)
        else:
            abc, cov = curve_fit(self.parabola, self.x, self.y, sigma=self.y_err, absolute_sigma=True)

        errores = np.sqrt(np.diag(cov))

        print('RESULTADOS DEL AJUSTE POLINÓMICO DE GRADO 2')
        print(f'Parámetro del término cuadrático: a={abc[0]}+-{errores[0]}')
        print(f'Parámetro del término lineal: b={abc[1]}+-{errores[1]}')
        print(f'Constante aditiva: c={abc[2]}+-{errores[2]}')

        y_par = self.parabola(self.x_ajuste, *abc)

        self.puntos()
        self.graf.plot(self.x_ajuste, y_par, label=f'Ajuste polinómico (G 2)', color='green')
        self.formato()

        plt.show()

    def exponencial(self):
        if self.y_err is None:
            abc, cov = curve_fit(self.exp, self.x, self.y)
        else:
            abc, cov = curve_fit(self.exp, self.x, self.y, sigma=self.y_err, absolute_sigma=True)

        errores = np.sqrt(np.diag(cov))

        print('RESULTADOS DEL AJUSTE EXPONENCIAL')
        print(f'Parámetro multiplicativo: a={abc[0]}+-{errores[0]}')
        print(f'Parámetro en el exponente: b={abc[1]}+-{errores[1]}')
        print(f'Parámetro aditivo: c={abc[2]}+-{errores[2]}')

        y_exp = self.exp(self.x_ajuste, *abc)

        self.puntos()
        self.graf.plot(self.x_ajuste, y_exp, label=f'Ajuste exponencial', color='green')
        self.formato()

        plt.show()


# CÁLCULO DE ERRORES:

def calculo_error(nombre_f, f_str, var):  # Se introduce la letra o símbolo representativo de la función em str, la función en sí en str y una lista con cada variable de la función en str.

    var_ord = sorted(var, key=len, reverse=True) #Es vital ordenar la lista de variables de mayor longitud a menor, para que a la hora de sustituir no se sustituyan letras que pertencen a palabras dentro de esas palabras.

    chill = [f'v_{i}' for i in range(len(var))]  # Cambiamos las variables de la función por unas variables genéricas y que no contienen símbolos extraños como barras para que sp.sympify no se pique.

    var_s = [sp.Symbol(v) for v in var]  # Creamos las variables simbólicas.
    var_s_ord = [sp.Symbol(v) for v in var_ord] #Variables simbólicas ordenadas.
    err_s = [sp.Symbol(rf'\Delta {e}') for e in var]  # cremos las variables simbólicas para los errores.

    f_chill = f_str
    for v, vc in zip(var_ord, chill):  # Creamos la versión de la función str con las variables seguras.
        f_chill = f_chill.replace(v, vc)

    func = sp.sympify(f_chill, locals=dict(zip(chill, var_s_ord)))  # Con sympify se sustituyen las variables str de la función con las variables simbólicas correspondientes.
    f = sp.Function(nombre_f)(*var_s)

    error0 = 0
    error1 = 0
    error2 = 0
    # Calculamos la ecuación simbólica del error:
    for v in range(len(var)):
        Dp = sp.Derivative(func, var_s[v])
        dp = sp.diff(func, var_s[v])

        error0 += (sp.Derivative(f, var_s[v]))
        error1 += (Dp * err_s[v]) ** 2
        error2 += (dp * err_s[v]) ** 2

    ec_error0 = sp.sqrt(error0)
    ec_error1 = sp.sqrt(error1)
    ec_error2 = sp.sqrt(error2)

    calc_f = sp.lambdify(var_s, func,'numpy')  # Con lambdify se convierten las ecuaciones simbólicas a funciones evaluables de python.
    calc_e = sp.lambdify(var_s + err_s, ec_error2, 'numpy')

    # Pasamos a latex las ecuaciones:
    f_latex = sp.latex(func)
    ec_error0_latex = sp.latex(ec_error0)
    ec_error1_latex = sp.latex(ec_error1)
    ec_error2_latex = sp.latex(ec_error2)

    print(f'Código en latex para la función: {nombre_f} = {f_latex}')
    print(
        fr'Código en latex para la ecuación del error: \Delta {nombre_f} ={ec_error0_latex} = {ec_error1_latex} = {ec_error2_latex}')

    return calc_f, calc_e


def matacifras(val, err):
    if err == 0 or np.isnan(err):
        return val, err

    orden = int(np.floor(np.log10(abs(err))))

    error = round(err, -orden)
    valor = round(val, -orden)

    return valor, error


def matacifras_np(val, err):
    valor = np.zeros_like(val)
    error = np.zeros_like(err)

    for i in range(len(err)):
        e = err[i]
        v = val[i]

        orden = int(np.floor(np.log10(abs(e))))

        error[i] = round(e, -orden)
        valor[i] = round(v, -orden)

    return valor, error


def cientifica(valor, exp, corte):
    coef = valor/(10**exp)

    coef_limpio = round(coef, 7)

    return f"${coef_limpio:.{corte}f}\\cdot 10^{{{exp}}}$"



def cientifica_np(valores, exp, corte):
    x = np.empty(len(valores), dtype=object)

    for i in range(len(valores)):
        x[i] = cientifica(valores[i], exp, corte)

    return x


def media_ponderada(valores, errores):
    pesos = np.array(1/errores**2)

    media = np.average(valores, weights=pesos)
    error = 1/np.sqrt(np.sum(pesos))

    mediaf, errorf = matacifras(media, error)

    return mediaf, errorf

"""
#LECTURA DE EXCEL:

P1_Gep = pd.read_excel('efecto_hall.xlsx', sheet_name='Parte_1Gep')

#Entre los corchetes se especifican las columnas o filas del excel donde se encuentran los datos
I = P1_Gep['Intensidad(mA)'].iloc[1:25]
U_h_bf = P1_Gep['U_h(mV)'].iloc[1:25]

I_err = 1
Uh_err = 0.01

B = P1_Gep['B(mT)'].values
U_h_if = P1_Gep['U_h'].values

B_err = 1

P1p_bf = Experimento(I, U_h_bf, I_err, 0, 'Intensidad frente a Voltaje Hall (Ge tipo p)', '$I(mA)$', '$U_H(mV)$')
P1p_bf.lineal()
P1p_bf.fig.savefig('Grafica IvsU(p).pdf')

P1p_if = Experimento(B, U_h_if, B_err, 0, 'Campo magnético frente a Voltaje Hall (Ge tipo p)', '$B(mT)$', '$U_H(mV)$')
P1p_if.lineal()
P1p_if.fig.savefig('Grafica BvsU(p).pdf')


Rh_1p_bf, e_1p_bf = calculo_error('R_h', 'm*d/B', ['m','d','B'])
Rh_1p_if, e_1p_if = calculo_error('R_h', 'm*d/I', ['m','d','I'])

print(f'El valor de la constante Hall con campo fijo (Tipo p) es: Rh = {Rh_1p_bf(0.88246, 0.00113, 199)} +- {e_1p_bf(0.88246, 0.00113, 199, 0.00005, 0.00001, 1)}')
print(f'El valor de la constante Hall con corriente fija (Tipo p) es: Rh = {Rh_1p_if(0.17066, 0.00113, 42)} +- {e_1p_if(0.17066, 0.00113, 42, 0.00002, 0.00001, 1)}')

P1_Gen = pd.read_excel('efecto_hall.xlsx', sheet_name='Parte_1Gen')

#Entre los corchetes se especifican las columnas o filas del excel donde se encuentran los datos
In = P1_Gen['Intensidad(mA)'].iloc[1:25]
U_h_bfn = P1_Gen['U_h()'].iloc[1:25]

Bn = P1_Gen['B(mT)'].values
U_h_ifn = P1_Gen['U_h'].values

P1n_bf = Experimento(In, U_h_bfn, I_err, 0, 'Intensidad frente a Voltaje Hall (Ge tipo n)', '$I(mA)$', '$U_H(mV)$')
P1n_bf.lineal()
P1n_bf.fig.savefig('Grafica IvsU(n).pdf')

P1n_if = Experimento(Bn, U_h_ifn, B_err, 0, 'Campo magnético frente a Voltaje Hall (Ge tipo n)', '$B(mT)$', '$U_H(mV)$')
P1n_if.lineal()
P1n_if.fig.savefig('Grafica BvsU(n).pdf')


Rh_1n_bf, e_1n_bf = calculo_error('R_h', 'm*d/B', ['m','d','B'])
Rh_1n_if, e_1n_if = calculo_error('R_h', 'm*d/I', ['m','d','I'])

print(f'El valor de la constante Hall con campo fijo (Tipo n) es: Rh = {Rh_1n_bf(-1.65678, 0.00113, 199)} +- {e_1n_bf(-1.65678, 0.00113, 199, 0.00005, 0.00001, 1)}')
print(f'El valor de la constante Hall con corriente fija (Tipo n) es: Rh = {Rh_1n_if(-0.31796, 0.00113, 41)} +- {e_1n_if(-0.31796, 0.00113, 41, 0.00002, 0.00001, 1)}')
"""


#Parte 2(dependencia con la temperatura):


#Región intrínseca (Tipo n):
T = np.array([29,32,35,37,40,42,45,47,50,52,55,57,60,62,65,67,70,72,75,77,
              80,82,85,87,90,92,95,97,100,102,105,107,110,112,115,117,120,122,
              125,127,130,132,135,137,140]) + 273.15

U_h = np.array([1.3609,1.3647,1.3743,1.3862,1.4013,1.4127,1.4292,1.4408,1.4583,1.4694,1.4851,1.4930,1.5044
                   ,1.5106,1.5175,1.5201,1.5219,1.5193,1.5118,1.5005,1.4758,1.4613,1.4278,1.4010,1.3571,1.3206
                   ,1.2656,1.2296,1.1717,1.1152,1.0587,1.0115,0.9547,0.9063,0.8556,0.8021,0.7542,0.7221,0.6725
                   ,0.6348,0.5832,0.5639,0.5200,0.4905,0.4523])

ln_sgm, ln_smg_e = calculo_error(r'ln(1/U)', 'ln(1/U)',['U'])
invT, e_invT = calculo_error(r'1/T', r'1/T', ['T'])

ln_sigma, ln_e_sigma = ln_sgm(U_h), ln_smg_e(U_h, 0.0001)
iT, e_iT = invT(T), e_invT(T, 1)

#print(f'Log de 1/U:{ln_sigma}+-{ln_e_sigma}')
#print(f'1/T:{iT}+-{e_iT}')

P2_fig, P2_graf= plt.subplots(figsize=(8,5), dpi=200)

P2_graf.errorbar(iT, ln_sigma, xerr=e_iT, label='Puntos experimentales', fmt='o', elinewidth=1, capsize=3, markersize='3', color='blue', ecolor='red', alpha=1)
P2_graf.set_xlabel(r'$1/T(K^{-1})$')
P2_graf.set_ylabel(r'$ln(1/U_{sample})$')
P2_graf.set_title('Dependencia con la temperatura del voltaje en Ge tipo n')
P2_graf.title.set_fontsize(13)
P2_graf.tick_params(axis='both', which='major', labelsize=9.5) #Personaliza las marcas en los ejes
P2_graf.grid(True, linestyle='--', alpha=0.7) #Crea una cuadrícula en la gráfica
P2_graf.legend(fontsize=8)

plt.show()
P2_fig.savefig('Graf_temp_n.pdf')

iT_high = iT[33:46]
ln_sigma_high = ln_sigma[33:46]

Reg_lineal_n = Experimento(iT_high, ln_sigma_high, e_iT[33:46], ln_e_sigma[33:46], 'Ajuste lineal en la región intrínseca (Ge tipo n)', r'$1/T(K^{-1})$', r'$ln(1/U_{sample})$')
Reg_lineal_n.lineal()
Reg_lineal_n.fig.savefig('Region lineal n.pdf')

Eg, Eg_e = calculo_error(r'E_g', r'-2*k_B*m',['k_B','m'])

print(fr'La energía del gap es Eg = {Eg(cte.k, -3878.4)} +- {Eg_e(cte.k, -3878.4, 0, 0.8)}')


#Región intrínseca (Tipo p):

U = np.array([0.9545,0.9617,0.9729,0.9879,1.0027,1.0144,1.0308,1.0437,1.0643,1.0749,1.0935,1.1078,1.1255,
              1.1360,1.1533,1.1668,1.1820,1.1906,1.2029,1.2132,1.2223,1.2279,1.2305,1.2309,1.2263,1.2202,
              1.2107,1.1949,1.1710,1.1499,1.1173,1.0914,1.0430,1.0108,0.9619,0.9279,0.8643,0.8419,0.7849,
              0.7464,0.6940,0.6635,0.6171,0.5897,0.5400])

ln_sigma_p, ln_e_sigma_p = ln_sgm(U), ln_smg_e(U, 0.0001)

print(f'Log de 1/U:{ln_sigma_p}+-{ln_e_sigma_p}')
#print(f'1/T:{iT}+-{e_iT}')

P2p_fig, P2p_graf= plt.subplots(figsize=(8,5), dpi=200)

P2p_graf.errorbar(iT, ln_sigma_p, xerr=e_iT, label='Puntos experimentales', fmt='o', elinewidth=1, capsize=3, markersize='3', color='blue', ecolor='red', alpha=1)
P2p_graf.set_xlabel(r'$1/T(K^{-1})$')
P2p_graf.set_ylabel(r'$ln(1/U_{sample})$')
P2p_graf.set_title('Dependencia con la temperatura del voltaje en Ge tipo p')
P2p_graf.title.set_fontsize(13)
P2p_graf.tick_params(axis='both', which='major', labelsize=9.5) #Personaliza las marcas en los ejes
P2p_graf.grid(True, linestyle='--', alpha=0.7) #Crea una cuadrícula en la gráfica
P2p_graf.legend(fontsize=8)

plt.show()
P2p_fig.savefig('Graf_temp_p.pdf')


Reg_lineal_p = Experimento(iT[33:46], ln_sigma_p[33:46], e_iT[33:46], ln_e_sigma_p[33:46], 'Ajuste lineal en la región intrínseca (Ge tipo p)', r'$1/T(K^{-1})$', r'$ln(1/U_{sample})$')
Reg_lineal_p.lineal()
Reg_lineal_p.fig.savefig('Region lineal p.pdf')

Eg, Eg_e = calculo_error(r'E_g', r'-2*k_B*m',['k_B','m'])

print(fr'La energía del gap es Eg = {Eg(cte.k, -3469.0)} +- {Eg_e(cte.k, -3469.0, 0, 0.7)}')


#Inversión de signo (Tipo p):
T2 = np.array([29,32,35,37,40,42,45,47,50,52,55,57,60,62,65,
               67,70,72,75,77,79,80,82,84,85,87,89,90,92,94,95,97,98,100,102,103,104,
               105,106,107,108,109,110,112,113,114,115,117,120,122,125,127,130,132,135,137,140]) + 273.15

Ui = np.array([28.45,28.43,28.47,28.52,28.63,28.71,28.84,28.93,29.03,29.12,29.24,29.31,29.37,29.38,29.34,
               29.31,29.07,28.94,28.57,28.26,27.95,27.61,27.20,26.21,24.92,24.08,23.17,21.11,20.51,19.35,
               17.63,17.06,16.03,14.07,12.93,11.03,9.92,9.16,7.99,7.36,6.82,5.79,5.25,4.34,3.49,2.32,1.98,
               0.79,-0.16,-0.98,-1.91,-2.36,-2.91,-3.13,-3.39,-3.55,-3.64])

P2i_fig, P2i_graf= plt.subplots(figsize=(8,5), dpi=200)

P2i_graf.errorbar(T2, Ui, xerr=1, label='Puntos experimentales', fmt='o', elinewidth=1, capsize=3, markersize='2.2', color='blue', ecolor='red', alpha=1)
P2i_graf.set_xlabel(r'$T(K)$')
P2i_graf.set_ylabel(r'$U_{sample}(mV)$')
P2i_graf.set_title(' $T$ frente a $U_{sample}$ en presencia de campo magnético (Ge tipo p)')
P2i_graf.title.set_fontsize(13)
P2i_graf.tick_params(axis='both', which='major', labelsize=9.5) #Personaliza las marcas en los ejes
P2i_graf.grid(True, linestyle='--', alpha=0.7) #Crea una cuadrícula en la gráfica
P2i_graf.legend(fontsize=8)

plt.show()
P2i_fig.savefig('Graf_temp_inv.pdf')
