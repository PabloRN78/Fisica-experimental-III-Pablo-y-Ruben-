import matplotlib.pyplot as plt
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
        if np.ndim(x_err) == 0 and x_err == 0:
            self.x_err = None
        else:
            self.x_err = np.array(x_err)

        if np.ndim(y_err) == 0 and y_err == 0:
            self.y_err = None
        else:
            self.y_err = np.array(y_err)
        self.titulo = titulo
        self.titulo_x = titulo_x
        self.titulo_y = titulo_y

        self.x_ajuste = np.linspace(min(self.x), max(self.x),
                                    100)  # Vector de puntos en el eje x para graficar la función del ajsute

        self.fig, self.graf = plt.subplots(figsize=(8, 5),
                                           dpi=200)  # Crea una ventana para una figura del tamaño indicado pero no la mostramos aún.

    def puntos(self):
        self.graf.errorbar(self.x, self.y, xerr=self.x_err, yerr=self.y_err, label='Puntos experimentales', fmt='o',
                           elinewidth=1,
                           capsize=3, markersize='3', color='blue', ecolor='red', alpha=1)
        # Plotea los datos x, y con sus errores xerr e yerr. El resto de parámetros personalizan el estilo.

    def formato(self):  #
        self.graf.set_xlabel(self.titulo_x)
        self.graf.set_ylabel(self.titulo_y)
        self.graf.set_title(self.titulo)
        self.graf.title.set_fontsize(13)
        self.graf.tick_params(axis='both', which='major', labelsize=9.5)  # Personaliza las marcas en los ejes
        self.graf.grid(True, linestyle='--', alpha=0.7)  # Crea una cuadrícula en la gráfica
        self.graf.legend(fontsize=8)

    # MODELOS PARA EL AJUSTE:

    def recta(self, z, m, n):
        return m * z + n

    def parabola(self, z, a, b, c):
        return a * z ** 2 + b * z + c

    def exp(self, z, a, b, c):
        return a * np.exp(b * z) + c

    def inv(self, z, a, b):
        return a/(z**b)

    # TIPOS DE AJUSTE:

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

        plt.tight_layout()
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

        plt.tight_layout()
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

        plt.tight_layout()
        plt.show()

    def inversa(self):
        if self.y_err is None:
            abc, cov = curve_fit(self.inv, self.x, self.y)
        else:
            abc, cov = curve_fit(self.inv, self.x, self.y, sigma=self.y_err, absolute_sigma=True)

        errores = np.sqrt(np.diag(cov))

        print('RESULTADOS DEL AJUSTE EXPONENCIAL')
        print(f'Parámetro multiplicativo: a={abc[0]}+-{errores[0]}')
        print(f'Parámetro en el exponente: b={abc[1]}+-{errores[1]}')

        y_exp = self.inv(self.x_ajuste, *abc)

        self.puntos()
        self.graf.plot(self.x_ajuste, y_exp, label=f'Ajuste a $y=a/x^b$', color='green')
    
        self.formato()

        plt.tight_layout()
        plt.show()



def calculo_error(nombre_f, f_str,
                  var):  # Se introduce la letra o símbolo representativo de la función em str, la función en sí en str y una lista con cada variable de la función en str.

    var_ord = sorted(var, key=len,
                     reverse=True)  # Es vital ordenar la lista de variables de mayor longitud a menor, para que a la hora de sustituir no se sustituyan letras que pertencen a palabras dentro de esas palabras.

    chill = [f'v_{i}' for i in range(
        len(var))]  # Cambiamos las variables de la función por unas variables genéricas y que no contienen símbolos extraños como barras para que sp.sympify no se pique.

    var_s = [sp.Symbol(v) for v in var]  # Creamos las variables simbólicas.
    var_s_ord = [sp.Symbol(v) for v in var_ord]  # Variables simbólicas ordenadas.
    err_s = [sp.Symbol(rf'\Delta {e}') for e in var]  # cremos las variables simbólicas para los errores.

    f_chill = f_str
    for v, vc in zip(var_ord, chill):  # Creamos la versión de la función str con las variables seguras.
        f_chill = f_chill.replace(v, vc)

    func = sp.sympify(f_chill, locals=dict(zip(chill,
                                               var_s_ord)))  # Con sympify se sustituyen las variables str de la función con las variables simbólicas correspondientes.
    f = sp.Function(nombre_f)(*var_s)

    error0 = 0
    error1 = 0
    error2 = 0
    # Calculamos la ecuación simbólica del error:
    for v in range(len(var)):
        Dp = sp.Derivative(func, var_s[v])
        dp = sp.diff(func, var_s[v])

        error0 += (err_s[v])**2*(sp.Derivative(f, var_s[v]))**2
        error1 += (Dp * err_s[v]) ** 2
        error2 += (dp * err_s[v]) ** 2

    ec_error0 = sp.sqrt(error0)
    ec_error1 = sp.sqrt(error1)
    ec_error2 = sp.sqrt(error2)

    calc_f = sp.lambdify(var_s, func,
                         'numpy')  # Con lambdify se convierten las ecuaciones simbólicas a funciones evaluables de python.
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


def cientifica(valor, exp):
    coef = valor / (10 ** exp)

    coef_limpio = round(coef, 7)

    return f"${coef_limpio:g}\\cdot 10^{{{exp}}}$"


def cientifica_np(valores, exp):
    x = np.empty(len(valores), dtype=object)

    for i in range(len(valores)):
        x[i] = cientifica(valores[i], exp)

    return x


def media_ponderada(valores, errores):
    pesos = np.array(1 / errores ** 2)

    media = np.average(valores, weights=pesos)
    error = 1 / np.sqrt(np.sum(pesos))

    mediaf, errorf = matacifras(media, error)

    return mediaf, errorf


#RESITENCIA DEL FILAMENTO:

R = pd.read_excel('stefan.xlsx', sheet_name='R')

#Entre los corchetes se especifican las columnas del excel donde se encuentran los datos
U = R[r'$U(V)$'].values
I = R[r'$I(A)$'].values

I_err = 0.001
U_err = np.concatenate((np.array([1,1,1,1,1])*0.0001,np.array([1,1,1,1,1,1,1])*0.001))

Resistencia = Experimento(I[0:12], U[0:12], 0.001, 0.001, 'Intensidad frente a tensión en la lámpara', 'I $(A)$', 'U $(V)$')
Resistencia.lineal()
Resistencia.fig.savefig('resistencia.pdf')

#LEY DE STEFAN:

L1 = pd.read_excel ('stefan.xlsx', sheet_name='ley1')
U1 = L1[r'U(V)'].values
I1 = L1[r'I(A)'].values
Uth = L1[f'U_th(mV)']

R, eR = calculo_error('R', 'U/I', ['U', 'I'])
R0, eR0 = calculo_error('R_0', r'R(t_R)/(1+\alpha*t_R+\beta*(t_R)**2)', [r'\alpha', r'\beta', r'R(t_R)', r't_R'])
temp, etemp = calculo_error('T', r'273 + 1/(2*\beta)*((\alpha**2 + 4*\beta*(R/R_0-1))**(1/2)-\alpha)', [r'\alpha', r'\beta', r'R', r'R_0'])
ln, eln = calculo_error('log', 'ln(x)', ['x'])


Rt, eRt = R(U1, I1), eR(U1, I1, 0.01, 0.001)

R_tr = 0.2677

r0, er0 = R0(4.82*10**(-3), 6.76*10**(-7), R_tr, 300), eR0(4.82*10**(-3), 6.76*10**(-7), R_tr, 300, 0, 0, 0.0007, 1)

T, eT = temp(4.82*10**(-3), 6.76*10**(-7), Rt, r0), etemp(4.82*10**(-3), 6.76*10**(-7), Rt, r0, 0, 0, eRt, er0)

lnU, elnU = ln(Uth), eln(Uth, 0.01)
lnT, elnT = ln(T), eln(T, eT)

print('Ley de Stefan-Boltzman')

print(f'R_0 = {matacifras(r0, er0)[0]} +- {matacifras(r0, er0)[1]}')
print(f'R(t) = {matacifras_np(Rt, eRt)[0]} +- {matacifras_np(Rt, eRt)[1]}')
print(f'T = {matacifras_np(T, eT)[0]} +- {matacifras_np(T, eT)[1]}')
print(f'ln(U_therm) =  {matacifras_np(lnU, elnU)[0]} +- {matacifras_np(lnU, elnU)[1]}')
print(f'ln(T) = {matacifras_np(lnT, elnT)[0]} +- {matacifras_np(lnT, elnT)[1]}')

ley1 = Experimento(lnT, lnU, elnT, elnU, 'ln$(T)$ frente a ln$(U_{therm})$', r'ln$(T)$', r'ln$(U_{therm}$)')
ley1.lineal()
ley1.fig.savefig('ley.pdf')

#DISTANCIA:

D =  pd.read_excel('stefan.xlsx', sheet_name='ley2')

Uth1 = D['U_th1(mV)'].values
Uth2 = D['U_th2(mV)'].values 
D1 = D['D1(m)'].values
D2 = D['D2(m)'].values


lnUth1, elnUth1 = ln(Uth1), eln(Uth1, 0.01)
lnD1, elnD1 = ln(D1), eln(D1, 0.001)

print(f'ln(D1) = {lnD1}+-{elnD1}')
print(f'ln(Uth1) = {lnUth1}+-{elnUth1}')

UD1 = Experimento(D1, Uth1, 0.001, 0, r'$U_{therm}$ frente a $D$, con $U = (4,84 \pm 0,01) V$', r'$D (m)$', r'$U_{therm} (mV)$')
UD1.inversa()
UD1.fig.savefig('distancia1.pdf')
UD1 = Experimento(lnD1, lnUth1, elnD1, 0, r'ln$(U_{therm})$ frente a ln$(D)$, con $U = (4,84 \pm 0,01) V$', r'ln$(D)$', r'ln$(U_{therm})$')
UD1.lineal()
UD1.fig.savefig('distancia1log.pdf')

lnUth2, elnUth2 = ln(Uth2), eln(Uth2, 0.01)
lnD2, elnD2 = ln(D2), eln(D2, 0.001)

print(f'ln(D2) = {lnD2}+-{elnD2}')
print(f'ln(Uth2) = {lnUth2}+-{elnUth2}')


UD2 = Experimento(D2, Uth2, 0.001, 0.01, r'$U_{therm}$ frente a $D$, con $U = (4,19 \pm 0,01) V$', r'$D (m)$', r'$U_{therm} (mV)$')
UD2.inversa()
UD2.fig.savefig('distancia2.pdf')
UD2 = Experimento(lnD2, lnUth2, elnD2, elnUth2, r'ln$(U_{therm})$ frente a ln$(D)$, con $U = (4,19 \pm 0,01) V$', r'ln$(D)$', r'ln$(U_{therm})$')
UD2.lineal()
UD2.fig.savefig('distancia2log.pdf')



