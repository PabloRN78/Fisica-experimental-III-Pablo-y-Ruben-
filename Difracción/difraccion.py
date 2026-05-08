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


#LECTURA DE EXCEL:

R1 = pd.read_excel('Rendija.xlsx', sheet_name='R1')

#Entre los corchetes se especifican las columnas del excel donde se encuentran los datos
D = R1[r'$D(mm) \pm 0.5 mm $'].values
X_m = R1[r'$X_{min}(mm) \pm 0.25 mm $'].values
dosX_m = R1[r'$2X_{min}(mm) \pm 0.5 mm $'].values

D_err = 0.5
Xm_err = 0.25

print('Rendija 1 ():')

Rendija1 = Experimento(D, X_m, 0, Xm_err, 'Rendija 1 ($2a_{teo} = 0.100 mm$)', '$D (mm)$', r'$X_{max} (mm)$')
Rendija1.lineal()

R1[R1.columns[[0,2]]] = R1[R1.columns[[0,2]]].astype(object)
R1[r'$X_{min}(mm) \pm 0.25 mm $'] = np.array([f'{x:.2f}' for x in X_m])
R1[r'$D(mm) \pm 0.5 mm $'] = np.array([f'{x:.1f}' for x in D])
R1[r'$2X_{min}(mm) \pm 0.5 mm $'] = np.array([f'{x:.1f}' for x in dosX_m])

print('CÓDIGO LATEX DE LA TABLA 1:')
print(R1.iloc[0:9, 0:3].style.hide(axis='index').to_latex(column_format="ccccc", caption="Difracción con rendija 1", label="tab:R1", hrules=True))


R2 = pd.read_excel('Rendija.xlsx', sheet_name='R2')

D2 = R2[r'$D(mm) \pm 0.5 mm $'].values
X_m2 = R2[r'$X_{min}(mm) \pm 0.25 mm $'].values
dosX_m2 = R2[r'$2X_{min}(mm) \pm 0.5 mm $'].values


print('Rendija 2 ():')

Rendija2 = Experimento(D2, X_m2, 0, Xm_err, 'Rendija 2 ($2a_{teo} = 0.025mm$)', '$D (mm)$', r'$X_{max} (mm)$')
Rendija2.lineal()

R2[R2.columns[[0,2]]] = R2[R2.columns[[0,2]]].astype(object)
R2[r'$X_{min}(mm) \pm 0.5 mm $'] = np.array([f'{x:g}' for x in X_m2])
R2[r'$D(mm) \pm 0.5 mm $'] = np.array([f'{x:.1f}' for x in D2])
R2[r'$2X_{min}(mm) \pm 0.25 mm $'] = np.array([f'{x:.3f}' for x in dosX_m2])


print('CÓDIGO LATEX DE LA TABLA 2:')
print(R2.iloc[0:9, 0:3].style.hide(axis='index').to_latex(column_format="ccccc", caption="Difracción con rendija 2", label="tab:R2", hrules=True))



Red = pd.read_excel('Rendija.xlsx', sheet_name='Red')

D3 = Red[r'$D(mm) \pm 0.5 mm $'].values
X_m3 = Red[r'$X_{min}(mm) \pm 0.5 mm $'].values


RD = Experimento(D3, X_m3, 0, 0.5, 'Red de difracción 1 ($p_{teo}=300 líneas/mm$)', '$D(mm)$', r'$X_{max}(mm)$')
RD.lineal()

Red[Red.columns[[0,2]]] = Red[Red.columns[[0,2]]].astype(object)
Red[r'$X_{min}(mm) \pm 0.5 mm $'] = np.array([f'{x:.1f}' for x in X_m3])
Red[r'$D(mm) \pm 0.5 mm $'] = np.array([f'{x:.1f}' for x in D3])


print('CÓDIGO LATEX DE LA TABLA 3:')
print(Red.iloc[0:9, 0:2].style.hide(axis='index').to_latex(column_format="ccccc", caption="Difracción por red", label="tab:Red", hrules=True))


Red2 = pd.read_excel('Rendija.xlsx', sheet_name='Red2')

D32 = Red2[r'$D(mm) \pm 0.5 mm $'].values
X_m32 = Red2[r'$X_{min}(mm) \pm 0.5 mm $'].values


RD2 = Experimento(D32, X_m32, 0, 0.5, 'Red de difracción 2 ($p_{teo}=100 líneas/mm$)', '$D(mm)$', r'$X_{max}(mm)$')
RD2.lineal()

Red2[Red2.columns[[0,2]]] = Red2[Red2.columns[[0,2]]].astype(object)
Red2[r'$X_{min}(mm) \pm 0.5 mm $'] = np.array([f'{x:.1f}' for x in X_m32])
Red2[r'$D(mm) \pm 0.5 mm $'] = np.array([f'{x:.1f}' for x in D32])


print('CÓDIGO LATEX DE LA TABLA 4:')
print(Red2.iloc[0:9, 0:2].style.hide(axis='index').to_latex(column_format="ccccc", caption="Difracción por red", label="tab:Red", hrules=True))


#Cálculos:

grosor, egrosor = calculo_error('2a_{exp}', r'3/2*\lambda/m',['m',r'\lambda'])
par_red, epar_red = calculo_error('p',r'm/\lambda',['m',r'\lambda'])
gonio, egonio = calculo_error(r'\lambda_{exp}', r'sin(\theta)*2 /p', [r'\theta','p'])

ang = (np.array([5.4, 5.6, 5.8, 6.0, 7.0, 8.0]))*(2*np.pi/360)

print(f'Grosor de la rendija 1: {grosor(0.0103, 635*10**(-6))} +- {egrosor(0.0103, 635*10**(-6), 0.0004, 5*10**(-6))}')
print(f'Grosor de la rendija 2: {grosor(0.0459, 635*10**(-6))} +- {egrosor(0.0459, 635*10**(-6), 0.0004, 5*10**(-6))}')
print(f'Parámetro de la red 1: {par_red(0.1949, 635*10**(-6))} +- {epar_red(0.1949, 635*10**(-6), 0.0004, 5*10**(-6))}')
print(f'Parámetro de la red 2: {par_red(0.0587, 635*10**(-6))} +- {epar_red(0.0587, 635*10**(-6), 0.0009, 5*10**(-6))}')
print(f'Longitud espectral: {gonio(ang, 531.5*10**(-6))} +- {egonio(ang, 531.5*10**(-6), 0.05*(2*np.pi/360), 0.5*10**(-6))}')

Rendija1.fig.savefig('Ajuste Rendija 1.pdf')
Rendija2.fig.savefig('Ajuste Rendija 2.pdf')
RD.fig.savefig('Ajuste Red 1.pdf')
RD2.fig.savefig('Ajuste Red 2.pdf')
