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




calc_g, calc_eg = calculo_error('g',r'(\lambda*(d**2+l**2)**(1/2))/l', [r'\lambda', 'd', 'l'])


#MERCURIO CON RED DE 300l/mm:
Hg300 = pd.read_excel('Serie de Balmer.xlsx', sheet_name = 'Hg300')

L_hg300 = np.array(Hg300['l(m)'].iloc[0:5])
lamb = np.array(Hg300[r'\lambda_{ref}(nm)'].iloc[0:5])*10**(-9)
d_hg300 = 0.502

g_hg300, eg_hg300 = matacifras_np(calc_g(lamb, d_hg300, L_hg300)*10**6, calc_eg(lamb, d_hg300, L_hg300, 0.1*10**(-9), 0.0005, 0.00025)*10**6)

G300, eG300 = media_ponderada(g_hg300, eg_hg300)

Hg300[r'g (\mu m)'] = Hg300[r'g (\mu m)'].astype(object)
Hg300.iloc[0:5, Hg300.columns.get_loc(r'g (\mu m)')] = g_hg300
Hg300.insert(5, r'\Delta g (\mu m)', np.nan)
Hg300.iloc[0:5, Hg300.columns.get_loc(r'\Delta g (\mu m)')] = eg_hg300


print('MERCURIO CON RED DE 300l/mm:')
print(g_hg300)
print(eg_hg300)
print(f'g media = {G300} +- {eG300}')
print('CÓDIGO LATEX DE LA TABLA:')
print(Hg300.iloc[0:5, 0:6].style.format(formatter="{:g}", subset=Hg300.columns[1:6]).hide(axis='index').to_latex(column_format="ccccc", caption="Mis datos de la Serie de Balmer", label="tab:hg300", hrules=True))


#MERCURIO CON RED DE 600l/mm:
Hg600 = pd.read_excel('Serie de Balmer.xlsx', sheet_name = 'Hg600')

L_hg600 = np.array(Hg600['l(m)'].iloc[0:5])
d_hg600 = 0.500

g_hg600, eg_hg600 = matacifras_np(calc_g(lamb, d_hg600, L_hg600)*10**6, calc_eg(lamb, d_hg600, L_hg600, 0.1*10**(-9), 0.0005, 0.00025)*10**6)

G600, eG600 = media_ponderada(g_hg600, eg_hg600)

Hg600[r'g (\mu m)'] = Hg600[r'g (\mu m)'].astype(object)
Hg600.iloc[0:5, Hg600.columns.get_loc(r'g (\mu m)')] = g_hg600
Hg600.insert(5, r'\Delta g (\mu m)', np.nan)
Hg600.iloc[0:5, Hg600.columns.get_loc(r'\Delta g (\mu m)')] = eg_hg600

print('MERCURIO CON RED DE 600l/mm:')
print(g_hg600)
print(eg_hg600)
print(f'g media = {G600} +- {eG600}')
print('CÓDIGO LATEX DE LA TABLA:')
print(Hg600.iloc[0:5, 0:6].style.format(formatter="{:g}", subset=Hg600.columns[1:6]).hide(axis='index').to_latex(column_format="ccccc", caption="Mis datos de la Serie de Balmer", label="tab:hg600", hrules=True))


#MERCURIO CON RED DESCONOCIDA:
Hg = pd.read_excel('Serie de Balmer.xlsx', sheet_name = 'Hg')

L_hg = np.array(Hg['l(m)'].iloc[0:3])
lamb_ = np.array(Hg300[r'\lambda_{ref}(nm)'].iloc[0:3])*10**(-9)
d_hg = 0.500

g_hg, eg_hg = matacifras_np(calc_g(lamb_, d_hg, L_hg)*10**6, calc_eg(lamb_, d_hg, L_hg, 0.1*10**(-9), 0.0005, 0.00025)*10**6)


G, eG = media_ponderada(g_hg, eg_hg)

Hg[r'g (\mu m)'] = Hg[r'g (\mu m)'].astype(object)
Hg.iloc[0:3, Hg.columns.get_loc(r'g (\mu m)')] = g_hg
Hg.insert(5, r'\Delta g (\mu m)', np.nan)
Hg.iloc[0:3, Hg.columns.get_loc(r'\Delta g (\mu m)')] = eg_hg




print('MERCURIO CON RED DESCONOCIDA:')
print(g_hg)
print(eg_hg)
print(f'g media = {G} +- {eG}')
print('CÓDIGO LATEX DE LA TABLA:')
print(Hg.iloc[0:3, 0:6].style.format(formatter="{:g}", subset=Hg.columns[1:6]).hide(axis='index').to_latex(column_format="ccccc", caption="Mis datos de la Serie de Balmer", label="tab:hg", hrules=True))


#BALMER 100l/mm:

calc_lam, calc_elam = calculo_error(r'\lambda_{exp}',r'g*l*(l**2+d**2)**(-1/2)', ['g', 'd', 'l'])
calc_ilam, calc_eilam = calculo_error(r'1/\lambda_{exp}', r'1/\lambda_{exp}', [r'\lambda_{exp}'])
calc_R, calc_eR = calculo_error('R',r'(1/\lambda_{exp})*(1/n**2-1/m**2)**(-1)', ['1/\lambda_{exp}', 'n', 'm'])

Bal100 = pd.read_excel('Serie de Balmer.xlsx', sheet_name='Bal100')

L_B100 = np.array(Bal100['l(m)'].iloc[0:3])
m = np.array([3, 4, 5])
g_100 = 1/100000
d_100 = 0.537

lam_100, elam_100 = matacifras_np(calc_lam(g_100, d_100, L_B100)*10**9, calc_elam(g_100, d_100, L_B100, 0.01*10**-6, 0.0005, 0.00025)*10**9)

ilam_100, eilam_100 = matacifras_np(calc_ilam(lam_100)*10**9, calc_eilam(lam_100, elam_100)*10**9)

R_100, eR_100 = matacifras_np(calc_R(ilam_100[0:3], 2, m), calc_eR(ilam_100[0:3], 2, m, eilam_100[0:3], 0, 0))

Rf100, eRf100 = media_ponderada(R_100, eR_100)

Bal100[Bal100.columns[[4,5,6]]] = Bal100[Bal100.columns[[4,5,6]]].astype(object)
Bal100.iloc[0:3, Bal100.columns.get_loc(r'$\lambda_{exp}(nm)$')] = lam_100
Bal100.iloc[0:3, Bal100.columns.get_loc(r'1/$\lambda_{exp}$')] = cientifica_np(ilam_100, 7)
Bal100.iloc[0:3, Bal100.columns.get_loc(r'R')] = cientifica_np(R_100, 7)


print('Serie de Balmer con 100l/mm:')
print(np.array(Bal100['Línea'].iloc[0:3]))
print(f'{lam_100} +- {elam_100}')
print(f'{ilam_100} +- {eilam_100}')
print(f'{R_100}+-{eR_100}')
print(f'R media = {Rf100} +- {eRf100}')
print('CÓDIGO LATEX DE LA TABLA:')
#print(Bal100.iloc[0:4, 0:6].style.format(formatter="{:g}", subset=Bal100.columns[1:6]).hide(axis='index').to_latex(column_format="ccccc", caption="Mis datos de la Serie de Balmer", label="tab:balmer100", hrules=True))


#BALMER 300l/mm:

Bal300 = pd.read_excel('Serie de Balmer.xlsx', sheet_name='Bal300')

L_B300 = np.array(Bal300['l(m)'].iloc[0:3])
g_300 = 3.41*10**-6
d_300 = 0.495

lam_300 = calc_lam(g_300, d_300, L_B300)
elam_300 = calc_elam(g_300, d_300, L_B300, 0.001*10**-6, 0.0005, 0.00025)

ilam_300 = calc_ilam(lam_300)
eilam_300 = calc_eilam(lam_300, elam_300)

R_300 = calc_R(ilam_300[0:3], 2, m)
eR_300 = calc_eR(ilam_300[0:3], 2, m, eilam_300[0:3], 0, 0)

Rf300, eRf300 = media_ponderada(R_300, eR_300)

print('Serie de Balmer con 300l/mm:')
print(np.array(Bal300['Línea'].iloc[0:3]))
print(f'{lam_300} +- {elam_300}')
print(f'{ilam_300} +- {eilam_300}')
print(f'{R_300}+-{eR_300}')
print(f'R media = {Rf300} +- {eRf300}')

#BALMER 500l/mm:

Bal500 = pd.read_excel('Serie de Balmer.xlsx', sheet_name='Bal500')

L_B500 = np.array(Bal500['l(m)'].iloc[0:3])
g_500 = 1/500000
d_500 = 0.492

lam_500 = calc_lam(g_500, d_500, L_B500)
elam_500 = calc_elam(g_500, d_500, L_B500, 0.001*10**-6, 0.0005, 0.00025)

ilam_500 = calc_ilam(lam_500)
eilam_500 = calc_eilam(lam_500, elam_500)

R_500 = calc_R(ilam_500[0:3], 2, m)
eR_500 = calc_eR(ilam_500[0:3], 2, m, eilam_500[0:3], 0, 0)

Rf500, eRf500 = media_ponderada(R_500, eR_500)

print('Serie de Balmer con 500l/mm:')
print(np.array(Bal500['Línea'].iloc[0:3]))
print(f'{lam_500} +- {elam_500}')
print(f'{ilam_500} +- {eilam_500}')
print(f'{R_500}+-{eR_500}')
print(f'R media = {Rf500} +- {eRf500}')

#BALMER 600l/mm:

Bal600 = pd.read_excel('Serie de Balmer.xlsx', sheet_name='Bal600')

L_B600 = np.array(Bal600['l(m)'].iloc[0:3])
g_600 = 1.63*10**-6
d_600 = 0.495

lam_600 = calc_lam(g_600, d_600, L_B600)
elam_600 = calc_elam(g_600, d_600, L_B600, 0.001**10*-6, 0.0005, 0.00025)

ilam_600 = calc_ilam(lam_600)
eilam_600 = calc_eilam(lam_600, elam_600)

R_600 = calc_R(ilam_600[0:3], 2, m)
eR_600 = calc_eR(ilam_600[0:3], 2, m, eilam_600[0:3], 0, 0)

Rf600, eRf600 = media_ponderada(R_600, eR_600)

print('Serie de Balmer con 600l/mm:')
print(np.array(Bal600['Línea'].iloc[0:3]))
print(f'{lam_600} +- {elam_600}')
print(f'{ilam_600} +- {eilam_600}')
print(f'{R_600}+-{eR_600}')
print(f'R media = {Rf600} +- {eRf600}')

print(media_ponderada(np.array([1.111,1.087,1.073,1.090]),np.array([0.001,0.002,0.005,0.001])))





