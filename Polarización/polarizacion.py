import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import sympy as sp
from scipy.optimize import curve_fit
import pandas as pd
from scipy.io import loadmat
from sympy.printing.tree import print_node

class Experimento:

    def __init__(self, x, y, x_err, y_err, titulo, titulo_x, titulo_y):
        self.x = np.array(x)
        self.y = np.array(y)
        self.x_err = np.array(x_err)
        self.y_err = np.array(y_err)
        self.titulo = titulo
        self.titulo_x = titulo_x
        self.titulo_y = titulo_y

        self.x_ajuste = np.linspace(min(self.x), max(self.x), 1000) #Vector de puntos en el eje x para graficar la función del ajsute

        self.fig, self.graf = plt.subplots(figsize=(8,5), dpi=200) #Crea una ventana para una figura del tamaño indicado pero no la mostramos aún.

    def puntos(self):
        self.graf.errorbar(self.x, self.y, xerr=self.x_err, yerr=self.y_err, label='Errores', fmt='o', elinewidth=1,
                           capsize=3, markersize='4', color='blue', ecolor='red', alpha=1)
        #Plotea los datos x, y con sus errores xerr e yerr. El resto de parámetros personalizan el estilo.

    def formato(self): #
        self.graf.set_xlabel(self.titulo_x)
        self.graf.set_ylabel(self.titulo_y)
        self.graf.set_title(self.titulo)
        self.graf.title.set_fontsize(13)
        self.graf.tick_params(axis='both', which='major', labelsize=9) #Personaliza las marcas en los ejes
        self.graf.grid(True, linestyle='--', alpha=0.7) #Crea una cuadrícula en la gráfica
        self.graf.legend(fontsize=8)

    #MODELOS PARA EL AJUSTE:

    def recta(self, z, m, n):
        return m*z+n

    def parabola(self, z, a, b, c):
        return a*z**2+b*z+c

    def exp(self, z, a, b, c):
        return a*np.exp(b*z)+c

    def cos2(self, z, a, b, c, d):
        return a*np.cos(b*z+c)**2+d

    #TIPOS DE AJUSTE:

    def lineal(self):
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
        abc, cov = curve_fit(self.parabola , self.x, self.y, sigma=self.y_err, absolute_sigma=True)
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
        abc, cov = curve_fit(self.exp , self.x, self.y, sigma=self.y_err, absolute_sigma=True)
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

    def coseno2(self):
        abcd, cov = curve_fit(self.cos2 , self.x, self.y, sigma=self.y_err, absolute_sigma=True)
        errores = np.sqrt(np.diag(cov))

        print('RESULTADOS DEL AJUSTE A COS^2')
        print(f'Parámetro multiplicativo: a={abcd[0]}+-{errores[0]}')
        print(f'Parámetro en el argumento 1: b={abcd[1]}+-{errores[1]}')
        print(f'Parámetro en el argumento 2: c={abcd[2]}+-{errores[2]}')
        print(f'Parámetro aditivo: d={abcd[3]}+-{errores[3]}')

        y_cos = self.cos2(self.x_ajuste, *abcd)

        self.puntos()
        self.graf.plot(self.x_ajuste, y_cos, label=f'Ajuste a $cos^2$', color='green')
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


#PARTE 1:
#Luz p (paralela al plano de incidencia):
th1_p = np.array([5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85])
I1_p = np.array([0.55,0.63,0.57,0.55,0.47,0.41,0.35,0.30,0.24,0.18,0.12,0.24,0.43,0.89,1.95,4.11,8.18])

I_err = 0.01
th_err = 0.5

P1p_fig, P1p_graf = plt.subplots(figsize=(8,5), dpi=200)
P1p_graf.errorbar(th1_p, I1_p, xerr=th_err, fmt='.',elinewidth=1, capsize=2, markersize=5, ecolor='red', label=r'Puntos experimentales')

P1p_graf.set_xlabel(r'$\theta (^\circ)$')
P1p_graf.set_ylabel(r'$I_r (\mu W)$')
P1p_graf.set_title(r'$I_r$ vs $\theta$ (Luz p)')
P1p_graf.title.set_fontsize(13)
P1p_graf.tick_params(axis='both', which='major', labelsize=9)
P1p_graf.grid(True, linestyle='--', alpha=0.7)

ticks_x = [10,20,30,40,50,55,60,70,80]
labels_x = ['10','20','30','40','50',r'$\theta_{B}$','60','70','80']
P1p_graf.yaxis.set_major_locator(ticker.MultipleLocator(1))

plt.xticks(ticks_x, labels_x)

#Añadimos una línea vertical punteada para resaltar la posición
plt.vlines(55, 0, 0.1, colors='gray', linestyles='dashed', alpha=0.6)

P1p_graf.legend(fontsize=8)
plt.show()
P1p_fig.savefig('Grafica Luz p.pdf')

# Parámetros
n1 = 1.0  # Aire (aproximadamente)
n2 = 1.5  # Según el valor n_ti de tu imagen
theta_i = np.linspace(0, np.pi/2, 500) # Ángulos de 0 a 90 grados

# Ley de Snell para encontrar el ángulo de transmisión (theta_t)
# n1 * sin(theta_i) = n2 * sin(theta_t)
sin_theta_t = (n1 / n2) * np.sin(theta_i)
theta_t = np.arcsin(sin_theta_t)

# Coeficiente de reflexión de Fresnel para polarización paralela (p)
r_parallel = (n2 * np.cos(theta_i) - n1 * np.cos(theta_t)) / \
             (n2 * np.cos(theta_i) + n1 * np.cos(theta_t))

# Reflectancia (R) y Transmitancia (T)
R_p = np.abs(r_parallel)**2
T_p = 1 - R_p

# Conversión a grados para el eje X
theta_deg = np.degrees(theta_i)

# Crear la gráfica
Trans = plt.figure(figsize=(8, 6))
plt.plot(theta_deg, T_p, 'k-', label=r'$T_{\parallel}$', linewidth=2)
plt.plot(theta_deg, R_p, 'k-', label=r'$R_{\parallel}$', linewidth=2)

# Formatear el gráfico para que se parezca al original
plt.tick_params(axis='both', which='major', labelsize=12)
plt.xlim(0, 90)
plt.ylim(0, 1.05)
plt.xticks([0, 30, 60, 90])
plt.yticks([0, 0.5, 1.0])

# Etiquetas y anotaciones
plt.title(r'$T_{\parallel}$,$R_{\parallel}$ vs $\theta_i$', fontsize=13)
plt.xlabel('$\\theta_i(^\circ)$', fontsize=12)
plt.ylabel('Reflectancia\ny Transmitancia', fontsize=12)
plt.text(45, 0.9, r'$T_{\parallel}$', fontsize=14)
plt.text(45, 0.1, r'$R_{\parallel}$', fontsize=14)
plt.text(15, 0.5, '$n_{prisma} = 1.5$', fontsize=12)
plt.text(56.3, 0.03, '$\\theta_B$', fontsize=12) # Ángulo de Brewster

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
Trans.savefig('Transmitancia.pdf')


#Luz s (perpendicular al plano de incidencia):
I1_s = np.array([2.85,2.92,3.02,3.16,3.54,3.79,4.18,4.37,5.05,7.49,9.50,12.0,17.0,25.0,40,58])
th1_s = np.array([5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80])

I_errs = np.array([0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.1,0.1,0.1,1,1])

P1s_fig, P1s_graf = plt.subplots(figsize=(8,5), dpi=200)
P1s_graf.errorbar(th1_s, I1_s, yerr=I_errs, xerr=th_err, fmt='.',elinewidth=1, capsize=2, markersize=5, ecolor='red', label=r'Puntos experimentales')

P1s_graf.set_xlabel(r'$\theta (^\circ)$')
P1s_graf.set_ylabel(r'$I_r (\mu W)$')
P1s_graf.set_title(r'$I_r$ vs $\theta$ (Luz s)')
P1s_graf.title.set_fontsize(13)
P1s_graf.tick_params(axis='both', which='major', labelsize=9)
P1s_graf.grid(True, linestyle='--', alpha=0.7)

P1s_graf.legend(fontsize=8)
plt.show()
P1s_fig.savefig('Grafica Luz s.pdf')

#Parte 2 (Ley de Malus):

th_m = np.array([60,70,80,90,100,110,120,130,140])*(2*np.pi)/360
I_m = np.array([0.014,0.012,0.0920,0.0620,0.0348,0.0150,0.0042,0.0038,0.0014])

malus = Experimento(th_m, I_m, th_err*(2*np.pi)/360, 0.0001, r'Ley de Malus', r'$\theta(rad)$', r'$I(mW)$')
malus.coseno2()
