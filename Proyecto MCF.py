import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import Funciones_MCF as MCF
from scipy.stats import kurtosis, skew ,norm, t

M7 = ['AAPL']


df_precios = MCF.obtener_datos(M7)
print(df_precios)

df_rendimientos = MCF.calcular_rendimientos(df_precios)
print(df_rendimientos)

promedio_rendi_diario = df_rendimientos['AAPL'].mean()
print("el promedioo de rendimiento diario es:", promedio_rendi_diario)

kurtosis = kurtosis(df_rendimientos['AAPL'])
skew = skew(df_rendimientos['AAPL'])
print("La kurtosis es:", kurtosis)
print("el sesgo es:", skew)

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm, t

def obtener_datos(stocks):
    '''Obtenemos los datos históricos de los precios de cierre de un activo'''
    df = yf.download(stocks, start="2010-01-01")['Close']
    return df

def calcular_rendimientos(df):
    '''Calculamos los rendimientos diarios'''
    return df.pct_change().dropna()

def calcular_var_parametrico(returns, alpha, distrib='normal'):
    '''Calcula VaR paramétrico bajo una distribución normal o t-Student'''
    mean = np.mean(returns)
    stdev = np.std(returns)
    
    if distrib == 'normal':
        VaR = norm.ppf(1-alpha, mean, stdev)
    elif distrib == 't':
        df_t = 10  # Grados de libertad, se puede ajustar según el caso
        VaR = t.ppf(1-alpha, df_t, mean, stdev)
    return VaR

def calcular_var_historico(returns, alpha):
    hVaR = returns.quantile(1-alpha)
    return hVaR

def calcular_var_montecarlo(returns, alpha, n_sims=100000):
    '''Calcula VaR usando simulaciones de Monte Carlo'''
    mean = np.mean(returns)
    stdev = np.std(returns)
    sim_returns = np.random.normal(mean, stdev, n_sims)
    return np.percentile(sim_returns, (1-alpha)*100)

def calcular_cvar(returns, hVaR):
    return returns[returns <= hVaR].mean()

# Datos y rendimientos
stocks = ['AAPL']
df_precios = obtener_datos(stocks)
df_rendimientos = calcular_rendimientos(df_precios)

# Intervalos de confianza
alpha_vals = [0.95, 0.975, 0.99]

# Crear un DataFrame vacío para almacenar los resultados
resultados_df = pd.DataFrame(columns=['VaR (Normal)', 'VaR (t-Student)', 'VaR (Histórico)', 'VaR (Monte Carlo)', 
                                      'CVaR (Normal)', 'CVaR (t-Student)', 'CVaR (Histórico)', 'CVaR (Monte Carlo)'])

# Cálculos para cada intervalo de confianza
for alpha in alpha_vals:
    # Calculamos VaR para cada método
    var_normal = calcular_var_parametrico(df_rendimientos['AAPL'], alpha, distrib='normal')
    var_t = calcular_var_parametrico(df_rendimientos['AAPL'], alpha, distrib='t')
    var_historico = calcular_var_historico(df_rendimientos['AAPL'], alpha)
    var_montecarlo = calcular_var_montecarlo(df_rendimientos['AAPL'], alpha)
    
    # Calculamos CVaR para cada método usando el VaR correspondiente
    cvar_normal = calcular_cvar(df_rendimientos['AAPL'], var_normal)
    cvar_t = calcular_cvar(df_rendimientos['AAPL'], var_t)
    cvar_historico = calcular_cvar(df_rendimientos['AAPL'], var_historico)
    cvar_montecarlo = calcular_cvar(df_rendimientos['AAPL'], var_montecarlo)
    
    # Almacenar los resultados en el DataFrame
    resultados_df.loc[f'{int(alpha*100)}% Confidence'] = [
        var_normal * 100, var_t * 100, var_historico * 100, var_montecarlo * 100,
        cvar_normal * 100, cvar_t * 100, cvar_historico * 100, cvar_montecarlo * 100
    ]

# Mostrar los resultados en una tabla
print("Resultados de VaR y CVaR por intervalo de confianza:")
print(resultados_df)

# Graficar el histograma de los rendimientos y los valores de VaR y CVaR
plt.figure(figsize=(10, 6))
n, bins, patches = plt.hist(df_rendimientos['AAPL'], bins=50, color='blue', alpha=0.7, label='Returns')

# Para el VaR histórico
for bin_left, bin_right, patch in zip(bins, bins[1:], patches):
    if bin_left < var_historico:
        patch.set_facecolor('red')

# Marcar los diferentes VaR y CVaR
plt.axvline(x=var_normal, color='skyblue', linestyle='--', label='VaR 95% (Normal)')
plt.axvline(x=var_montecarlo, color='grey', linestyle='--', label='VaR 95% (Monte Carlo)')
plt.axvline(x=var_historico, color='green', linestyle='--', label='VaR 95% (Histórico)')
plt.axvline(x=cvar_normal, color='purple', linestyle='-.', label='CVaR 95% (Normal)')

plt.title('Histograma de Rendimientos con VaR y CVaR')
plt.xlabel('Rendimientos')
plt.ylabel('Frecuencia')
plt.legend()

# Mostrar el gráfico
plt.show()

# Definimos el tamaño de la ventana móvil de 252 días
window_size = 252

# Cálculo de la media y desviación estándar de los retornos en la ventana móvil
rolling_mean = df_rendimientos['AAPL'].rolling(window=window_size).mean()
rolling_std = df_rendimientos['AAPL'].rolling(window=window_size).std()

# Cálculo del VaR paramétrico al 95% y 99% usando la distribución normal
VaR_95_rolling = norm.ppf(1-0.95, rolling_mean, rolling_std)
VaR_99_rolling = norm.ppf(1-0.99, rolling_mean, rolling_std)

# Cálculo del VaR histórico al 95% y 99%
VaR_95_rolling_hist = df_rendimientos['AAPL'].rolling(window=window_size).quantile(0.05)
VaR_99_rolling_hist = df_rendimientos['AAPL'].rolling(window=window_size).quantile(0.01)

# Cálculo del Expected Shortfall (ES) para el 95% y 99% en cada ventana
def calcular_ES(returns, var_rolling):
    return returns[returns <= var_rolling].mean()

# Para alpha = 0.95
ES_95_rolling = [calcular_ES(df_rendimientos['AAPL'][i-window_size:i], VaR_95_rolling[i]) for i in range(window_size, len(df_rendimientos))]

# Para alpha = 0.99
ES_99_rolling = [calcular_ES(df_rendimientos['AAPL'][i-window_size:i], VaR_99_rolling[i]) for i in range(window_size, len(df_rendimientos))]
# Cálculo del ES histórico al 95% y 99%
def calcular_ES_hist(returns, var_hist_rolling):
    return returns[returns <= var_hist_rolling].mean()

# Para alpha = 0.95 (histórico)
ES_95_rolling_hist = [calcular_ES_hist(df_rendimientos['AAPL'][i-window_size:i], VaR_95_rolling_hist[i]) for i in range(window_size, len(df_rendimientos))]

# Para alpha = 0.99 (histórico)
ES_99_rolling_hist = [calcular_ES_hist(df_rendimientos['AAPL'][i-window_size:i], VaR_99_rolling_hist[i]) for i in range(window_size, len(df_rendimientos))]

# Crear un DataFrame para almacenar los resultados de VaR y ES
rolling_results_df = pd.DataFrame({
    'Date': df_rendimientos.index[window_size:],
    'VaR_95_Rolling': VaR_95_rolling[window_size:],
    'VaR_99_Rolling': VaR_99_rolling[window_size:],
    'VaR_95_Rolling_Hist': VaR_95_rolling_hist[window_size:],
    'VaR_99_Rolling_Hist': VaR_99_rolling_hist[window_size:],
    'ES_95_Rolling': ES_95_rolling,
    'ES_99_Rolling': ES_99_rolling,
    'ES_95_Rolling_Hist': ES_95_rolling_hist,
    'ES_99_Rolling_Hist': ES_99_rolling_hist
})
rolling_results_df.set_index('Date', inplace=True)

# Graficar los resultados
plt.figure(figsize=(14, 7))

# Graficar los rendimientos diarios
plt.plot(df_rendimientos.index, df_rendimientos['AAPL'] * 100, label='Rendimientos Diarios (%)', color='blue', alpha=0.5)

# Graficar el VaR Rolling para 95% y 99% (paramétrico y histórico)
plt.plot(rolling_results_df.index, rolling_results_df['VaR_95_Rolling'] * 100, label='VaR 95% Rolling Paramétrico', color='red')
plt.plot(rolling_results_df.index, rolling_results_df['VaR_99_Rolling'] * 100, label='VaR 99% Rolling Paramétrico', color='green')
plt.plot(rolling_results_df.index, rolling_results_df['VaR_95_Rolling_Hist'] * 100, label='VaR 95% Rolling Histórico', color='orange')
plt.plot(rolling_results_df.index, rolling_results_df['VaR_99_Rolling_Hist'] * 100, label='VaR 99% Rolling Histórico', color='purple')

# Graficar el Expected Shortfall (ES) para 95% y 99% (paramétrico y histórico)
plt.plot(rolling_results_df.index, rolling_results_df['ES_95_Rolling'] * 100, label='ES 95% Rolling Paramétrico', color='cyan')
plt.plot(rolling_results_df.index, rolling_results_df['ES_99_Rolling'] * 100, label='ES 99% Rolling Paramétrico', color='magenta')
plt.plot(rolling_results_df.index, rolling_results_df['ES_95_Rolling_Hist'] * 100, label='ES 95% Rolling Histórico', color='pink')
plt.plot(rolling_results_df.index, rolling_results_df['ES_99_Rolling_Hist'] * 100, label='ES 99% Rolling Histórico', color='yellow')

# Título y etiquetas
plt.title('Rendimientos Diarios y VaR/ES Rolling Window (252 días)')
plt.xlabel('Fecha')
plt.ylabel('Valor (%)')

# Agregar la leyenda
plt.legend()

# Mostrar la gráfica
plt.tight_layout()
plt.show()


#E)

# Función para contar violaciones
def contar_violaciones(returns, risk_measure):
    """
    Cuenta cuántas veces los rendimientos reales fueron peores que la medida de riesgo estimada.
    
    Args:
        returns: Serie de rendimientos reales
        risk_measure: Serie de medidas de riesgo estimadas (VaR o ES)
        
    Returns:
        Número de violaciones y porcentaje de violaciones
    """
    violations = returns < risk_measure
    num_violations = violations.sum()
    violation_percentage = (num_violations / len(returns)) * 100
    return num_violations, violation_percentage

# Preparar los datos para el análisis de violaciones
# Necesitamos alinear los rendimientos con las medidas de riesgo estimadas
returns_for_test = df_rendimientos['AAPL'].iloc[window_size:]

# Crear DataFrame para resultados de violaciones
violation_results = pd.DataFrame(columns=['VaR 95% Paramétrico', 'VaR 99% Paramétrico',
                                         'VaR 95% Histórico', 'VaR 99% Histórico',
                                         'ES 95% Paramétrico', 'ES 99% Paramétrico',
                                         'ES 95% Histórico', 'ES 99% Histórico'],
                                index=['Número de violaciones', 'Porcentaje de violaciones'])

# Calcular violaciones para cada medida de riesgo
var_95_violations, var_95_percent = contar_violaciones(returns_for_test, rolling_results_df['VaR_95_Rolling'])
var_99_violations, var_99_percent = contar_violaciones(returns_for_test, rolling_results_df['VaR_99_Rolling'])
var_95_hist_violations, var_95_hist_percent = contar_violaciones(returns_for_test, rolling_results_df['VaR_95_Rolling_Hist'])
var_99_hist_violations, var_99_hist_percent = contar_violaciones(returns_for_test, rolling_results_df['VaR_99_Rolling_Hist'])

es_95_violations, es_95_percent = contar_violaciones(returns_for_test, rolling_results_df['ES_95_Rolling'])
es_99_violations, es_99_percent = contar_violaciones(returns_for_test, rolling_results_df['ES_99_Rolling'])
es_95_hist_violations, es_95_hist_percent = contar_violaciones(returns_for_test, rolling_results_df['ES_95_Rolling_Hist'])
es_99_hist_violations, es_99_hist_percent = contar_violaciones(returns_for_test, rolling_results_df['ES_99_Rolling_Hist'])

# Almacenar resultados
violation_results.loc['Número de violaciones'] = [
    var_95_violations, var_99_violations,
    var_95_hist_violations, var_99_hist_violations,
    es_95_violations, es_99_violations,
    es_95_hist_violations, es_99_hist_violations
]

violation_results.loc['Porcentaje de violaciones'] = [
    var_95_percent, var_99_percent,
    var_95_hist_percent, var_99_hist_percent,
    es_95_percent, es_99_percent,
    es_95_hist_percent, es_99_hist_percent
]

# Mostrar resultados
print("\nResultados de violaciones de VaR y ES:")
print(violation_results)

#F)

# Inciso (f) - VaR con volatilidad móvil y distribución normal (versión corregida)

# Definimos los niveles de significancia
alpha_1 = 0.05  # Para VaR 95%
alpha_2 = 0.01   # Para VaR 99%

# Calculamos la desviación estándar móvil de 252 días (volatilidad)
rolling_volatility = df_rendimientos['AAPL'].rolling(window=252).std()

# Calculamos los percentiles de la distribución normal
q_alpha1 = norm.ppf(alpha_1)
q_alpha2 = norm.ppf(alpha_2)

# Calculamos el VaR móvil para ambos niveles de confianza
VaR_95_vol_movil = q_alpha1 * rolling_volatility
VaR_99_vol_movil = q_alpha2 * rolling_volatility

# Graficamos los resultados
plt.figure(figsize=(14, 7))

# Graficar los rendimientos diarios
plt.plot(df_rendimientos.index, df_rendimientos['AAPL'] * 100, 
         label='Rendimientos Diarios (%)', color='blue', alpha=0.3)

# Graficar el VaR con volatilidad móvil
plt.plot(VaR_95_vol_movil.index, VaR_95_vol_movil * 100, 
         label='VaR 95% con Volatilidad Móvil', color='red')
plt.plot(VaR_99_vol_movil.index, VaR_99_vol_movil * 100, 
         label='VaR 99% con Volatilidad Móvil', color='darkred')

# Configuración del gráfico
plt.title('Rendimientos Diarios y VaR con Volatilidad Móvil (252 días)')
plt.xlabel('Fecha')
plt.ylabel('Valor (%)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Función modificada para contar violaciones que alinea los índices
def contar_violaciones_alineadas(returns, risk_measure):
    """
    Versión modificada que alinea los índices antes de comparar
    
    Args:
        returns: Serie de rendimientos reales
        risk_measure: Serie de medidas de riesgo estimadas (VaR o ES)
        
    Returns:
        Número de violaciones y porcentaje de violaciones
    """
    # Convertimos a DataFrame para hacer merge
    df_returns = returns.to_frame(name='returns')
    df_risk = risk_measure.to_frame(name='risk')
    
    # Unimos las series por índice
    merged = pd.merge(df_returns, df_risk, left_index=True, right_index=True)
    
    # Calculamos violaciones
    violations = merged['returns'] < merged['risk']
    num_violations = violations.sum()
    violation_percentage = (num_violations / len(merged)) * 100
    
    return num_violations, violation_percentage

# Preparamos los datos para el análisis de violaciones
# Aseguramos que usemos el mismo rango de fechas
returns_for_test_vol = df_rendimientos['AAPL'].loc[VaR_95_vol_movil.dropna().index]

# Calculamos violaciones con la función modificada
vol_movil_violations_95, vol_movil_percent_95 = contar_violaciones_alineadas(
    returns_for_test_vol, VaR_95_vol_movil.dropna())
vol_movil_violations_99, vol_movil_percent_99 = contar_violaciones_alineadas(
    returns_for_test_vol, VaR_99_vol_movil.dropna())

# Mostramos los resultados de violaciones
print("\nResultados de violaciones para VaR con volatilidad móvil:")
print(f"VaR 95% con volatilidad móvil: {vol_movil_violations_95} violaciones ({vol_movil_percent_95:.2f}%)")
print(f"VaR 99% con volatilidad móvil: {vol_movil_violations_99} violaciones ({vol_movil_percent_99:.2f}%)")

# Evaluación de la calidad de las estimaciones
print("\nEvaluación de la calidad de las estimaciones con volatilidad móvil:")
print(f"Para VaR 95%: esperado ~5% de violaciones, obtenido {vol_movil_percent_95:.2f}%")
print(f"Para VaR 99%: esperado ~1% de violaciones, obtenido {vol_movil_percent_99:.2f}%")
print("Nota: Un buen modelo debería tener menos del 2.5% de violaciones en general.")