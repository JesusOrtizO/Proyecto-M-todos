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
print("el promedio de rendimiento diario es:", promedio_rendi_diario)

kurtosis = kurtosis(df_rendimientos['AAPL'])
skew = skew(df_rendimientos['AAPL'])
print("La kurtosis es:", kurtosis)
print("el sesgo es:", skew)

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
    '''Calcula VaR histórico usando percentil'''
    return returns.quantile(1-alpha)

def calcular_var_montecarlo(returns, alpha, n_sims=100000):
    '''Calcula VaR usando simulaciones de Monte Carlo'''
    mean = np.mean(returns)
    stdev = np.std(returns)
    sim_returns = np.random.normal(mean, stdev, n_sims)
    return np.percentile(sim_returns, (1-alpha)*100)

def calcular_cvar(returns, var):
    '''Calcula el CVaR (Conditional VaR)'''
    return returns[returns <= var].mean()

# Datos y rendimientos
stocks = ['AAPL']
df_precios = obtener_datos(stocks)
df_rendimientos = calcular_rendimientos(df_precios)

# Intervalos de confianza
alpha_vals = [0.95, 0.975, 0.99]

# Cálculos
resultados = {}
for alpha in alpha_vals:
    var_normal = calcular_var_parametrico(df_rendimientos['AAPL'], alpha, distrib='normal')
    var_t = calcular_var_parametrico(df_rendimientos['AAPL'], alpha, distrib='t')
    var_historico = calcular_var_historico(df_rendimientos['AAPL'], alpha)
    var_montecarlo = calcular_var_montecarlo(df_rendimientos['AAPL'], alpha)
    
    cvar_normal = calcular_cvar(df_rendimientos['AAPL'], var_normal)
    cvar_t = calcular_cvar(df_rendimientos['AAPL'], var_t)
    cvar_historico = calcular_cvar(df_rendimientos['AAPL'], var_historico)
    cvar_montecarlo = calcular_cvar(df_rendimientos['AAPL'], var_montecarlo)
    
    resultados[alpha] = {
        'VaR_Normal': var_normal,
        'VaR_t': var_t,
        'VaR_Historico': var_historico,
        'VaR_MonteCarlo': var_montecarlo,
        'CVaR_Normal': cvar_normal,
        'CVaR_t': cvar_t,
        'CVaR_Historico': cvar_historico,
        'CVaR_MonteCarlo': cvar_montecarlo
    }

# Imprimir resultados
for alpha, result in resultados.items():
    print(f"Resultados para alpha = {alpha*100}%:")
    for key, value in result.items():
        print(f"{key}: {value:.4f}")
    print("\n")

# Graficar los histogramas y marcas de VaR y CVaR
plt.figure(figsize=(10, 6))
n, bins, patches = plt.hist(df_rendimientos['AAPL'], bins=50, color='blue', alpha=0.7, label='Returns')

# Identificar los bins a la izquierda de los VaR y CVaR y colorearlos
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