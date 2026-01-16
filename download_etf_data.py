"""
Script para descargar datos de ETFs de Yahoo Finance y calcular retornos diarios.
Cubre 14 ETFs que representan diferentes regiones y sectores del mundo.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.stattools import durbin_watson
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo de gráficos
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('ggplot')
sns.set_palette("husl")

# Definir los ETFs seleccionados (cubren diferentes regiones, bonos y materias primas)
ETFS = {
    # ACCIONES - EEUU
    'SPY': 'S&P 500 (EEUU)',
    'QQQ': 'Nasdaq 100 (Tecnología EEUU)',
    'IWM': 'Russell 2000 (Pequeñas Empresas EEUU)',
    # ACCIONES - Internacional
    'EFA': 'EAFE - Europa, Asia, Lejano Oriente',
    'EEM': 'Mercados Emergentes',
    'VGK': 'Europa (Vanguard)',
    'VPL': 'Asia-Pacífico (Vanguard)',
    # BONOS
    'LQD': 'Bonos Corporativos Investment Grade',
    'HYG': 'Bonos High Yield (Basura)',
    'TLT': 'Bonos del Tesoro EEUU 20+ años',
    'AGG': 'Total Bond Market (Bonos Agregados)',
    # TASA LIBRE DE RIESGO (RF)
    'SHY': 'Tasa Libre de Riesgo - Treasury 1-3 años (iShares)',
    # Alternativa: 'BIL': 'Tasa Libre de Riesgo - T-Bill 1-3 meses (SPDR)'
    # MATERIAS PRIMAS
    'GLD': 'Oro (SPDR Gold Trust)',
    'SLV': 'Plata (iShares Silver Trust)',
    'USO': 'Petróleo (United States Oil Fund)'
}

def download_etf_returns(etf_symbols, years=10):
    """
    Descarga datos históricos de ETFs y calcula retornos diarios.
    
    Parameters:
    -----------
    etf_symbols : dict
        Diccionario con símbolos de ETFs y sus descripciones
    years : int
        Número de años de datos históricos a descargar (default: 10)
    
    Returns:
    --------
    dict
        Diccionario con nombre del ETF y sus retornos diarios
    """
    # Calcular fecha de inicio (hace 'years' años)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    
    returns_dict = {}
    
    print(f"Descargando datos de {len(etf_symbols)} ETFs desde {start_date.date()} hasta {end_date.date()}...")
    print("-" * 80)
    
    for symbol, description in etf_symbols.items():
        try:
            print(f"Descargando {symbol} - {description}...")
            
            # Descargar datos históricos
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                print(f"  ⚠️  No se encontraron datos para {symbol}")
                continue
            
            # Calcular retornos diarios (porcentaje)
            # Retorno = (Precio_t - Precio_{t-1}) / Precio_{t-1} * 100
            data['Returns'] = data['Close'].pct_change() * 100
            
            # Eliminar el primer valor (NaN)
            returns = data['Returns'].dropna()
            
            # Guardar en el diccionario
            returns_dict[symbol] = returns
            
            print(f"  ✓ {symbol}: {len(returns)} días de retornos calculados")
            print(f"    Período: {returns.index[0].date()} a {returns.index[-1].date()}")
            print(f"    Retorno promedio diario: {returns.mean():.4f}%")
            print(f"    Volatilidad diaria: {returns.std():.4f}%")
            print()
            
        except Exception as e:
            print(f"  ✗ Error descargando {symbol}: {str(e)}")
            print()
    
    print("-" * 80)
    print(f"✓ Proceso completado. {len(returns_dict)} ETFs descargados exitosamente.")
    
    return returns_dict

def calculate_excess_returns(returns_dict, rf_symbol='SHY'):
    """
    Calcula retornos en exceso (excess returns) restando la tasa libre de riesgo.
    
    Parameters:
    -----------
    returns_dict : dict
        Diccionario con retornos de ETFs
    rf_symbol : str
        Símbolo del ETF a usar como tasa libre de riesgo (default: 'SHY')
    
    Returns:
    --------
    dict
        Diccionario con retornos en exceso para cada ETF
    """
    if rf_symbol not in returns_dict:
        print(f"⚠️  {rf_symbol} no encontrado en returns_dict. No se calcularán retornos en exceso.")
        return None
    
    rf_returns = returns_dict[rf_symbol]
    excess_returns_dict = {}
    
    print(f"\nCalculando retornos en exceso usando {rf_symbol} como tasa libre de riesgo...")
    
    for symbol, returns in returns_dict.items():
        if symbol == rf_symbol:
            continue  # No calcular exceso para el propio RF
        
        # Alinear fechas
        aligned_data = pd.DataFrame({
            'asset': returns,
            'rf': rf_returns
        }).dropna()
        
        if len(aligned_data) > 0:
            excess_returns = aligned_data['asset'] - aligned_data['rf']
            excess_returns_dict[symbol] = excess_returns
            print(f"  ✓ {symbol}: {len(excess_returns)} observaciones")
    
    return excess_returns_dict

def save_returns_to_dict_file(returns_dict, data_dir='data', filename='etf_returns_dict.pkl'):
    """
    Guarda el diccionario de retornos en un archivo pickle.
    
    Parameters:
    -----------
    returns_dict : dict
        Diccionario con retornos de ETFs
    data_dir : str
        Directorio donde guardar los archivos
    filename : str
        Nombre del archivo donde guardar
    """
    import pickle
    
    # Crear directorio si no existe
    os.makedirs(data_dir, exist_ok=True)
    
    filepath = os.path.join(data_dir, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(returns_dict, f)
    print(f"\n✓ Diccionario guardado en {filepath}")

def plot_all_etfs_comparison(returns_dict, data_dir='data'):
    """
    Crea gráficos comparativos de todos los ETFs.
    
    Parameters:
    -----------
    returns_dict : dict
        Diccionario con retornos de ETFs
    data_dir : str
        Directorio donde guardar los gráficos
    """
    # Crear directorio si no existe
    os.makedirs(data_dir, exist_ok=True)
    
    # Convertir diccionario a DataFrame para facilitar el análisis
    returns_df = pd.DataFrame(returns_dict)
    
    # Alinear fechas (usar intersección de fechas comunes)
    returns_df = returns_df.dropna()
    
    print("\n" + "=" * 80)
    print("GENERANDO GRÁFICOS COMPARATIVOS")
    print("=" * 80)
    
    # 1. Gráfico 1: Series temporales de retornos acumulados
    print("\n1. Generando gráfico de retornos acumulados...")
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Calcular retornos acumulados (normalizados a 100)
    cumulative_returns = (1 + returns_df / 100).cumprod() * 100
    
    for col in cumulative_returns.columns:
        ax.plot(cumulative_returns.index, cumulative_returns[col], 
                label=col, linewidth=1.5, alpha=0.8)
    
    ax.set_title('Retornos Acumulados de ETFs (Base 100)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Fecha', fontsize=12)
    ax.set_ylabel('Valor Acumulado (Base 100)', fontsize=12)
    ax.legend(loc='best', ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filepath = os.path.join(data_dir, '1_retornos_acumulados.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"   ✓ Guardado: {filepath}")
    plt.close()
    
    # 2. Gráfico 2: Matriz de correlación
    print("2. Generando matriz de correlación...")
    fig, ax = plt.subplots(figsize=(12, 10))
    
    correlation_matrix = returns_df.corr()
    
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, vmin=-1, vmax=1, square=True, 
                linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    
    ax.set_title('Matriz de Correlación entre ETFs', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    filepath = os.path.join(data_dir, '2_matriz_correlacion.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"   ✓ Guardado: {filepath}")
    plt.close()
    
    # 3. Gráfico 3: Distribución de retornos (boxplot)
    print("3. Generando boxplot de distribuciones...")
    fig, ax = plt.subplots(figsize=(14, 8))
    
    returns_df_melted = returns_df.melt(var_name='ETF', value_name='Retorno')
    sns.boxplot(data=returns_df_melted, x='ETF', y='Retorno', ax=ax)
    
    ax.set_title('Distribución de Retornos Diarios por ETF', fontsize=16, fontweight='bold')
    ax.set_xlabel('ETF', fontsize=12)
    ax.set_ylabel('Retorno Diario (%)', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    filepath = os.path.join(data_dir, '3_distribucion_retornos.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"   ✓ Guardado: {filepath}")
    plt.close()
    
    # 4. Gráfico 4: Scatter plots de todos contra todos (matriz de dispersión)
    print("4. Generando matriz de scatter plots...")
    fig = sns.pairplot(returns_df, diag_kind='kde', plot_kws={'alpha': 0.3, 's': 10})
    fig.fig.suptitle('Matriz de Scatter Plots: Retornos Diarios entre ETFs', 
                     fontsize=16, fontweight='bold', y=1.02)
    
    filepath = os.path.join(data_dir, '4_matriz_scatter.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"   ✓ Guardado: {filepath}")
    plt.close()
    
    # 5. Gráfico 5: Retorno vs Volatilidad (Risk-Return)
    print("5. Generando gráfico Risk-Return...")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calcular estadísticas anualizadas
    # Los retornos están en porcentaje, primero convertir a decimales
    returns_decimal = returns_df / 100
    annual_returns = returns_decimal.mean() * 252 * 100  # Retorno anualizado (%)
    annual_vol = returns_decimal.std() * np.sqrt(252) * 100  # Volatilidad anualizada (%)
    
    scatter = ax.scatter(annual_vol, annual_returns, s=200, alpha=0.6, 
                        c=range(len(annual_returns)), cmap='viridis')
    
    # Etiquetar cada punto
    for i, symbol in enumerate(annual_returns.index):
        ax.annotate(symbol, (annual_vol[i], annual_returns[i]), 
                   fontsize=10, ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Volatilidad Anualizada (%)', fontsize=12)
    ax.set_ylabel('Retorno Anualizado (%)', fontsize=12)
    ax.set_title('Risk-Return: Retorno vs Volatilidad Anualizada', 
                fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Índice ETF')
    plt.tight_layout()
    
    filepath = os.path.join(data_dir, '5_risk_return.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"   ✓ Guardado: {filepath}")
    plt.close()
    
    # 6. Gráfico 6: Series temporales de retornos diarios (todas superpuestas)
    print("6. Generando series temporales de retornos diarios...")
    fig, ax = plt.subplots(figsize=(16, 8))
    
    for col in returns_df.columns:
        ax.plot(returns_df.index, returns_df[col], label=col, alpha=0.6, linewidth=0.8)
    
    ax.set_title('Retornos Diarios de Todos los ETFs', fontsize=16, fontweight='bold')
    ax.set_xlabel('Fecha', fontsize=12)
    ax.set_ylabel('Retorno Diario (%)', fontsize=12)
    ax.legend(loc='best', ncol=3, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    plt.tight_layout()
    
    filepath = os.path.join(data_dir, '6_retornos_diarios.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"   ✓ Guardado: {filepath}")
    plt.close()
    
    print("\n✓ Todos los gráficos generados exitosamente en el directorio 'data/'")

def analyze_time_series_properties(returns_dict, market_factor='SPY'):
    """
    Analiza las propiedades estadísticas y de series temporales de cada ETF.
    
    Parameters:
    -----------
    returns_dict : dict
        Diccionario con retornos de ETFs
    market_factor : str
        Símbolo del ETF a usar como factor de mercado (default: 'SPY')
    
    Returns:
    --------
    pd.DataFrame
        DataFrame con todas las propiedades analizadas para cada ETF
    """
    print("\n" + "=" * 80)
    print("ANÁLISIS DE PROPIEDADES ESTADÍSTICAS Y DE SERIES TEMPORALES")
    print("=" * 80)
    
    # Convertir a DataFrame para facilitar cálculos
    returns_df = pd.DataFrame(returns_dict)
    returns_df = returns_df.dropna()
    
    # Obtener retornos del factor de mercado si existe
    market_returns = None
    if market_factor in returns_df.columns:
        market_returns = returns_df[market_factor]
    
    properties_list = []
    
    for symbol in returns_df.columns:
        print(f"\nAnalizando {symbol}...")
        returns = returns_df[symbol].dropna()
        
        if len(returns) < 100:  # Necesitamos suficientes datos
            print(f"  ⚠️  Datos insuficientes para {symbol}")
            continue
        
        properties = {'ETF': symbol}
        
        # ========== 1. ESTACIONARIEDAD Y RAÍZ UNITARIA ==========
        try:
            # Test ADF (Augmented Dickey-Fuller)
            adf_result = adfuller(returns, autolag='AIC')
            properties['adf_statistic'] = adf_result[0]
            properties['adf_pvalue'] = adf_result[1]
            properties['adf_stationary'] = adf_result[1] < 0.05
            
            # Test KPSS
            try:
                kpss_result = kpss(returns, regression='ct', nlags='auto')
                properties['kpss_statistic'] = kpss_result[0]
                properties['kpss_pvalue'] = kpss_result[1]
                properties['kpss_stationary'] = kpss_result[1] > 0.05
            except:
                properties['kpss_statistic'] = np.nan
                properties['kpss_pvalue'] = np.nan
                properties['kpss_stationary'] = np.nan
        except Exception as e:
            print(f"    ⚠️  Error en tests de estacionariedad: {e}")
            properties['adf_statistic'] = np.nan
            properties['adf_pvalue'] = np.nan
            properties['adf_stationary'] = np.nan
            properties['kpss_statistic'] = np.nan
            properties['kpss_pvalue'] = np.nan
            properties['kpss_stationary'] = np.nan
        
        # ========== 2. AUTOCORRELACIÓN Y AUTOCORRELACIÓN PARCIAL ==========
        try:
            # ACF y PACF para lags 1, 5, 10, 20
            acf_values = acf(returns, nlags=20, fft=True)
            pacf_values = pacf(returns, nlags=20)
            
            properties['acf_lag1'] = acf_values[1] if len(acf_values) > 1 else np.nan
            properties['acf_lag5'] = acf_values[5] if len(acf_values) > 5 else np.nan
            properties['acf_lag10'] = acf_values[10] if len(acf_values) > 10 else np.nan
            properties['acf_lag20'] = acf_values[20] if len(acf_values) > 20 else np.nan
            
            properties['pacf_lag1'] = pacf_values[1] if len(pacf_values) > 1 else np.nan
            properties['pacf_lag5'] = pacf_values[5] if len(pacf_values) > 5 else np.nan
            properties['pacf_lag10'] = pacf_values[10] if len(pacf_values) > 10 else np.nan
            properties['pacf_lag20'] = pacf_values[20] if len(pacf_values) > 20 else np.nan
            
            # Test de Ljung-Box para autocorrelación serial
            try:
                lb_result = acorr_ljungbox(returns, lags=10, return_df=True)
                properties['ljung_box_pvalue'] = lb_result['lb_pvalue'].iloc[-1]
                properties['has_autocorrelation'] = lb_result['lb_pvalue'].iloc[-1] < 0.05
            except:
                properties['ljung_box_pvalue'] = np.nan
                properties['has_autocorrelation'] = np.nan
        except Exception as e:
            print(f"    ⚠️  Error en análisis de autocorrelación: {e}")
            properties['acf_lag1'] = np.nan
            properties['acf_lag5'] = np.nan
            properties['acf_lag10'] = np.nan
            properties['acf_lag20'] = np.nan
            properties['pacf_lag1'] = np.nan
            properties['pacf_lag5'] = np.nan
            properties['pacf_lag10'] = np.nan
            properties['pacf_lag20'] = np.nan
            properties['ljung_box_pvalue'] = np.nan
            properties['has_autocorrelation'] = np.nan
        
        # ========== 3. CLUSTERING DE VOLATILIDAD Y PERSISTENCIA ==========
        try:
            # Calcular volatilidad (desviación estándar de retornos)
            # Los retornos están en porcentaje, convertir a decimales para cálculos
            returns_decimal = returns / 100
            volatility_daily_decimal = returns_decimal.std()
            properties['volatility_daily'] = volatility_daily_decimal * 100  # Volatilidad diaria (%)
            properties['volatility_annualized'] = volatility_daily_decimal * np.sqrt(252) * 100  # Volatilidad anualizada (%)
            
            # ARCH effects: correlación de retornos al cuadrado (proxy de clustering)
            returns_squared = returns ** 2
            acf_squared = acf(returns_squared, nlags=5, fft=True)
            properties['arch_effect_lag1'] = acf_squared[1] if len(acf_squared) > 1 else np.nan
            properties['arch_effect_lag5'] = acf_squared[5] if len(acf_squared) > 5 else np.nan
            properties['has_volatility_clustering'] = acf_squared[1] > 0.1 if len(acf_squared) > 1 else np.nan
            
            # Persistencia de volatilidad (ratio de varianza de largo plazo vs corto plazo)
            # Dividir en dos períodos
            mid_point = len(returns) // 2
            vol_short = returns.iloc[:mid_point].std()
            vol_long = returns.iloc[mid_point:].std()
            properties['volatility_persistence_ratio'] = vol_long / vol_short if vol_short > 0 else np.nan
        except Exception as e:
            print(f"    ⚠️  Error en análisis de volatilidad: {e}")
            properties['volatility_daily'] = np.nan
            properties['volatility_annualized'] = np.nan
            properties['arch_effect_lag1'] = np.nan
            properties['arch_effect_lag5'] = np.nan
            properties['has_volatility_clustering'] = np.nan
            properties['volatility_persistence_ratio'] = np.nan
        
        # ========== 4. MOMENTOS SUPERIORES (SKEWNESS, KURTOSIS) ==========
        try:
            properties['skewness'] = stats.skew(returns)
            properties['kurtosis'] = stats.kurtosis(returns)
            properties['excess_kurtosis'] = properties['kurtosis'] - 3  # Exceso de curtosis
            properties['is_leptokurtic'] = properties['kurtosis'] > 3  # Colas pesadas
            properties['is_negatively_skewed'] = properties['skewness'] < 0  # Sesgo negativo
        except Exception as e:
            print(f"    ⚠️  Error en análisis de momentos: {e}")
            properties['skewness'] = np.nan
            properties['kurtosis'] = np.nan
            properties['excess_kurtosis'] = np.nan
            properties['is_leptokurtic'] = np.nan
            properties['is_negatively_skewed'] = np.nan
        
        # ========== 5. CAMBIOS DE RÉGIMEN Y NO LINEALIDADES ==========
        try:
            # Test de cambio estructural: dividir en dos períodos y comparar medias
            mid_point = len(returns) // 2
            period1 = returns.iloc[:mid_point]
            period2 = returns.iloc[mid_point:]
            
            # Test t para diferencia de medias
            t_stat, p_value = stats.ttest_ind(period1, period2)
            properties['regime_shift_tstat'] = t_stat
            properties['regime_shift_pvalue'] = p_value
            properties['has_regime_shift'] = p_value < 0.05
            
            # Ratio de volatilidad entre períodos
            vol1 = period1.std()
            vol2 = period2.std()
            properties['regime_volatility_ratio'] = vol2 / vol1 if vol1 > 0 else np.nan
            
            # Test de no linealidad: correlación entre retornos y retornos al cuadrado
            returns_squared = returns ** 2
            corr_linear = returns.corr(returns_squared)
            properties['nonlinearity_correlation'] = corr_linear
            properties['has_nonlinearity'] = abs(corr_linear) > 0.1
        except Exception as e:
            print(f"    ⚠️  Error en análisis de régimen: {e}")
            properties['regime_shift_tstat'] = np.nan
            properties['regime_shift_pvalue'] = np.nan
            properties['has_regime_shift'] = np.nan
            properties['regime_volatility_ratio'] = np.nan
            properties['nonlinearity_correlation'] = np.nan
            properties['has_nonlinearity'] = np.nan
        
        # ========== 6. EXPOSICIÓN A FACTORES DE MERCADO ==========
        try:
            if market_returns is not None and symbol != market_factor:
                # CAPM: Beta y Alpha
                # Alinear fechas
                aligned_data = pd.DataFrame({
                    'market': market_returns,
                    'asset': returns
                }).dropna()
                
                if len(aligned_data) > 50:
                    # Regresión CAPM: R_asset = alpha + beta * R_market + epsilon
                    X = aligned_data['market'].values
                    y = aligned_data['asset'].values
                    
                    # Añadir constante para intercepto
                    X_with_const = np.column_stack([np.ones(len(X)), X])
                    
                    try:
                        coefficients = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
                        alpha = coefficients[0]  # Intercepto
                        beta = coefficients[1]  # Pendiente
                        properties['market_beta'] = beta
                        properties['market_alpha'] = alpha
                        
                        # Correlación con mercado
                        properties['market_correlation'] = aligned_data['market'].corr(aligned_data['asset'])
                        
                        # R-squared
                        y_pred = alpha + beta * X
                        ss_res = np.sum((y - y_pred) ** 2)
                        ss_tot = np.sum((y - np.mean(y)) ** 2)
                        properties['market_rsquared'] = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
                    except:
                        properties['market_beta'] = np.nan
                        properties['market_alpha'] = np.nan
                        properties['market_correlation'] = np.nan
                        properties['market_rsquared'] = np.nan
                else:
                    properties['market_beta'] = np.nan
                    properties['market_alpha'] = np.nan
                    properties['market_correlation'] = np.nan
                    properties['market_rsquared'] = np.nan
            else:
                properties['market_beta'] = 1.0 if symbol == market_factor else np.nan
                properties['market_alpha'] = 0.0 if symbol == market_factor else np.nan
                properties['market_correlation'] = 1.0 if symbol == market_factor else np.nan
                properties['market_rsquared'] = 1.0 if symbol == market_factor else np.nan
        except Exception as e:
            print(f"    ⚠️  Error en análisis de factores de mercado: {e}")
            properties['market_beta'] = np.nan
            properties['market_alpha'] = np.nan
            properties['market_correlation'] = np.nan
            properties['market_rsquared'] = np.nan
        
        # ========== ESTADÍSTICAS BÁSICAS ==========
        # Los retornos están en porcentaje
        properties['mean_return_daily'] = returns.mean()  # Retorno diario (%)
        # Convertir a decimales para anualizar correctamente
        returns_decimal = returns / 100
        properties['mean_return_annualized'] = returns_decimal.mean() * 252 * 100  # Retorno anualizado (%)
        properties['n_observations'] = len(returns)
        properties['min_return'] = returns.min()
        properties['max_return'] = returns.max()
        properties['median_return'] = returns.median()
        
        properties_list.append(properties)
        print(f"  ✓ {symbol} analizado")
    
    # Crear DataFrame
    properties_df = pd.DataFrame(properties_list)
    
    # Reordenar columnas para mejor visualización
    column_order = [
        'ETF', 'n_observations',
        'mean_return_daily', 'mean_return_annualized', 'volatility_daily', 'volatility_annualized',
        'min_return', 'max_return', 'median_return',
        'skewness', 'kurtosis', 'excess_kurtosis', 'is_leptokurtic', 'is_negatively_skewed',
        'adf_statistic', 'adf_pvalue', 'adf_stationary',
        'kpss_statistic', 'kpss_pvalue', 'kpss_stationary',
        'acf_lag1', 'acf_lag5', 'acf_lag10', 'acf_lag20',
        'pacf_lag1', 'pacf_lag5', 'pacf_lag10', 'pacf_lag20',
        'ljung_box_pvalue', 'has_autocorrelation',
        'arch_effect_lag1', 'arch_effect_lag5', 'has_volatility_clustering', 'volatility_persistence_ratio',
        'regime_shift_tstat', 'regime_shift_pvalue', 'has_regime_shift', 'regime_volatility_ratio',
        'nonlinearity_correlation', 'has_nonlinearity',
        'market_beta', 'market_alpha', 'market_correlation', 'market_rsquared'
    ]
    
    # Mantener solo las columnas que existen
    available_columns = [col for col in column_order if col in properties_df.columns]
    other_columns = [col for col in properties_df.columns if col not in column_order]
    final_order = available_columns + other_columns
    
    properties_df = properties_df[final_order]
    
    return properties_df

def main():
    """Función principal"""
    # Directorio para guardar datos
    data_dir = 'data'
    
    print("=" * 80)
    print("DESCARGA DE DATOS DE ETFs - YAHOO FINANCE")
    print("=" * 80)
    print()
    
    # Descargar datos y calcular retornos
    returns_dict = download_etf_returns(ETFS, years=20)
    
    # Mostrar resumen del diccionario
    print("\n" + "=" * 80)
    print("RESUMEN DEL DICCIONARIO DE RETORNOS")
    print("=" * 80)
    print(f"\nEstructura del diccionario:")
    print(f"  - Claves: {list(returns_dict.keys())}")
    print(f"  - Tipo de valores: {type(returns_dict[list(returns_dict.keys())[0]])}")
    print(f"\nEjemplo de datos para {list(returns_dict.keys())[0]}:")
    print(returns_dict[list(returns_dict.keys())[0]].head(10))
    
    # Crear directorio si no existe
    os.makedirs(data_dir, exist_ok=True)
    
    # Guardar el diccionario
    save_returns_to_dict_file(returns_dict, data_dir, 'etf_returns_dict.pkl')
    
    # También guardar como CSV para cada ETF
    print(f"\nGuardando datos individuales en archivos CSV en '{data_dir}/'...")
    for symbol, returns in returns_dict.items():
        filepath = os.path.join(data_dir, f'{symbol}_returns.csv')
        returns.to_csv(filepath, header=['Returns'])
    print("✓ Archivos CSV guardados")
    
    # Generar gráficos comparativos
    plot_all_etfs_comparison(returns_dict, data_dir)
    
    # Análisis de propiedades estadísticas y de series temporales
    properties_df = analyze_time_series_properties(returns_dict, market_factor='SPY')
    
    # Guardar el DataFrame de propiedades
    properties_filepath = os.path.join(data_dir, 'etf_properties_analysis.csv')
    properties_df.to_csv(properties_filepath, index=False)
    print(f"\n✓ Propiedades guardadas en {properties_filepath}")
    
    # Imprimir el DataFrame completo
    print("\n" + "=" * 80)
    print("RESUMEN DE PROPIEDADES ESTADÍSTICAS Y DE SERIES TEMPORALES")
    print("=" * 80)
    print("\nDataFrame completo de propiedades:")
    print(properties_df.to_string())
    
    # Imprimir resumen por categorías
    print("\n" + "=" * 80)
    print("RESUMEN POR CATEGORÍAS")
    print("=" * 80)
    
    print("\n1. ESTACIONARIEDAD:")
    print(properties_df[['ETF', 'adf_stationary', 'kpss_stationary', 'adf_pvalue', 'kpss_pvalue']].to_string())
    
    print("\n2. AUTOCORRELACIÓN:")
    print(properties_df[['ETF', 'acf_lag1', 'pacf_lag1', 'has_autocorrelation', 'ljung_box_pvalue']].to_string())
    
    print("\n3. VOLATILIDAD Y CLUSTERING:")
    print(properties_df[['ETF', 'volatility_annualized', 'has_volatility_clustering', 'arch_effect_lag1', 'volatility_persistence_ratio']].to_string())
    
    print("\n4. MOMENTOS SUPERIORES:")
    print(properties_df[['ETF', 'skewness', 'kurtosis', 'excess_kurtosis', 'is_leptokurtic', 'is_negatively_skewed']].to_string())
    
    print("\n5. CAMBIOS DE RÉGIMEN:")
    print(properties_df[['ETF', 'has_regime_shift', 'regime_shift_pvalue', 'regime_volatility_ratio', 'has_nonlinearity']].to_string())
    
    print("\n6. EXPOSICIÓN A FACTORES DE MERCADO:")
    print(properties_df[['ETF', 'market_beta', 'market_alpha', 'market_correlation', 'market_rsquared']].to_string())
    
    return returns_dict, properties_df

if __name__ == "__main__":
    # Ejecutar el script
    etf_returns, properties_df = main()
    
    # Mostrar el diccionario completo (opcional, puede ser muy largo)
    print("\n" + "=" * 80)
    print("DICCIONARIO COMPLETO DE RETORNOS")
    print("=" * 80)
    for symbol, returns in etf_returns.items():
        print(f"\n{symbol}:")
        print(f"  Tipo: {type(returns)}")
        print(f"  Forma: {returns.shape}")
        print(f"  Primeros 5 valores:")
        print(returns.head())

