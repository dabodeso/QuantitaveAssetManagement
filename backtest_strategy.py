"""
Script para hacer backtesting del sistema completo de predicción y optimización.

Este script:
1. Simula el trading histórico usando el modelo entrenado
2. Re-balancea el portafolio periódicamente basado en predicciones
3. Calcula métricas de performance (Sharpe, retorno, drawdown, etc.)
4. Compara con benchmarks (SPY, 1/N, etc.)
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Importar funciones de otros módulos
from train_sharpe_predictor import load_ml_dataset, prepare_data, split_data_temporal
from optimize_portfolio import load_models, predict_sharpe_ratios, optimize_portfolio
import os

def load_returns_data(data_dir='data'):
    """Carga retornos históricos de ETFs"""
    returns_file = os.path.join(data_dir, 'etf_returns_dict.pkl')
    
    if not os.path.exists(returns_file):
        raise FileNotFoundError(f"No se encontró {returns_file}")
    
    with open(returns_file, 'rb') as f:
        returns_dict = pickle.load(f)
    
    return returns_dict

def calculate_portfolio_returns(weights_dict, returns_dict, geographies):
    """
    Calcula retornos del portafolio basado en pesos y retornos históricos.
    
    Parameters:
    -----------
    weights_dict : dict
        Pesos por geografía (puede variar en el tiempo)
    returns_dict : dict
        Retornos históricos por ETF
    geographies : list
        Lista de geografías
    
    Returns:
    --------
    pd.Series: Retornos del portafolio
    """
    # Definir ETFs por geografía
    ETFS_BY_GEOGRAPHY = {
        'USA': ['SPY', 'QQQ', 'IWM'],
        'EUROPA': ['VGK'],
        'ASIA_PACIFICO': ['VPL'],
        'EMERGENTES': ['EEM'],
        'BONOS': ['TLT', 'LQD', 'HYG', 'SHY'],
        'MATERIAS_PRIMAS': ['GLD', 'USO', 'DJP'],
        'REAL_ESTATE': ['VNQ']
    }
    
    # Calcular retornos por geografía
    geo_returns = {}
    
    for geo in geographies:
        if geo not in ETFS_BY_GEOGRAPHY:
            continue
        
        etfs = ETFS_BY_GEOGRAPHY[geo]
        available_etfs = [etf for etf in etfs if etf in returns_dict]
        
        if len(available_etfs) == 0:
            continue
        
        # Retorno promedio de la geografía
        geo_returns_list = []
        for etf in available_etfs:
            if etf in returns_dict:
                geo_returns_list.append(returns_dict[etf])
        
        if len(geo_returns_list) > 0:
            geo_returns_df = pd.DataFrame(geo_returns_list).T
            geo_returns[geo] = geo_returns_df.mean(axis=1, skipna=True)
    
    # Crear DataFrame con retornos por geografía
    returns_df = pd.DataFrame(geo_returns)
    returns_df = returns_df.dropna()
    
    # Normalizar índices de fecha (eliminar timezone si existe)
    if isinstance(returns_df.index, pd.DatetimeIndex):
        if returns_df.index.tz is not None:
            returns_df.index = returns_df.index.tz_localize(None)
        returns_df.index = returns_df.index.normalize()
    
    # Calcular retornos del portafolio
    if isinstance(weights_dict, dict):
        # Pesos constantes
        weights = np.array([weights_dict.get(geo, 0) for geo in returns_df.columns])
        portfolio_returns = (returns_df * weights).sum(axis=1)
    else:
        # Pesos variables en el tiempo (DataFrame)
        # Normalizar índice de weights también
        if isinstance(weights_dict.index, pd.DatetimeIndex):
            if weights_dict.index.tz is not None:
                weights_dict.index = weights_dict.index.tz_localize(None)
            weights_dict.index = weights_dict.index.normalize()
        # Alinear fechas
        aligned_weights = weights_dict.reindex(returns_df.index, method='ffill')
        portfolio_returns = (returns_df * aligned_weights).sum(axis=1)
    
    return portfolio_returns

def calculate_metrics(returns, risk_free_rate=0.02):
    """
    Calcula métricas de performance del portafolio.
    
    Parameters:
    -----------
    returns : pd.Series
        Retornos del portafolio
    risk_free_rate : float
        Tasa libre de riesgo anualizada
    
    Returns:
    --------
    dict: Métricas de performance
    """
    returns = returns.dropna()
    
    if len(returns) == 0:
        return {
            'total_return': 0,
            'annualized_return': 0,
            'volatility': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'win_rate': 0
        }
    
    # Detectar y manejar valores extremos (outliers)
    # Retornos diarios típicamente están entre -10% y +10%
    # Valores fuera de este rango pueden ser errores
    q1 = returns.quantile(0.01)
    q99 = returns.quantile(0.99)
    
    # Filtrar outliers extremos (más de 3 desviaciones estándar)
    mean_ret = returns.mean()
    std_ret = returns.std()
    outlier_threshold = 3 * std_ret
    
    # Identificar outliers pero no eliminarlos todavía (solo para reporte)
    outliers = returns[(returns.abs() > abs(mean_ret) + outlier_threshold)]
    
    # Los retornos del portafolio vienen de calculate_portfolio_returns
    # que calcula: (returns_df * weights).sum(axis=1)
    # donde returns_df contiene retornos en porcentaje (ej: 0.5 = 0.5%)
    # Por lo tanto, los retornos del portafolio también están en porcentaje
    
    # Verificar formato basado en el rango típico
    sample_values = returns.dropna()
    if len(sample_values) > 0:
        max_abs = sample_values.abs().max()
        mean_abs = sample_values.abs().mean()
        
        # Retornos diarios típicos: -5% a +5% (valores entre -5 y 5)
        # Si están en decimal: -0.05 a +0.05 (valores entre -0.05 y 0.05)
        if mean_abs < 0.1 and max_abs < 1:  # Probablemente en decimal
            returns_pct = returns * 100  # Convertir a porcentaje
        elif mean_abs > 50 or max_abs > 100:  # Valores extremos, error
            print(f"    [WARNING] Valores extremos detectados (max_abs={max_abs:.2f})")
            # Capar valores extremos
            returns_pct = returns.clip(lower=-50, upper=50)  # Limitar a ±50%
        else:  # Ya están en porcentaje (rango típico)
            returns_pct = returns
    else:
        returns_pct = returns
    
    # Verificar si hay valores extremos después de normalizar
    if len(outliers) > 0 and len(outliers) < len(returns) * 0.05:  # Menos del 5% son outliers
        # Reemplazar outliers con valores límite (winsorization)
        returns_pct_clean = returns_pct.copy()
        lower_bound = returns_pct.quantile(0.01)
        upper_bound = returns_pct.quantile(0.99)
        returns_pct_clean = returns_pct_clean.clip(lower=lower_bound, upper=upper_bound)
        returns_pct = returns_pct_clean
    
    # Retorno total (asumiendo retornos en porcentaje)
    total_return = (1 + returns_pct / 100).prod() - 1
    
    # Retorno anualizado
    years = len(returns) / 252
    annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    
    # Volatilidad anualizada
    # Los retornos están en porcentaje diario (ej: 0.5 = 0.5%)
    # std() da desviación estándar diaria en porcentaje
    # Multiplicar por sqrt(252) para anualizar
    volatility_daily = returns_pct.std()
    volatility = volatility_daily * np.sqrt(252)  # Anualizada, en porcentaje
    
    # Sharpe Ratio
    # annualized_return está en decimal (ej: 0.1 = 10%)
    # volatility está en porcentaje (ej: 15 = 15%)
    # Convertir volatility a decimal para el cálculo
    excess_return = annualized_return - risk_free_rate
    volatility_decimal = volatility / 100  # Convertir a decimal para Sharpe
    sharpe_ratio = excess_return / volatility_decimal if volatility_decimal > 0 else 0
    
    # Drawdown máximo (usar returns_pct que ya está normalizado)
    cumulative = (1 + returns_pct / 100).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Win rate
    win_rate = (returns > 0).sum() / len(returns)
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,  # En porcentaje anualizado
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'num_trades': len(returns)
    }

def backtest_walk_forward(dataset, models, returns_dict, 
                          rebalance_frequency='quarterly',  # ESTRATEGIA 7: Más conservador
                          train_years=5,
                          start_date=None,
                          end_date=None,
                          min_weight_change=0.1,  # ESTRATEGIA 7: Solo re-balancear si cambio > 10%
                          stop_loss_threshold=-0.15):  # ESTRATEGIA 4.4: Stop-loss a -15%
    """
    Backtesting walk-forward: entrena y prueba en ventanas deslizantes.
    
    Parameters:
    -----------
    dataset : pd.DataFrame
        Dataset completo con features y targets
    models : dict
        Modelos entrenados (se re-entrenan en cada ventana)
    returns_dict : dict
        Retornos históricos
    rebalance_frequency : str
        'daily', 'weekly', 'monthly'
    train_years : int
        Años de datos para entrenar en cada ventana
    start_date : datetime, optional
        Fecha de inicio del backtest
    end_date : datetime, optional
        Fecha de fin del backtest
    """
    print("=" * 80)
    print("BACKTESTING WALK-FORWARD")
    print("=" * 80)
    
    # Preparar datos
    feature_cols = [col for col in dataset.columns if not col.startswith('target_')]
    target_cols = [col for col in dataset.columns if col.startswith('target_')]
    
    # Normalizar índices de fecha (eliminar timezone si existe)
    if isinstance(dataset.index, pd.DatetimeIndex):
        if dataset.index.tz is not None:
            dataset.index = dataset.index.tz_localize(None)
        dataset.index = dataset.index.normalize()
    
    # Filtrar por fechas
    if start_date:
        if isinstance(start_date, pd.Timestamp) and start_date.tz is not None:
            start_date = start_date.tz_localize(None)
        dataset = dataset[dataset.index >= start_date]
    if end_date:
        if isinstance(end_date, pd.Timestamp) and end_date.tz is not None:
            end_date = end_date.tz_localize(None)
        dataset = dataset[dataset.index <= end_date]
    
    dataset = dataset.sort_index()
    
    # ESTRATEGIA 7: Determinar fechas de re-balanceo (más conservador)
    if rebalance_frequency == 'daily':
        rebalance_dates = dataset.index.tolist()
    elif rebalance_frequency == 'weekly':
        rebalance_dates = dataset.index[::5].tolist()  # Cada 5 días hábiles
    elif rebalance_frequency == 'monthly':
        # Obtener fechas mensuales
        monthly_periods = dataset.index.to_period('M').drop_duplicates()
        rebalance_dates = []
        for period in monthly_periods:
            month_start = period.to_timestamp()
            month_dates = dataset[dataset.index >= month_start].index
            if len(month_dates) > 0:
                rebalance_dates.append(month_dates[0])
        rebalance_dates = sorted(list(set(rebalance_dates)))
    elif rebalance_frequency == 'quarterly':  # ESTRATEGIA 7: Trimestral
        quarterly_periods = dataset.index.to_period('Q').drop_duplicates()
        rebalance_dates = []
        for period in quarterly_periods:
            quarter_start = period.to_timestamp()
            quarter_dates = dataset[dataset.index >= quarter_start].index
            if len(quarter_dates) > 0:
                rebalance_dates.append(quarter_dates[0])
        rebalance_dates = sorted(list(set(rebalance_dates)))
    else:
        raise ValueError(f"Frecuencia desconocida: {rebalance_frequency}")
    
    print(f"\nFechas de re-balanceo: {len(rebalance_dates)}")
    print(f"  Primera: {rebalance_dates[0]}")
    print(f"  Última: {rebalance_dates[-1]}")
    
    # Almacenar pesos y retornos en el tiempo
    portfolio_weights = {}
    portfolio_returns_list = []
    previous_weights = None  # ESTRATEGIA 4.2: Para turnover constraint
    cumulative_value = 1.0  # ESTRATEGIA 4.4: Para stop-loss
    
    # Backtesting
    print("\nEjecutando backtesting...")
    print(f"  [ESTRATEGIA 7] Re-balanceo: {rebalance_frequency}")
    print(f"  [ESTRATEGIA 7] Cambio mínimo para re-balancear: {min_weight_change*100:.0f}%")
    print(f"  [ESTRATEGIA 4.4] Stop-loss: {stop_loss_threshold*100:.0f}%")
    
    successful_predictions = 0
    failed_predictions = 0
    skipped_rebalances = 0  # ESTRATEGIA 7: Contador de re-balanceos saltados
    
    for i, rebalance_date in enumerate(rebalance_dates):
        if i % 50 == 0:
            print(f"  Procesando fecha {i+1}/{len(rebalance_dates)}: {rebalance_date}")
        
        # Normalizar fecha de rebalanceo
        if isinstance(rebalance_date, pd.Timestamp):
            if rebalance_date.tz is not None:
                rebalance_date = rebalance_date.tz_localize(None)
            rebalance_date = rebalance_date.normalize()
        
        # Obtener datos hasta esta fecha (sin data leakage)
        data_until_date = dataset[dataset.index <= rebalance_date]
        
        if len(data_until_date) < train_years * 252:
            # No hay suficientes datos para entrenar
            continue
        
        # Obtener features actuales
        X_current = data_until_date.iloc[[-1]]
        
        # Generar predicciones
        try:
            # ESTRATEGIA 9: Obtener VIX si está disponible para gestión de riesgo dinámica
            vix_level = None
            if 'VIX_value' in X_current.columns:
                vix_val = X_current['VIX_value'].iloc[-1]
                if not np.isnan(vix_val):
                    vix_level = vix_val
            
            # Cargar métricas de modelos si están disponibles (ESTRATEGIA 8)
            model_metrics = None
            try:
                import json
                models_dir = 'models'
                metrics_files = [f for f in os.listdir(models_dir) if f.startswith('metrics_') and f.endswith('.json')]
                if len(metrics_files) > 0:
                    model_metrics = {}
                    for metrics_file in metrics_files:
                        geo = metrics_file.replace('metrics_', '').replace('.json', '')
                        metrics_path = os.path.join(models_dir, metrics_file)
                        with open(metrics_path, 'r') as f:
                            metrics_data = json.load(f)
                            model_metrics[geo] = {'r2': metrics_data.get('best_r2_val', 0)}
            except Exception as e:
                pass
            
            sharpe_predictions = predict_sharpe_ratios(models, X_current, model_metrics=model_metrics)
            
            # Verificar que hay predicciones válidas
            valid_predictions = {k: v for k, v in sharpe_predictions.items() if not np.isnan(v)}
            if len(valid_predictions) == 0:
                failed_predictions += 1
                continue
            
            # ESTRATEGIA 4.4: Aplicar stop-loss si el portafolio ha caído demasiado
            if cumulative_value < (1 + stop_loss_threshold):
                print(f"    [ESTRATEGIA 4.4] Stop-loss activado (valor: {cumulative_value:.4f}), reduciendo exposición")
                # Reducir todos los pesos a la mitad y aumentar cash (bonos)
                if previous_weights:
                    weights = {k: v * 0.5 for k, v in previous_weights.items()}
                    # El resto va a bonos (SHY)
                    total_reduced = sum(weights.values())
                    if 'BONOS' in weights:
                        weights['BONOS'] += (1.0 - total_reduced)
                    else:
                        weights['BONOS'] = (1.0 - total_reduced)
                else:
                    # Si no hay pesos anteriores, usar pesos conservadores
                    weights = {'BONOS': 0.7, 'USA': 0.2, 'EUROPA': 0.1}
            else:
                # Optimizar portafolio con mejoras
                weights = optimize_portfolio(
                    sharpe_predictions,
                    min_weight=0.0,
                    max_weight=0.3,  # ESTRATEGIA 4: Más conservador (30% en lugar de 40%)
                    returns_dict=returns_dict,
                    previous_weights=previous_weights,  # ESTRATEGIA 4.2: Turnover constraint
                    max_volatility=0.25,  # ESTRATEGIA 4.1: Límite de volatilidad
                    max_turnover=0.2,  # ESTRATEGIA 4.2: Máximo 20% de cambio
                    vix_level=vix_level  # ESTRATEGIA 9: Gestión de riesgo dinámica
                )
            
            if weights is None or len(weights) == 0:
                failed_predictions += 1
                continue
            
            # ESTRATEGIA 7: Solo re-balancear si el cambio es significativo
            if previous_weights is not None:
                # Calcular cambio total
                total_change = sum(abs(weights.get(geo, 0) - previous_weights.get(geo, 0)) 
                                 for geo in set(list(weights.keys()) + list(previous_weights.keys())))
                
                if total_change < min_weight_change:
                    # Cambio muy pequeño, mantener pesos anteriores
                    skipped_rebalances += 1
                    weights = previous_weights.copy()
                else:
                    # Cambio significativo, usar nuevos pesos
                    portfolio_weights[rebalance_date] = weights
                    previous_weights = weights.copy()
                    successful_predictions += 1
            else:
                # Primera vez, usar los pesos optimizados
                portfolio_weights[rebalance_date] = weights
                previous_weights = weights.copy()
                successful_predictions += 1
            
            # Calcular retorno del portafolio para este período
            # (hasta el próximo re-balanceo o fin de datos)
            if i < len(rebalance_dates) - 1:
                next_date = rebalance_dates[i + 1]
                # Normalizar next_date también
                if isinstance(next_date, pd.Timestamp):
                    if next_date.tz is not None:
                        next_date = next_date.tz_localize(None)
                    next_date = next_date.normalize()
                
                # Filtrar retornos del período
                period_returns = calculate_portfolio_returns(
                    weights, returns_dict, list(weights.keys())
                )
                
                # Normalizar índice de period_returns si es necesario
                if isinstance(period_returns.index, pd.DatetimeIndex):
                    if period_returns.index.tz is not None:
                        period_returns.index = period_returns.index.tz_localize(None)
                    period_returns.index = period_returns.index.normalize()
                
                # Filtrar por rango de fechas
                period_mask = (period_returns.index > rebalance_date) & (period_returns.index <= next_date)
                period_returns_filtered = period_returns[period_mask]
            else:
                # Último período: desde rebalance_date hasta el fin
                period_returns = calculate_portfolio_returns(
                    weights, returns_dict, list(weights.keys())
                )
                
                # Normalizar índice
                if isinstance(period_returns.index, pd.DatetimeIndex):
                    if period_returns.index.tz is not None:
                        period_returns.index = period_returns.index.tz_localize(None)
                    period_returns.index = period_returns.index.normalize()
                
                period_mask = period_returns.index > rebalance_date
                period_returns_filtered = period_returns[period_mask]
            
            if len(period_returns_filtered) > 0:
                portfolio_returns_list.append(period_returns_filtered)
                
                # ESTRATEGIA 4.4: Actualizar valor acumulado para stop-loss
                period_cumulative = (1 + period_returns_filtered / 100).prod()
                cumulative_value *= period_cumulative
        
        except Exception as e:
            failed_predictions += 1
            if i % 50 == 0 or i < 5:  # Mostrar primeros errores y luego cada 50
                print(f"    [WARNING] Error en {rebalance_date}: {e}")
                import traceback
                if i < 5:  # Mostrar traceback completo para primeros errores
                    traceback.print_exc()
            continue
    
    print(f"\n  Predicciones exitosas: {successful_predictions}")
    print(f"  Predicciones fallidas: {failed_predictions}")
    print(f"  Re-balanceos saltados (cambio < {min_weight_change*100:.0f}%): {skipped_rebalances}")
    
    # Combinar todos los retornos
    if len(portfolio_returns_list) > 0:
        try:
            portfolio_returns = pd.concat(portfolio_returns_list).sort_index()
            portfolio_returns = portfolio_returns[~portfolio_returns.index.duplicated(keep='first')]
            print(f"\n  Retornos generados: {len(portfolio_returns)} observaciones")
            print(f"  Rango de fechas: {portfolio_returns.index.min()} a {portfolio_returns.index.max()}")
        except Exception as e:
            print(f"\n[ERROR] Error combinando retornos: {e}")
            print(f"  Número de listas: {len(portfolio_returns_list)}")
            if len(portfolio_returns_list) > 0:
                print(f"  Primer elemento: {type(portfolio_returns_list[0])}, longitud: {len(portfolio_returns_list[0])}")
            portfolio_returns = pd.Series(dtype=float)
    else:
        print(f"\n[WARNING] No se generaron retornos del portafolio")
        print(f"  Predicciones exitosas: {successful_predictions}")
        print(f"  Predicciones fallidas: {failed_predictions}")
        portfolio_returns = pd.Series(dtype=float)
    
    return portfolio_returns, portfolio_weights

def compare_with_benchmarks(portfolio_returns, returns_dict, risk_free_rate=0.02):
    """Compara performance con benchmarks"""
    benchmarks = {}
    
    # 1. SPY (mercado)
    if 'SPY' in returns_dict:
        benchmarks['SPY'] = returns_dict['SPY']
    
    # 2. Portafolio 1/N (igualmente ponderado)
    geographies = ['USA', 'EUROPA', 'ASIA_PACIFICO', 'EMERGENTES', 'BONOS', 'MATERIAS_PRIMAS', 'REAL_ESTATE']
    ETFS_BY_GEOGRAPHY = {
        'USA': ['SPY', 'QQQ', 'IWM'],
        'EUROPA': ['VGK'],
        'ASIA_PACIFICO': ['VPL'],
        'EMERGENTES': ['EEM'],
        'BONOS': ['TLT', 'LQD', 'HYG', 'SHY'],
        'MATERIAS_PRIMAS': ['GLD', 'USO', 'DJP'],
        'REAL_ESTATE': ['VNQ']
    }
    
    # Calcular retornos 1/N
    geo_returns_list = []
    for geo in geographies:
        if geo not in ETFS_BY_GEOGRAPHY:
            continue
        etfs = [etf for etf in ETFS_BY_GEOGRAPHY[geo] if etf in returns_dict]
        if len(etfs) > 0:
            geo_ret = pd.DataFrame({etf: returns_dict[etf] for etf in etfs}).mean(axis=1)
            geo_returns_list.append(geo_ret)
    
    if len(geo_returns_list) > 0:
        geo_returns_df = pd.DataFrame(geo_returns_list).T
        equal_weights = np.ones(len(geo_returns_df.columns)) / len(geo_returns_df.columns)
        benchmarks['1/N Portfolio'] = (geo_returns_df * equal_weights).sum(axis=1)
    
    # Normalizar índices de fecha antes de alinear
    # Portfolio returns
    if isinstance(portfolio_returns.index, pd.DatetimeIndex):
        if portfolio_returns.index.tz is not None:
            portfolio_returns.index = portfolio_returns.index.tz_localize(None)
        portfolio_returns.index = portfolio_returns.index.normalize()
    
    # Benchmarks
    benchmarks_normalized = {}
    for bench_name, bench_returns in benchmarks.items():
        if isinstance(bench_returns.index, pd.DatetimeIndex):
            bench_returns_copy = bench_returns.copy()
            if bench_returns_copy.index.tz is not None:
                bench_returns_copy.index = bench_returns_copy.index.tz_localize(None)
            bench_returns_copy.index = bench_returns_copy.index.normalize()
            benchmarks_normalized[bench_name] = bench_returns_copy
        else:
            benchmarks_normalized[bench_name] = bench_returns
    
    # Alinear fechas (ahora todas sin timezone)
    all_dates = set(portfolio_returns.index)
    for bench_name, bench_returns in benchmarks_normalized.items():
        all_dates.update(bench_returns.index)
    
    # Convertir a lista y ordenar (ahora todas son tz-naive)
    common_dates = sorted([pd.Timestamp(d).normalize() if isinstance(d, pd.Timestamp) else d for d in all_dates])
    
    # Filtrar a fechas comunes
    portfolio_aligned = portfolio_returns.reindex(common_dates).dropna()
    benchmarks_aligned = {}
    for bench_name, bench_returns in benchmarks_normalized.items():
        bench_aligned = bench_returns.reindex(common_dates).dropna()
        if len(bench_aligned) > 0:
            benchmarks_aligned[bench_name] = bench_aligned
    
    # Calcular métricas
    results = {}
    
    # Portfolio
    portfolio_metrics = calculate_metrics(portfolio_aligned, risk_free_rate)
    results['Portfolio Optimizado'] = portfolio_metrics
    
    # Benchmarks
    for bench_name, bench_returns in benchmarks_aligned.items():
        # Intersectar fechas
        common = portfolio_aligned.index.intersection(bench_returns.index)
        if len(common) > 0:
            bench_metrics = calculate_metrics(bench_returns.loc[common], risk_free_rate)
            results[bench_name] = bench_metrics
    
    return results

def main():
    """Función principal"""
    print("=" * 80)
    print("BACKTESTING DEL SISTEMA DE PREDICCIÓN Y OPTIMIZACIÓN")
    print("=" * 80)
    
    data_dir = 'data'
    models_dir = 'models'
    
    # 1. Cargar modelos (ESTRATEGIA 6: Intentar cargar ensembles)
    print("\n1. Cargando modelos...")
    models = load_models(models_dir, use_ensemble=True)
    if models is None:
        print("[ERROR] No se pudieron cargar modelos")
        return
    
    # Verificar tipo de modelos cargados
    if models.get('type') == 'ensemble':
        print(f"  [ESTRATEGIA 6] Usando ensembles: {len(models['ensembles'])} geografías")
    else:
        print(f"  Usando modelos individuales: {len(models.get('models', {}))} geografías")
    
    # 2. Cargar dataset
    print("\n2. Cargando dataset...")
    dataset = load_ml_dataset(data_dir)
    
    # 3. Cargar retornos
    print("\n3. Cargando retornos históricos...")
    returns_dict = load_returns_data(data_dir)
    
    # 4. Ejecutar backtesting
    print("\n4. Ejecutando backtesting...")
    portfolio_returns, portfolio_weights = backtest_walk_forward(
        dataset, models, returns_dict,
        rebalance_frequency='quarterly',  # ESTRATEGIA 7: Re-balancear trimestralmente
        train_years=5,
        min_weight_change=0.1,  # ESTRATEGIA 7: Solo re-balancear si cambio > 10%
        stop_loss_threshold=-0.15  # ESTRATEGIA 4.4: Stop-loss a -15%
    )
    
    if len(portfolio_returns) == 0:
        print("[ERROR] No se generaron retornos del portafolio")
        return
    
    # 5. Diagnóstico de retornos
    print("\n5. Diagnóstico de retornos del portafolio...")
    if len(portfolio_returns) > 0:
        print(f"  Observaciones: {len(portfolio_returns):,}")
        print(f"  Rango de valores: [{portfolio_returns.min():.4f}, {portfolio_returns.max():.4f}]")
        print(f"  Media: {portfolio_returns.mean():.4f}")
        print(f"  Mediana: {portfolio_returns.median():.4f}")
        print(f"  Desviación estándar (diaria): {portfolio_returns.std():.4f}")
        print(f"  Percentiles: P1={portfolio_returns.quantile(0.01):.4f}, P99={portfolio_returns.quantile(0.99):.4f}")
        
        # Detectar posibles problemas
        if portfolio_returns.std() > 50:
            print(f"  [WARNING] Desviación estándar muy alta ({portfolio_returns.std():.2f})")
            print(f"            Esto puede indicar valores extremos o error en el cálculo")
        
        # Mostrar algunos valores de ejemplo
        print(f"  Primeros 5 valores:")
        for i, (date, val) in enumerate(portfolio_returns.head().items()):
            print(f"    {date}: {val:.4f}")
    
    # 6. Calcular métricas
    print("\n6. Calculando métricas de performance...")
    portfolio_metrics = calculate_metrics(portfolio_returns)
    
    print("\n" + "=" * 80)
    print("MÉTRICAS DEL PORTAFOLIO OPTIMIZADO")
    print("=" * 80)
    print(f"\nPeríodo: {portfolio_returns.index.min()} a {portfolio_returns.index.max()}")
    print(f"Retorno Total:        {portfolio_metrics['total_return']*100:>8.2f}%")
    print(f"Retorno Anualizado:   {portfolio_metrics['annualized_return']*100:>8.2f}%")
    print(f"Volatilidad:           {portfolio_metrics['volatility']*100:>8.2f}%")
    print(f"Sharpe Ratio:         {portfolio_metrics['sharpe_ratio']:>8.4f}")
    print(f"Drawdown Máximo:      {portfolio_metrics['max_drawdown']*100:>8.2f}%")
    print(f"Win Rate:             {portfolio_metrics['win_rate']*100:>8.2f}%")
    
    # 7. Comparar con benchmarks
    print("\n7. Comparando con benchmarks...")
    comparison = compare_with_benchmarks(portfolio_returns, returns_dict)
    
    print("\n" + "=" * 80)
    print("COMPARACIÓN CON BENCHMARKS")
    print("=" * 80)
    print(f"\n{'Estrategia':<25} {'Retorno An.':<12} {'Sharpe':<10} {'Drawdown':<10}")
    print("-" * 60)
    
    for strategy, metrics in comparison.items():
        ret = metrics['annualized_return'] * 100
        sharpe = metrics['sharpe_ratio']
        dd = metrics['max_drawdown'] * 100
        print(f"{strategy:<25} {ret:>10.2f}% {sharpe:>9.4f} {dd:>9.2f}%")
    
    # 7. Guardar resultados
    results = {
        'portfolio_returns': portfolio_returns,
        'portfolio_weights': portfolio_weights,
        'metrics': portfolio_metrics,
        'comparison': comparison
    }
    
    results_file = os.path.join(data_dir, 'backtest_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n[OK] Resultados guardados en {results_file}")
    
    # 8. Visualizar resultados
    print("\n8. Generando visualizaciones...")
    try:
        import sys
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, current_dir)
        from visualize_backtest_results import visualize_backtest_results
        visualize_backtest_results(portfolio_returns, results, data_dir)
    except ImportError as e:
        print(f"    [WARNING] No se pudo importar visualize_backtest_results: {e}")
        print("    Instala matplotlib: pip install matplotlib")
    except Exception as e:
        print(f"    [WARNING] Error en visualización: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("BACKTESTING COMPLETADO")
    print("=" * 80)

if __name__ == "__main__":
    main()
