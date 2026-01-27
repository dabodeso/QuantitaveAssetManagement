"""
Script para optimizar asignaciones de portafolio basado en predicciones de Sharpe Ratio.

Este script:
1. Carga los modelos entrenados
2. Genera predicciones de Sharpe Ratio futuro por geografía
3. Optimiza las asignaciones de capital para maximizar el Sharpe Ratio del portafolio
4. Considera restricciones de diversificación y riesgo
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Variable global para controlar warnings
_WARNING_SHOWN = False

try:
    import cvxpy as cp
    # Verificar que hay al menos un solver disponible
    try:
        # Intentar importar un solver común
        import ecos
        CVXPY_AVAILABLE = True
        CVXPY_SOLVERS_AVAILABLE = True
    except ImportError:
        try:
            import scs
            CVXPY_AVAILABLE = True
            CVXPY_SOLVERS_AVAILABLE = True
        except ImportError:
            CVXPY_AVAILABLE = True
            CVXPY_SOLVERS_AVAILABLE = False
            if not _WARNING_SHOWN:
                print("[INFO] CVXPy está instalado pero no hay solvers disponibles.")
                print("       Usando scipy.optimize como fallback.")
                print("       Para mejor performance: pip install ecos o scs")
    from scipy.optimize import minimize
except ImportError:
    CVXPY_AVAILABLE = False
    CVXPY_SOLVERS_AVAILABLE = False
    if not _WARNING_SHOWN:
        print("[INFO] CVXPy no está instalado. Usando optimización con scipy.optimize")
        print("       Para mejor performance: pip install cvxpy ecos")
    from scipy.optimize import minimize

def load_models(models_dir='models', use_ensemble=True):
    """
    Carga todos los modelos entrenados.
    
    Parameters:
    -----------
    use_ensemble : bool
        Si True, intenta cargar ensembles. Si False, carga modelos individuales.
    """
    models = {}
    ensembles = {}
    
    if not os.path.exists(models_dir):
        print(f"[ERROR] Directorio {models_dir} no existe. Ejecuta train_sharpe_predictor.py primero.")
        return None
    
    print(f"\nCargando modelos desde {models_dir}...")
    
    # ESTRATEGIA 6: Intentar cargar ensembles primero
    if use_ensemble:
        ensemble_files = [f for f in os.listdir(models_dir) if f.startswith('ensemble_') and f.endswith('.pkl')]
        
        if len(ensemble_files) > 0:
            print(f"  [ESTRATEGIA 6] Cargando ensembles ({len(ensemble_files)} encontrados)...")
            for ensemble_file in ensemble_files:
                geo = ensemble_file.replace('ensemble_', '').replace('.pkl', '')
                filepath = os.path.join(models_dir, ensemble_file)
                
                try:
                    with open(filepath, 'rb') as f:
                        ensemble_data = pickle.load(f)
                    ensembles[geo] = ensemble_data
                    print(f"    [OK] {geo}: ensemble con {len(ensemble_data['models'])} modelos")
                except Exception as e:
                    print(f"    [ERROR] Error cargando {ensemble_file}: {e}")
    
    # Si no hay ensembles o use_ensemble=False, cargar modelos individuales
    if len(ensembles) == 0:
        model_files = [f for f in os.listdir(models_dir) if f.startswith('sharpe_predictor_') and f.endswith('.pkl')]
        
        if len(model_files) == 0:
            print(f"[ERROR] No se encontraron modelos en {models_dir}")
            return None
        
        for model_file in model_files:
            # Extraer geografía del nombre: sharpe_predictor_{GEO}_{model_type}.pkl
            parts = model_file.replace('sharpe_predictor_', '').replace('.pkl', '').split('_')
            geo = parts[0]
            model_type = '_'.join(parts[1:]) if len(parts) > 1 else 'unknown'
            
            filepath = os.path.join(models_dir, model_file)
            
            try:
                with open(filepath, 'rb') as f:
                    model = pickle.load(f)
                models[geo] = {
                    'model': model,
                    'type': model_type
                }
                print(f"  [OK] {geo}: {model_type}")
            except Exception as e:
                print(f"  [ERROR] Error cargando {model_file}: {e}")
    
    # Retornar ensembles si están disponibles, sino modelos individuales
    if len(ensembles) > 0:
        return {'ensembles': ensembles, 'type': 'ensemble'}
    elif len(models) > 0:
        return {'models': models, 'type': 'individual'}
    else:
        return None

def load_ml_dataset(data_dir='data'):
    """Carga el dataset ML para obtener features actuales"""
    pkl_file = os.path.join(data_dir, 'ml_dataset.pkl')
    csv_file = os.path.join(data_dir, 'ml_dataset.csv')
    
    if os.path.exists(pkl_file):
        with open(pkl_file, 'rb') as f:
            dataset = pickle.load(f)
        return dataset
    elif os.path.exists(csv_file):
        dataset = pd.read_csv(csv_file, index_col=0, parse_dates=True)
        return dataset
    else:
        raise FileNotFoundError(f"No se encontró ml_dataset.pkl ni ml_dataset.csv en {data_dir}")

def predict_sharpe_ratios(models_dict, X_current, model_metrics=None):
    """
    Genera predicciones de Sharpe Ratio para todas las geografías.
    ESTRATEGIA 6: Usa ensemble si está disponible.
    ESTRATEGIA 8: Filtros de calidad de predicciones.
    
    Parameters:
    -----------
    models_dict : dict
        Diccionario con modelos o ensembles por geografía
    X_current : pd.DataFrame
        Features actuales (última fila del dataset)
    model_metrics : dict, optional
        Métricas de los modelos (R²) para filtros de calidad
    
    Returns:
    --------
    dict: Predicciones de Sharpe Ratio por geografía
    """
    predictions = {}
    prediction_confidence = {}
    
    # Obtener features (eliminar targets)
    feature_cols = [col for col in X_current.columns if not col.startswith('target_')]
    X = X_current[feature_cols].copy()
    
    # Manejar NaN (usar mediana de todo el dataset)
    X_clean = X.fillna(X.median())
    
    # Determinar si tenemos ensembles o modelos individuales
    if models_dict.get('type') == 'ensemble':
        ensembles = models_dict['ensembles']
        
        for geo, ensemble_data in ensembles.items():
            try:
                models = ensemble_data['models']
                weights = ensemble_data.get('weights', {})
                selected_features = ensemble_data.get('selected_features', None)
                
                # Si hay feature selector, aplicar selección
                if selected_features:
                    X_selected = X_clean[[f for f in selected_features if f in X_clean.columns]]
                    if len(X_selected.columns) < len(selected_features) * 0.8:
                        print(f"  [WARNING] {geo}: Faltan muchas features seleccionadas, usando todas")
                        X_selected = X_clean
                else:
                    X_selected = X_clean
                
                # ESTRATEGIA 6: Ensemble - combinar predicciones
                ensemble_predictions = []
                valid_models = 0
                
                for model_name, model in models.items():
                    try:
                        if model_name == 'lgbm':
                            pred = model.predict(X_selected.iloc[[-1]])[0]
                        else:
                            pred = model.predict(X_selected.iloc[[-1]])[0]
                        
                        weight = weights.get(model_name, 1.0 / len(models))
                        ensemble_predictions.append(pred * weight)
                        valid_models += 1
                    except Exception as e:
                        continue
                
                if valid_models > 0:
                    final_pred = sum(ensemble_predictions)
                    predictions[geo] = final_pred
                    
                    # ESTRATEGIA 8: Calcular confianza (basada en número de modelos válidos)
                    confidence = valid_models / len(models)
                    prediction_confidence[geo] = confidence
                else:
                    predictions[geo] = np.nan
                    prediction_confidence[geo] = 0.0
                    
            except Exception as e:
                print(f"[WARNING] Error prediciendo para {geo} (ensemble): {e}")
                predictions[geo] = np.nan
                prediction_confidence[geo] = 0.0
    
    else:
        # Modelos individuales (comportamiento original)
        models = models_dict.get('models', {})
        
        for geo, model_info in models.items():
            model = model_info['model']
            model_type = model_info['type']
            
            try:
                # Predecir según el tipo de modelo
                if model_type == 'lgbm':
                    pred = model.predict(X_clean.iloc[[-1]])[0]
                else:
                    pred = model.predict(X_clean.iloc[[-1]])[0]
                
                predictions[geo] = pred
                
                # ESTRATEGIA 8: Obtener R² del modelo si está disponible
                if model_metrics and geo in model_metrics:
                    r2 = model_metrics[geo].get('r2', 0)
                    prediction_confidence[geo] = max(0, min(1, r2))  # Normalizar R² a [0,1]
                else:
                    prediction_confidence[geo] = 0.5  # Confianza media por defecto
                    
            except Exception as e:
                print(f"[WARNING] Error prediciendo para {geo}: {e}")
                predictions[geo] = np.nan
                prediction_confidence[geo] = 0.0
    
    # ESTRATEGIA 8: Filtrar predicciones de baja calidad
    filtered_predictions = {}
    min_confidence = 0.3  # Mínimo R² de 0.3
    min_prediction_value = -5.0  # Sharpe mínimo razonable
    max_prediction_value = 5.0   # Sharpe máximo razonable
    
    for geo, pred in predictions.items():
        confidence = prediction_confidence.get(geo, 0.0)
        
        # Validar predicción
        if np.isnan(pred):
            continue
        
        if pred < min_prediction_value or pred > max_prediction_value:
            print(f"  [WARNING] {geo}: Predicción fuera de rango ({pred:.4f}), ignorando")
            continue
        
        # Si la confianza es muy baja, usar predicción conservadora (Sharpe = 0)
        if confidence < min_confidence:
            print(f"  [WARNING] {geo}: Confianza baja ({confidence:.3f}), usando Sharpe = 0")
            filtered_predictions[geo] = 0.0
        else:
            filtered_predictions[geo] = pred
    
    return filtered_predictions

def calculate_covariance_matrix(returns_dict, geographies, window=252):
    """
    Calcula matriz de covarianza de retornos por geografía.
    
    Parameters:
    -----------
    returns_dict : dict
        Diccionario con retornos por ETF
    geographies : list
        Lista de geografías
    window : int
        Ventana para calcular covarianza (días)
    
    Returns:
    --------
    pd.DataFrame: Matriz de covarianza
    """
    # Definir ETFs por geografía (debe coincidir con generate_ml_features.py)
    ETFS_BY_GEOGRAPHY = {
        'USA': ['SPY', 'QQQ', 'IWM'],
        'EUROPA': ['VGK'],
        'ASIA_PACIFICO': ['VPL'],
        'EMERGENTES': ['EEM'],
        'BONOS': ['TLT', 'LQD', 'HYG', 'SHY'],
        'MATERIAS_PRIMAS': ['GLD', 'USO', 'DJP'],
        'REAL_ESTATE': ['VNQ']
    }
    
    # Calcular retornos promedio por geografía
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
    
    if len(returns_df) < window:
        window = len(returns_df)
    
    # Calcular covarianza usando últimos N días
    recent_returns = returns_df.iloc[-window:]
    cov_matrix = recent_returns.cov() * 252  # Anualizar
    
    return cov_matrix

def optimize_portfolio_cvxpy(sharpe_predictions, cov_matrix, min_weight=0.0, max_weight=0.4,
                            previous_weights=None, max_volatility=0.25, max_turnover=0.2):
    """
    Optimiza portafolio usando CVXPy (optimización convexa).
    ESTRATEGIA 4: Incluye restricciones de riesgo y turnover.
    
    Maximiza: (w' × μ) / sqrt(w' × Σ × w)
    donde μ = Sharpe Ratio predicho, Σ = matriz de covarianza
    """
    geographies = list(sharpe_predictions.keys())
    n = len(geographies)
    
    # Vector de Sharpe predicho
    mu = np.array([sharpe_predictions[geo] for geo in geographies])
    
    # Asegurar que la matriz de covarianza esté alineada
    cov_aligned = cov_matrix.loc[geographies, geographies].values
    
    # Variables de optimización
    w = cp.Variable(n)
    
    # Función objetivo: maximizar Sharpe Ratio
    # Sharpe = (w' × μ) / sqrt(w' × Σ × w)
    portfolio_return = w @ mu
    portfolio_risk = cp.quad_form(w, cov_aligned)
    
    # Usar aproximación: maximizar (w' × μ) - λ × sqrt(w' × Σ × w)
    # donde λ es un parámetro de aversión al riesgo
    risk_aversion = 1.0
    objective = cp.Maximize(portfolio_return - risk_aversion * cp.sqrt(portfolio_risk))
    
    # ESTRATEGIA 4: Restricciones mejoradas
    constraints = [
        cp.sum(w) == 1,  # Suma de pesos = 1
        w >= min_weight,  # Sin ventas en corto
        w <= max_weight   # Diversificación (máximo por geografía)
    ]
    
    # ESTRATEGIA 4.1: Restricción de volatilidad máxima
    if max_volatility is not None and max_volatility > 0:
        portfolio_volatility = cp.sqrt(portfolio_risk)
        constraints.append(portfolio_volatility <= max_volatility)
    
    # ESTRATEGIA 4.2: Turnover constraint (si hay pesos anteriores)
    if previous_weights is not None and max_turnover is not None:
        prev_weights_array = np.array([previous_weights.get(geo, 0) for geo in geographies])
        turnover = cp.sum(cp.abs(w - prev_weights_array))
        constraints.append(turnover <= max_turnover)
    
    # Resolver - intentar múltiples solvers
    problem = cp.Problem(objective, constraints)
    
    # Lista de solvers a intentar (en orden de preferencia)
    solvers_to_try = [
        cp.ECOS,
        cp.SCS,
        cp.OSQP,
        cp.CLARABEL
    ]
    
    # Intentar cada solver hasta que uno funcione
    for solver in solvers_to_try:
        try:
            problem.solve(solver=solver, verbose=False)
            if problem.status not in ["infeasible", "unbounded", "solver_error"]:
                weights = {geo: w.value[i] for i, geo in enumerate(geographies)}
                return weights, problem.value
        except Exception as e:
            # Intentar siguiente solver
            continue
    
    # Si ningún solver funcionó
    global _WARNING_SHOWN
    if not _WARNING_SHOWN:
        print(f"[INFO] Ningún solver de CVXPy funcionó. Usando scipy.optimize como fallback.")
        _WARNING_SHOWN = True
    return None, None

def optimize_portfolio_scipy(sharpe_predictions, cov_matrix, min_weight=0.0, max_weight=0.4,
                             previous_weights=None, max_volatility=0.25, max_turnover=0.2):
    """
    Optimiza portafolio usando scipy.optimize (fallback si CVXPy no está disponible).
    ESTRATEGIA 4: Incluye restricciones de riesgo y turnover.
    """
    geographies = list(sharpe_predictions.keys())
    n = len(geographies)
    
    # Vector de Sharpe predicho
    mu = np.array([sharpe_predictions[geo] for geo in geographies])
    
    # Matriz de covarianza alineada
    cov_aligned = cov_matrix.loc[geographies, geographies].values
    
    # Función objetivo: minimizar -Sharpe Ratio
    def objective(w):
        portfolio_return = np.dot(w, mu)
        portfolio_risk = np.sqrt(np.dot(w, np.dot(cov_aligned, w)))
        if portfolio_risk == 0:
            return -portfolio_return if portfolio_return > 0 else 1e10
        sharpe = portfolio_return / portfolio_risk
        return -sharpe  # Minimizar negativo = maximizar positivo
    
    # Restricciones
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Suma = 1
    ]
    
    # ESTRATEGIA 4.1: Restricción de volatilidad máxima
    if max_volatility is not None and max_volatility > 0:
        def volatility_constraint(w):
            portfolio_risk = np.sqrt(np.dot(w, np.dot(cov_aligned, w)))
            return max_volatility - portfolio_risk
        constraints.append({'type': 'ineq', 'fun': volatility_constraint})
    
    # ESTRATEGIA 4.2: Turnover constraint
    if previous_weights is not None:
        prev_weights_array = np.array([previous_weights.get(geo, 0) for geo in geographies])
        def turnover_constraint(w):
            turnover = np.sum(np.abs(w - prev_weights_array))
            return max_turnover - turnover
        constraints.append({'type': 'ineq', 'fun': turnover_constraint})
    
    # Límites
    bounds = [(min_weight, max_weight) for _ in range(n)]
    
    # Punto inicial (usar pesos anteriores si están disponibles, sino igualmente ponderado)
    if previous_weights is not None:
        x0 = np.array([previous_weights.get(geo, 1.0/n) for geo in geographies])
        # Normalizar
        x0 = x0 / x0.sum()
    else:
        x0 = np.ones(n) / n
    
    # Optimizar
    try:
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, 
                         options={'maxiter': 1000, 'ftol': 1e-6})
        
        if result.success:
            weights = {geo: result.x[i] for i, geo in enumerate(geographies)}
            # Normalizar pesos (asegurar que sumen 1)
            total = sum(weights.values())
            if total > 0:
                weights = {geo: w / total for geo, w in weights.items()}
            return weights, -result.fun
        else:
            print(f"[WARNING] Optimización scipy falló: {result.message}")
            # Intentar con método alternativo
            try:
                result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
                if result.success:
                    weights = {geo: result.x[i] for i, geo in enumerate(geographies)}
                    total = sum(weights.values())
                    if total > 0:
                        weights = {geo: w / total for geo, w in weights.items()}
                    return weights, -result.fun
            except:
                pass
            return None, None
    except Exception as e:
        print(f"[ERROR] Error en optimización scipy: {e}")
        return None, None

def optimize_portfolio(sharpe_predictions, cov_matrix=None, min_weight=0.0, max_weight=0.3, 
                       risk_free_rate=0.0, returns_dict=None, previous_weights=None,
                       max_volatility=0.25, max_turnover=0.2, vix_level=None):
    """
    Optimiza portafolio basado en predicciones de Sharpe Ratio.
    ESTRATEGIA 4: Mejoras de optimización (restricciones de riesgo, turnover, etc.)
    ESTRATEGIA 9: Gestión de riesgo dinámica basada en VIX
    
    Parameters:
    -----------
    sharpe_predictions : dict
        Predicciones de Sharpe Ratio por geografía
    cov_matrix : pd.DataFrame, optional
        Matriz de covarianza. Si None, se calcula desde returns_dict
    min_weight : float
        Peso mínimo por geografía (0 = sin ventas en corto)
    max_weight : float
        Peso máximo por geografía (diversificación) - default 0.3 (30%)
    risk_free_rate : float
        Tasa libre de riesgo (para calcular Sharpe)
    returns_dict : dict, optional
        Diccionario con retornos históricos (para calcular covarianza)
    previous_weights : dict, optional
        Pesos anteriores (para turnover constraint)
    max_volatility : float
        Volatilidad máxima permitida (default: 0.25 = 25%)
    max_turnover : float
        Turnover máximo permitido (default: 0.2 = 20%)
    vix_level : float, optional
        Nivel de VIX para ajuste dinámico de riesgo
    
    Returns:
    --------
    dict: Pesos optimizados por geografía
    """
    # ESTRATEGIA 9: Gestión de riesgo dinámica
    if vix_level is not None:
        if vix_level > 30:  # Mercado volátil
            max_weight = min(max_weight, 0.2)  # Reducir exposición máxima
            max_volatility = 0.15  # 15% máximo
            print(f"  [ESTRATEGIA 9] VIX alto ({vix_level:.1f}), reduciendo exposición (max_weight={max_weight:.1%})")
        elif vix_level < 15:  # Mercado tranquilo
            max_weight = min(max_weight, 0.4)  # Aumentar exposición
            max_volatility = 0.25  # 25% máximo
            print(f"  [ESTRATEGIA 9] VIX bajo ({vix_level:.1f}), aumentando exposición (max_weight={max_weight:.1%})")
    geographies = list(sharpe_predictions.keys())
    
    # Si no hay matriz de covarianza, calcularla o usar identidad
    if cov_matrix is None:
        if returns_dict is not None:
            cov_matrix = calculate_covariance_matrix(returns_dict, geographies)
        else:
            # Usar matriz identidad (asumir correlación baja)
            cov_matrix = pd.DataFrame(
                np.eye(len(geographies)),
                index=geographies,
                columns=geographies
            )
    
    # Filtrar geografías que están en ambas estructuras
    valid_geos = [g for g in geographies if g in cov_matrix.index]
    
    if len(valid_geos) == 0:
        print("[ERROR] No hay geografías válidas en la matriz de covarianza")
        return None
    
    # Filtrar predicciones y covarianza
    sharpe_filtered = {g: sharpe_predictions[g] for g in valid_geos}
    cov_filtered = cov_matrix.loc[valid_geos, valid_geos]
    
    # Optimizar - intentar CVXPy primero, luego scipy como fallback
    weights = None
    objective_value = None
    
    if CVXPY_AVAILABLE:
        weights, objective_value = optimize_portfolio_cvxpy(
            sharpe_filtered, cov_filtered, min_weight, max_weight,
            previous_weights=previous_weights,
            max_volatility=max_volatility,
            max_turnover=max_turnover
        )
    
    # Si CVXPy falló o no está disponible, usar scipy
    if weights is None:
        global _WARNING_SHOWN
        if not _WARNING_SHOWN:
            print("[INFO] Usando scipy.optimize para optimización...")
            _WARNING_SHOWN = True
        weights, objective_value = optimize_portfolio_scipy(
            sharpe_filtered, cov_filtered, min_weight, max_weight,
            previous_weights=previous_weights,
            max_volatility=max_volatility,
            max_turnover=max_turnover
        )
    
    # ESTRATEGIA 4.4: Validar pesos (deben sumar 1.0)
    if weights is not None:
        total = sum(weights.values())
        if abs(total - 1.0) > 0.01:  # Tolerancia del 1%
            print(f"  [WARNING] Pesos no suman 1.0 (suman {total:.4f}), normalizando...")
            weights = {geo: w / total for geo, w in weights.items()}
    
    return weights

def main():
    """Función principal"""
    print("=" * 80)
    print("OPTIMIZACIÓN DE PORTAFOLIO BASADO EN PREDICCIONES DE SHARPE RATIO")
    print("=" * 80)
    
    data_dir = 'data'
    models_dir = 'models'
    
    # 1. Cargar modelos
    print("\n1. Cargando modelos entrenados...")
    models = load_models(models_dir)
    
    if models is None or len(models) == 0:
        print("[ERROR] No se pudieron cargar modelos. Ejecuta train_sharpe_predictor.py primero.")
        return
    
    # 2. Cargar dataset para obtener features actuales
    print("\n2. Cargando dataset para features actuales...")
    dataset = load_ml_dataset(data_dir)
    print(f"   [OK] Dataset cargado: {dataset.shape[0]} filas")
    
    # 3. Obtener última fila (features más recientes)
    X_current = dataset.iloc[[-1]]
    current_date = X_current.index[0]
    print(f"   Fecha actual: {current_date}")
    
    # 4. Generar predicciones
    print("\n3. Generando predicciones de Sharpe Ratio...")
    
    # Cargar métricas de modelos si están disponibles
    model_metrics = None
    try:
        import json
        metrics_files = [f for f in os.listdir(models_dir) if f.startswith('metrics_') and f.endswith('.json')]
        if len(metrics_files) > 0:
            model_metrics = {}
            for metrics_file in metrics_files:
                geo = metrics_file.replace('metrics_', '').replace('.json', '')
                with open(os.path.join(models_dir, metrics_file), 'r') as f:
                    metrics_data = json.load(f)
                    model_metrics[geo] = {'r2': metrics_data.get('best_r2_val', 0)}
    except:
        pass
    
    sharpe_predictions = predict_sharpe_ratios(models, X_current, model_metrics=model_metrics)
    
    print("\nPredicciones de Sharpe Ratio futuro (20 días):")
    for geo, sharpe in sharpe_predictions.items():
        print(f"  {geo:20s}: {sharpe:8.4f}")
    
    # 5. Cargar retornos históricos para calcular covarianza
    print("\n4. Cargando retornos históricos para matriz de covarianza...")
    try:
        with open(os.path.join(data_dir, 'etf_returns_dict.pkl'), 'rb') as f:
            returns_dict = pickle.load(f)
        print(f"   [OK] Retornos cargados para {len(returns_dict)} ETFs")
    except Exception as e:
        print(f"   [WARNING] No se pudieron cargar retornos: {e}")
        returns_dict = None
    
    # 6. Optimizar portafolio
    print("\n5. Optimizando asignaciones de portafolio...")
    weights = optimize_portfolio(
        sharpe_predictions,
        min_weight=0.0,  # Sin ventas en corto
        max_weight=0.4,  # Máximo 40% por geografía
        returns_dict=returns_dict
    )
    
    if weights is None:
        print("[ERROR] No se pudo optimizar el portafolio")
        return
    
    # 7. Mostrar resultados
    print("\n" + "=" * 80)
    print("ASIGNACIONES OPTIMIZADAS DE PORTAFOLIO")
    print("=" * 80)
    print(f"\nFecha: {current_date}")
    print(f"\n{'Geografía':<20} {'Peso (%)':<12} {'Sharpe Predicho':<15}")
    print("-" * 50)
    
    total_weight = 0
    for geo in sorted(weights.keys()):
        weight_pct = weights[geo] * 100
        sharpe = sharpe_predictions.get(geo, np.nan)
        print(f"{geo:<20} {weight_pct:>10.2f}% {sharpe:>14.4f}")
        total_weight += weights[geo]
    
    print("-" * 50)
    print(f"{'TOTAL':<20} {total_weight*100:>10.2f}%")
    
    # 8. Guardar resultados
    results = {
        'date': current_date,
        'predictions': sharpe_predictions,
        'weights': weights,
        'total_weight': total_weight
    }
    
    results_file = os.path.join(data_dir, 'portfolio_weights_latest.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n[OK] Resultados guardados en {results_file}")
    
    print("\n" + "=" * 80)
    print("OPTIMIZACIÓN COMPLETADA")
    print("=" * 80)

if __name__ == "__main__":
    main()
