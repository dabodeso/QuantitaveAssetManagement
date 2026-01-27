"""
Script para entrenar modelos de Machine Learning que predicen Sharpe Ratio futuro por geografía.

Este script:
1. Carga el dataset ML preparado
2. Divide los datos en train/validation/test con validación temporal
3. Entrena múltiples modelos (XGBoost, LightGBM, Random Forest)
4. Evalúa y compara los modelos
5. Guarda el mejor modelo y sus métricas
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Importar modelos de ML
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
except ImportError:
    print("[ERROR] scikit-learn no está instalado. Ejecuta: pip install scikit-learn")
    exit(1)

try:
    import xgboost as xgb
except ImportError:
    print("[WARNING] XGBoost no está instalado. Ejecuta: pip install xgboost")
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    print("[WARNING] LightGBM no está instalado. Ejecuta: pip install lightgbm")
    lgb = None

def load_ml_dataset(data_dir='data'):
    """Carga el dataset ML completo"""
    pkl_file = os.path.join(data_dir, 'ml_dataset.pkl')
    csv_file = os.path.join(data_dir, 'ml_dataset.csv')
    
    if os.path.exists(pkl_file):
        print(f"[OK] Cargando dataset desde {pkl_file}...")
        with open(pkl_file, 'rb') as f:
            dataset = pickle.load(f)
        return dataset
    elif os.path.exists(csv_file):
        print(f"[OK] Cargando dataset desde {csv_file}...")
        dataset = pd.read_csv(csv_file, index_col=0, parse_dates=True)
        return dataset
    else:
        raise FileNotFoundError(f"No se encontró ml_dataset.pkl ni ml_dataset.csv en {data_dir}")

def prepare_data(dataset, target_geography=None):
    """
    Prepara los datos para entrenamiento.
    
    Parameters:
    -----------
    dataset : pd.DataFrame
        Dataset completo con features y targets
    target_geography : str, optional
        Si se especifica, entrena modelo solo para esa geografía.
        Si None, entrena modelos separados para cada geografía.
    
    Returns:
    --------
    dict: Diccionario con datos preparados por geografía
    """
    # Separar features y targets
    feature_cols = [col for col in dataset.columns if not col.startswith('target_')]
    target_cols = [col for col in dataset.columns if col.startswith('target_')]
    
    print(f"\nFeatures disponibles: {len(feature_cols)}")
    print(f"Targets disponibles: {len(target_cols)}")
    
    # Extraer nombres de geografías de los targets
    geographies = []
    for col in target_cols:
        # Formato: target_{GEO}_sharpe
        geo = col.replace('target_', '').replace('_sharpe', '')
        geographies.append(geo)
    
    print(f"Geografías encontradas: {geographies}")
    
    # Preparar datos por geografía
    prepared_data = {}
    
    for geo in geographies:
        target_col = f'target_{geo}_sharpe'
        
        if target_geography and geo != target_geography:
            continue
        
        if target_col not in dataset.columns:
            print(f"[WARNING] Target {target_col} no encontrado, saltando {geo}")
            continue
        
        # Extraer features y target
        X = dataset[feature_cols].copy()
        y = dataset[target_col].copy()
        
        # Eliminar filas donde el target es NaN
        valid_mask = y.notna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) == 0:
            print(f"[WARNING] No hay datos válidos para {geo}, saltando")
            continue
        
        # Eliminar features con demasiados NaN (>50%)
        missing_pct = X.isna().sum() / len(X)
        valid_features = missing_pct[missing_pct < 0.5].index
        X = X[valid_features]
        
        print(f"\n{geo}:")
        print(f"  Observaciones válidas: {len(X):,}")
        print(f"  Features válidas (antes de selección): {len(X.columns)}")
        print(f"  Rango de fechas: {X.index.min()} a {X.index.max()}")
        print(f"  Target - Media: {y.mean():.4f}, Std: {y.std():.4f}")
        
        prepared_data[geo] = {
            'X': X,
            'y': y,
            'feature_names': X.columns.tolist()
        }
    
    return prepared_data

def split_data_temporal(X, y, train_years=10, val_years=4, test_years=5):
    """
    Divide los datos en train/validation/test respetando el orden temporal.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target
    train_years : int
        Años de datos para entrenamiento
    val_years : int
        Años de datos para validación
    test_years : int
        Años de datos para test
    
    Returns:
    --------
    dict: Diccionario con splits
    """
    # Ordenar por fecha
    X = X.sort_index()
    y = y.sort_index()
    
    # Calcular fechas de corte
    start_date = X.index.min()
    end_date = X.index.max()
    
    # Calcular días (asumiendo ~252 días hábiles por año)
    train_days = int(train_years * 252)
    val_days = int(val_years * 252)
    test_days = int(test_years * 252)
    
    # Dividir por fecha
    dates = X.index.unique().sort_values()
    
    if len(dates) < train_days + val_days + test_days:
        # Si no hay suficientes datos, ajustar proporciones
        total_days = len(dates)
        train_days = int(total_days * 0.6)
        val_days = int(total_days * 0.2)
        test_days = total_days - train_days - val_days
        print(f"[INFO] Ajustando splits: train={train_days}, val={val_days}, test={test_days}")
    
    train_end_idx = train_days
    val_end_idx = train_days + val_days
    
    train_dates = dates[:train_end_idx]
    val_dates = dates[train_end_idx:val_end_idx]
    test_dates = dates[val_end_idx:val_end_idx+test_days] if val_end_idx+test_days <= len(dates) else dates[val_end_idx:]
    
    X_train = X.loc[train_dates]
    y_train = y.loc[train_dates]
    X_val = X.loc[val_dates]
    y_val = y.loc[val_dates]
    X_test = X.loc[test_dates]
    y_test = y.loc[test_dates]
    
    print(f"\nDivisión temporal:")
    print(f"  Train: {X_train.index.min()} a {X_train.index.max()} ({len(X_train):,} observaciones)")
    print(f"  Val:   {X_val.index.min()} a {X_val.index.max()} ({len(X_val):,} observaciones)")
    print(f"  Test:  {X_test.index.min()} a {X_test.index.max()} ({len(X_test):,} observaciones)")
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test
    }

def train_xgboost(X_train, y_train, X_val, y_val):
    """Entrena modelo XGBoost con regularización mejorada"""
    if xgb is None:
        return None
    
    print("\nEntrenando XGBoost...")
    
    # Manejar NaN en features (XGBoost puede manejarlos, pero es mejor imputar)
    X_train_clean = X_train.fillna(X_train.median())
    X_val_clean = X_val.fillna(X_train.median())
    
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=4,  # Reducido de 6 a 4 para reducir overfitting
        learning_rate=0.03,  # Reducido para mejor generalización
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,  # L1 regularization
        reg_lambda=1.0,  # L2 regularization
        min_child_weight=5,  # Aumentado para más regularización
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50,  # Más agresivo (de 20 a 50)
        eval_metric='rmse'
    )
    
    model.fit(
        X_train_clean, y_train,
        eval_set=[(X_val_clean, y_val)],
        verbose=False
    )
    
    return model

def train_lightgbm(X_train, y_train, X_val, y_val):
    """Entrena modelo LightGBM con regularización mejorada"""
    if lgb is None:
        return None
    
    print("\nEntrenando LightGBM...")
    
    # Manejar NaN
    X_train_clean = X_train.fillna(X_train.median())
    X_val_clean = X_val.fillna(X_train.median())
    
    train_data = lgb.Dataset(X_train_clean, label=y_train)
    val_data = lgb.Dataset(X_val_clean, label=y_val, reference=train_data)
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 15,  # Reducido de 31 a 15 para menos overfitting
        'learning_rate': 0.03,  # Reducido de 0.05
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 1.0,  # L1 regularization
        'lambda_l2': 1.0,  # L2 regularization
        'min_child_samples': 20,  # Aumentado para más regularización
        'verbose': -1,
        'random_state': 42
    }
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=300,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)]  # Más agresivo
    )
    
    return model

def train_random_forest(X_train, y_train, X_val, y_val):
    """Entrena modelo Random Forest con regularización mejorada"""
    print("\nEntrenando Random Forest...")
    
    # Manejar NaN
    X_train_clean = X_train.fillna(X_train.median())
    X_val_clean = X_val.fillna(X_train.median())
    
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,  # Reducido de 10 a 8
        min_samples_split=10,  # Aumentado de 5 a 10
        min_samples_leaf=4,  # Aumentado de 2 a 4
        max_features='sqrt',  # Limitar features por árbol
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_clean, y_train)
    
    return model

def evaluate_model(model, X, y, model_name='Model', is_lgbm=False):
    """Evalúa un modelo y retorna métricas"""
    # Manejar NaN
    X_clean = X.fillna(X.median())
    
    # Predecir
    if is_lgbm:
        y_pred = model.predict(X_clean)
    else:
        y_pred = model.predict(X_clean)
    
    # Calcular métricas
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    correlation = np.corrcoef(y, y_pred)[0, 1]
    
    metrics = {
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'correlation': correlation
    }
    
    print(f"\n{model_name} - Métricas:")
    print(f"  R²:           {r2:.4f}")
    print(f"  MAE:          {mae:.4f}")
    print(f"  RMSE:         {rmse:.4f}")
    print(f"  Correlación:  {correlation:.4f}")
    
    return metrics, y_pred

def select_features(X_train, y_train, X_val, k=50):
    """
    Selecciona las k mejores features usando SelectKBest.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Features de entrenamiento
    y_train : pd.Series
        Target de entrenamiento
    X_val : pd.DataFrame
        Features de validación
    k : int
        Número de features a seleccionar (default: 50)
    
    Returns:
    --------
    tuple: (X_train_selected, X_val_selected, selector, selected_features)
    """
    # Imputar NaN antes de feature selection
    X_train_imputed = X_train.fillna(X_train.median())
    X_val_imputed = X_val.fillna(X_train.median())
    
    # Si ya tenemos menos de k features, no hacer selección
    if X_train_imputed.shape[1] <= k:
        print(f"  [INFO] Ya tenemos {X_train_imputed.shape[1]} features (<= {k}), saltando selección")
        return X_train_imputed, X_val_imputed, None, X_train_imputed.columns.tolist()
    
    # Seleccionar features usando f_regression
    print(f"  Seleccionando top {k} features de {X_train_imputed.shape[1]} disponibles...")
    selector = SelectKBest(score_func=f_regression, k=min(k, X_train_imputed.shape[1]))
    
    X_train_selected = selector.fit_transform(X_train_imputed, y_train)
    X_val_selected = selector.transform(X_val_imputed)
    
    # Obtener nombres de features seleccionadas
    selected_mask = selector.get_support()
    selected_features = X_train_imputed.columns[selected_mask].tolist()
    
    print(f"  [OK] Seleccionadas {len(selected_features)} features")
    
    # Convertir de array a DataFrame manteniendo nombres
    X_train_selected = pd.DataFrame(X_train_selected, index=X_train_imputed.index, columns=selected_features)
    X_val_selected = pd.DataFrame(X_val_selected, index=X_val_imputed.index, columns=selected_features)
    
    return X_train_selected, X_val_selected, selector, selected_features

def train_models_for_geography(geo, data_dict, models_dir='models', n_features=50):
    """Entrena modelos para una geografía específica"""
    print("\n" + "=" * 80)
    print(f"ENTRENANDO MODELOS PARA: {geo}")
    print("=" * 80)
    
    X = data_dict['X']
    y = data_dict['y']
    
    # Dividir datos
    splits = split_data_temporal(X, y)
    
    # ESTRATEGIA 2: Feature selection (reducir a 50 features)
    print(f"\n[ESTRATEGIA 2] Selección de features (top {n_features})...")
    X_train_selected, X_val_selected, feature_selector, selected_features = select_features(
        splits['X_train'], splits['y_train'], splits['X_val'], k=n_features
    )
    
    # Actualizar splits con features seleccionadas
    splits['X_train'] = X_train_selected
    splits['X_val'] = X_val_selected
    splits['X_test'] = splits['X_test'][selected_features].fillna(splits['X_train'].median())
    
    # Entrenar múltiples modelos
    models = {}
    results = {}
    
    # 1. XGBoost
    if xgb is not None:
        model_xgb = train_xgboost(
            splits['X_train'], splits['y_train'],
            splits['X_val'], splits['y_val']
        )
        if model_xgb is not None:
            models['xgb'] = model_xgb
            results['xgb'] = {}
            results['xgb']['val'], _ = evaluate_model(
                model_xgb, splits['X_val'], splits['y_val'], 'XGBoost (Val)'
            )
            results['xgb']['test'], _ = evaluate_model(
                model_xgb, splits['X_test'], splits['y_test'], 'XGBoost (Test)'
            )
    
    # 2. LightGBM
    if lgb is not None:
        model_lgbm = train_lightgbm(
            splits['X_train'], splits['y_train'],
            splits['X_val'], splits['y_val']
        )
        if model_lgbm is not None:
            models['lgbm'] = model_lgbm
            results['lgbm'] = {}
            results['lgbm']['val'], _ = evaluate_model(
                model_lgbm, splits['X_val'], splits['y_val'], 'LightGBM (Val)', is_lgbm=True
            )
            results['lgbm']['test'], _ = evaluate_model(
                model_lgbm, splits['X_test'], splits['y_test'], 'LightGBM (Test)', is_lgbm=True
            )
    
    # 3. Random Forest
    model_rf = train_random_forest(
        splits['X_train'], splits['y_train'],
        splits['X_val'], splits['y_val']
    )
    models['rf'] = model_rf
    results['rf'] = {}
    results['rf']['val'], _ = evaluate_model(
        model_rf, splits['X_val'], splits['y_val'], 'Random Forest (Val)'
    )
    results['rf']['test'], _ = evaluate_model(
        model_rf, splits['X_test'], splits['y_test'], 'Random Forest (Test)'
    )
    
    # ESTRATEGIA 6: Ensemble de Modelos
    # En lugar de seleccionar solo el mejor, crear ensemble
    print(f"\n[ESTRATEGIA 6] Creando ensemble de modelos...")
    
    # Calcular pesos del ensemble basados en R² de validación
    ensemble_weights = {}
    total_r2 = 0
    
    for model_name, result in results.items():
        val_r2 = result['val']['r2']
        if val_r2 > 0:  # Solo incluir modelos con R² positivo
            ensemble_weights[model_name] = val_r2
            total_r2 += val_r2
    
    # Normalizar pesos
    if total_r2 > 0:
        ensemble_weights = {k: v / total_r2 for k, v in ensemble_weights.items()}
        print(f"  Pesos del ensemble:")
        for model_name, weight in sorted(ensemble_weights.items(), key=lambda x: x[1], reverse=True):
            print(f"    {model_name}: {weight:.3f}")
    else:
        # Si todos tienen R² negativo, usar pesos iguales
        ensemble_weights = {k: 1.0/len(results) for k in results.keys()}
        print(f"  [WARNING] Todos los modelos tienen R² negativo, usando pesos iguales")
    
    # Seleccionar mejor modelo individual también (para comparación)
    best_model_name = None
    best_r2 = -np.inf
    
    for model_name, result in results.items():
        val_r2 = result['val']['r2']
        if val_r2 > best_r2:
            best_r2 = val_r2
            best_model_name = model_name
    
    print(f"\n{'='*80}")
    print(f"MEJOR MODELO INDIVIDUAL: {best_model_name.upper()} (R² val: {best_r2:.4f})")
    print(f"{'='*80}")
    
    # Guardar todos los modelos y el ensemble
    os.makedirs(models_dir, exist_ok=True)
    
    # Guardar mejor modelo individual
    best_model = models[best_model_name]
    model_file = os.path.join(models_dir, f'sharpe_predictor_{geo}_{best_model_name}.pkl')
    if best_model_name == 'lgbm':
        best_model.save_model(model_file.replace('.pkl', '.txt'))
        with open(model_file, 'wb') as f:
            pickle.dump(best_model, f)
    else:
        with open(model_file, 'wb') as f:
            pickle.dump(best_model, f)
    
    print(f"[OK] Mejor modelo guardado en {model_file}")
    
    # Guardar ensemble (todos los modelos + pesos)
    ensemble_file = os.path.join(models_dir, f'ensemble_{geo}.pkl')
    ensemble_data = {
        'models': models,
        'weights': ensemble_weights,
        'best_model_name': best_model_name,
        'feature_selector': feature_selector,
        'selected_features': selected_features
    }
    with open(ensemble_file, 'wb') as f:
        pickle.dump(ensemble_data, f)
    print(f"[OK] Ensemble guardado en {ensemble_file}")
    
    # Guardar métricas
    metrics_file = os.path.join(models_dir, f'metrics_{geo}.json')
    import json
    with open(metrics_file, 'w') as f:
        json.dump({
            'geography': geo,
            'best_model': best_model_name,
            'best_r2_val': best_r2,
            'results': {k: {split: {m: float(v) for m, v in metrics.items()} 
                           for split, metrics in splits.items()} 
                       for k, splits in results.items()}
        }, f, indent=2)
    
    print(f"[OK] Métricas guardadas en {metrics_file}")
    
    return {
        'best_model': best_model,
        'best_model_name': best_model_name,
        'models': models,
        'results': results,
        'splits': splits,
        'ensemble_weights': ensemble_weights,
        'selected_features': selected_features
    }

def main():
    """Función principal"""
    print("=" * 80)
    print("ENTRENAMIENTO DE MODELOS PARA PREDECIR SHARPE RATIO FUTURO")
    print("=" * 80)
    
    data_dir = 'data'
    models_dir = 'models'
    
    # 1. Cargar dataset
    print("\n1. Cargando dataset ML...")
    dataset = load_ml_dataset(data_dir)
    print(f"   [OK] Dataset cargado: {dataset.shape[0]} filas, {dataset.shape[1]} columnas")
    
    # 2. Preparar datos
    print("\n2. Preparando datos...")
    prepared_data = prepare_data(dataset)
    
    if len(prepared_data) == 0:
        print("[ERROR] No se encontraron datos válidos para entrenar")
        return
    
    # 3. Entrenar modelos para cada geografía
    print("\n3. Entrenando modelos por geografía...")
    all_results = {}
    
    # ESTRATEGIA 2: Usar 50 features (configurable)
    n_features = 50
    print(f"\n[CONFIGURACIÓN] Usando top {n_features} features por geografía")
    
    for geo, data_dict in prepared_data.items():
        try:
            results = train_models_for_geography(geo, data_dict, models_dir, n_features=n_features)
            all_results[geo] = results
        except Exception as e:
            print(f"[ERROR] Error entrenando modelo para {geo}: {e}")
            import traceback
            traceback.print_exc()
    
    # 4. Resumen final
    print("\n" + "=" * 80)
    print("RESUMEN FINAL")
    print("=" * 80)
    
    for geo, results in all_results.items():
        best_model = results['best_model_name']
        test_r2 = results['results'][best_model]['test']['r2']
        test_corr = results['results'][best_model]['test']['correlation']
        print(f"\n{geo}:")
        print(f"  Mejor modelo: {best_model}")
        print(f"  R² (test): {test_r2:.4f}")
        print(f"  Correlación (test): {test_corr:.4f}")
    
    print("\n" + "=" * 80)
    print("ENTRENAMIENTO COMPLETADO")
    print("=" * 80)
    print(f"\nModelos guardados en: {models_dir}/")
    print("Usa estos modelos en optimize_portfolio.py para optimizar asignaciones.")

if __name__ == "__main__":
    main()
