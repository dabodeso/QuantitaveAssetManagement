"""
Script para analizar y mostrar estadísticas de las features generadas

Este script:
1. Lista todas las features generadas
2. Muestra estadísticas descriptivas de cada feature (media, mediana, desviación, etc.)
3. Identifica features con muchos valores faltantes
4. Genera un reporte completo
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_ml_dataset(data_dir='data'):
    """Carga el dataset ML completo"""
    # Intentar cargar desde pickle (más rápido)
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
        print(f"[WARNING] No se encontró ml_dataset.pkl ni ml_dataset.csv en {data_dir}")
        print("Intentando cargar desde diccionarios individuales...")
        return load_from_dicts(data_dir)

def load_from_dicts(data_dir='data'):
    """Carga features desde diccionarios individuales y crea dataset"""
    dataset = pd.DataFrame()
    
    # Cargar features individuales
    features_file = os.path.join(data_dir, 'ml_features_dict.pkl')
    if os.path.exists(features_file):
        with open(features_file, 'rb') as f:
            features_dict = pickle.load(f)
        
        print(f"[OK] Cargadas features individuales para {len(features_dict)} activos")
        for symbol, features_df in features_dict.items():
            for col in features_df.columns:
                if symbol in ['VIX', 'DXY', 'TNX_10Y', 'IRX_3M', 'YIELD_SPREAD']:
                    dataset[f'{symbol}_{col}'] = features_df[col]
                else:
                    dataset[f'{symbol}_{col}'] = features_df[col]
    
    # Cargar features de geografía
    geo_file = os.path.join(data_dir, 'geography_features_dict.pkl')
    if os.path.exists(geo_file):
        with open(geo_file, 'rb') as f:
            geo_features = pickle.load(f)
        
        print(f"[OK] Cargadas features de geografía para {len(geo_features)} geografías")
        for geo, features_df in geo_features.items():
            for col in features_df.columns:
                dataset[col] = features_df[col]
    
    # Cargar targets
    target_file = os.path.join(data_dir, 'target_by_geography_dict.pkl')
    if os.path.exists(target_file):
        with open(target_file, 'rb') as f:
            target_dict = pickle.load(f)
        
        print(f"[OK] Cargados targets para {len(target_dict)} geografías")
        for geo, target_series in target_dict.items():
            dataset[f'target_{geo}_sharpe'] = target_series
    
    return dataset

def calculate_statistics(series):
    """Calcula estadísticas descriptivas de una serie"""
    # Filtrar valores no nulos
    valid_data = series.dropna()
    
    if len(valid_data) == 0:
        return {
            'count': 0,
            'null_count': len(series),
            'null_pct': 100.0,
            'mean': np.nan,
            'median': np.nan,
            'std': np.nan,
            'min': np.nan,
            'max': np.nan,
            'p25': np.nan,
            'p50': np.nan,
            'p75': np.nan,
            'p90': np.nan,
            'p95': np.nan,
            'skewness': np.nan,
            'kurtosis': np.nan
        }
    
    stats_dict = {
        'count': len(valid_data),
        'null_count': len(series) - len(valid_data),
        'null_pct': (len(series) - len(valid_data)) / len(series) * 100,
        'mean': valid_data.mean(),
        'median': valid_data.median(),
        'std': valid_data.std(),
        'min': valid_data.min(),
        'max': valid_data.max(),
        'p25': valid_data.quantile(0.25),
        'p50': valid_data.quantile(0.50),
        'p75': valid_data.quantile(0.75),
        'p90': valid_data.quantile(0.90),
        'p95': valid_data.quantile(0.95),
        'skewness': valid_data.skew(),
        'kurtosis': valid_data.kurtosis()
    }
    
    return stats_dict

def analyze_features(dataset, output_file=None):
    """Analiza todas las features y genera reporte"""
    print("\n" + "=" * 100)
    print("ANÁLISIS DE FEATURES GENERADAS")
    print("=" * 100)
    
    if dataset.empty:
        print("[ERROR] El dataset está vacío. Ejecuta generate_ml_features.py primero.")
        return
    
    print(f"\nDataset cargado: {dataset.shape[0]} filas, {dataset.shape[1]} columnas")
    print(f"Rango de fechas: {dataset.index.min()} a {dataset.index.max()}")
    
    # Separar features y targets
    feature_cols = [col for col in dataset.columns if not col.startswith('target_')]
    target_cols = [col for col in dataset.columns if col.startswith('target_')]
    
    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Targets: {len(target_cols)}")
    
    # Calcular estadísticas para cada feature
    print("\n" + "=" * 100)
    print("ESTADÍSTICAS DESCRIPTIVAS POR FEATURE")
    print("=" * 100)
    
    all_stats = []
    
    for col in feature_cols:
        stats = calculate_statistics(dataset[col])
        stats['feature'] = col
        all_stats.append(stats)
    
    # Crear DataFrame con todas las estadísticas
    stats_df = pd.DataFrame(all_stats)
    
    # Reordenar columnas
    column_order = ['feature', 'count', 'null_count', 'null_pct', 'mean', 'median', 'std', 
                    'min', 'p25', 'p50', 'p75', 'p90', 'p95', 'max', 'skewness', 'kurtosis']
    stats_df = stats_df[column_order]
    
    # Mostrar resumen compacto primero
    print("\n" + "-" * 100)
    print("RESUMEN COMPACTO DE FEATURES")
    print("-" * 100)
    print(f"\n{'Feature':<50} {'Valores':<12} {'Faltantes %':<12} {'Media':<12} {'Mediana':<12} {'Desv. Est.':<12}")
    print("-" * 100)
    
    for _, row in stats_df.iterrows():
        feature_name = row['feature'][:48]  # Truncar si es muy largo
        count = int(row['count'])
        null_pct = row['null_pct']
        mean = row['mean'] if not pd.isna(row['mean']) else np.nan
        median = row['median'] if not pd.isna(row['median']) else np.nan
        std = row['std'] if not pd.isna(row['std']) else np.nan
        
        mean_str = f"{mean:.4f}" if not pd.isna(mean) else "NaN"
        median_str = f"{median:.4f}" if not pd.isna(median) else "NaN"
        std_str = f"{std:.4f}" if not pd.isna(std) else "NaN"
        
        print(f"{feature_name:<50} {count:<12,} {null_pct:<12.2f} {mean_str:<12} {median_str:<12} {std_str:<12}")
    
    # Mostrar estadísticas detalladas
    print("\n" + "-" * 100)
    print("ESTADÍSTICAS DETALLADAS POR FEATURE")
    print("-" * 100)
    
    # Configurar pandas para mostrar más columnas
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 50)
    pd.set_option('display.float_format', lambda x: f'{x:.4f}' if not pd.isna(x) else 'NaN')
    
    # Mostrar todas las features con sus estadísticas
    for idx, row in stats_df.iterrows():
        print(f"\n{'='*100}")
        print(f"FEATURE: {row['feature']}")
        print(f"{'='*100}")
        print(f"  Valores válidos: {int(row['count']):,} ({100-row['null_pct']:.2f}%)")
        print(f"  Valores faltantes: {int(row['null_count']):,} ({row['null_pct']:.2f}%)")
        
        if row['count'] > 0:
            print("\n  Estadísticas centrales:")
            print(f"    Media:        {row['mean']:>12.4f}")
            print(f"    Mediana:      {row['median']:>12.4f}")
            print(f"    Desv. Est.:   {row['std']:>12.4f}")
            
            print("\n  Rango:")
            print(f"    Mínimo:       {row['min']:>12.4f}")
            print(f"    Máximo:       {row['max']:>12.4f}")
            
            print("\n  Percentiles:")
            print(f"    P25 (Q1):     {row['p25']:>12.4f}")
            print(f"    P50 (Mediana): {row['p50']:>12.4f}")
            print(f"    P75 (Q3):     {row['p75']:>12.4f}")
            print(f"    P90:          {row['p90']:>12.4f}")
            print(f"    P95:          {row['p95']:>12.4f}")
            
            print("\n  Forma de distribución:")
            print(f"    Asimetría:    {row['skewness']:>12.4f}")
            print(f"    Curtosis:     {row['kurtosis']:>12.4f}")
        else:
            print("  [WARNING] No hay valores válidos para esta feature")
    
    # Resumen de targets
    if target_cols:
        print("\n" + "=" * 100)
        print("ESTADÍSTICAS DE TARGETS (VARIABLES OBJETIVO)")
        print("=" * 100)
        
        for col in target_cols:
            stats = calculate_statistics(dataset[col])
            print(f"\n{'-'*100}")
            print(f"TARGET: {col}")
            print(f"{'-'*100}")
            print(f"  Valores válidos: {int(stats['count']):,} ({100-stats['null_pct']:.2f}%)")
            print(f"  Valores faltantes: {int(stats['null_count']):,} ({stats['null_pct']:.2f}%)")
            
            if stats['count'] > 0:
                print("\n  Estadísticas:")
                print(f"    Media:        {stats['mean']:>12.4f}")
                print(f"    Mediana:      {stats['median']:>12.4f}")
                print(f"    Desv. Est.:   {stats['std']:>12.4f}")
                print(f"    Mínimo:       {stats['min']:>12.4f}")
                print(f"    Máximo:       {stats['max']:>12.4f}")
                print(f"    P25:          {stats['p25']:>12.4f}")
                print(f"    P75:          {stats['p75']:>12.4f}")
    
    # Resumen general
    print("\n" + "=" * 100)
    print("RESUMEN GENERAL")
    print("=" * 100)
    
    print(f"\nTotal de features: {len(feature_cols)}")
    print(f"Total de targets: {len(target_cols)}")
    print(f"Total de observaciones: {len(dataset):,}")
    
    # Features con muchos valores faltantes
    high_missing = stats_df[stats_df['null_pct'] > 50]
    if len(high_missing) > 0:
        print(f"\n[ADVERTENCIA] Features con más del 50% de valores faltantes: {len(high_missing)}")
        for _, row in high_missing.iterrows():
            print(f"  - {row['feature']}: {row['null_pct']:.2f}% faltantes")
    
    # Features sin valores faltantes
    no_missing = stats_df[stats_df['null_pct'] == 0]
    print(f"\nFeatures sin valores faltantes: {len(no_missing)}")
    
    # Guardar reporte en archivo
    if output_file:
        print(f"\n[OK] Guardando reporte completo en {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write("REPORTE DE ANÁLISIS DE FEATURES\n")
            f.write("=" * 100 + "\n")
            f.write(f"\nFecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"\nDataset: {dataset.shape[0]} filas, {dataset.shape[1]} columnas\n")
            f.write(f"Rango de fechas: {dataset.index.min()} a {dataset.index.max()}\n")
            
            f.write("\n" + "=" * 100 + "\n")
            f.write("ESTADÍSTICAS POR FEATURE\n")
            f.write("=" * 100 + "\n\n")
            f.write(stats_df.to_string())
            
            if target_cols:
                f.write("\n\n" + "=" * 100 + "\n")
                f.write("ESTADÍSTICAS DE TARGETS\n")
                f.write("=" * 100 + "\n\n")
                for col in target_cols:
                    stats = calculate_statistics(dataset[col])
                    f.write(f"\n{col}:\n")
                    for key, value in stats.items():
                        if key != 'feature':
                            f.write(f"  {key}: {value}\n")
        
        print(f"[OK] Reporte guardado exitosamente")
    
    # Guardar también CSV con estadísticas
    csv_output = output_file.replace('.txt', '.csv') if output_file else 'features_statistics.csv'
    stats_df.to_csv(csv_output, index=False)
    print(f"[OK] Estadísticas guardadas en CSV: {csv_output}")
    
    return stats_df

def main():
    """Función principal"""
    data_dir = 'data'
    
    print("=" * 100)
    print("ANÁLISIS DE FEATURES GENERADAS")
    print("=" * 100)
    
    # Cargar dataset
    print("\n1. Cargando dataset ML...")
    dataset = load_ml_dataset(data_dir)
    
    if dataset.empty:
        print("\n❌ No se pudo cargar el dataset. Asegúrate de ejecutar generate_ml_features.py primero.")
        return
    
    # Analizar features
    print("\n2. Analizando features...")
    analyze_features(dataset, output_file=os.path.join(data_dir, 'features_analysis_report.txt'))
    
    print("\n" + "=" * 100)
    print("ANÁLISIS COMPLETADO")
    print("=" * 100)
    print("\nRevisa el reporte generado para ver todas las estadísticas detalladas.")

if __name__ == "__main__":
    main()
