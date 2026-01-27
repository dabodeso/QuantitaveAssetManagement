"""
Script de Visualización y Exploración de Datos

Este script genera visualizaciones completas de los datos para:
1. Detectar valores faltantes (NaN)
2. Visualizar distribuciones
3. Analizar correlaciones
4. Explorar series temporales
5. Verificar calidad de datos
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('ggplot')

sns.set_palette("husl")

def load_data(data_dir='data'):
    """Carga todos los datos disponibles"""
    data = {}
    
    # 1. Retornos de ETFs
    etf_file = os.path.join(data_dir, 'etf_returns_dict.pkl')
    if os.path.exists(etf_file):
        with open(etf_file, 'rb') as f:
            data['etf_returns'] = pickle.load(f)
        print(f"[OK] Cargados {len(data['etf_returns'])} ETFs")
    
    # 2. Features técnicas
    features_file = os.path.join(data_dir, 'ml_features_dict.pkl')
    if os.path.exists(features_file):
        with open(features_file, 'rb') as f:
            data['technical_features'] = pickle.load(f)
        print(f"[OK] Cargadas features técnicas para {len(data['technical_features'])} activos")
    
    # 3. Features de geografía
    geo_file = os.path.join(data_dir, 'geography_features_dict.pkl')
    if os.path.exists(geo_file):
        with open(geo_file, 'rb') as f:
            data['geography_features'] = pickle.load(f)
        print(f"[OK] Cargadas features de geografía para {len(data['geography_features'])} geografías")
    
    # 4. Dataset ML completo
    ml_file = os.path.join(data_dir, 'ml_dataset.pkl')
    if os.path.exists(ml_file):
        data['ml_dataset'] = pd.read_pickle(ml_file)
        print(f"[OK] Dataset ML cargado: {data['ml_dataset'].shape}")
    else:
        # Intentar CSV
        ml_csv = os.path.join(data_dir, 'ml_dataset.csv')
        if os.path.exists(ml_csv):
            data['ml_dataset'] = pd.read_csv(ml_csv, index_col=0, parse_dates=True)
            print(f"[OK] Dataset ML cargado desde CSV: {data['ml_dataset'].shape}")
    
    # 5. Targets
    targets_file = os.path.join(data_dir, 'target_by_geography_dict.pkl')
    if os.path.exists(targets_file):
        with open(targets_file, 'rb') as f:
            data['targets'] = pickle.load(f)
        print(f"[OK] Cargados targets para {len(data['targets'])} geografías")
    
    return data

def analyze_missing_data(df, title="Dataset"):
    """Analiza y visualiza datos faltantes"""
    print(f"\n{'='*80}")
    print(f"ANÁLISIS DE DATOS FALTANTES: {title}")
    print(f"{'='*80}")
    
    # Estadísticas básicas
    total_cells = df.size
    missing_cells = df.isna().sum().sum()
    missing_pct = (missing_cells / total_cells) * 100
    
    print(f"\nResumen General:")
    print(f"  Total de celdas: {total_cells:,}")
    print(f"  Celdas faltantes: {missing_cells:,} ({missing_pct:.2f}%)")
    print(f"  Filas: {len(df)}")
    print(f"  Columnas: {len(df.columns)}")
    
    # Por columna
    missing_by_col = df.isna().sum()
    missing_pct_by_col = (missing_by_col / len(df)) * 100
    
    print(f"\nTop 20 Columnas con más datos faltantes:")
    top_missing = missing_by_col.sort_values(ascending=False).head(20)
    for col, count in top_missing.items():
        pct = missing_pct_by_col[col]
        print(f"  {col}: {count:,} ({pct:.2f}%)")
    
    # Por fila
    missing_by_row = df.isna().sum(axis=1)
    missing_pct_by_row = (missing_by_row / len(df.columns)) * 100
    
    print(f"\nFilas con datos faltantes:")
    print(f"  Filas con al menos 1 NaN: {(missing_by_row > 0).sum()} ({(missing_by_row > 0).sum()/len(df)*100:.2f}%)")
    print(f"  Filas completamente vacías: {(missing_by_row == len(df.columns)).sum()}")
    print(f"  Promedio de NaN por fila: {missing_by_row.mean():.2f}")
    
    return {
        'total_cells': total_cells,
        'missing_cells': missing_cells,
        'missing_pct': missing_pct,
        'missing_by_col': missing_by_col,
        'missing_by_row': missing_by_row
    }

def plot_missing_data_heatmap(df, title="Datos Faltantes", save_path=None):
    """Crea heatmap de datos faltantes"""
    # Validar que el DataFrame tenga datos
    if df.empty or len(df) == 0:
        print(f"  [WARNING] DataFrame vacío para {title}. No se puede generar heatmap.")
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    
    # 1. Heatmap de datos faltantes (muestra solo una muestra de columnas si hay muchas)
    if len(df.columns) > 50:
        # Muestra solo las columnas con más datos faltantes
        missing_by_col = df.isna().sum().sort_values(ascending=False)
        top_cols = missing_by_col.head(50).index.tolist()
        df_sample = df[top_cols]
        title_suffix = " (Top 50 columnas con más NaN)"
    else:
        df_sample = df
        title_suffix = ""
    
    # Muestra solo una muestra de filas si hay muchas
    if len(df) > 1000:
        step = len(df) // 1000
        df_sample = df_sample.iloc[::step]
        title_suffix += f" (muestra de {len(df_sample)} filas)"
    
    # Heatmap
    sns.heatmap(df_sample.isna(), yticklabels=False, cbar=True, 
                cmap='viridis', ax=axes[0])
    axes[0].set_title(f'Heatmap de Datos Faltantes{title_suffix}', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Columnas', fontsize=12)
    axes[0].set_ylabel('Filas (muestra)', fontsize=12)
    
    # 2. Porcentaje de datos faltantes por columna
    missing_pct = (df.isna().sum() / len(df)) * 100
    top_missing = missing_pct.sort_values(ascending=False).head(30)
    
    axes[1].barh(range(len(top_missing)), top_missing.values)
    axes[1].set_yticks(range(len(top_missing)))
    axes[1].set_yticklabels(top_missing.index, fontsize=8)
    axes[1].set_xlabel('Porcentaje de Datos Faltantes (%)', fontsize=12)
    axes[1].set_title('Top 30 Columnas con Más Datos Faltantes', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Guardado: {save_path}")
    
    plt.close()

def plot_data_coverage(df, title="Cobertura de Datos", save_path=None):
    """Visualiza la cobertura temporal de los datos"""
    # Validar que el DataFrame tenga datos
    if df.empty or len(df) == 0:
        print(f"  [WARNING] DataFrame vacío para {title}. No se puede generar gráfico de cobertura.")
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # 1. Cobertura por columna (fechas con datos)
    coverage_by_col = {}
    for col in df.columns:
        non_null = df[col].notna()
        if non_null.any():
            first_date = df[non_null].index[0]
            last_date = df[non_null].index[-1]
            coverage_by_col[col] = {
                'first': first_date,
                'last': last_date,
                'days': (last_date - first_date).days,
                'pct': non_null.sum() / len(df) * 100
            }
    
    # Ordenar por porcentaje de cobertura
    sorted_cols = sorted(coverage_by_col.items(), 
                        key=lambda x: x[1]['pct'], 
                        reverse=True)[:30]
    
    # Gráfico de barras de cobertura
    cols = [x[0] for x in sorted_cols]
    pcts = [x[1]['pct'] for x in sorted_cols]
    
    axes[0].barh(range(len(cols)), pcts)
    axes[0].set_yticks(range(len(cols)))
    axes[0].set_yticklabels(cols, fontsize=8)
    axes[0].set_xlabel('Porcentaje de Cobertura (%)', fontsize=12)
    axes[0].set_title('Cobertura de Datos por Columna (Top 30)', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')
    axes[0].axvline(x=100, color='red', linestyle='--', alpha=0.5, label='100% Cobertura')
    axes[0].legend()
    
    # 2. Timeline de cobertura (para algunas columnas clave)
    key_cols = [col for col in df.columns if any(x in col.lower() for x in ['spy', 'returns', 'target', 'vix', 'dxy'])]
    if len(key_cols) > 10:
        key_cols = key_cols[:10]
    
    for col in key_cols:
        if col in coverage_by_col:
            first = coverage_by_col[col]['first']
            last = coverage_by_col[col]['last']
            pct = coverage_by_col[col]['pct']
            axes[1].plot([first, last], [col, col], 
                        linewidth=3, alpha=0.7, 
                        label=f"{col} ({pct:.1f}%)")
    
    axes[1].set_xlabel('Fecha', fontsize=12)
    axes[1].set_ylabel('Columna', fontsize=12)
    axes[1].set_title('Timeline de Cobertura (Columnas Clave)', fontsize=14, fontweight='bold')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Guardado: {save_path}")
    
    plt.close()

def plot_distributions(df, title="Distribuciones", save_path=None, max_cols=20):
    """Visualiza distribuciones de las variables"""
    # Validar que el DataFrame tenga datos
    if df.empty or len(df) == 0:
        print(f"  [WARNING] DataFrame vacío para {title}. No se puede generar gráfico de distribuciones.")
        return
    
    # Seleccionar columnas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) > max_cols:
        # Seleccionar columnas con menos NaN
        missing_pct = (df[numeric_cols].isna().sum() / len(df)) * 100
        numeric_cols = missing_pct.nsmallest(max_cols).index.tolist()
    
    n_cols = min(4, len(numeric_cols))
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for idx, col in enumerate(numeric_cols):
        ax = axes[idx]
        data = df[col].dropna()
        
        if len(data) > 0:
            ax.hist(data, bins=50, alpha=0.7, edgecolor='black')
            ax.set_title(f'{col}\n(n={len(data)}, NaN={df[col].isna().sum()})', 
                        fontsize=9, fontweight='bold')
            ax.set_xlabel('Valor', fontsize=8)
            ax.set_ylabel('Frecuencia', fontsize=8)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Sin datos', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(col, fontsize=9)
    
    # Ocultar ejes extra
    for idx in range(len(numeric_cols), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Distribuciones de Variables: {title}', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Guardado: {save_path}")
    
    plt.close()

def plot_correlations(df, title="Correlaciones", save_path=None, max_cols=30):
    """Visualiza matriz de correlación"""
    # Validar que el DataFrame tenga datos
    if df.empty or len(df) == 0:
        print(f"  [WARNING] DataFrame vacío para {title}. No se puede generar matriz de correlación.")
        return
    
    # Seleccionar solo columnas numéricas
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Si hay muchas columnas, seleccionar las más importantes
    if len(numeric_df.columns) > max_cols:
        # Seleccionar columnas con menos NaN y más variabilidad
        missing_pct = (numeric_df.isna().sum() / len(numeric_df)) * 100
        std_values = numeric_df.std()
        # Score: baja missing, alta std
        score = (100 - missing_pct) * std_values
        top_cols = score.nlargest(max_cols).index.tolist()
        numeric_df = numeric_df[top_cols]
    
    # Calcular correlación (solo donde hay suficientes datos)
    corr_matrix = numeric_df.corr()
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    sns.heatmap(corr_matrix, annot=False, fmt='.2f', cmap='coolwarm', 
                center=0, vmin=-1, vmax=1, square=True, 
                linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    
    ax.set_title(f'Matriz de Correlación: {title}', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Guardado: {save_path}")
    
    plt.close()

def plot_time_series(df, title="Series Temporales", save_path=None, max_cols=10):
    """Visualiza series temporales de las variables"""
    # Validar que el DataFrame tenga datos
    if df.empty or len(df) == 0:
        print(f"  [WARNING] DataFrame vacío para {title}. No se puede generar gráfico de series temporales.")
        return
    
    if not isinstance(df.index, pd.DatetimeIndex):
        print("[WARNING] El índice no es de tipo fecha. No se pueden graficar series temporales.")
        return
    
    # Seleccionar columnas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) > max_cols:
        # Seleccionar columnas clave
        key_words = ['returns', 'target', 'sharpe', 'vix', 'dxy', 'spread']
        selected = []
        for word in key_words:
            selected.extend([col for col in numeric_cols if word in col.lower()])
        selected = list(set(selected))[:max_cols]
        if len(selected) < max_cols:
            # Completar con columnas con menos NaN
            missing_pct = (df[numeric_cols].isna().sum() / len(df)) * 100
            remaining = missing_pct.nsmallest(max_cols - len(selected)).index.tolist()
            selected.extend(remaining)
        numeric_cols = selected[:max_cols]
    
    n_cols = min(2, len(numeric_cols))
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for idx, col in enumerate(numeric_cols):
        ax = axes[idx]
        data = df[col].dropna()
        
        if len(data) > 0:
            ax.plot(data.index, data.values, linewidth=1, alpha=0.7)
            ax.set_title(f'{col}\n(n={len(data)}, NaN={df[col].isna().sum()})', 
                        fontsize=10, fontweight='bold')
            ax.set_xlabel('Fecha', fontsize=9)
            ax.set_ylabel('Valor', fontsize=9)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Sin datos', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(col, fontsize=10)
    
    # Ocultar ejes extra
    for idx in range(len(numeric_cols), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Series Temporales: {title}', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Guardado: {save_path}")
    
    plt.close()

def generate_summary_report(data, output_dir='data/visualizations'):
    """Genera reporte completo de visualización"""
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("GENERANDO REPORTE DE VISUALIZACIÓN")
    print("="*80)
    
    # 1. Dataset ML completo
    if 'ml_dataset' in data:
        df = data['ml_dataset']
        print(f"\n1. Analizando Dataset ML ({df.shape[0]} filas, {df.shape[1]} columnas)...")
        
        if df.empty or len(df) == 0:
            print("  [WARNING] Dataset ML está vacío. Esto puede indicar que:")
            print("    - generate_ml_features.py no se ejecutó correctamente")
            print("    - Hubo un error al crear el dataset")
            print("    - Los datos no se alinearon correctamente")
            print("  [INFO] Continuando con otros datasets...")
        else:
            # Análisis de datos faltantes
            missing_stats = analyze_missing_data(df, "Dataset ML")
            
            # Visualizaciones
            plot_missing_data_heatmap(df, "Dataset ML", 
                                     os.path.join(output_dir, '1_missing_data_heatmap.png'))
            plot_data_coverage(df, "Dataset ML", 
                              os.path.join(output_dir, '2_data_coverage.png'))
            plot_distributions(df, "Dataset ML", 
                              os.path.join(output_dir, '3_distributions.png'))
            plot_correlations(df, "Dataset ML", 
                             os.path.join(output_dir, '4_correlations.png'))
            plot_time_series(df, "Dataset ML", 
                            os.path.join(output_dir, '5_time_series.png'))
    
    # 2. Retornos de ETFs
    if 'etf_returns' in data:
        returns_dict = data['etf_returns']
        returns_df = pd.DataFrame(returns_dict)
        returns_df = returns_df.dropna(how='all')
        
        print(f"\n2. Analizando Retornos de ETFs ({returns_df.shape[0]} filas, {returns_df.shape[1]} columnas)...")
        
        missing_stats = analyze_missing_data(returns_df, "Retornos de ETFs")
        
        plot_missing_data_heatmap(returns_df, "Retornos de ETFs", 
                                 os.path.join(output_dir, '6_etf_returns_missing.png'))
        plot_time_series(returns_df, "Retornos de ETFs", 
                        os.path.join(output_dir, '7_etf_returns_timeseries.png'))
        plot_correlations(returns_df, "Retornos de ETFs", 
                        os.path.join(output_dir, '8_etf_returns_correlations.png'))
    
    # 3. Features técnicas (muestra algunas)
    if 'technical_features' in data:
        features_dict = data['technical_features']
        print(f"\n3. Analizando Features Técnicas ({len(features_dict)} activos)...")
        
        # Combinar todas las features en un DataFrame
        all_features = []
        for symbol, features_df in list(features_dict.items())[:5]:  # Primeros 5 activos
            for col in features_df.columns:
                all_features.append(features_df[col].rename(f'{symbol}_{col}'))
        
        if all_features:
            features_combined = pd.concat(all_features, axis=1)
            features_combined = features_combined.dropna(how='all')
            
            print(f"   Muestra: {features_combined.shape[0]} filas, {features_combined.shape[1]} columnas")
            missing_stats = analyze_missing_data(features_combined, "Features Técnicas (muestra)")
            
            plot_missing_data_heatmap(features_combined, "Features Técnicas", 
                                     os.path.join(output_dir, '9_technical_features_missing.png'))
    
    # 4. Targets
    if 'targets' in data:
        targets_dict = data['targets']
        targets_df = pd.DataFrame(targets_dict)
        targets_df = targets_df.dropna(how='all')
        
        print(f"\n4. Analizando Targets ({targets_df.shape[0]} filas, {targets_df.shape[1]} columnas)...")
        
        missing_stats = analyze_missing_data(targets_df, "Targets")
        
        plot_missing_data_heatmap(targets_df, "Targets", 
                                 os.path.join(output_dir, '10_targets_missing.png'))
        plot_time_series(targets_df, "Targets", 
                        os.path.join(output_dir, '11_targets_timeseries.png'))
        plot_distributions(targets_df, "Targets", 
                          os.path.join(output_dir, '12_targets_distributions.png'))
    
    print(f"\n{'='*80}")
    print("REPORTE COMPLETADO")
    print(f"{'='*80}")
    print(f"\n[OK] Todas las visualizaciones guardadas en: {output_dir}/")
    print(f"     Total de archivos generados: {len(os.listdir(output_dir))}")

def main():
    """Función principal"""
    data_dir = 'data'
    output_dir = 'data/visualizations'
    
    print("="*80)
    print("VISUALIZACIÓN Y EXPLORACIÓN DE DATOS")
    print("="*80)
    
    # Cargar datos
    print("\nCargando datos...")
    data = load_data(data_dir)
    
    if not data:
        print("\n[ERROR] No se encontraron datos. Ejecuta primero:")
        print("  1. download_etf_data.py")
        print("  2. generate_ml_features.py")
        return
    
    # Generar reporte
    generate_summary_report(data, output_dir)
    
    print("\n[OK] Proceso completado!")

if __name__ == "__main__":
    main()
