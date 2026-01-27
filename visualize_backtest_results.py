"""
Funciones para visualizar resultados del backtesting
"""

import pandas as pd
import numpy as np
import os
import pickle

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("[WARNING] matplotlib no está instalado. Instala con: pip install matplotlib")

def visualize_backtest_results(portfolio_returns, results_dict, data_dir='data', save_dir='data/visualizations'):
    """
    Genera visualizaciones de los resultados del backtesting.
    
    Parameters:
    -----------
    portfolio_returns : pd.Series
        Retornos del portafolio optimizado
    results_dict : dict
        Diccionario con resultados del backtesting
    data_dir : str
        Directorio de datos
    save_dir : str
        Directorio para guardar gráficos
    """
    if not MATPLOTLIB_AVAILABLE:
        print("    [WARNING] matplotlib no disponible, saltando visualizaciones")
        return
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Curva de equity (valor acumulado)
    print("  Generando curva de equity...")
    cumulative = (1 + portfolio_returns / 100).cumprod()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1.1 Curva de equity
    ax1 = axes[0, 0]
    ax1.plot(cumulative.index, cumulative.values, linewidth=2, label='Portafolio Optimizado')
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Break-even')
    ax1.set_title('Curva de Equity - Portafolio Optimizado', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Fecha')
    ax1.set_ylabel('Valor Acumulado (Base = 1.0)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 1.2 Retornos diarios
    ax2 = axes[0, 1]
    ax2.plot(portfolio_returns.index, portfolio_returns.values, alpha=0.6, linewidth=0.5)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax2.set_title('Retornos Diarios del Portafolio', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Fecha')
    ax2.set_ylabel('Retorno Diario (%)')
    ax2.grid(True, alpha=0.3)
    
    # 1.3 Distribución de retornos
    ax3 = axes[1, 0]
    ax3.hist(portfolio_returns.values, bins=50, alpha=0.7, edgecolor='black')
    ax3.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    ax3.axvline(x=portfolio_returns.mean(), color='green', linestyle='--', 
                label=f'Media: {portfolio_returns.mean():.3f}%')
    ax3.set_title('Distribución de Retornos Diarios', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Retorno Diario (%)')
    ax3.set_ylabel('Frecuencia')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 1.4 Drawdown
    ax4 = axes[1, 1]
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max * 100
    ax4.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
    ax4.plot(drawdown.index, drawdown.values, color='red', linewidth=1)
    ax4.set_title('Drawdown del Portafolio', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Fecha')
    ax4.set_ylabel('Drawdown (%)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'backtest_portfolio_analysis.png'), dpi=300, bbox_inches='tight')
    print(f"    [OK] Gráfico guardado: {save_dir}/backtest_portfolio_analysis.png")
    plt.close()
    
    # 2. Comparación con benchmarks
    print("  Generando comparación con benchmarks...")
    try:
        with open(os.path.join(data_dir, 'etf_returns_dict.pkl'), 'rb') as f:
            returns_dict = pickle.load(f)
        
        # SPY
        if 'SPY' in returns_dict:
            spy_returns = returns_dict['SPY']
            # Normalizar índice
            if isinstance(spy_returns.index, pd.DatetimeIndex):
                if spy_returns.index.tz is not None:
                    spy_returns.index = spy_returns.index.tz_localize(None)
                spy_returns.index = spy_returns.index.normalize()
            
            # Alinear fechas
            common_dates = portfolio_returns.index.intersection(spy_returns.index)
            if len(common_dates) > 0:
                portfolio_aligned = portfolio_returns.loc[common_dates]
                spy_aligned = spy_returns.loc[common_dates]
                
                # Calcular valores acumulados
                portfolio_cum = (1 + portfolio_aligned / 100).cumprod()
                spy_cum = (1 + spy_aligned / 100).cumprod()
                
                fig, ax = plt.subplots(figsize=(14, 8))
                ax.plot(portfolio_cum.index, portfolio_cum.values, linewidth=2, 
                       label='Portafolio Optimizado', color='blue')
                ax.plot(spy_cum.index, spy_cum.values, linewidth=2, 
                       label='SPY (Benchmark)', color='orange')
                ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
                ax.set_title('Comparación: Portafolio Optimizado vs SPY', 
                           fontsize=16, fontweight='bold')
                ax.set_xlabel('Fecha')
                ax.set_ylabel('Valor Acumulado (Base = 1.0)')
                ax.legend(fontsize=12)
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, 'backtest_vs_benchmark.png'), 
                          dpi=300, bbox_inches='tight')
                print(f"    [OK] Gráfico guardado: {save_dir}/backtest_vs_benchmark.png")
                plt.close()
    except Exception as e:
        print(f"    [WARNING] No se pudo generar comparación con benchmarks: {e}")
    
    # 3. División train/test
    print("  Generando gráfico train/test...")
    try:
        # Dividir en train (70%) y test (30%)
        split_idx = int(len(portfolio_returns) * 0.7)
        train_returns = portfolio_returns.iloc[:split_idx]
        test_returns = portfolio_returns.iloc[split_idx:]
        
        train_cum = (1 + train_returns / 100).cumprod()
        test_cum = (1 + test_returns / 100).cumprod()
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Train
        ax1 = axes[0]
        ax1.plot(train_cum.index, train_cum.values, linewidth=2, color='blue', label='Train')
        ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_title(f'Período de Entrenamiento (Train) - {len(train_returns)} días', 
                     fontsize=14, fontweight='bold')
        ax1.set_ylabel('Valor Acumulado')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Test
        ax2 = axes[1]
        ax2.plot(test_cum.index, test_cum.values, linewidth=2, color='red', label='Test')
        ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_title(f'Período de Prueba (Test) - {len(test_returns)} días', 
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel('Fecha')
        ax2.set_ylabel('Valor Acumulado')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'backtest_train_test.png'), 
                   dpi=300, bbox_inches='tight')
        print(f"    [OK] Gráfico guardado: {save_dir}/backtest_train_test.png")
        plt.close()
        
        # Calcular métricas por período
        from backtest_strategy import calculate_metrics
        train_metrics = calculate_metrics(train_returns)
        test_metrics = calculate_metrics(test_returns)
        
        print(f"\n  Métricas Train:")
        print(f"    Retorno Anualizado: {train_metrics['annualized_return']*100:.2f}%")
        print(f"    Sharpe Ratio: {train_metrics['sharpe_ratio']:.4f}")
        print(f"    Volatilidad: {train_metrics['volatility']:.2f}%")
        
        print(f"\n  Métricas Test:")
        print(f"    Retorno Anualizado: {test_metrics['annualized_return']*100:.2f}%")
        print(f"    Sharpe Ratio: {test_metrics['sharpe_ratio']:.4f}")
        print(f"    Volatilidad: {test_metrics['volatility']:.2f}%")
        
    except Exception as e:
        print(f"    [WARNING] Error generando gráfico train/test: {e}")
        import traceback
        traceback.print_exc()
