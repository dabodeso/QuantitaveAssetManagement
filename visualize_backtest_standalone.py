"""
Script standalone para visualizar resultados del backtesting
Ejecutar: python visualize_backtest_standalone.py
"""

import pandas as pd
import numpy as np
import pickle
import os

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("[ERROR] matplotlib no está instalado. Instala con: pip install matplotlib")
    exit(1)

def load_backtest_results(data_dir='data'):
    """Carga los resultados del backtesting"""
    results_file = os.path.join(data_dir, 'backtest_results.pkl')
    
    if not os.path.exists(results_file):
        print(f"[ERROR] No se encontró {results_file}")
        print("Ejecuta backtest_strategy.py primero para generar los resultados.")
        return None
    
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    
    return results

def visualize_backtest_results(portfolio_returns, results_dict, data_dir='data', save_dir='data/visualizations'):
    """
    Genera visualizaciones de los resultados del backtesting.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\nGenerando visualizaciones en {save_dir}...")
    
    # 1. Curva de equity (valor acumulado)
    print("  1. Curva de equity...")
    cumulative = (1 + portfolio_returns / 100).cumprod()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1.1 Curva de equity
    ax1 = axes[0, 0]
    ax1.plot(cumulative.index, cumulative.values, linewidth=2, label='Portafolio Optimizado', color='blue')
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Break-even')
    ax1.set_title('Curva de Equity - Portafolio Optimizado', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Fecha')
    ax1.set_ylabel('Valor Acumulado (Base = 1.0)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 1.2 Retornos diarios
    ax2 = axes[0, 1]
    ax2.plot(portfolio_returns.index, portfolio_returns.values, alpha=0.6, linewidth=0.5, color='blue')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax2.set_title('Retornos Diarios del Portafolio', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Fecha')
    ax2.set_ylabel('Retorno Diario (%)')
    ax2.grid(True, alpha=0.3)
    
    # 1.3 Distribución de retornos
    ax3 = axes[1, 0]
    ax3.hist(portfolio_returns.values, bins=50, alpha=0.7, edgecolor='black', color='blue')
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
    output_file = os.path.join(save_dir, 'backtest_portfolio_analysis.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"      [OK] Guardado: {output_file}")
    plt.close()
    
    # 2. Comparación con benchmarks
    print("  2. Comparación con benchmarks...")
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
                output_file = os.path.join(save_dir, 'backtest_vs_benchmark.png')
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                print(f"      [OK] Guardado: {output_file}")
                plt.close()
    except Exception as e:
        print(f"      [WARNING] No se pudo generar comparación con benchmarks: {e}")
    
    # 3. División train/test
    print("  3. Gráfico train/test...")
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
        output_file = os.path.join(save_dir, 'backtest_train_test.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"      [OK] Guardado: {output_file}")
        plt.close()
        
        # Calcular métricas por período
        def calculate_simple_metrics(returns):
            """Calcula métricas simples"""
            returns_pct = returns
            cumulative = (1 + returns_pct / 100).cumprod()
            total_return = cumulative.iloc[-1] - 1
            years = len(returns) / 252
            annualized = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
            volatility = returns_pct.std() * np.sqrt(252)
            sharpe = (annualized * 100) / volatility if volatility > 0 else 0
            
            return {
                'annualized_return': annualized,
                'volatility': volatility,
                'sharpe_ratio': sharpe
            }
        
        train_metrics = calculate_simple_metrics(train_returns)
        test_metrics = calculate_simple_metrics(test_returns)
        
        print(f"\n  Métricas Train:")
        print(f"    Retorno Anualizado: {train_metrics['annualized_return']*100:.2f}%")
        print(f"    Sharpe Ratio: {train_metrics['sharpe_ratio']:.4f}")
        print(f"    Volatilidad: {train_metrics['volatility']:.2f}%")
        
        print(f"\n  Métricas Test:")
        print(f"    Retorno Anualizado: {test_metrics['annualized_return']*100:.2f}%")
        print(f"    Sharpe Ratio: {test_metrics['sharpe_ratio']:.4f}")
        print(f"    Volatilidad: {test_metrics['volatility']:.2f}%")
        
    except Exception as e:
        print(f"      [WARNING] Error generando gráfico train/test: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n[OK] Visualizaciones completadas. Revisa los archivos en {save_dir}/")

def main():
    """Función principal"""
    print("=" * 80)
    print("VISUALIZACIÓN DE RESULTADOS DE BACKTESTING")
    print("=" * 80)
    
    data_dir = 'data'
    
    # Cargar resultados
    print("\n1. Cargando resultados del backtesting...")
    results = load_backtest_results(data_dir)
    
    if results is None:
        return
    
    portfolio_returns = results['portfolio_returns']
    
    print(f"   [OK] Retornos cargados: {len(portfolio_returns)} observaciones")
    print(f"   Rango: {portfolio_returns.index.min()} a {portfolio_returns.index.max()}")
    
    # Generar visualizaciones
    print("\n2. Generando visualizaciones...")
    visualize_backtest_results(portfolio_returns, results, data_dir)
    
    print("\n" + "=" * 80)
    print("VISUALIZACIÓN COMPLETADA")
    print("=" * 80)

if __name__ == "__main__":
    main()
